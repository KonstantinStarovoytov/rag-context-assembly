"""Evaluate saved answers without running retrieval or generation again."""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Evaluation, Langfuse
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


# Any date before the first experiment; the API requires a lower bound.
EXPERIMENTS_SINCE = datetime(2026, 9, 1, tzinfo=timezone.utc)


def _as_json(value):
    """Experiment items return input/output as JSON strings; the retired
    trace endpoint returned them parsed. Accept both."""
    if isinstance(value, str):
        return json.loads(value)
    return value


def _experiment_items(client, run_name):
    """Every item of one experiment run, via the non-legacy Experiments API.

    Replaces per-item `client.api.trace.get()`, which Langfuse v4 retires on
    2026-11-16 and rate-limits to 5 req/min on Hobby from 2026-09-21 - a
    16-item run would already throttle. One paginated call instead of one
    legacy call per item.
    """
    cursor = None
    while True:
        kwargs = {
            "experiment_name": run_name,
            "from_start_time": EXPERIMENTS_SINCE,
            "fields": "io",
            "limit": 100,
        }
        if cursor:
            kwargs["cursor"] = cursor
        response = client.api.experiments.list_items(**kwargs)
        yield from response.data
        cursor = getattr(getattr(response, "meta", None), "cursor", None)
        if not cursor:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dataset", default="rag/evidence-coverage-v1")
    parser.add_argument("--judge-model", default="gpt-5.6-luna")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    client = Langfuse()
    rows = []
    for item in _experiment_items(client, args.run_name):
        output = _as_json(item.output)
        if not isinstance(output, dict) or not output.get("contexts"):
            raise ValueError(f"Missing saved context for {item.trace_id}")
        rows.append(
            {
                "input": _as_json(item.input),
                "expected_output": None,
                "metadata": {"source_trace_id": item.trace_id},
                "saved_output": output,
            }
        )
    if not rows:
        raise ValueError(f"No items found for experiment run {args.run_name!r}")

    async def evaluate(*, input, output, **_kwargs):
        async with AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]) as api:
            llm = llm_factory(args.judge_model, client=api)
            score = await Faithfulness(llm=llm).ascore(
                user_input=input["question"],
                response=output["answer"],
                retrieved_contexts=[x["content"] for x in output["contexts"]],
            )
        value = float(score.value)
        if not math.isfinite(value):
            raise ValueError("Ragas returned an undefined faithfulness score")
        print(f"Faithfulness={value:.3f}: {input['question']}", flush=True)
        return Evaluation(
            name="ragas_faithfulness",
            value=value,
            metadata={"ragas_version": "0.4.3", "judge_model": args.judge_model},
        )

    result = client.run_experiment(
        name="ragas-saved-answer-faithfulness-v1",
        data=rows,
        task=lambda *, item, **kwargs: item["saved_output"],
        evaluators=[evaluate],
        max_concurrency=1,
        metadata={
            "source_run": args.run_name,
            "judge_model": args.judge_model,
            "ragas_version": "0.4.3",
            "reused_answers": True,
        },
    )
    print(result.format())
    for item in result.item_results:
        print(client.get_trace_url(trace_id=item.trace_id))
    client.flush()
    if any(not item.evaluations for item in result.item_results):
        raise RuntimeError("Some answers were not scored; inspect experiment errors")


if __name__ == "__main__":
    main()
