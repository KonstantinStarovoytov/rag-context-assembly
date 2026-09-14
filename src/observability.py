"""Optional Langfuse tracing; never creates a client when tracing is disabled."""

from collections.abc import Callable
from contextlib import nullcontext
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from functools import lru_cache, wraps
from typing import Any

from src.config import settings

last_trace_id = ContextVar("last_trace_id", default=None)


@lru_cache(maxsize=1)
def get_langfuse():
    from langfuse import Langfuse

    missing = []
    if not settings.langfuse_public_key:
        missing.append("LANGFUSE_PUBLIC_KEY")
    if settings.langfuse_secret_key is None:
        missing.append("LANGFUSE_SECRET_KEY")
    if missing:
        raise ValueError(
            f"{' and '.join(missing)} must be set when Langfuse is enabled"
        )

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
    )


def serialize(value):
    if is_dataclass(value):
        return serialize(asdict(value))
    if hasattr(value, "model_dump"):
        return serialize(value.model_dump())
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    return value


def traced(
    name: str,
    kind: str = "span",
    *,
    input_factory: Callable[..., Any] | None = None,
    metadata_factory: Callable[..., dict[str, Any]] | None = None,
    output_factory: Callable[[Any], Any] | None = None,
):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if not settings.tracing_enabled:
                return fn(*args, **kwargs)
            inputs = (
                serialize(input_factory(*args, **kwargs))
                if input_factory is not None
                else {
                    "args": [
                        serialize(a)
                        for a in args
                        if not callable(a) and not hasattr(a, "__dict__")
                    ],
                    "kwargs": {
                        k: serialize(v) for k, v in kwargs.items() if not callable(v)
                    },
                }
            )
            observation = {
                "name": name,
                "as_type": kind,
                "input": inputs,
            }
            if metadata_factory is not None:
                observation["metadata"] = serialize(metadata_factory(*args, **kwargs))
            with get_langfuse().start_as_current_observation(**observation) as span:
                last_trace_id.set(span.trace_id)
                try:
                    result = fn(*args, **kwargs)
                except Exception as error:
                    # Keep failed CLI/API calls diagnosable from the same trace URL.
                    span.update(level="ERROR", status_message=str(error))
                    raise
                output = output_factory(result) if output_factory else result
                span.update(output=serialize(output))
                return result

        return wrapped

    return decorate


def model_config(run_name: str | None = None):
    if not settings.tracing_enabled:
        return {}
    from langfuse.langchain import CallbackHandler

    get_langfuse()
    if not settings.langfuse_public_key:
        raise ValueError("LANGFUSE_PUBLIC_KEY must be set when tracing is enabled")
    config = {"callbacks": [CallbackHandler(public_key=settings.langfuse_public_key)]}
    if run_name:
        config["run_name"] = run_name
    return config


def prompt_context(managed_prompt):
    if not settings.tracing_enabled:
        return nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(
        prompt=managed_prompt.langfuse_prompt,
        metadata={
            k: str(v) for k, v in managed_prompt.metadata.items() if v is not None
        },
    )


def flush():
    if settings.tracing_enabled:
        get_langfuse().flush()


def trace_url():
    trace_id = last_trace_id.get()
    return (
        get_langfuse().get_trace_url(trace_id=trace_id)
        if settings.tracing_enabled and trace_id
        else None
    )
