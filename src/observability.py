"""Optional Langfuse tracing; never creates a client when tracing is disabled."""

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from functools import lru_cache, wraps
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar, cast

from langchain_core.runnables.config import RunnableConfig

from src.config import settings

if TYPE_CHECKING:
    from src.prompts.managed import ManagedPrompt

P = ParamSpec("P")
R = TypeVar("R")

last_trace_id: ContextVar[str | None] = ContextVar("last_trace_id", default=None)


@lru_cache(maxsize=1)
def get_langfuse() -> Any:
    from langfuse import Langfuse

    missing: list[str] = []
    if not settings.langfuse_public_key:
        missing.append("LANGFUSE_PUBLIC_KEY")
    secret_key = settings.langfuse_secret_key
    if secret_key is None:
        missing.append("LANGFUSE_SECRET_KEY")
    if missing:
        raise ValueError(
            f"{' and '.join(missing)} must be set when Langfuse is enabled"
        )
    assert secret_key is not None

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=secret_key.get_secret_value(),
        base_url=settings.langfuse_base_url,
    )


def _propagate_attributes(**attributes: Any) -> AbstractContextManager[Any]:
    from langfuse import propagate_attributes

    return propagate_attributes(**attributes)


def serialize(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return serialize(asdict(value))
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return serialize(model_dump())
    if isinstance(value, Mapping):
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
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
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
            observation: dict[str, Any] = {
                "name": name,
                "as_type": kind,
                "input": inputs,
            }
            if metadata_factory is not None:
                observation["metadata"] = serialize(metadata_factory(*args, **kwargs))
            client = get_langfuse()
            # v4 is observations-first: the trace name is only searchable on
            # child observations when propagated from the root scope.
            is_root = client.get_current_trace_id() is None
            with (
                client.start_as_current_observation(**observation) as span,
                _propagate_attributes(trace_name=name) if is_root else nullcontext(),
            ):
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


def model_config(run_name: str | None = None) -> RunnableConfig:
    if not settings.tracing_enabled:
        return {}
    from langfuse.langchain import CallbackHandler

    get_langfuse()
    if not settings.langfuse_public_key:
        raise ValueError("LANGFUSE_PUBLIC_KEY must be set when tracing is enabled")
    config: dict[str, Any] = {
        "callbacks": [CallbackHandler(public_key=settings.langfuse_public_key)]
    }
    if run_name:
        config["run_name"] = run_name
    return cast(RunnableConfig, config)


def prompt_context(managed_prompt: "ManagedPrompt") -> AbstractContextManager[Any]:
    if not settings.tracing_enabled:
        return nullcontext()
    from langfuse import propagate_attributes

    return propagate_attributes(
        prompt=managed_prompt.langfuse_prompt,
        metadata={
            k: str(v) for k, v in managed_prompt.metadata.items() if v is not None
        },
    )


def flush() -> None:
    if settings.tracing_enabled:
        get_langfuse().flush()


def trace_url() -> str | None:
    trace_id = last_trace_id.get()
    return (
        get_langfuse().get_trace_url(trace_id=trace_id)
        if settings.tracing_enabled and trace_id
        else None
    )
