from __future__ import annotations

from typing import Any

from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from loguru import logger


DEFAULT_USER_ID = "anonymous"


def _resolve_user_id(user_id: str | None) -> str:
    return DEFAULT_USER_ID if user_id is None else user_id


def _create_langfuse_config(
    *, max_concurrency: int | None = None
) -> dict[str, Any]:
    config: dict[str, Any] = {"callbacks": [CallbackHandler()]}
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    return config


@observe
def run_graph_with_langfuse(
    graph,
    state,
    *,
    trace_name,
    session_id,
    user_id: str | None = None,
    tags: list[str] | None = None,
    max_concurrency: int | None = None,
):
    langfuse_config = _create_langfuse_config(max_concurrency=max_concurrency)
    configured_graph = graph.with_config(langfuse_config)
    resolved_user_id = _resolve_user_id(user_id)

    is_batch = isinstance(state, list)
    if is_batch and len(state) == 1:
        is_batch = False
        state = state[0]

    if is_batch:
        logger.info("Running graph in batch mode with {} states.", len(state))

    with propagate_attributes(
        trace_name=trace_name,
        session_id=session_id,
        user_id=resolved_user_id,
        tags=tags or [],
    ):
        if is_batch:
            return configured_graph.batch(state)

        return configured_graph.invoke(state)


@observe
def run_generator_with_langfuse(
    generator,
    input_data,
    *,
    trace_name,
    session_id,
    user_id: str | None = None,
    tags: list[str] | None = None,
    max_concurrency: int | None = None,
):
    langfuse_config = _create_langfuse_config(max_concurrency=max_concurrency)
    resolved_user_id = _resolve_user_id(user_id)

    is_batch = isinstance(input_data, list)
    if is_batch and len(input_data) == 1:
        is_batch = False
        input_data = input_data[0]

    with propagate_attributes(
        trace_name=trace_name,
        session_id=session_id,
        user_id=resolved_user_id,
        tags=tags or [],
    ):
        if is_batch:
            return generator.batch(input_data, config=langfuse_config)

        return generator.invoke(input_data, config=langfuse_config)


@observe
def run_with_langfuse(
    run_func,
    input_data,
    *,
    trace_name,
    session_id,
    user_id: str | None = None,
    tags: list[str] | None = None,
    max_concurrency: int | None = None,
):
    is_graph = type(run_func).__name__ in ("CompiledGraph", "CompiledStateGraph")
    langfuse_runner = (
        run_graph_with_langfuse if is_graph else run_generator_with_langfuse
    )

    return langfuse_runner(
        run_func,
        input_data,
        trace_name=trace_name,
        session_id=session_id,
        user_id=user_id,
        tags=tags,
        max_concurrency=max_concurrency,
    )
