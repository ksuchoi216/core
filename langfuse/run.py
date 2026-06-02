from __future__ import annotations

import os
from typing import Any

from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from loguru import logger


DEFAULT_USER_ID = "anonymous"
FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _resolve_user_id(user_id: str | None) -> str:
    return DEFAULT_USER_ID if user_id is None else user_id


def _langfuse_enabled() -> bool:
    return os.getenv("LANGFUSE_ENABLED", "true").strip().lower() not in FALSE_ENV_VALUES


def _create_langfuse_config(
    *, max_concurrency: int | None = None
) -> dict[str, Any]:
    config: dict[str, Any] = {"callbacks": [CallbackHandler()]}
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    return config


def _create_plain_config(*, max_concurrency: int | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    return config


def _run_graph(
    graph,
    state,
    *,
    langfuse_config: dict[str, Any],
):
    configured_graph = graph.with_config(langfuse_config)

    is_batch = isinstance(state, list)
    if is_batch and len(state) == 1:
        is_batch = False
        state = state[0]

    if is_batch:
        logger.info("Running graph in batch mode with {} states.", len(state))

    if is_batch:
        return configured_graph.batch(state)

    return configured_graph.invoke(state)


def _run_generator(
    generator,
    input_data,
    *,
    langfuse_config: dict[str, Any],
):
    is_batch = isinstance(input_data, list)
    if is_batch and len(input_data) == 1:
        is_batch = False
        input_data = input_data[0]

    if is_batch:
        return generator.batch(input_data, config=langfuse_config)

    return generator.invoke(input_data, config=langfuse_config)


@observe
def _run_graph_with_langfuse_observed(
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
    resolved_user_id = _resolve_user_id(user_id)

    with propagate_attributes(
        trace_name=trace_name,
        session_id=session_id,
        user_id=resolved_user_id,
        tags=tags or [],
    ):
        return _run_graph(graph, state, langfuse_config=langfuse_config)


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
    if not _langfuse_enabled():
        logger.info("Langfuse tracing disabled by LANGFUSE_ENABLED.")
        return _run_graph(
            graph,
            state,
            langfuse_config=_create_plain_config(max_concurrency=max_concurrency),
        )

    return _run_graph_with_langfuse_observed(
        graph,
        state,
        trace_name=trace_name,
        session_id=session_id,
        user_id=user_id,
        tags=tags,
        max_concurrency=max_concurrency,
    )


@observe
def _run_generator_with_langfuse_observed(
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

    with propagate_attributes(
        trace_name=trace_name,
        session_id=session_id,
        user_id=resolved_user_id,
        tags=tags or [],
    ):
        return _run_generator(
            generator,
            input_data,
            langfuse_config=langfuse_config,
        )


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
    if not _langfuse_enabled():
        logger.info("Langfuse tracing disabled by LANGFUSE_ENABLED.")
        return _run_generator(
            generator,
            input_data,
            langfuse_config=_create_plain_config(max_concurrency=max_concurrency),
        )

    return _run_generator_with_langfuse_observed(
        generator,
        input_data,
        trace_name=trace_name,
        session_id=session_id,
        user_id=user_id,
        tags=tags,
        max_concurrency=max_concurrency,
    )


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
