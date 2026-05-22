from __future__ import annotations

from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from loguru import logger


@observe
def run_graph_with_langfuse(
    graph,
    state,
    *,
    trace_name,
    session_id,
    user_id=None,
    tags=None,
    # call_type: Literal["batch", "invoke"] = "invoke",
    is_batch: bool = False,
):
    langfuse_handler = CallbackHandler()

    graph = graph.with_config(
        {
            "callbacks": [langfuse_handler],
        }
    )
    if user_id is None:
        user_id = "anonymous"

    if is_batch:
        # check state is list for batch call
        if not isinstance(state, list):
            raise ValueError("State must be a list for batch call.")
        # log length of state for batch call
        logger.info("Running graph in batch mode with {} states.", len(state))

    with propagate_attributes(
        trace_name=trace_name,
        session_id=session_id,
        user_id=user_id,
        tags=tags or [],
    ):
        if not is_batch:
            return graph.invoke(state)
        else:
            return graph.batch(state)

        # langfuse.set_current_trace_io(
        #     input=state,
        #     output=result,
        # )


@observe
def run_with_langfuse(
    generator,
    input_data,
    *,
    trace_name,
    session_id,
    user_id: str | None = None,
    tags: list[str] | None = None,
    # call_type: Literal["batch", "invoke"] = "invoke",
    is_batch: bool = False,
):
    langfuse_handler = CallbackHandler()
    if user_id is None:
        user_id = "anonymous"

    if is_batch:
        # check state is list for batch call
        if not isinstance(input_data, list):
            raise ValueError("State must be a list for batch call.")

    with propagate_attributes(
        trace_name=trace_name,
        session_id=session_id,
        user_id=user_id,
        tags=tags or [],
    ):
        if not is_batch:
            return generator.invoke(
                input_data, config={"callbacks": [langfuse_handler]}
            )
        else:
            return generator.batch(
                [input_data], config={"callbacks": [langfuse_handler]}
            )
