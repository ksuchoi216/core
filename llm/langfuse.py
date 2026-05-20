from __future__ import annotations

from pathlib import Path

from core.support.file import load_file
import os

from langchain_core.prompts import BasePromptTemplate
from langfuse import observe, propagate_attributes
from langfuse.langchain import CallbackHandler
from loguru import logger

PROMPT_FILE_SUFFIX = ".txt"
DEFAULT_LOCAL_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def _resolve_local_prompt_path(
    prompt_key: str,
    prompt_dir: str | Path | None = None,
) -> Path:
    base_path = Path(prompt_dir) if prompt_dir is not None else DEFAULT_LOCAL_PROMPT_DIR
    if base_path.is_file():
        return base_path
    return base_path / f"{prompt_key}{PROMPT_FILE_SUFFIX}"


#
def load_prompt(
    prompt_key: str,
    *,
    local_prompt: bool = False,
    prompt_dir: str | Path | None = None,
) -> BasePromptTemplate | str:
    if local_prompt or prompt_dir is not None:
        prompt_path = _resolve_local_prompt_path(prompt_key, prompt_dir)
        prompt_text = prompt_path.read_text(encoding="utf-8")
        logger.info("Prompt {} loaded from local file {}.", prompt_key, prompt_path)
        return prompt_text

    try:
        from langfuse import get_client
    except ImportError as exc:
        raise RuntimeError(
            "Langfuse is required to load prompt keys. Install langfuse or monkeypatch "
            "load_prompt() in tests."
        ) from exc

    prompt = get_client().get_prompt(prompt_key).get_langchain_prompt()
    logger.info("Prompt {} loaded from Langfuse.", prompt_key)
    # log 200 char of prompt
    logger.info("Prompt {} preview: {}.", prompt_key, prompt[:200])
    if isinstance(prompt, BasePromptTemplate):
        return prompt
    if isinstance(prompt, str):
        return prompt
    raise TypeError(f"Unsupported Langfuse prompt type: {type(prompt)!r}")


def download_prompt(
    prompt_key: str,
    *,
    prompt_dir: str | Path | None = None,
) -> None:
    try:
        from langfuse import get_client
    except ImportError as exc:
        raise RuntimeError(
            "Langfuse is required to download prompt keys. Install langfuse first."
        ) from exc

    prompt = get_client().get_prompt(prompt_key).get_langchain_prompt()
    if isinstance(prompt, BasePromptTemplate):
        prompt_text = prompt.format()
    elif isinstance(prompt, str):
        prompt_text = prompt
    else:
        raise TypeError(f"Unsupported Langfuse prompt type: {type(prompt)!r}")

    prompt_path = _resolve_local_prompt_path(prompt_key, prompt_dir)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt_text, encoding="utf-8")
    logger.info("Prompt {} downloaded from Langfuse to {}.", prompt_key, prompt_path)


def download_prompts_by_prompt_keys(
    prompt_keys_path="configs/prompt_keys.yaml", prompt_dir="prompts"
):
    if not os.path.exists(prompt_keys_path):
        raise FileNotFoundError(f"Prompt list file not found at {prompt_keys_path}")

    prompt_keys = load_file(prompt_keys_path)
    for item in prompt_keys:
        if isinstance(item, dict):
            prompt_key = item.get("prompt_key")
        else:
            prompt_key = item
        if not prompt_key:
            continue
        try:
            download_prompt(prompt_key, prompt_dir=prompt_dir)
        except Exception as e:
            logger.exception("Failed to download prompt {}: {}", prompt_key, e)
            prompt_path = _resolve_local_prompt_path(prompt_key, prompt_dir)
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text("", encoding="utf-8")


def upload_prompt(
    prompt_key: str,
    *,
    prompt_dir: str | Path | None = "prompts",
    tags: list[str] | None = None,
    labels: list[str] | None = None,
) -> None:
    try:
        from langfuse import get_client
    except ImportError as exc:
        raise RuntimeError(
            "Langfuse is required to upload prompt keys. Install langfuse first."
        ) from exc

    prompt_path = _resolve_local_prompt_path(prompt_key, prompt_dir)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Local prompt file not found at {prompt_path}")
    prompt_text = prompt_path.read_text(encoding="utf-8")

    if labels is None:
        labels = ["production"]

    get_client().create_prompt(
        name=prompt_key,
        prompt=prompt_text,
        labels=labels,
        tags=tags,
        type="text",
    )
    logger.info("Prompt {} uploaded from {} to Langfuse.", prompt_key, prompt_path)


def upload_prompts_from_local_by_prompt_keys(
    prompt_keys_path="configs/prompt_keys.yaml",
    prompt_dir="prompts",
    tags: list[str] | None = None,
    labels: list[str] | None = None,
):
    if not os.path.exists(prompt_keys_path):
        raise FileNotFoundError(f"Prompt list file not found at {prompt_keys_path}")

    prompt_keys = load_file(prompt_keys_path)
    for item in prompt_keys:
        if isinstance(item, dict):
            prompt_key = item.get("prompt_key")
            item_tags = item.get("tags") or tags
            item_labels = item.get("labels") or labels
        else:
            prompt_key = item
            item_tags = tags
            item_labels = labels
        if not prompt_key:
            continue
        try:
            upload_prompt(
                prompt_key, prompt_dir=prompt_dir, tags=item_tags, labels=item_labels
            )
        except Exception as e:
            logger.exception("Failed to upload prompt {}: {}", prompt_key, e)


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
