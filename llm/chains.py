from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
import re
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.outputs import Generation
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel

from core.support.config import ClaudeNodeConfig, LLMNodeConfig, OpenAINodeConfig
from .callbacks import CacheUsageLogger
from .models import (
    OPENAI_REASONING_MODELS,
    ClaudeModelNames,
    OpenAIModelNames,
)

ParserType = PydanticOutputParser | StrOutputParser


def validate_model_name(model_name: str) -> OpenAIModelNames:
    try:
        return OpenAIModelNames(model_name)
    except ValueError as exc:
        supported_models = ", ".join(model.value for model in OpenAIModelNames)
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            f"Supported models: {supported_models}"
        ) from exc


def validate_claude_model_name(model_name: str) -> ClaudeModelNames:
    try:
        return ClaudeModelNames(model_name)
    except ValueError as exc:
        supported_models = ", ".join(model.value for model in ClaudeModelNames)
        raise ValueError(
            f"Unsupported Claude model_name: {model_name}. "
            f"Supported models: {supported_models}"
        ) from exc


def build_openai_llm(model_config: OpenAINodeConfig) -> ChatOpenAI:
    model_name = validate_model_name(model_config.model_name)

    llm_kwargs: dict[str, Any] = {
        "model": model_config.model_name,
        "use_responses_api": model_config.use_responses_api,
    }
    logger.info("model_name: {}", model_name)
    if model_name in OPENAI_REASONING_MODELS:
        if model_config.temperature is not None:
            raise ValueError("temperature is only supported for non-reasoning models.")
        if model_config.reasoning is not None:
            llm_kwargs["reasoning"] = model_config.reasoning.model_dump()
        if model_config.verbosity is not None:
            if model_config.use_responses_api:
                llm_kwargs["model_kwargs"] = {
                    "text": {"verbosity": model_config.verbosity}
                }
            else:
                llm_kwargs["verbosity"] = model_config.verbosity
    else:
        if model_config.reasoning is not None or model_config.verbosity is not None:
            raise ValueError(
                "reasoning and verbosity are only supported for reasoning models."
            )
        if model_config.temperature is not None:
            llm_kwargs["temperature"] = model_config.temperature

    if model_config.prompt_cache_key:
        llm_kwargs.setdefault("model_kwargs", {})["prompt_cache_key"] = (
            model_config.prompt_cache_key
        )
        logger.info("Using prompt_cache_key: {}", model_config.prompt_cache_key)

    llm_kwargs["callbacks"] = [CacheUsageLogger(label=model_config.model_name)]
    return ChatOpenAI(**llm_kwargs)


def build_claude_llm(model_config: ClaudeNodeConfig) -> ChatAnthropic:
    validate_claude_model_name(model_config.model_name)

    llm_kwargs: dict[str, Any] = {"model": model_config.model_name}
    logger.info("model_name: {}", model_config.model_name)

    if model_config.max_tokens is not None:
        llm_kwargs["max_tokens"] = model_config.max_tokens
    if model_config.temperature is not None:
        llm_kwargs["temperature"] = model_config.temperature
    if model_config.thinking is not None:
        llm_kwargs["thinking"] = model_config.thinking.model_dump(exclude_none=True)
    if model_config.output_config is not None:
        llm_kwargs["output_config"] = model_config.output_config.model_dump(
            exclude_none=True
        )
    if model_config.prompt_cache is not None:
        # The direct Anthropic API accepts a top-level `cache_control` request
        # param; ChatAnthropic forwards `model_kwargs` into the request payload.
        llm_kwargs["model_kwargs"] = {
            "cache_control": model_config.prompt_cache.model_dump(exclude_none=True)
        }
        logger.info("Using Claude prompt cache: {}", llm_kwargs["model_kwargs"])

    llm_kwargs["callbacks"] = [CacheUsageLogger(label=model_config.model_name)]
    return ChatAnthropic(**llm_kwargs)


def build_llm(model_config: LLMNodeConfig):
    if isinstance(model_config, ClaudeNodeConfig):
        return build_claude_llm(model_config)
    return build_openai_llm(model_config)


def build_bound_llm(
    model_config: LLMNodeConfig,
    *,
    tools: Sequence[Any] | None = None,
    tool_choice: Any = None,
):
    llm = build_llm(model_config)
    if not tools:
        return llm

    bind_kwargs: dict[str, Any] = {}
    if tool_choice is not None:
        bind_kwargs["tool_choice"] = tool_choice
    return llm.bind_tools(list(tools), **bind_kwargs)


class CleanPydanticOutputParser(PydanticOutputParser):
    def parse(self, text: str) -> Any:
        cleaned = text.strip()
        tags = ["pydantic_output_format", "format_instructions", "json"]
        for tag in tags:
            start_tag = f"<{tag}>"
            end_tag = f"</{tag}>"
            match = re.search(
                rf"{start_tag}(.*?){end_tag}", cleaned, re.DOTALL | re.IGNORECASE
            )
            if match:
                cleaned = match.group(1).strip()
                break
            else:
                cleaned = re.sub(
                    rf"^{start_tag}", "", cleaned, flags=re.IGNORECASE
                ).strip()
                cleaned = re.sub(
                    rf"{end_tag}$", "", cleaned, flags=re.IGNORECASE
                ).strip()
        return super().parse(cleaned)

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        for r in result:
            if hasattr(r, "text"):
                cleaned = r.text.strip()
                tags = ["pydantic_output_format", "format_instructions", "json"]
                for tag in tags:
                    start_tag = f"<{tag}>"
                    end_tag = f"</{tag}>"
                    match = re.search(
                        rf"{start_tag}(.*?){end_tag}",
                        cleaned,
                        re.DOTALL | re.IGNORECASE,
                    )
                    if match:
                        cleaned = match.group(1).strip()
                        break
                    else:
                        cleaned = re.sub(
                            rf"^{start_tag}", "", cleaned, flags=re.IGNORECASE
                        ).strip()
                        cleaned = re.sub(
                            rf"{end_tag}$", "", cleaned, flags=re.IGNORECASE
                        ).strip()
                r.text = cleaned
        return super().parse_result(result, partial=partial)


def build_output_parser(output_parser) -> ParserType:
    if isinstance(output_parser, type) and issubclass(output_parser, BaseModel):
        logger.info(
            "Using PydanticOutputParser with model: {}",
            output_parser.__name__,
        )
        return CleanPydanticOutputParser(pydantic_object=output_parser)
    logger.info("Using StrOutputParser for output parsing.")
    return StrOutputParser()


def build_chain(
    *,
    model_config: LLMNodeConfig,
    prompt_key: str,
    local_prompt: bool = False,
    local_prompt_dir: str | Path | None = None,
    output_parser=None,
    is_chat: bool = False,
    tools: Sequence[Any] | None = None,
    tool_choice: Any = None,
):
    # Imported lazily so constructing an LLM (build_llm) does not require the
    # Langfuse integration, which only the prompt-loading path below needs.
    from core.langfuse import load_prompt

    prompt = load_prompt(
        prompt_key,
        local_prompt=local_prompt,
        prompt_dir=local_prompt_dir,
    )
    llm = build_bound_llm(
        model_config,
        tools=tools,
        tool_choice=tool_choice,
    )
    parser = build_output_parser(output_parser)

    if is_chat:
        chain = llm | parser
    else:
        prompter = PromptTemplate.from_template(prompt)
        chain = prompter | llm | parser

    return prompt, parser, chain


# def build_chain_chat(self, prompt, llm, parser, is_chat: bool = False):
