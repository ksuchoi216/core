"""Shared node helpers for LangGraph sessions."""

# from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Literal

from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from loguru import logger

from .chains import build_chain
from core.support.config import OpenAINodeConfig

load_dotenv(find_dotenv(usecwd=True))


def _normalize_output_value(value: Any) -> Any:
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _normalize_output_value(value.model_dump())
    if isinstance(value, Mapping):
        return {
            _normalize_output_value(key): _normalize_output_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_output_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_output_value(item) for item in value]
    if isinstance(value, set):
        return [_normalize_output_value(item) for item in value]
    if isinstance(value, str) and type(value) is not str:
        return str(value)
    return value


class GeneralNode:
    def __init__(
        self,
        *,
        model_config: OpenAINodeConfig,
        state_to_input: Callable[[dict], dict[str, Any] | None],
        state_save_key: str,
        prompt_key: str,
        is_batch: bool = False,
        input_to_batch_input: (
            Callable[[dict[str, Any]], Sequence[dict[str, Any]]] | None
        ) = None,
        local_prompt: bool = True,
        local_prompt_dir: str | Path = "prompts",
        prepare_state: Callable[[dict], dict[str, Any]] | None = None,
        output_to_state: Callable[[Any, dict], dict[str, Any]] | None = None,
        node_name: str | None = None,
        output_parser=None,
        tools: Sequence[Any] | None = None,
        tool_choice: Any = None,
        state_type: Literal["dict", "list", "direct"] = "direct",
        state_dict_key: str | None = None,
        iter_key: str | None = None,
        return_parallel: bool = False,
    ) -> None:
        self.model_config = model_config
        self.prompt_key = prompt_key
        self.is_batch = is_batch
        self.input_to_batch_input = input_to_batch_input
        self.local_prompt = local_prompt
        self.local_prompt_dir = local_prompt_dir
        self.state_to_input = state_to_input
        self.prepare_state = prepare_state
        self.output_to_state = output_to_state
        self.output_parser = output_parser
        self.tools = tools
        self.tool_choice = tool_choice
        self.state_type = state_type
        self.state_dict_key = state_dict_key
        self.state_save_key = state_save_key
        self.node_name = node_name or self.prompt_key or "general_node"
        self.return_parallel = return_parallel
        self.iter_key = iter_key

        logger.info("node: {}", self.node_name)

    def _preprocess(self) -> None:
        self.prompt, self.parser, self.chain = build_chain(
            model_config=self.model_config,
            prompt_key=self.prompt_key,
            local_prompt=self.local_prompt,
            local_prompt_dir=self.local_prompt_dir,
            output_parser=self.output_parser,
            tools=self.tools,
            tool_choice=self.tool_choice,
        )

    def _prepare_state(self, state: dict[str, Any]) -> dict[str, Any]:
        if self.prepare_state is None:
            return state

        state_update = self.prepare_state(state)
        if not isinstance(state_update, Mapping):
            raise TypeError("prepare_state must return a mapping.")
        state.update(state_update)
        return state

    def _output_to_state(
        self,
        *,
        state: dict[str, Any],
        output: Any,
    ) -> dict[str, Any]:
        output = _normalize_output_value(output)

        if self.output_to_state is not None:
            state_update = self.output_to_state(output, state)
            if not isinstance(state_update, Mapping):
                raise TypeError("output_to_state must return a mapping.")
            logger.info("AI Answer:\n{}\n", output)
            state.update(state_update)
            return state

        logger.info("AI Answer:\n{}\n", output)

        if self.return_parallel:
            return {self.state_save_key: output}

        if self.state_type == "list":
            if isinstance(output, list):
                state.setdefault(self.state_save_key, []).extend(output)
            else:
                state.setdefault(self.state_save_key, []).append(output)
        elif self.state_type == "dict":
            state.setdefault(self.state_save_key, {})[self.state_dict_key] = output
        else:
            state[self.state_save_key] = output
        return state

    def _add_format_instructions(self, inputs: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(inputs)
        if isinstance(self.parser, PydanticOutputParser):
            resolved["format_instructions"] = self.parser.get_format_instructions()
        return resolved

    def batch(
        self, inputs: dict[str, Any], config: RunnableConfig | None = None
    ) -> Any:
        if self.input_to_batch_input is None:
            raise ValueError(
                "input_to_batch_input must be provided when is_batch is True."
            )

        batch_inputs = self.input_to_batch_input(inputs)
        if not isinstance(batch_inputs, Sequence) or isinstance(
            batch_inputs, (str, bytes)
        ):
            raise TypeError("input_to_batch_input must return a sequence of mappings.")

        resolved_inputs_list = [
            self._add_format_instructions(inp) for inp in batch_inputs
        ]
        return self.chain.batch(resolved_inputs_list, config=config)

    def invoke(
        self, inputs: dict[str, Any], config: RunnableConfig | None = None
    ) -> Any:
        return self.chain.invoke(self._add_format_instructions(inputs), config=config)

    def __call__(self):
        self._preprocess()

        def node(
            state: dict[str, Any],
            config: RunnableConfig | None = None,
        ) -> dict[str, Any]:
            state = self._prepare_state(state)
            logger.info("============= {} ==============", self.node_name)

            inputs = self.state_to_input(state)
            if inputs is None:
                inputs = {}

            if not isinstance(inputs, Mapping):
                raise TypeError("state_to_input must return a mapping or None.")

            if self.is_batch:
                output = self.batch(dict(inputs), config=config)
            else:
                output = self.invoke(dict(inputs), config=config)

            if self.iter_key:
                state[self.iter_key] += 1
                logger.info("Iter: {} from {}", state[self.iter_key], self.iter_key)

            return self._output_to_state(state=state, output=output)

        return node
