from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, Literal, Union
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from core.llm.models import CLAUDE_ADAPTIVE_THINKING_MODELS
from core.support.file import load_file


ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]
ClaudeEffort = Literal["low", "medium", "high", "xhigh", "max"]


class OpenAIReasoningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effort: ReasoningEffort


class OpenAINodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["openai"]
    model_name: str
    use_responses_api: bool = True
    prompt_cache_key: str | None = None
    temperature: float | None = None
    reasoning: OpenAIReasoningConfig | None = None
    verbosity: Literal["low", "medium", "high"] | None = None


class ClaudeThinkingConfig(BaseModel):
    # Allow forward-compatible Anthropic thinking keys without schema changes.
    model_config = ConfigDict(extra="allow")

    type: Literal["enabled", "adaptive", "disabled"] | None = None
    budget_tokens: int | None = None
    display: Literal["summarized", "raw"] | None = None


class ClaudeOutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effort: ClaudeEffort | None = None


class ClaudePromptCacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["ephemeral"] = "ephemeral"
    ttl: Literal["5m", "1h"] | None = None


class ClaudeNodeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["claude"]
    model_name: str
    temperature: float | None = None
    max_tokens: int | None = None
    thinking: ClaudeThinkingConfig | None = None
    output_config: ClaudeOutputConfig | None = None
    prompt_cache: ClaudePromptCacheConfig | None = None

    @model_validator(mode="after")
    def _validate_thinking(self) -> "ClaudeNodeConfig":
        if self.thinking is None:
            return self

        if self.thinking.type == "enabled":
            if self.thinking.budget_tokens is None:
                raise ValueError(
                    'thinking.budget_tokens is required when thinking.type is "enabled".'
                )
            if self.thinking.budget_tokens < 1024:
                raise ValueError("thinking.budget_tokens must be at least 1024.")
            if self.max_tokens is None:
                raise ValueError(
                    'max_tokens is required when thinking.type is "enabled".'
                )
            if self.thinking.budget_tokens >= self.max_tokens:
                raise ValueError("thinking.budget_tokens must be less than max_tokens.")

        if self.model_name in CLAUDE_ADAPTIVE_THINKING_MODELS:
            if self.thinking.type == "enabled":
                raise ValueError(
                    f"{self.model_name} does not support manual extended thinking "
                    '(thinking.type: "enabled"). Use thinking.type: "adaptive" with '
                    "output_config.effort instead."
                )
            if self.thinking.budget_tokens is not None:
                raise ValueError(
                    f"{self.model_name} does not support thinking.budget_tokens. "
                    "Use output_config.effort to control adaptive thinking instead."
                )
        return self


LLMNodeConfig = Annotated[
    Union[OpenAINodeConfig, ClaudeNodeConfig],
    Field(discriminator="provider"),
]

_NODE_CONFIG_ADAPTER: TypeAdapter[Any] = TypeAdapter(LLMNodeConfig)


def validate_node_config(node_data: Any) -> Any:
    """Validate one node's config dict into the provider-specific model.

    Dispatches on the required ``provider`` discriminator. A missing or unknown
    provider, or a field that belongs to the other provider, raises ``ValueError``.
    """
    try:
        return _NODE_CONFIG_ADAPTER.validate_python(node_data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


class OpenAIConfigCollection(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"Config item not found: {name}") from exc

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.keys()))


def to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: to_namespace(item) for key, item in value.items()}
        )

    if isinstance(value, list):
        return [to_namespace(item) for item in value]

    return value


def _parse_model_session(session_name: str, data: Any) -> OpenAIConfigCollection:
    if not isinstance(data, dict):
        raise ValueError(f"Invalid session config: {session_name}")

    configs = OpenAIConfigCollection()
    for node_name, node_data in data.items():
        if not isinstance(node_data, dict) or "model_name" not in node_data:
            raise ValueError(f"Invalid node config: {session_name}.{node_name}")
        try:
            configs[node_name] = validate_node_config(node_data)
        except ValueError as exc:
            raise ValueError(
                f"Invalid node config: {session_name}.{node_name}: {exc}"
            ) from exc
    return configs


def _apply_test_model_names(config: Any) -> None:
    if isinstance(config, OpenAINodeConfig):
        if not (
            config.model_name.endswith("-mini") or config.model_name.endswith("-nano")
        ):
            config.model_name = f"{config.model_name}-mini"
        return

    if isinstance(config, ClaudeNodeConfig):
        if "opus" in config.model_name:
            config.model_name = "claude-haiku-4-5"
            config.thinking = ClaudeThinkingConfig(
                type="enabled",
                budget_tokens=1024,
            )
            config.output_config = None
        return

    if isinstance(config, dict):
        for item in config.values():
            _apply_test_model_names(item)


@lru_cache(maxsize=None)
def load_model_config(
    config_path: str | Path = "configs/models.yaml",
) -> OpenAIConfigCollection:
    session_config = load_file(config_path) or {}

    configs = OpenAIConfigCollection()
    for session_name, session_data in session_config.items():
        configs[session_name] = _parse_model_session(session_name, session_data)

    return configs


def load_general_config(
    config_path: str | Path = "configs/general.yaml",
):

    config = load_file(config_path) or {}
    return to_namespace(config)


def load_config(
    config_dir: str | Path = "configs", is_test: bool = False
) -> SimpleNamespace:
    config_dir = Path(config_dir)

    general_config_path: str | Path = config_dir / "general.yaml"
    model_config_path: str | Path = config_dir / "models.yaml"

    general_config = (
        load_general_config(general_config_path)
        if general_config_path is not None
        else SimpleNamespace()
    )

    model_config = (
        load_model_config(model_config_path)
        if model_config_path is not None
        else OpenAIConfigCollection()
    )
    model_config = deepcopy(model_config)

    if is_test:
        _apply_test_model_names(model_config)

    return SimpleNamespace(
        **vars(general_config),
        models=model_config,
    )
