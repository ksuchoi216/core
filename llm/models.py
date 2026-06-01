from enum import Enum


class OpenAIModelNames(Enum):
    gpt_41_mini = "gpt-4.1-mini"
    gpt_41 = "gpt-4.1"
    gpt_5_nano = "gpt-5-nano"
    gpt_5_mini = "gpt-5-mini"
    gpt_5 = "gpt-5"
    gpt_54_nano = "gpt-5.4-nano"
    gpt_54_mini = "gpt-5.4-mini"
    gpt_54 = "gpt-5.4"
    # gpt_54_pro = "gpt-5.4-pro" # too expensive
    gpt_55_nano = "gpt-5.5-nano"
    gpt_55_mini = "gpt-5.5-mini"
    gpt_55 = "gpt-5.5"
    # gpt_55_pro = "gpt-5.5-pro" # too expensive


# Backwards-compatible alias for the original OpenAI-only enum name.
ModelNames = OpenAIModelNames


OPENAI_REASONING_MODELS = {
    OpenAIModelNames.gpt_5_nano,
    OpenAIModelNames.gpt_5_mini,
    OpenAIModelNames.gpt_5,
    OpenAIModelNames.gpt_54_nano,
    OpenAIModelNames.gpt_54_mini,
    OpenAIModelNames.gpt_54,
}

# Backwards-compatible alias for the original constant name.
REASONING_MODELS = OPENAI_REASONING_MODELS


class ClaudeModelNames(Enum):
    claude_haiku_45 = "claude-haiku-4-5"
    claude_haiku_45_20251001 = "claude-haiku-4-5-20251001"
    claude_sonnet_46 = "claude-sonnet-4-6"
    claude_opus_47 = "claude-opus-4-7"
    claude_opus_48 = "claude-opus-4-8"


# Claude models that drive thinking through adaptive thinking plus
# `output_config.effort` (Opus 4.7+) instead of manual extended thinking
# (`thinking.type: "enabled"` with `budget_tokens`). Stored as raw values so
# config validation can check the configured `model_name` string directly.
CLAUDE_ADAPTIVE_THINKING_MODELS = {
    ClaudeModelNames.claude_opus_47.value,
    ClaudeModelNames.claude_opus_48.value,
}
