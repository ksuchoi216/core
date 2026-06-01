"""LangChain callbacks for observing LLM behaviour at runtime."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from loguru import logger


class CacheUsageLogger(BaseCallbackHandler):
    """Log prompt-cache usage after each LLM call.

    Reads the provider-agnostic ``usage_metadata.input_token_details`` exposed by
    langchain_core, which carries ``cache_read`` (cache hits) and
    ``cache_creation`` (cache writes) for both Anthropic and OpenAI. Caching only
    yields ``cache_read > 0`` once the cached prefix (typically the system prompt)
    clears the model minimum: 1,024 tokens for Opus 4.8 / Sonnet 4.6 and 4,096 for
    Haiku 4.5.
    """

    def __init__(self, *, label: str) -> None:
        self.label = label

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if not usage:
                    continue

                details = usage.get("input_token_details") or {}
                cache_read = details.get("cache_read", 0)
                cache_creation = details.get("cache_creation", 0)
                input_tokens = usage.get("input_tokens", 0)

                cacheable = cache_read + cache_creation
                hit_ratio = cache_read / cacheable if cacheable else 0.0

                logger.info(
                    "[cache] {} | input={} cache_read={} cache_creation={} "
                    "hit_ratio={:.2f}",
                    self.label,
                    input_tokens,
                    cache_read,
                    cache_creation,
                    hit_ratio,
                )
