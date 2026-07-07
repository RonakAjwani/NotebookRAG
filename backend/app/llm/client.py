"""Provider-agnostic LLM client.

Cerebras / Groq / OpenAI are all OpenAI-compatible, so a single client wraps
them by swapping base_url + key + model. Adds:
  - exponential backoff on rate limits / transient errors (a 429 sleeps and
    retries; it never crashes a run mid-suite)
  - a prompt-hash cache so re-runs replay unchanged calls for free
  - a JSON-mode helper for structured calls (reranking, verification, evals)
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Optional

from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError

from app.config import Settings, settings
from app.llm.cache import PromptCache

# Provider name -> (api_key_attr, base_url_attr, model_attr)
_PROVIDERS = {
    "cerebras": ("cerebras_api_key", "cerebras_base_url", "cerebras_model"),
    "groq": ("groq_api_key", "groq_base_url", "groq_model"),
    "openai": ("openai_api_key", "openai_base_url", "openai_model"),
    "openrouter": ("openrouter_api_key", "openrouter_base_url", "openrouter_model"),
}

# Proactive per-provider pacing (shared across client instances): space calls to
# the provider's RPM so we rarely trip 429s instead of only backing off after.
import threading

_pace_lock = threading.Lock()
_last_call_at: dict[str, float] = {}


def _pace(pace_key: str, rpm: int) -> None:
    """Space calls sharing a rate-limit pool. Keyed on provider:model because
    Cerebras (and others) meter each model separately - routing roles to
    different models multiplies the effective daily budget."""
    if rpm <= 0:
        return
    min_interval = 60.0 / rpm
    with _pace_lock:
        now = time.monotonic()
        prev = _last_call_at.get(pace_key, 0.0)
        wait = prev + min_interval - now
        # Reserve our slot before sleeping so concurrent callers space out too.
        _last_call_at[pace_key] = max(now, prev + min_interval) if wait > 0 else now
    if wait > 0:
        time.sleep(wait)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        provider: str,
        model: str | None = None,
        cfg: Settings | None = None,
        cache: PromptCache | None = None,
    ):
        self.cfg = cfg or settings
        if provider not in _PROVIDERS:
            raise LLMError(f"Unknown LLM provider: {provider}")
        self.provider = provider

        key_attr, url_attr, model_attr = _PROVIDERS[provider]
        api_key = getattr(self.cfg, key_attr)
        if not api_key:
            raise LLMError(
                f"{key_attr.upper()} is not set. Add it to backend/.env before "
                f"using the '{provider}' provider."
            )
        self.model = model or getattr(self.cfg, model_attr)
        self._client = OpenAI(api_key=api_key, base_url=getattr(self.cfg, url_attr))
        self._cache = cache or PromptCache(self.cfg.cache_dir, self.cfg.llm_cache_enabled)

    # -- public API ---------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        model: str | None = None,
    ) -> str:
        """Return the assistant message text for a chat completion."""
        temperature = self.cfg.llm_temperature if temperature is None else temperature
        max_tokens = self.cfg.llm_max_tokens if max_tokens is None else max_tokens
        model = model or self.model

        cache_key = self._cache.make_key(
            {
                "provider": self.provider,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "json_mode": json_mode,
                "messages": messages,
            }
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        text = self._call_with_retry(kwargs)
        self._cache.set(cache_key, text, meta={"provider": self.provider, "model": model})
        return text

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> Any:
        """Chat completion parsed as JSON, tolerant of code-fenced output."""
        raw = self.complete(
            messages,
            temperature=0.0 if temperature is None else temperature,
            max_tokens=max_tokens,
            json_mode=True,
            model=model,
        )
        return _parse_json(raw)

    # -- internals ----------------------------------------------------------

    def _call_with_retry(self, kwargs: dict[str, Any]) -> str:
        last_exc: Exception | None = None
        rpm = getattr(self.cfg, f"{self.provider}_rpm", 0)
        pace_key = f"{self.provider}:{kwargs.get('model', self.model)}"
        for attempt in range(self.cfg.llm_max_retries):
            try:
                _pace(pace_key, rpm)
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                return (content or "").strip()
            except (RateLimitError, APIConnectionError) as exc:
                last_exc = exc
                self._sleep(attempt)
            except APIStatusError as exc:
                # Retry only server-side / throttling errors; fail fast on 4xx.
                if exc.status_code in (429, 500, 502, 503, 504):
                    last_exc = exc
                    self._sleep(attempt)
                else:
                    raise LLMError(f"{self.provider} API error {exc.status_code}: {exc}") from exc
        raise LLMError(
            f"{self.provider} call failed after {self.cfg.llm_max_retries} retries: {last_exc}"
        ) from last_exc

    def _sleep(self, attempt: int) -> None:
        delay = min(self.cfg.llm_backoff_base ** attempt, self.cfg.llm_backoff_max)
        delay += random.uniform(0, delay * 0.1)  # jitter
        time.sleep(delay)


def _parse_json(raw: str) -> Any:
    """Parse model output as JSON, tolerant of code fences, prose wrappers, and
    the trailing-comma / thinking-tag quirks some models emit. Always raises
    LLMError (never a bare JSONDecodeError) so callers can treat a malformed
    response uniformly as a judge failure."""
    for candidate in _json_candidates(raw):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise LLMError(f"LLM did not return valid JSON: {raw[:300]}")


def _json_candidates(raw: str):
    yield raw
    # Drop <think>...</think> blocks (reasoning models) and code fences.
    stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    stripped = re.sub(r"```(?:json)?|```", "", stripped).strip()
    yield stripped
    # Outermost object/array span.
    match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
    if match:
        block = match.group(1)
        yield block
        # Remove trailing commas before } or ].
        yield re.sub(r",(\s*[}\]])", r"\1", block)


# -- shared instances ------------------------------------------------------
# Built lazily so importing this module never requires API keys (ingestion and
# tests that don't touch the LLM must import cleanly without a .env).

_client_cache: dict[str, LLMClient] = {}


def _client_for_spec(spec: str, default_provider: str) -> LLMClient:
    """Resolve a routing spec into a cached client.

    spec is "" (use default_provider + its default model), "provider", or
    "provider/model". Clients are cached per resolved (provider, model)."""
    spec = (spec or "").strip()
    if not spec:
        provider, model = default_provider, None
    elif "/" in spec:
        provider, model = spec.split("/", 1)
    else:
        provider, model = spec, None
    key = f"{provider}:{model or 'default'}"
    if key not in _client_cache:
        _client_cache[key] = LLMClient(provider, model=model)
    return _client_cache[key]


def get_generator() -> LLMClient:
    return _client_for_spec("", settings.llm_provider)


def get_rerank_client() -> LLMClient:
    # Reranking orders passages; it does not judge generated text, so it is
    # exempt from self-preference concerns and can share the generator provider.
    return _client_for_spec(settings.rerank_llm, settings.judge_provider)


def get_verify_client() -> LLMClient:
    return _client_for_spec(settings.verify_llm, settings.judge_provider)


def get_completeness_client() -> LLMClient:
    return _client_for_spec(settings.completeness_llm, settings.judge_provider)


def get_eval_judge() -> LLMClient:
    return _client_for_spec(settings.eval_judge_llm, settings.judge_provider)


# Back-compat alias: pipeline judge default (used where role is generic).
def get_judge() -> LLMClient:
    return _client_for_spec(settings.judge_provider, settings.judge_provider)
