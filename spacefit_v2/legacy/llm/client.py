"""Minimal GPT-4o wrapper with JSON-mode, retry, and cost logging."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .. import config

try:
    from openai import OpenAI
    _AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _AVAILABLE = False


# Approximate pricing (USD per 1M tokens) as of 2025-11 for gpt-4o / gpt-4o-mini.
# Used only for best-effort cost logging; values can drift.
_PRICES = {
    "gpt-4o":      {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


class LLMClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 temperature: float = config.OPENAI_TEMPERATURE,
                 max_retries: int = config.OPENAI_MAX_RETRIES):
        if not _AVAILABLE:
            raise RuntimeError("openai package not available. Run `pip install openai`.")
        key = api_key or config.OPENAI_API_KEY
        if not key:
            raise ValueError("OPENAI_API_KEY is empty — set env var or pass api_key")
        self.client = OpenAI(api_key=key)
        self.model = model or config.OPENAI_MODEL
        self.temperature = temperature
        self.max_retries = max_retries
        self.usage = LLMUsage()

    def _log_usage(self, usage, tag):
        price = _PRICES.get(self.model, {"in": 0, "out": 0})
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        cost = (in_tok * price["in"] + out_tok * price["out"]) / 1_000_000
        self.usage.prompt_tokens += in_tok
        self.usage.completion_tokens += out_tok
        self.usage.total_cost_usd += cost
        self.usage.calls += 1
        self.usage.history.append({
            "tag": tag, "in": in_tok, "out": out_tok, "cost": cost,
        })

    def chat_json(self, system: str, user: str, tag: str = "") -> Dict[str, Any]:
        """Single-turn chat with JSON response. Retries on JSON parse failure."""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content or ""
                self._log_usage(resp.usage, tag=tag)
                return json.loads(content)
            except json.JSONDecodeError as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
            except Exception as e:
                last_err = e
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"LLM call '{tag}' failed after {self.max_retries} retries: {last_err}")
