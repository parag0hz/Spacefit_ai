"""OpenAI-chat-compatible wrapper around a local HuggingFace causal LM."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _extract_json(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    if text.startswith("{"):
        try:
            return json.dumps(json.loads(text), ensure_ascii=False)
        except Exception:
            pass
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_str = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : idx + 1]
                try:
                    return json.dumps(json.loads(candidate), ensure_ascii=False)
                except Exception:
                    return candidate
    return text


@dataclass
class LocalHFChatConfig:
    model_id: str = "Qwen/Qwen3-8B"
    max_new_tokens: int = 512
    temperature: float = 0.0
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True


class LocalHFChatClient:
    """Tiny subset of the OpenAI client API used by this project."""

    def __init__(self, config: LocalHFChatConfig):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)
        dtype = getattr(torch, config.torch_dtype, torch.bfloat16)
        model_config = AutoConfig.from_pretrained(config.model_id, trust_remote_code=config.trust_remote_code)
        architectures = {str(name) for name in getattr(model_config, "architectures", []) or []}
        model_type = str(getattr(model_config, "model_type", ""))
        if any("ConditionalGeneration" in name for name in architectures) or model_type in {"gemma4"}:
            self.model = AutoModelForImageTextToText.from_pretrained(
                config.model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=config.trust_remote_code,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=config.trust_remote_code,
            )
        self.model.eval()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _format_prompt(self, messages: Sequence[Mapping[str, Any]]) -> str:
        normalized = [{"role": m.get("role", "user"), "content": _message_text(m.get("content", ""))} for m in messages]
        try:
            return self.tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            pass
        try:
            return self.tokenizer.apply_chat_template(normalized, tokenize=False, add_generation_prompt=True)
        except Exception:
            return "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in normalized) + "\n\nASSISTANT:\n"

    def _create(self, model: str, messages: Sequence[Mapping[str, Any]], response_format: Any = None, temperature: float = 0.0, **_: Any) -> Any:
        import torch

        prompt = self._format_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        temp = float(temperature if temperature is not None else self.config.temperature)
        do_sample = temp > 1e-6
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=do_sample,
                temperature=temp if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if isinstance(response_format, Mapping) and response_format.get("type") == "json_object":
            text = _extract_json(text)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
