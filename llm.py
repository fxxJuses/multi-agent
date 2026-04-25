from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

ARK_DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    content: str
    usage: LLMUsage | None = None


def _resolve_api_key(explicit_api_key: str | None) -> str:
    if explicit_api_key:
        return explicit_api_key

    api_key = (
        os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        or (os.getenv("ARK_API_KEY") or os.getenv("VOLCENGINE_API_KEY") or "").strip()
    )
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY, DASHSCOPE_API_KEY, or ARK_API_KEY."
        )
    return api_key


def _resolve_base_url(explicit_base_url: str | None) -> str | None:
    if explicit_base_url:
        return explicit_base_url.rstrip("/")

    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    dash_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    ark_key = (os.getenv("ARK_API_KEY") or os.getenv("VOLCENGINE_API_KEY") or "").strip()

    base_url = (
        os.getenv("OPENAI_BASE_URL", "").strip()
        or os.getenv("DASHSCOPE_BASE_URL", "").strip()
        or os.getenv("ARK_BASE_URL", "").strip()
    )
    if base_url:
        return base_url.rstrip("/")
    if dash_key and not openai_key and not ark_key:
        return DASHSCOPE_DEFAULT_BASE_URL
    if ark_key and not openai_key and not dash_key:
        return ARK_DEFAULT_BASE_URL
    return None


def _resolve_model(explicit_model: str | None) -> str:
    if explicit_model:
        return explicit_model

    model = (
        os.getenv("LLM_MODEL", "").strip()
        or os.getenv("DASHSCOPE_MODEL", "").strip()
        or os.getenv("QWEN_MODEL", "").strip()
        or os.getenv("ARK_MODEL", "").strip()
        or os.getenv("VOLCENGINE_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
    )
    if model:
        return model
    if os.getenv("DASHSCOPE_API_KEY", "").strip():
        return "qwen-plus"
    return "gpt-4o-mini"


class LLM:
    """A thin wrapper around the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        self.model = _resolve_model(model)
        self.system_prompt = system_prompt

        client_kwargs: dict[str, Any] = {"api_key": _resolve_api_key(api_key)}
        resolved_base_url = _resolve_base_url(base_url)
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        self.client = OpenAI(**client_kwargs)

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        final_system_prompt = system_prompt or self.system_prompt
        if final_system_prompt:
            messages.append({"role": "system", "content": final_system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _extract_usage(usage: Any) -> LLMUsage | None:
        if not usage:
            return None
        return LLMUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
        )

    def chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system_prompt=system_prompt, history=history),
            temperature=temperature,
            **kwargs,
        )
        message = response.choices[0].message.content or ""
        return LLMResponse(
            content=message,
            usage=self._extract_usage(getattr(response, "usage", None)),
        )

    def stream_chat(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        temperature: float = 0.7,
        print_output: bool = True,
        **kwargs: Any,
    ) -> LLMResponse:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system_prompt=system_prompt, history=history),
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )

        parts: list[str] = []
        usage: LLMUsage | None = None

        for chunk in stream:
            if chunk.choices:
                content = chunk.choices[0].delta.content or ""
                if content:
                    parts.append(content)
                    if print_output:
                        print(content, end="", flush=True)
            elif chunk.usage:
                usage = self._extract_usage(chunk.usage)

        return LLMResponse(content="".join(parts), usage=usage)
