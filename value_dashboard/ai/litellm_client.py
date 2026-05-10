from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from litellm import completion


@dataclass
class LiteLLMClient:
    """Small application-owned wrapper around LiteLLM chat completions."""

    model: str
    api_key: str | None = None
    reasoning_effort: str | None = None
    verbosity: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)

    def _completion_params(self, **overrides: Any) -> dict[str, Any]:
        """Build LiteLLM completion parameters with configured overrides."""
        params = dict(self.extra_params)
        if self.api_key:
            params["api_key"] = self.api_key
        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
        if self.verbosity:
            params["verbosity"] = self.verbosity
        params.update({key: value for key, value in overrides.items() if value is not None})
        return params

    def complete_messages(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send chat messages to LiteLLM and return text content."""
        response = completion(
            model=self.model,
            messages=messages,
            **self._completion_params(**kwargs),
        )
        return response.choices[0].message.content or ""

    def complete_text(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Send a prompt to LiteLLM with an optional system message."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.complete_messages(messages, **kwargs)
