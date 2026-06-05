from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    model_id: str

    def generate(self, prompt: str, *, problem_id: str, attempt: int) -> str:
        """Return model response text."""


@dataclass(slots=True)
class ScriptedLLMClient:
    """Deterministic fake LLM for testing the evaluator loop."""

    scripts: dict[str, list[str]]
    model_id: str = "scripted-demo-model"

    def generate(self, prompt: str, *, problem_id: str, attempt: int) -> str:
        responses = self.scripts.get(problem_id)
        if not responses:
            raise KeyError(f"No scripted responses for problem_id={problem_id!r}")
        index = min(attempt - 1, len(responses) - 1)
        return responses[index]


@dataclass(slots=True)
class OpenAIChatClient:
    """Small OpenAI chat-completions adapter.

    Environment variables:
    - OPENAI_API_KEY must be set.
    - OPENAI_MODEL can override the model argument.
    """

    model_id: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1200

    def generate(self, prompt: str, *, problem_id: str, attempt: int) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the openai package or use ScriptedLLMClient.") from exc

        model = os.getenv("OPENAI_MODEL", self.model_id)
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a coding assistant. Return only Python code.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        return content or ""
