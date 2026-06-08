from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

_SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Always return your solution as a complete Python code block inside ```python ... ``` markers. "
    "Include only the function definition and any helper code — no prose or commentary."
)


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
class AnthropicChatClient:
    """Anthropic Messages API adapter.

    Environment variables:
    - ANTHROPIC_API_KEY must be set.
    - ANTHROPIC_MODEL can override the model argument.
    """

    model_id: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    max_tokens: int = 1200

    def generate(self, prompt: str, *, problem_id: str, attempt: int) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Install the anthropic package: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set."
            )

        model = os.getenv("ANTHROPIC_MODEL", self.model_id)
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text if message.content else ""


@dataclass(slots=True)
class HuggingFaceChatClient:
    """Hugging Face Inference API adapter (free tier).

    Environment variables:
    - HF_TOKEN must be set (get one at huggingface.co/settings/tokens).

    Good free coding models:
    - "Qwen/Qwen2.5-Coder-7B-Instruct"  (recommended)
    - "microsoft/Phi-3-mini-4k-instruct"
    - "mistralai/Mistral-7B-Instruct-v0.3"
    """

    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    temperature: float = 0.1
    max_tokens: int = 1200

    def generate(self, prompt: str, *, problem_id: str, attempt: int) -> str:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "Install the huggingface_hub package: pip install huggingface_hub"
            ) from exc

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise EnvironmentError(
                "HF_TOKEN environment variable is not set. "
                "Get a free token at huggingface.co/settings/tokens"
            )

        client = InferenceClient(model=self.model_id, token=token)
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        return content or ""


def validate_model(model_id: str, *, warn_size_gb: float = 8.0) -> None:
    """Check whether a HuggingFace model is likely to work with RecoverFlow.

    Raises ValueError with a clear message if the model is definitely incompatible.
    Prints warnings for things that might cause problems (large size, base model name).
    Does not download weights — only fetches the Hub metadata JSON.
    """
    try:
        from huggingface_hub import model_info as hf_model_info
        from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub: pip install huggingface_hub") from exc

    print(f"  Checking model {model_id!r} on HuggingFace Hub...")

    try:
        info = hf_model_info(model_id)
    except RepositoryNotFoundError:
        raise ValueError(
            f"Model {model_id!r} was not found on HuggingFace Hub.\n"
            "Check the model ID for typos at https://huggingface.co/models"
        )
    except GatedRepoError:
        raise ValueError(
            f"Model {model_id!r} is gated (requires acceptance of terms).\n"
            "Visit the model page on HuggingFace and accept the terms, then set HF_TOKEN."
        )
    except Exception as exc:
        print(f"  WARNING: Could not fetch Hub metadata ({exc}). Proceeding anyway.")
        return

    # Check pipeline tag
    pipeline_tag = getattr(info, "pipeline_tag", None)
    if pipeline_tag and pipeline_tag != "text-generation":
        raise ValueError(
            f"Model {model_id!r} has pipeline_tag={pipeline_tag!r}.\n"
            "RecoverFlow requires a text-generation model."
        )

    # Warn if this looks like a base model (no instruct/chat in name)
    name_lower = model_id.lower()
    instruct_hints = ("instruct", "chat", "-it", "sft", "rlhf", "dpo", "assistant")
    if not any(h in name_lower for h in instruct_hints):
        print(
            f"  WARNING: {model_id!r} does not look like an instruction-tuned model\n"
            "  (no 'instruct'/'chat' in the name). Base models will not follow prompts\n"
            "  and will produce garbage output. Use an -Instruct variant instead."
        )

    # Check safetensors size to estimate parameter count
    safetensors = [
        s for s in (info.safetensors or {}).get("parameters", {}).items()
    ] if info.safetensors else []

    total_params = None
    if info.safetensors and hasattr(info.safetensors, "total"):
        total_params = info.safetensors.total
    elif info.safetensors and isinstance(info.safetensors, dict):
        total_params = info.safetensors.get("total")

    if total_params:
        size_gb = total_params * 2 / 1e9  # bfloat16 estimate
        if size_gb > warn_size_gb:
            print(
                f"  WARNING: Model has ~{total_params/1e9:.1f}B parameters "
                f"(~{size_gb:.1f} GB in bfloat16).\n"
                f"  This exceeds {warn_size_gb:.0f} GB — it may OOM on free Colab (15 GB VRAM)\n"
                "  or run very slowly on CPU. Consider a smaller model."
            )
        else:
            print(f"  Size: ~{total_params/1e9:.1f}B parameters (~{size_gb:.1f} GB). OK.")
    else:
        print("  Could not determine model size from Hub metadata.")

    print(f"  Model {model_id!r} looks compatible.\n")


@dataclass
class LocalHuggingFaceClient:
    """Run a HuggingFace model locally via transformers (no API credits needed).

    Good small models that fit on CPU or a modest GPU:
    - "Qwen/Qwen2.5-Coder-0.5B-Instruct"  (500M, fast on CPU)
    - "Qwen/Qwen2.5-Coder-1.5B-Instruct"  (1.5B, still manageable on CPU)
    - "microsoft/Phi-3-mini-4k-instruct"   (3.8B, needs ~8GB RAM)
    """

    model_id: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    temperature: float = 0.1
    max_new_tokens: int = 1200
    _pipeline: object = None

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers and torch: pip install transformers torch"
            ) from exc
        pipe = pipeline(
            "text-generation",
            model=self.model_id,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self._pipeline = pipe
        return pipe

    def generate(self, prompt: str, *, problem_id: str, attempt: int) -> str:
        pipe = self._get_pipeline()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        output = pipe(
            messages,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        return output[0]["generated_text"][-1]["content"]


@dataclass(slots=True)
class OpenAIChatClient:
    """OpenAI chat-completions adapter.

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
            raise RuntimeError(
                "Install the openai package: pip install openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )

        model = os.getenv("OPENAI_MODEL", self.model_id)
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        return content or ""
