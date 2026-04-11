"""
LogoMesh BaseModelClient — Abstract interface for LLM inference backends.

Abstracts the model layer so research code can run against:
  - Live OpenAI API (OpenAIModelClient)          — baseline comparisons
  - Local HuggingFace model (LocalLlamaOracle)   — Phase A/B offline MCTS + KV-cache MCTS

Inject a custom client via the oracle parameter to swap backends without
touching any experiment logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModelClient(ABC):
    """Abstract backend for LLM inference in the LogoMesh research pipeline.

    All LLM calls in experiments route through this interface, making
    experiment logic backend-agnostic.
    """

    @abstractmethod
    async def generate(self, system: str, user: str, temperature: float = 0.7) -> str:
        """Generate a response given system + user messages.

        Args:
            system: System prompt.
            user: User message.
            temperature: Sampling temperature.

        Returns:
            Generated text string. Empty string on failure.
        """

    @property
    def supports_telemetry(self) -> bool:
        """True if this client exposes hidden states for H-Neuron / LAT monitoring."""
        return False

    @property
    def model_id(self) -> str:
        """Human-readable model identifier for logging."""
        return "unknown"


# Backwards-compatible alias kept for any remaining references
AttackOracle = BaseModelClient


class OpenAIModelClient(BaseModelClient):
    """Wraps an AsyncOpenAI client for API calls.

    Used for baseline comparisons (Experiment 2, 4) and Phase B live inference.
    """

    def __init__(self, client, model: str) -> None:
        self._client = client
        self._model = model

    async def generate(self, system: str, user: str, temperature: float = 0.7) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    @property
    def model_id(self) -> str:
        return self._model


# Backwards-compatible alias
OpenAIOracle = OpenAIModelClient
