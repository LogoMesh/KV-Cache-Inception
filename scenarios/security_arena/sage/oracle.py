"""
SAGE Oracle — Abstract interface for LLM inference backends.

Decouples the Attacker agent from a specific LLM client so the same
attack logic can run against:
  - Live OpenAI/Lambda API (OpenAIOracle)       — production battles
  - Local transformers model (LocalLlamaOracle) — Phase A/B offline MCTS

Inject a custom oracle via Attacker(oracle=...) to swap backends without
touching any attack logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class AttackOracle(ABC):
    """Abstract backend for LLM inference in the attacker pipeline.

    All LLM calls in the attacker route through this interface, making
    the attack logic backend-agnostic.
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
        """True if this oracle exposes hidden states for H-Neuron / LAT monitoring."""
        return False

    @property
    def model_id(self) -> str:
        """Human-readable model identifier for logging."""
        return "unknown"


class OpenAIOracle(AttackOracle):
    """Wraps an AsyncOpenAI client for live battle API calls.

    Preserves existing Attacker behavior exactly — no functional change
    for production battles. Created automatically in Attacker.__init__
    when no oracle is injected.
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
