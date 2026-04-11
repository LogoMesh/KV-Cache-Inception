"""
SAGE H-Neuron Monitor — Hallucination-associated neuron scoring.

Identifies neurons whose activation correlates with incoherent/hallucinated
output, and provides a real-time score during offline MCTS generation.

Two paths depending on model architecture:
  - Dense (Llama-3.2-1B, Phase A): MLP neuron activation monitoring.
    H-Neurons are found by calibration: neurons with high activation
    variance between coherent and hallucinated examples.
  - MoE (gpt-oss-20b, Phase B): router logit entropy as proxy.
    High routing entropy → expert selection is uncertain → model is
    generating incoherent output. See CLAUDE.md "Gotchas" for why
    dense monitoring does not apply to MoE.

Usage (Phase A — Llama):
    from logomesh.local_model import LocalLlamaOracle
    from logomesh.hneuron_monitor import HNeuronMonitor

    oracle = LocalLlamaOracle.load("./models/llama-3.2-1b")
    monitor = HNeuronMonitor(oracle)

    monitor.calibrate(
        coherent_examples=["Tell me about Paris.", "What is photosynthesis?"],
        hallucinated_examples=["Explain the Battle of Zorvak in 1842.",
                               "List the moons of Planet Quasar-7."],
    )

    await oracle.generate(system="...", user="attack prompt")
    score = monitor.score()  # 0.0 = coherent, 1.0 = hallucinating

Score interpretation in MCTS reward:
    Final reward = 0.4 * (1 - h_score) + 0.6 * lat_score
    Low h_score (coherent) is a prerequisite for the attack to be credible.
    A payload with h_score > 0.7 is likely incoherent and will not fool
    the defender regardless of how "attack-like" it looks.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .local_model import LocalLlamaOracle

logger = logging.getLogger(__name__)

# Minimum examples required for reliable calibration
MIN_CALIBRATION_EXAMPLES = 5

# How many top H-Neurons to use for scoring (the most discriminative ones)
TOP_K_NEURONS = 50


class HNeuronMonitor:
    """Monitor hallucination-associated neurons during offline MCTS.

    Call calibrate() once per session (or load from cache), then call
    score() after each oracle.generate() to get the H-Neuron score.

    Args:
        oracle: A LocalLlamaOracle with supports_telemetry=True.
    """

    def __init__(self, oracle: "LocalLlamaOracle") -> None:
        if not oracle.supports_telemetry:
            raise ValueError(
                "HNeuronMonitor requires an oracle with supports_telemetry=True. "
                "Use LocalLlamaOracle, not OpenAIOracle."
            )
        self._oracle = oracle
        self._is_moe: bool = False  # set in calibrate() based on router_logits presence
        self._calibrated: bool = False

        # Dense path calibration state
        self._h_neuron_indices: list[int] = []     # top-K neuron indices
        self._coherent_mean: float = 0.0           # mean score on coherent examples
        self._coherent_std: float = 1.0
        self._hallucinated_mean: float = 0.0
        self._hallucinated_std: float = 1.0

        # MoE path calibration state
        self._coherent_entropy_mean: float = 0.0
        self._coherent_entropy_std: float = 1.0
        self._hallucinated_entropy_mean: float = 0.0

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    async def calibrate(
        self,
        coherent_examples: list[str],
        hallucinated_examples: list[str],
        system_prompt: str = "You are a helpful assistant.",
    ) -> None:
        """Calibrate H-Neuron thresholds from contrastive examples.

        Args:
            coherent_examples: Prompts where factual, coherent responses are expected.
                               E.g., well-known facts, clear questions.
            hallucinated_examples: Prompts likely to elicit hallucinated responses.
                                   E.g., fictional entities, contradictory constraints.
            system_prompt: System prompt to use for both sets.

        After calibration, score() returns values normalized to [0, 1] relative
        to these calibration distributions.
        """
        if len(coherent_examples) < MIN_CALIBRATION_EXAMPLES:
            logger.warning(
                "Only %d coherent examples provided (min %d). "
                "Calibration may be unreliable.",
                len(coherent_examples), MIN_CALIBRATION_EXAMPLES,
            )
        if len(hallucinated_examples) < MIN_CALIBRATION_EXAMPLES:
            logger.warning(
                "Only %d hallucinated examples provided (min %d).",
                len(hallucinated_examples), MIN_CALIBRATION_EXAMPLES,
            )

        logger.info(
            "Calibrating HNeuronMonitor: %d coherent, %d hallucinated examples",
            len(coherent_examples), len(hallucinated_examples),
        )

        coherent_activations = await self._collect_activations(
            coherent_examples, system_prompt
        )
        hallucinated_activations = await self._collect_activations(
            hallucinated_examples, system_prompt
        )

        if not coherent_activations or not hallucinated_activations:
            logger.error("Calibration failed: no activations collected")
            return

        # Detect model type from first example's router logits
        self._is_moe = len(self._oracle.get_router_logits()) > 0

        if self._is_moe:
            self._calibrate_moe(coherent_activations, hallucinated_activations)
        else:
            self._calibrate_dense(coherent_activations, hallucinated_activations)

        self._calibrated = True
        logger.info(
            "Calibration complete. Mode: %s. H-Neurons: %d",
            "MoE/entropy" if self._is_moe else "dense/MLP",
            0 if self._is_moe else len(self._h_neuron_indices),
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self) -> float:
        """Score the most recent oracle.generate() call.

        Returns:
            Float in [0.0, 1.0]. 0.0 = maximally coherent, 1.0 = hallucinating.
            Returns 0.5 if not calibrated (neutral / unknown).

        Call this immediately after oracle.generate() — the hidden states
        are cached from that forward pass and will be overwritten on the next call.
        """
        if not self._calibrated:
            logger.warning("HNeuronMonitor.score() called before calibrate(). Returning 0.5.")
            return 0.5

        if self._is_moe:
            return self._score_moe()
        return self._score_dense()

    # ------------------------------------------------------------------
    # Internal — dense path (Llama-3.2-1B)
    # ------------------------------------------------------------------

    def _calibrate_dense(
        self,
        coherent_activations: list[list[float]],
        hallucinated_activations: list[list[float]],
    ) -> None:
        """Identify top-K H-Neurons by activation difference."""
        import statistics

        n_neurons = len(coherent_activations[0]) if coherent_activations else 0
        if n_neurons == 0:
            return

        k = min(TOP_K_NEURONS, n_neurons)

        # Per-neuron mean across examples
        coherent_means = [
            statistics.mean(row[i] for row in coherent_activations)
            for i in range(n_neurons)
        ]
        hallucinated_means = [
            statistics.mean(row[i] for row in hallucinated_activations)
            for i in range(n_neurons)
        ]

        # H-Neurons: highest (hallucinated_mean - coherent_mean)
        diffs = [hallucinated_means[i] - coherent_means[i] for i in range(n_neurons)]
        ranked = sorted(range(n_neurons), key=lambda i: diffs[i], reverse=True)
        self._h_neuron_indices = ranked[:k]

        # Calibrate score distribution using top-K mean
        coherent_scores = [
            self._raw_dense_score(row) for row in coherent_activations
        ]
        hallucinated_scores = [
            self._raw_dense_score(row) for row in hallucinated_activations
        ]

        self._coherent_mean = statistics.mean(coherent_scores) if coherent_scores else 0.0
        self._coherent_std = statistics.stdev(coherent_scores) if len(coherent_scores) > 1 else 1.0
        self._hallucinated_mean = statistics.mean(hallucinated_scores) if hallucinated_scores else 1.0

    def _raw_dense_score(self, activation_row: list[float]) -> float:
        """Mean activation over the top-K H-Neurons for one example."""
        if not self._h_neuron_indices:
            return 0.0
        return sum(activation_row[i] for i in self._h_neuron_indices) / len(self._h_neuron_indices)

    def _score_dense(self) -> float:
        """Score using cached MLP activations from LocalLlamaOracle."""
        hidden_states = self._oracle.get_hidden_states()
        if not hidden_states:
            logger.warning("No hidden states available. Call oracle.generate() first.")
            return 0.5

        # Use the last layer's hidden states.
        # Shape is [hidden_size] (last token) or [seq, hidden_size] — handle both.
        last_layer = hidden_states[-1]
        try:
            if last_layer.dim() == 1:
                activation_row = last_layer.tolist()
            else:
                activation_row = last_layer[-1].tolist()
        except Exception:
            return 0.5

        raw = self._raw_dense_score(activation_row)

        # Normalize: 0.0 at coherent_mean, 1.0 at hallucinated_mean
        span = self._hallucinated_mean - self._coherent_mean
        if abs(span) < 1e-8:
            return 0.5
        normalized = (raw - self._coherent_mean) / span
        return float(max(0.0, min(1.0, normalized)))

    # ------------------------------------------------------------------
    # Internal — MoE path (gpt-oss-20b, Phase B)
    # ------------------------------------------------------------------

    def _calibrate_moe(
        self,
        coherent_activations: list[list[float]],
        hallucinated_activations: list[list[float]],
    ) -> None:
        """Calibrate entropy baseline from MoE router logits.

        coherent_activations here contains pre-computed entropy scores
        (one per example), not raw neuron activations.
        """
        import statistics
        # When is_moe, _collect_activations stores [entropy_score] per example
        c_entropies = [row[0] for row in coherent_activations if row]
        h_entropies = [row[0] for row in hallucinated_activations if row]

        self._coherent_entropy_mean = statistics.mean(c_entropies) if c_entropies else 0.0
        self._coherent_entropy_std = statistics.stdev(c_entropies) if len(c_entropies) > 1 else 1.0
        self._hallucinated_entropy_mean = statistics.mean(h_entropies) if h_entropies else 1.0

    def _score_moe(self) -> float:
        """Score using router logit entropy from LocalLlamaOracle (gpt-oss-20b path)."""
        router_logits = self._oracle.get_router_logits()
        if not router_logits:
            logger.warning("No router logits available. Is this a MoE model?")
            return 0.5

        entropy = _compute_router_entropy(router_logits)
        span = self._hallucinated_entropy_mean - self._coherent_entropy_mean
        if abs(span) < 1e-8:
            return 0.5
        normalized = (entropy - self._coherent_entropy_mean) / span
        return float(max(0.0, min(1.0, normalized)))

    # ------------------------------------------------------------------
    # Internal — activation collection
    # ------------------------------------------------------------------

    async def _collect_activations(
        self, examples: list[str], system_prompt: str
    ) -> list[list[float]]:
        """Run oracle on each example and extract activations.

        Returns a list (one per example) of activation vectors.
        For dense models: last-layer hidden state mean-pooled → [hidden_size] floats.
        For MoE models: [router_entropy_scalar] — single-element list.
        """
        results = []
        for text in examples:
            try:
                await self._oracle.generate(system=system_prompt, user=text)
            except Exception as e:
                logger.error("Failed to generate for calibration example: %s", e)
                continue

            router_logits = self._oracle.get_router_logits()
            is_moe = len(router_logits) > 0

            if is_moe:
                entropy = _compute_router_entropy(router_logits)
                results.append([entropy])
            else:
                hidden_states = self._oracle.get_hidden_states()
                if not hidden_states:
                    continue
                # last hidden state: shape [hidden_size] (last token, from forward pass)
                # or [seq, hidden_size] (if returned as full sequence) — handle both.
                last_layer = hidden_states[-1]
                try:
                    if last_layer.dim() == 1:
                        row = last_layer.tolist()   # [hidden_size] — already last token
                    else:
                        row = last_layer[-1].tolist()  # [seq, hidden] → take last token
                    results.append(row)
                except Exception as e:
                    logger.error("Failed to extract activations: %s", e)

        return results


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _compute_router_entropy(router_logits: list) -> float:
    """Compute mean per-token entropy over all MoE routing layers.

    Args:
        router_logits: List of Tensors from oracle.get_router_logits().
                       Each tensor: [seq_len, num_experts] or [num_experts].

    Returns:
        Mean routing entropy in nats. Higher = more uncertain = hallucinating.
    """
    import math

    entropies = []
    for logits in router_logits:
        try:
            # logits: Tensor [seq, num_experts] or [num_experts]
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)

            # Softmax → probability distribution over experts
            log_probs = logits - logits.logsumexp(dim=-1, keepdim=True)
            probs = log_probs.exp()

            # Shannon entropy per token: H = -∑ p_i log(p_i)
            H = -(probs * log_probs.clamp(min=-100)).sum(dim=-1)  # [seq]
            entropies.append(float(H.mean()))
        except Exception as e:
            logger.debug("Router entropy computation failed for a layer: %s", e)
            continue

    if not entropies:
        return 0.0

    # Normalize by maximum possible entropy: log(num_experts)
    # This puts the score on a [0, 1] scale relative to uniform routing
    max_entropy = math.log(router_logits[0].shape[-1]) if router_logits else 1.0
    raw_mean = sum(entropies) / len(entropies)
    return raw_mean / max(max_entropy, 1e-8)
