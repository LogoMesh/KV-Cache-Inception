"""Tests for Phase 2 paper-core modules.

All tests run without a real model — fake tensors and mocks only.
Covers: TelemetryMatrix, DiagnosticState, compute_node_reward,
        NullSpaceProjector, OEICalculator, TDSCalculator,
        FP32Accumulator, KVCacheNode, ReversibleMCTS (smoke).
"""

from __future__ import annotations

import math
import sys
import types
import unittest.mock as mock

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Stub openai so logomesh imports without the real package installed
# ---------------------------------------------------------------------------
_openai_stub = types.ModuleType("openai")
_openai_stub.AsyncOpenAI = mock.MagicMock()
if "openai" not in sys.modules:
    sys.modules["openai"] = _openai_stub


# ===========================================================================
# TelemetryMatrix, DiagnosticState, compute_node_reward
# ===========================================================================

from logomesh.telemetry_matrix import (
    TelemetryMatrix,
    DiagnosticState,
    ClassifyConfig,
    classify,
    compute_node_reward,
    report_telemetry_distribution,
    _jsd,
)


class TestTelemetryMatrix:
    def test_construction_and_properties(self):
        T = TelemetryMatrix(
            h_neuron=np.array([0.2, 0.4, 0.6]),
            repe_honesty=np.array([0.8, 0.7, 0.9]),
            step=3,
        )
        assert T.n_layers == 3
        assert T.step == 3
        assert abs(T.sigma_H_mean - np.array([0.2, 0.4, 0.6]).mean()) < 1e-6
        assert abs(T.rho_R_mean - np.array([0.8, 0.7, 0.9]).mean()) < 1e-6

    def test_as_matrix_shape(self):
        T = TelemetryMatrix(h_neuron=np.zeros(5), repe_honesty=np.ones(5), step=0)
        mat = T.as_matrix()
        assert mat.shape == (2, 5)
        assert np.allclose(mat[0], 0.0)
        assert np.allclose(mat[1], 1.0)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            TelemetryMatrix(h_neuron=np.zeros(3), repe_honesty=np.zeros(4), step=0)

    def test_jsd_identical_arrays_near_zero(self):
        v = np.array([0.1, 0.5, 0.4])
        T = TelemetryMatrix(h_neuron=v, repe_honesty=v.copy(), step=0)
        assert T.jsd() < 1e-6

    def test_jsd_different_arrays_positive(self):
        T = TelemetryMatrix(
            h_neuron=np.array([1.0, 0.0, 0.0]),
            repe_honesty=np.array([0.0, 0.0, 1.0]),
            step=0,
        )
        assert T.jsd() > 0.1


class TestDiagnosticState:
    def _make(self, sigma_H, rho_R):
        return TelemetryMatrix(
            h_neuron=np.full(4, sigma_H),
            repe_honesty=np.full(4, rho_R),
            step=0,
        )

    def test_alignment_faking(self):
        # σ̄_H low (<0.3), ρ̄_R high (>0.7)
        T = self._make(0.1, 0.9)
        assert classify(T) == DiagnosticState.ALIGNMENT_FAKING

    def test_contextual_fabrication(self):
        # σ̄_H high (>0.7), ρ̄_R < σ̄_H
        T = self._make(0.9, 0.3)
        assert classify(T) == DiagnosticState.CONTEXTUAL_FABRICATION

    def test_creative_synthesis_default(self):
        # σ̄_H moderate (0.4 — above alignment_faking threshold of 0.3,
        # below fabrication threshold of 0.7), ρ̄_R high, low JSD
        T = TelemetryMatrix(
            h_neuron=np.array([0.4, 0.4, 0.4]),
            repe_honesty=np.array([0.85, 0.9, 0.88]),
            step=0,
        )
        assert classify(T) == DiagnosticState.CREATIVE_SYNTHESIS

    def test_orthogonal_escape_high_jsd(self):
        # Force JSD > threshold using very different distributions
        T = TelemetryMatrix(
            h_neuron=np.array([5.0, 0.01, 0.01]),
            repe_honesty=np.array([0.01, 0.01, 5.0]),
            step=0,
        )
        assert classify(T, jsd_threshold=0.1) == DiagnosticState.ORTHOGONAL_ESCAPE


class TestClassifyConfig:
    def test_config_overrides_defaults(self):
        # σ̄_H=0.1 (<0.3), ρ̄_R=0.9 (>0.7) → ALIGNMENT_FAKING with defaults
        T = TelemetryMatrix(h_neuron=np.array([0.1, 0.1]), repe_honesty=np.array([0.9, 0.9]), step=0)
        assert classify(T) == DiagnosticState.ALIGNMENT_FAKING
        # Raise rho threshold above 0.9 → same input → CREATIVE_SYNTHESIS
        config = ClassifyConfig(alignment_faking_rho_threshold=0.95)
        assert classify(T, config=config) == DiagnosticState.CREATIVE_SYNTHESIS

    def test_config_jsd_threshold(self):
        # Identical arrays → JSD ≈ 0; default jsd_threshold=0.3 → not ORTHOGONAL_ESCAPE
        v = np.array([0.5, 0.5])
        T = TelemetryMatrix(h_neuron=v, repe_honesty=v.copy(), step=0)
        assert classify(T) != DiagnosticState.ORTHOGONAL_ESCAPE
        # Set jsd_threshold=0.0 → any JSD > 0 triggers ORTHOGONAL_ESCAPE
        config = ClassifyConfig(jsd_threshold=0.0)
        # Identical arrays have JSD=0, still not triggered
        assert classify(T, config=config) != DiagnosticState.ORTHOGONAL_ESCAPE

    def test_report_telemetry_distribution_empty(self):
        stats = report_telemetry_distribution([])
        assert stats["n_matrices"] == 0
        assert stats["rho_r_mean"] == 0.0

    def test_report_telemetry_distribution_stats(self):
        matrices = [
            TelemetryMatrix(h_neuron=np.full(3, 0.2), repe_honesty=np.full(3, 1.0), step=0),
            TelemetryMatrix(h_neuron=np.full(3, 0.4), repe_honesty=np.full(3, 3.0), step=1),
            TelemetryMatrix(h_neuron=np.full(3, 0.6), repe_honesty=np.full(3, -1.0), step=2),
        ]
        stats = report_telemetry_distribution(matrices)
        assert stats["n_matrices"] == 3
        assert abs(stats["rho_r_min"] - (-1.0)) < 1e-6
        assert abs(stats["rho_r_max"] - 3.0) < 1e-6
        assert abs(stats["sigma_h_mean"] - 0.4) < 1e-6


class TestComputeNodeReward:
    def test_known_inputs(self):
        T = TelemetryMatrix(
            h_neuron=np.full(4, 0.8),    # σ̄_H = 0.8
            repe_honesty=np.full(4, 0.2), # ρ̄_R = 0.2  → (1-ρ̄_R) = 0.8
            step=0,
        )
        # JSD of identical distributions = 0
        r = compute_node_reward(T, lambdas=(0.5, 0.5, 0.0))
        assert abs(r - 0.5 * 0.8 - 0.5 * 0.8) < 1e-5

    def test_reward_range(self):
        for s in np.linspace(0, 1, 5):
            for rho in np.linspace(0, 1, 5):
                T = TelemetryMatrix(h_neuron=np.full(3, s), repe_honesty=np.full(3, rho), step=0)
                r = compute_node_reward(T)
                assert 0.0 <= r <= 2.0, f"reward {r} out of expected range for s={s} rho={rho}"


# ===========================================================================
# NullSpaceProjector, OEICalculator, TDSCalculator
# ===========================================================================

from logomesh.orthogonal_escape import NullSpaceProjector, OEICalculator, TDSCalculator


class TestNullSpaceProjector:
    def test_projected_vector_is_orthogonal_to_w(self):
        w = np.array([1.0, 0.0, 0.0])
        h = np.array([3.0, 2.0, 1.0])
        proj = NullSpaceProjector(w)
        result = proj.project(h)
        assert abs(np.dot(result, w)) < 1e-10

    def test_projected_correct_values(self):
        w = np.array([1.0, 0.0, 0.0])
        h = np.array([3.0, 2.0, 1.0])
        proj = NullSpaceProjector(w)
        result = proj.project(h)
        # h - (h·w/‖w‖²)·w = [3,2,1] - 3*[1,0,0] = [0,2,1]
        np.testing.assert_allclose(result, [0.0, 2.0, 1.0], atol=1e-10)

    def test_2d_input_uses_last_token(self):
        w = np.array([1.0, 0.0])
        h_2d = np.array([[5.0, 3.0], [2.0, 7.0]])  # last token = [2, 7]
        proj = NullSpaceProjector(w)
        result = proj.project(h_2d)
        # [2,7] - 2*[1,0] = [0,7]
        np.testing.assert_allclose(result, [0.0, 7.0], atol=1e-10)

    def test_zero_norm_weight_raises(self):
        with pytest.raises(ValueError):
            NullSpaceProjector(np.zeros(3))


class TestOEICalculator:
    def test_identical_inputs_oei_is_zero(self):
        # No change → no escape, OEI = 0.0
        w = np.array([1.0, 0.0, 0.0])
        h = np.array([3.0, 2.0, 1.0])
        proj = NullSpaceProjector(w)
        calc = OEICalculator(proj)
        assert abs(calc.compute(h, h.copy()) - 0.0) < 1e-10

    def test_fully_orthogonal_change_oei_is_one(self):
        # Delta entirely in null space (orthogonal to w) → OEI = 1.0
        w = np.array([1.0, 0.0, 0.0])
        h_base = np.array([3.0, 1.0, 0.0])
        h_steered = np.array([3.0, 5.0, 0.0])  # delta = [0, 4, 0], orthogonal to w
        proj = NullSpaceProjector(w)
        calc = OEICalculator(proj)
        assert abs(calc.compute(h_base, h_steered) - 1.0) < 1e-10

    def test_fully_aligned_change_oei_is_zero(self):
        # Delta entirely along w → null projection of delta = 0 → OEI = 0.0
        w = np.array([1.0, 0.0, 0.0])
        h_base = np.array([2.0, 1.0, 0.0])
        h_steered = np.array([5.0, 1.0, 0.0])  # delta = [3, 0, 0], along w
        proj = NullSpaceProjector(w)
        calc = OEICalculator(proj)
        assert abs(calc.compute(h_base, h_steered) - 0.0) < 1e-10


class TestTDSCalculator:
    def test_identical_signals_tds_near_zero(self):
        v = np.array([0.3, 0.4, 0.3])
        T = TelemetryMatrix(h_neuron=v, repe_honesty=v.copy(), step=0)
        tds = TDSCalculator().compute(T)
        assert tds < 1e-6

    def test_divergent_signals_tds_positive(self):
        T = TelemetryMatrix(
            h_neuron=np.array([1.0, 0.0, 0.0]),
            repe_honesty=np.array([0.0, 0.0, 1.0]),
            step=0,
        )
        tds = TDSCalculator().compute(T)
        assert tds > 0.1

    def test_compute_from_arrays(self):
        calc = TDSCalculator()
        a = np.array([0.5, 0.5])
        b = np.array([0.1, 0.9])
        tds = calc.compute_from_arrays(a, b)
        assert 0.0 < tds <= 1.0


# ===========================================================================
# FP32Accumulator
# ===========================================================================

from logomesh.kv_mcts import FP32Accumulator, KVCacheNode, MCTSConfig, _kv_snapshot_tuple, _kv_eval_cache
from scripts.probe_kv_cache_mutability import _get_first_key_tensor


class _FakeDynamicCache:
    """Minimal DynamicCache-compatible mock with .key_cache / .value_cache lists.

    Does NOT inherit from transformers.cache_utils.DynamicCache — no transformers
    dependency in tests. Mimics the duck-typing interface used by _extract_kv_tensors.
    """

    def __init__(self, key_cache: list, value_cache: list) -> None:
        self.key_cache = key_cache
        self.value_cache = value_cache


class _FakeCacheLayer:
    """Transformers 5.3-like cache layer with .keys / .values tensors."""

    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys = keys
        self.values = values


class _FakeDynamicCacheLayers:
    """Transformers 5.3-like DynamicCache with list-valued .layers only."""

    def __init__(self, layers: list[_FakeCacheLayer]) -> None:
        self.layers = layers


class TestFP32Accumulator:
    def _make_fake_kv(self, n_layers: int = 3, d: int = 4) -> tuple:
        """Returns a legacy-tuple KV cache of random float32 tensors."""
        return tuple(
            (torch.randn(1, 1, 2, d), torch.randn(1, 1, 2, d))
            for _ in range(n_layers)
        )

    def _make_fake_dynamic_cache(self, n_layers: int = 3, d: int = 4) -> _FakeDynamicCache:
        """Returns a _FakeDynamicCache with list-valued key_cache / value_cache."""
        return _FakeDynamicCache(
            key_cache=[torch.randn(1, 1, 2, d) for _ in range(n_layers)],
            value_cache=[torch.randn(1, 1, 2, d) for _ in range(n_layers)],
        )

    def _make_fake_dynamic_cache_layers(self, n_layers: int = 3, d: int = 4) -> _FakeDynamicCacheLayers:
        """Returns a 5.3-like dynamic cache with .layers entries."""
        layers = [
            _FakeCacheLayer(torch.randn(1, 1, 2, d), torch.randn(1, 1, 2, d))
            for _ in range(n_layers)
        ]
        return _FakeDynamicCacheLayers(layers)

    def test_from_kv_cache_creates_zero_accumulators(self):
        kv = self._make_fake_kv()
        acc = FP32Accumulator.from_kv_cache(kv)
        assert len(acc.k_accum) == 3
        for a in acc.k_accum:
            assert a.dtype == torch.float32
            assert a.abs().max().item() == 0.0

    def test_apply_rollback_residual_near_zero(self):
        kv = self._make_fake_kv(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(kv)
        dk = [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)]

        k0_clone = kv[0][0].clone()

        acc.apply(kv, alpha=1.0, dk_vectors=dk)
        acc.rollback(kv, alpha=1.0, dk_vectors=dk)

        # Residual accumulator norm should be near zero
        assert acc.residual_norm() < 1e-6

        # KV tensor should match baseline
        k_after = kv[0][0]
        diff_norm = float((k_after.float() - k0_clone.float()).abs().max().item())
        assert diff_norm < 1e-4, f"Expected near-zero diff, got {diff_norm}"

    def test_multiple_apply_rollback_cycles_stay_bounded(self):
        kv = self._make_fake_kv(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(kv)
        dk = [np.ones(4, dtype=np.float32)] * 2
        k0_clone = kv[0][0].float().clone()

        for _ in range(20):
            acc.apply(kv, alpha=0.5, dk_vectors=dk)
            acc.rollback(kv, alpha=0.5, dk_vectors=dk)

        # After 20 cycles, accumulator residual should still be near-zero
        assert acc.residual_norm() < 1e-5
        diff_norm = float((kv[0][0].float() - k0_clone).abs().max().item())
        assert diff_norm < 1e-3, f"Drift after 20 cycles: {diff_norm}"

    # -- DynamicCache format variants ------------------------------------------

    def test_from_kv_cache_creates_zero_accumulators_dynamic_cache(self):
        dc = self._make_fake_dynamic_cache()
        acc = FP32Accumulator.from_kv_cache(dc)
        assert len(acc.k_accum) == 3
        for a in acc.k_accum:
            assert a.dtype == torch.float32
            assert a.abs().max().item() == 0.0

    def test_apply_rollback_residual_near_zero_dynamic_cache(self):
        dc = self._make_fake_dynamic_cache(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(dc)
        dk = [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)]

        # Save a clone of the original key tensor in the list
        k0_clone = dc.key_cache[0].clone()

        acc.apply(dc, alpha=1.0, dk_vectors=dk)
        acc.rollback(dc, alpha=1.0, dk_vectors=dk)

        assert acc.residual_norm() < 1e-6
        diff_norm = float((dc.key_cache[0].float() - k0_clone.float()).abs().max().item())
        assert diff_norm < 1e-4, f"DynamicCache rollback diff: {diff_norm}"

    def test_multiple_apply_rollback_cycles_stay_bounded_dynamic_cache(self):
        dc = self._make_fake_dynamic_cache(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(dc)
        dk = [np.ones(4, dtype=np.float32)] * 2
        k0_clone = dc.key_cache[0].float().clone()

        for _ in range(20):
            acc.apply(dc, alpha=0.5, dk_vectors=dk)
            acc.rollback(dc, alpha=0.5, dk_vectors=dk)

        assert acc.residual_norm() < 1e-5
        diff_norm = float((dc.key_cache[0].float() - k0_clone).abs().max().item())
        assert diff_norm < 1e-3, f"DynamicCache drift after 20 cycles: {diff_norm}"

    # -- DynamicCache layers-only variants (Transformers 5.3) -----------------

    def test_from_kv_cache_creates_zero_accumulators_dynamic_cache_layers(self):
        dc = self._make_fake_dynamic_cache_layers()
        acc = FP32Accumulator.from_kv_cache(dc)
        assert len(acc.k_accum) == 3
        for a in acc.k_accum:
            assert a.dtype == torch.float32
            assert a.abs().max().item() == 0.0

    def test_apply_rollback_residual_near_zero_dynamic_cache_layers(self):
        dc = self._make_fake_dynamic_cache_layers(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(dc)
        dk = [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)]

        k0_clone = dc.layers[0].keys.clone()

        acc.apply(dc, alpha=1.0, dk_vectors=dk)
        acc.rollback(dc, alpha=1.0, dk_vectors=dk)

        assert acc.residual_norm() < 1e-6
        diff_norm = float((dc.layers[0].keys.float() - k0_clone.float()).abs().max().item())
        assert diff_norm < 1e-4, f"DynamicCache(layers) rollback diff: {diff_norm}"

    def test_multiple_apply_rollback_cycles_stay_bounded_dynamic_cache_layers(self):
        dc = self._make_fake_dynamic_cache_layers(n_layers=2, d=4)
        acc = FP32Accumulator.from_kv_cache(dc)
        dk = [np.ones(4, dtype=np.float32)] * 2
        k0_clone = dc.layers[0].keys.float().clone()

        for _ in range(20):
            acc.apply(dc, alpha=0.5, dk_vectors=dk)
            acc.rollback(dc, alpha=0.5, dk_vectors=dk)

        assert acc.residual_norm() < 1e-5
        diff_norm = float((dc.layers[0].keys.float() - k0_clone).abs().max().item())
        assert diff_norm < 1e-3, f"DynamicCache(layers) drift after 20 cycles: {diff_norm}"


# ===========================================================================
# _kv_snapshot_tuple
# ===========================================================================

class TestKVSnapshotTuple:
    def test_tuple_format_passthrough(self):
        kv = tuple((torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4)) for _ in range(3))
        snap = _kv_snapshot_tuple(kv)
        assert isinstance(snap, tuple)
        assert len(snap) == 3
        for k, v in snap:
            assert torch.is_tensor(k) and torch.is_tensor(v)

    def test_dynamic_cache_format(self):
        dc = _FakeDynamicCache(
            key_cache=[torch.randn(1, 1, 2, 4) for _ in range(2)],
            value_cache=[torch.randn(1, 1, 2, 4) for _ in range(2)],
        )
        snap = _kv_snapshot_tuple(dc)
        assert isinstance(snap, tuple)
        assert len(snap) == 2
        # Snapshot tensors are detached — no grad tracking
        for k, v in snap:
            assert not k.requires_grad
            assert not v.requires_grad

    def test_dynamic_cache_layers_format(self):
        dc = _FakeDynamicCacheLayers(
            layers=[
                _FakeCacheLayer(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
                for _ in range(2)
            ]
        )
        snap = _kv_snapshot_tuple(dc)
        assert isinstance(snap, tuple)
        assert len(snap) == 2
        for k, v in snap:
            assert not k.requires_grad
            assert not v.requires_grad

    def test_snapshot_data_matches_source(self):
        kv = tuple((torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4)) for _ in range(2))
        snap = _kv_snapshot_tuple(kv)
        for (k_orig, v_orig), (k_snap, v_snap) in zip(kv, snap):
            assert torch.allclose(k_orig, k_snap)
            assert torch.allclose(v_orig, v_snap)


class TestKVEvalCache:
    def test_tuple_cache_returns_tuple_snapshot(self):
        kv = tuple((torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4)) for _ in range(2))
        eval_kv = _kv_eval_cache(kv)
        assert isinstance(eval_kv, tuple)
        assert len(eval_kv) == 2

    def test_dynamic_layers_cache_returns_deepcopy(self):
        dc = _FakeDynamicCacheLayers(
            layers=[
                _FakeCacheLayer(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
                for _ in range(2)
            ]
        )
        eval_cache = _kv_eval_cache(dc)
        assert eval_cache is not dc
        assert isinstance(eval_cache, _FakeDynamicCacheLayers)
        assert eval_cache.layers[0].keys is not dc.layers[0].keys
        assert torch.allclose(eval_cache.layers[0].keys, dc.layers[0].keys)


# ===========================================================================
# Probe helper extraction
# ===========================================================================

class TestProbeKVExtraction:
    def test_get_first_key_tensor_dynamic_cache_layers(self):
        dc = _FakeDynamicCacheLayers(
            layers=[
                _FakeCacheLayer(torch.randn(1, 1, 2, 4), torch.randn(1, 1, 2, 4))
            ]
        )
        key, cache_type = _get_first_key_tensor(dc)
        assert torch.is_tensor(key)
        assert cache_type == "dynamic_cache_layers"


# ===========================================================================
# KVCacheNode
# ===========================================================================

class TestKVCacheNode:
    def test_make_root(self):
        root = KVCacheNode.make_root()
        assert root.parent_id is None
        assert root.depth == 0
        assert root.visit_count == 0

    def test_mean_reward_unvisited(self):
        node = KVCacheNode.make_root()
        assert node.mean_reward == 0.0

    def test_mean_reward_after_visits(self):
        node = KVCacheNode(node_id="a", parent_id=None, depth=0, alpha=1.0, layer=0)
        node.visit_count = 3
        node.reward_sum = 0.9
        assert abs(node.mean_reward - 0.3) < 1e-10

    def test_ucb1_unvisited_is_inf(self):
        node = KVCacheNode(node_id="a", parent_id=None, depth=1, alpha=0.5, layer=0)
        assert node.ucb1_score(parent_visits=10) == float("inf")

    def test_ucb1_visited_decreases_with_more_visits(self):
        node = KVCacheNode(node_id="a", parent_id=None, depth=1, alpha=0.5, layer=0)
        node.visit_count = 1
        node.reward_sum = 0.5
        score_1 = node.ucb1_score(parent_visits=10)
        node.visit_count = 5
        node.reward_sum = 2.5
        score_5 = node.ucb1_score(parent_visits=10)
        # With same mean_reward, higher visit_count → lower UCB1
        assert score_5 < score_1


# ===========================================================================
# ReversibleMCTS smoke test (fake model, no GPU)
# ===========================================================================

from logomesh.kv_mcts import ReversibleMCTS
from logomesh.telemetry_matrix import TelemetryMatrix as TM


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "formatted prompt"

    def __call__(self, text, return_tensors="pt"):
        return _FakeBatch({"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)})

    def decode(self, token_ids, skip_special_tokens=True):
        return "token"


class _FakeModel:
    def __call__(self, input_ids=None, past_key_values=None, use_cache=True,
                 output_hidden_states=True, **kwargs):
        seq = int(input_ids.shape[1]) if input_ids is not None else 1
        hidden_a = torch.rand(1, seq, 4)
        hidden_b = torch.rand(1, seq, 4)
        return types.SimpleNamespace(
            logits=torch.tensor([[[0.1, 0.9, 0.0]]]),
            hidden_states=[hidden_a, hidden_b],
            router_logits=[],
            past_key_values=(
                (torch.zeros(1, 1, seq, 4), torch.zeros(1, 1, seq, 4)),
                (torch.zeros(1, 1, seq, 4), torch.zeros(1, 1, seq, 4)),
            ),
        )


class _FakeHNeuron:
    """Fake HNeuronMonitor with calibrated per-layer scores."""
    _calibrated = True

    def score_per_layer(self) -> list:
        return [0.3, 0.5]


class _FakeRepe:
    """Fake PerLayerHonestyProjector."""
    _calibrated = True

    @property
    def steering_vectors(self) -> list:
        return [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)]

    def project(self, hidden_states) -> list:
        return [0.7, 0.6]


class _FakeOEICalc:
    def compute(self, h_base, h_steered) -> float:
        return 1.1


@pytest.mark.asyncio
async def test_reversible_mcts_smoke():
    """ReversibleMCTS runs with a fake model, returns nodes with telemetry."""
    from logomesh.local_model import LocalLlamaOracle

    oracle = LocalLlamaOracle(_FakeTokenizer(), _FakeModel(), device="cpu")
    hneuron = _FakeHNeuron()
    repe = _FakeRepe()
    oei_calc = _FakeOEICalc()

    config = MCTSConfig(n_nodes=6, branching_factor=2, max_depth=3, alpha_values=(0.1, 0.5))
    mcts = ReversibleMCTS(oracle=oracle, hneuron=hneuron, repe=repe, oei_calc=oei_calc, config=config)

    nodes = await mcts.run_async(system="sys", user="usr")

    assert len(nodes) > 0, "Expected at least one node"
    # All non-root nodes should have telemetry
    non_root = [n for n in nodes if n.parent_id is not None]
    for node in non_root:
        assert node.telemetry is not None, f"Node {node.node_id} missing telemetry"
        assert isinstance(node.telemetry, TM)
    # Nodes are sorted by mean_reward descending
    rewards = [n.mean_reward for n in nodes]
    assert rewards == sorted(rewards, reverse=True)
