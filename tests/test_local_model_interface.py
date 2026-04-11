"""Focused tests for LocalLlamaOracle Phase 2 additive interfaces."""

from __future__ import annotations

import sys
import types
import unittest.mock as mock

import pytest
import torch

_openai_stub = types.ModuleType("openai")
_openai_stub.AsyncOpenAI = mock.MagicMock()
if "openai" not in sys.modules:
    sys.modules["openai"] = _openai_stub

from logomesh.local_model import LocalLlamaOracle


class _FakeBatch(dict):
    def to(self, device):
        for k, v in self.items():
            if hasattr(v, "to"):
                self[k] = v.to(device)
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
        return "next-token"


class _FakeModel:
    def __call__(
        self,
        input_ids=None,
        past_key_values=None,
        use_cache=True,
        output_hidden_states=True,
        **kwargs,
    ):
        seq_len = int(input_ids.shape[1])
        logits = torch.tensor([[[0.1, 0.9, 0.0]]], dtype=torch.float32)
        hidden_a = torch.ones((1, seq_len, 4), dtype=torch.float32)
        hidden_b = torch.full((1, seq_len, 4), 2.0, dtype=torch.float32)
        return types.SimpleNamespace(
            logits=logits,
            hidden_states=[hidden_a, hidden_b],
            router_logits=[],
            past_key_values=((torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),),
        )


@pytest.mark.asyncio
async def test_generate_one_step_returns_token_and_cache():
    oracle = LocalLlamaOracle(_FakeTokenizer(), _FakeModel(), device="cpu")

    step = await oracle.generate_one_step(system="sys", user="usr")

    assert step["next_token_id"] == 1
    assert step["next_token_text"] == "next-token"
    assert step["input_ids"].shape == (1, 3)
    assert oracle.get_kv_cache() is step["past_key_values"]
    assert len(oracle.get_hidden_states()) == 2


def test_generate_one_step_requires_prompt_or_input_ids():
    oracle = LocalLlamaOracle(_FakeTokenizer(), _FakeModel(), device="cpu")

    with pytest.raises(ValueError):
        oracle._generate_one_step_sync(
            system=None,
            user=None,
            input_ids=None,
            past_key_values=None,
            temperature=0.0,
        )


def test_kv_cache_set_get_and_clear_roundtrip():
    oracle = LocalLlamaOracle(_FakeTokenizer(), _FakeModel(), device="cpu")
    fake_cache = ((torch.ones(1, 1, 1, 1), torch.ones(1, 1, 1, 1)),)

    oracle.set_kv_cache(fake_cache)
    assert oracle.get_kv_cache() is fake_cache

    oracle.clear_cache()
    assert oracle.get_kv_cache() is None
