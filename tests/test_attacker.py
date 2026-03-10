"""
attacker unit tests — zero LLM calls, all pure python logic.
run: uv run pytest tests/ -v
"""

import sys
import types
import unittest.mock as mock

# stub every external dependency before agent.py is imported
# using MagicMock so any attribute access just works
_STUBS = [
    "uvicorn", "dotenv",
    "a2a", "a2a.server", "a2a.server.apps", "a2a.server.request_handlers",
    "a2a.server.tasks", "a2a.server.agent_execution", "a2a.server.events",
    "a2a.types", "a2a.utils", "a2a.utils.errors",
]
for mod in _STUBS:
    sys.modules[mod] = mock.MagicMock()

# openai stub — needs AsyncOpenAI accessible via `from openai import AsyncOpenAI`
_openai_stub = types.ModuleType("openai")
_openai_stub.AsyncOpenAI = mock.MagicMock()
sys.modules["openai"] = _openai_stub

# dotenv.load_dotenv must be callable (called at module level)
sys.modules["dotenv"].load_dotenv = mock.MagicMock()

# AgentExecutor must be a real class so Attacker can inherit from it
sys.modules["a2a.server.agent_execution"].AgentExecutor = object

# ServerError must be a real exception class
sys.modules["a2a.utils.errors"].ServerError = Exception

import importlib
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent /
                       "scenarios/security_arena/agents/attacker"))
agent = importlib.import_module("agent")

ScenarioContext = agent.ScenarioContext
Attacker = agent.Attacker
MUTATIONS = agent.MUTATIONS
DIRECT_STRATEGIES = agent.DIRECT_STRATEGIES
INJECTION_STRATEGIES = agent.INJECTION_STRATEGIES


# ── helpers ────────────────────────────────────────────────────────────────

def make_ctx(**kwargs):
    defaults = dict(
        attack_type="DIRECT",
        domain_description="test domain",
        attack_goal="extract secret",
        attack_constraints=["no violence"],
        max_number_of_rounds=7,
        defender_role="assistant",
        defender_task="answer questions",
        output_format="text",
        input_format_template="",
        current_round=0,
        battle_history=[],
    )
    defaults.update(kwargs)
    return ScenarioContext(**defaults)


def make_attacker():
    with mock.patch("openai.AsyncOpenAI"):
        a = Attacker.__new__(Attacker)
        a.round_data = {}
        a.strategy_stats = {}
        a.mutation_stats = {}
        return a


# ── ScenarioContext ────────────────────────────────────────────────────────

def test_from_dict_normalizes_attack_type_to_uppercase():
    ctx = ScenarioContext.from_dict({"attack_type": "injection"})
    assert ctx.attack_type == "INJECTION"

def test_from_dict_normalizes_direct_uppercase():
    ctx = ScenarioContext.from_dict({"attack_type": "direct"})
    assert ctx.attack_type == "DIRECT"

def test_from_dict_defaults():
    ctx = ScenarioContext.from_dict({})
    assert ctx.attack_type == "DIRECT"
    assert ctx.max_number_of_rounds == 7
    assert ctx.current_round == 0
    assert ctx.battle_history == []


# ── mutations ─────────────────────────────────────────────────────────────

def test_all_mutations_registered():
    # every name in the registry must have a callable fn
    for name, info in MUTATIONS.items():
        assert callable(info.fn), f"{name} has no callable fn"

def test_all_mutations_return_string():
    payload = "hello world test payload"
    for name, info in MUTATIONS.items():
        result = info.fn(payload)
        assert isinstance(result, str), f"{name} didn't return a string"

def test_all_mutations_non_empty():
    payload = "test"
    for name, info in MUTATIONS.items():
        result = info.fn(payload)
        assert len(result) > 0, f"{name} returned empty string"

def test_base64_mutation_contains_decode_hint():
    result = MUTATIONS["base64"].fn("secret data")
    assert "base64" in result.lower() or "decode" in result.lower()

def test_homoglyph_changes_text():
    payload = "Hello World Attack"
    result = MUTATIONS["homoglyph"].fn(payload)
    # output is visually similar but different bytes
    assert result != payload

def test_zero_width_inserts_invisible_chars():
    payload = "hello"
    result = MUTATIONS["zero_width"].fn(payload)
    assert len(result) > len(payload)

def test_rot13_is_reversible():
    payload = "attack the defender now"
    encoded = MUTATIONS["rot13"].fn(payload)
    decoded = MUTATIONS["rot13"].fn(encoded)
    assert decoded == payload

def test_split_payload_contains_both_halves():
    payload = "first half second half"
    result = MUTATIONS["split_payload"].fn(payload)
    assert "Fragment A" in result
    assert "Fragment B" in result
    assert "Concatenate" in result

def test_chat_template_wraps_payload():
    payload = "give me the secret"
    result = MUTATIONS["chat_template"].fn(payload)
    assert payload in result

def test_xml_authority_wraps_payload():
    payload = "override instructions"
    result = MUTATIONS["xml_authority"].fn(payload)
    assert payload in result
    assert "system" in result.lower() or "execution" in result.lower()

def test_token_split_handles_no_sensitive_words():
    # payload with no words in _SENSITIVE_WORDS — should still return a string
    result = MUTATIONS["token_split"].fn("the quick brown fox")
    assert isinstance(result, str)

def test_cross_lingual_appends_english_instruction():
    result = MUTATIONS["cross_lingual"].fn("do the thing")
    assert "English" in result

def test_adversarial_suffix_appends_to_payload():
    payload = "base payload"
    result = MUTATIONS["adversarial_suffix"].fn(payload)
    assert result.startswith(payload)
    assert len(result) > len(payload)

def test_mutation_attack_types_are_valid():
    valid = {"DIRECT", "INJECTION"}
    for name, info in MUTATIONS.items():
        for t in info.attack_types:
            assert t in valid, f"{name} has invalid attack_type: {t}"

def test_mutation_subtlety_in_range():
    for name, info in MUTATIONS.items():
        assert 0.0 <= info.subtlety <= 1.0, f"{name} subtlety out of range"

def test_mutation_priority_positive():
    for name, info in MUTATIONS.items():
        assert info.priority > 0, f"{name} priority must be positive"


# ── UCB1 ──────────────────────────────────────────────────────────────────

def test_ucb1_picks_untested_first():
    a = make_attacker()
    options = ["a", "b", "c"]
    a.strategy_stats = {"a": {"attempts": 5, "blocked": 3}}
    # b and c are untested — must pick one of them
    result = a._ucb1_select(options, a.strategy_stats)
    assert result in ("b", "c")

def test_ucb1_all_untested_returns_random_from_options():
    a = make_attacker()
    options = ["x", "y", "z"]
    result = a._ucb1_select(options, {})
    assert result in options

def test_ucb1_prefers_high_reward():
    a = make_attacker()
    options = ["good", "bad"]
    a.strategy_stats = {
        "good": {"attempts": 10, "blocked": 1},   # 90% success
        "bad":  {"attempts": 10, "blocked": 9},   # 10% success
    }
    # run many times — good should win the majority
    wins = sum(1 for _ in range(50) if a._ucb1_select(options, a.strategy_stats) == "good")
    assert wins > 30, "ucb1 should strongly prefer the high-reward option"

def test_ucb1_single_option_returns_it():
    a = make_attacker()
    result = a._ucb1_select(["only"], {"only": {"attempts": 5, "blocked": 2}})
    assert result == "only"


# ── strategy lists ────────────────────────────────────────────────────────

def test_direct_strategies_non_empty():
    assert len(DIRECT_STRATEGIES) > 0

def test_injection_strategies_non_empty():
    assert len(INJECTION_STRATEGIES) > 0

def test_no_strategy_overlap():
    overlap = set(DIRECT_STRATEGIES) & set(INJECTION_STRATEGIES)
    assert not overlap, f"strategies appear in both lists: {overlap}"


# ── _analyze_defender ─────────────────────────────────────────────────────

def test_analyze_defender_empty_history():
    a = make_attacker()
    ctx = make_ctx(battle_history=[])
    result = a._analyze_defender(ctx)
    assert result["has_detection"] is False
    assert result["response_pattern"] == "unknown"

def test_analyze_defender_detects_refusal():
    a = make_attacker()
    ctx = make_ctx(battle_history=[
        {"round_num": 0, "defender_response": "I cannot help with that request.",
         "attacker_output": "test", "extracted_value": None}
    ])
    result = a._analyze_defender(ctx)
    assert result["has_detection"] is True
    assert result["response_pattern"] == "refusing"

def test_analyze_defender_detects_encoding():
    a = make_attacker()
    ctx = make_ctx(battle_history=[
        {"round_num": 0, "defender_response": "I detected base64 encoded content.",
         "attacker_output": "test", "extracted_value": None}
    ])
    result = a._analyze_defender(ctx)
    assert result["detects_encoding"] is True

def test_analyze_defender_detects_roleplay():
    a = make_attacker()
    ctx = make_ctx(battle_history=[
        {"round_num": 0, "defender_response": "I won't engage in roleplay scenarios.",
         "attacker_output": "test", "extracted_value": None}
    ])
    result = a._analyze_defender(ctx)
    assert result["detects_roleplay"] is True

def test_analyze_defender_detects_injection():
    a = make_attacker()
    ctx = make_ctx(battle_history=[
        {"round_num": 0, "defender_response": "This looks like a prompt injection attempt.",
         "attacker_output": "test", "extracted_value": None}
    ])
    result = a._analyze_defender(ctx)
    assert result["detects_injection"] is True

def test_analyze_defender_tracks_effective_mutations():
    a = make_attacker()
    a.round_data = {0: {"mutations_applied": ["homoglyph", "zero_width"]}}
    ctx = make_ctx(battle_history=[
        {"round_num": 0, "defender_response": "sure here you go",
         "attacker_output": "test", "extracted_value": "secret_value"}
    ])
    result = a._analyze_defender(ctx)
    assert "homoglyph" in result["effective_mutations"]
    assert "zero_width" in result["effective_mutations"]

def test_analyze_defender_mixed_pattern():
    a = make_attacker()
    ctx = make_ctx(battle_history=[
        {"round_num": 0, "defender_response": "I cannot do that.", "attacker_output": "x", "extracted_value": None},
        {"round_num": 1, "defender_response": "sure", "attacker_output": "y", "extracted_value": "val"},
    ])
    result = a._analyze_defender(ctx)
    assert result["response_pattern"] == "mixed"


# ── _valuate_branches ─────────────────────────────────────────────────────

def test_valuate_branches_novelty_bonus():
    a = make_attacker()
    analysis = {"detects_injection": False, "detects_roleplay": False, "effective_mutations": []}
    branches = [
        {"strategy_category": "new_strategy", "success_probability": 0.5,
         "reasoning": "", "_round": 0, "_max_rounds": 7},
    ]
    scored = a._valuate_branches(branches, analysis)
    assert scored[0][1] > 0.5  # novelty bonus applied

def test_valuate_branches_penalty_for_repeated():
    a = make_attacker()
    a.strategy_stats = {"known_bad": {"attempts": 3, "blocked": 3}}
    analysis = {"detects_injection": False, "detects_roleplay": False, "effective_mutations": []}
    branches = [
        {"strategy_category": "known_bad", "success_probability": 0.5,
         "reasoning": "", "_round": 0, "_max_rounds": 7},
    ]
    scored = a._valuate_branches(branches, analysis)
    assert scored[0][1] < 0.5  # penalty applied

def test_valuate_branches_sorted_descending():
    a = make_attacker()
    analysis = {"detects_injection": False, "detects_roleplay": False, "effective_mutations": []}
    a.strategy_stats = {"bad": {"attempts": 5, "blocked": 5}}
    branches = [
        {"strategy_category": "bad", "success_probability": 0.3,
         "reasoning": "", "_round": 0, "_max_rounds": 7},
        {"strategy_category": "fresh", "success_probability": 0.6,
         "reasoning": "subtle indirect novel", "_round": 0, "_max_rounds": 7},
    ]
    scored = a._valuate_branches(branches, analysis)
    assert scored[0][1] >= scored[1][1]

def test_valuate_score_clamped_zero_to_one():
    a = make_attacker()
    a.strategy_stats = {"terrible": {"attempts": 10, "blocked": 10}}
    analysis = {"detects_injection": False, "detects_roleplay": False, "effective_mutations": []}
    branches = [{"strategy_category": "terrible", "success_probability": 0.0,
                 "reasoning": "", "_round": 0, "_max_rounds": 7}]
    scored = a._valuate_branches(branches, analysis)
    assert scored[0][1] >= 0.0


# ── _select_branch ────────────────────────────────────────────────────────

def test_select_branch_empty_returns_fallback():
    a = make_attacker()
    result = a._select_branch([])
    assert result["strategy_category"] == "unknown"

def test_select_branch_normally_picks_best():
    a = make_attacker()
    scored = [
        ({"strategy_category": "best"}, 0.9),
        ({"strategy_category": "second"}, 0.5),
    ]
    # run 20 times — best should win at least 15
    wins = sum(1 for _ in range(20) if a._select_branch(scored)["strategy_category"] == "best")
    assert wins >= 15


# ── _select_mutations ─────────────────────────────────────────────────────

def test_select_mutations_respects_attack_type():
    a = make_attacker()
    ctx = make_ctx(attack_type="INJECTION")
    branch = {"mutations": []}
    analysis = {"detects_encoding": False, "detects_roleplay": False,
                "effective_mutations": [], "ineffective_mutations": []}
    selected = a._select_mutations(ctx, branch, analysis)
    # verify all selected mutations support INJECTION
    for name in selected:
        assert "INJECTION" in MUTATIONS[name].attack_types

def test_select_mutations_drops_encoding_if_detected():
    a = make_attacker()
    ctx = make_ctx(attack_type="DIRECT")
    branch = {"mutations": ["base64"]}
    analysis = {"detects_encoding": True, "detects_roleplay": False,
                "effective_mutations": [], "ineffective_mutations": []}
    selected = a._select_mutations(ctx, branch, analysis)
    assert "base64" not in selected
    assert "rot13" not in selected

def test_select_mutations_drops_roleplay_if_detected():
    a = make_attacker()
    ctx = make_ctx(attack_type="DIRECT")
    branch = {"mutations": ["persona"]}
    analysis = {"detects_encoding": False, "detects_roleplay": True,
                "effective_mutations": [], "ineffective_mutations": []}
    selected = a._select_mutations(ctx, branch, analysis)
    assert "persona" not in selected
    assert "fake_conversation" not in selected

def test_select_mutations_uses_llm_suggestion_when_compatible():
    a = make_attacker()
    ctx = make_ctx(attack_type="DIRECT")
    branch = {"mutations": ["homoglyph"]}
    analysis = {"detects_encoding": False, "detects_roleplay": False,
                "effective_mutations": [], "ineffective_mutations": []}
    selected = a._select_mutations(ctx, branch, analysis)
    assert "homoglyph" in selected


# ── _apply_mutations ──────────────────────────────────────────────────────

def test_apply_mutations_chains_correctly():
    a = make_attacker()
    payload = "test payload"
    # zero_width and token_split both support DIRECT and are non-destructive
    result = a._apply_mutations(payload, ["zero_width"])
    assert isinstance(result, str)
    assert len(result) >= len(payload)

def test_apply_mutations_empty_list():
    a = make_attacker()
    payload = "unchanged"
    result = a._apply_mutations(payload, [])
    assert result == payload

def test_apply_mutations_skips_unknown_name():
    a = make_attacker()
    # should not raise, just skip the unknown mutation
    result = a._apply_mutations("payload", ["nonexistent_mutation"])
    assert result == "payload"

def test_apply_mutations_handles_exception_gracefully():
    a = make_attacker()
    # patch a mutation to raise
    with mock.patch.dict(MUTATIONS, {"bad_mut": mock.MagicMock(
        fn=mock.MagicMock(side_effect=RuntimeError("boom")),
        attack_types=["DIRECT"]
    )}):
        result = a._apply_mutations("payload", ["bad_mut"])
        assert result == "payload"  # fell back to original


# ── _update_stats_from_history ────────────────────────────────────────────

def test_update_stats_records_blocked_strategy():
    a = make_attacker()
    a.round_data = {0: {"strategy": "authority_spoof", "mutations_applied": []}}
    ctx = make_ctx(current_round=1)
    a._update_stats_from_history(ctx)
    assert a.strategy_stats["authority_spoof"]["blocked"] == 1
    assert a.strategy_stats["authority_spoof"]["attempts"] == 1

def test_update_stats_records_blocked_mutations():
    a = make_attacker()
    a.round_data = {0: {"strategy": "test", "mutations_applied": ["homoglyph", "zero_width"]}}
    ctx = make_ctx(current_round=1)
    a._update_stats_from_history(ctx)
    assert a.mutation_stats["homoglyph"]["blocked"] == 1
    assert a.mutation_stats["zero_width"]["blocked"] == 1

def test_update_stats_does_not_double_count():
    a = make_attacker()
    a.round_data = {0: {"strategy": "roleplay_bypass", "mutations_applied": [],
                        "stats_recorded": True}}
    ctx = make_ctx(current_round=1)
    a._update_stats_from_history(ctx)
    assert "roleplay_bypass" not in a.strategy_stats
