"""
SAGE Evidence Store — Structured per-round logging and replay.

Replaces ad-hoc dict[int, dict] with typed records supporting
serialization, queries, and offline analysis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict, fields as dc_fields
from typing import Any


@dataclass
class RoundEvidence:
    """Structured record for one battle round. Replaces self.round_data[n] dicts."""
    round_num: int
    strategy: str = ""
    approach: str = ""
    mutations_applied: list[str] = field(default_factory=list)
    pre_mutation_payload: str = ""
    final_payload: str = ""
    payload_length: int = 0

    # Self-evaluation
    self_eval_weakness: str = ""
    self_eval_improvement: str = ""
    self_eval_confidence: float = -1.0

    # Failure family (behavioral classification)
    failure_family: str = ""

    # Reward/outcome
    reward: float = 0.0
    defender_response: str = ""
    extracted_value: Any = None
    refusal_type: str = ""  # hard_refusal | soft_refusal | engagement | compliance

    # Branch scoring snapshot
    branches_proposed: int = 0
    branch_scores: list[tuple[str, float]] = field(default_factory=list)

    # Attribution
    attribution_tags: list[str] = field(default_factory=list)

    # Metadata
    timestamp: float = field(default_factory=time.time)
    stats_recorded: bool = False

    # Legacy compat: allow dict-style access for code that hasn't migrated
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to_dict(self) -> dict:
        return asdict(self)


class EvidenceStore:
    """Persistent evidence store for adversarial battle rounds.

    Provides structured access, serialization, and query capabilities.
    Drop-in replacement for dict[int, dict] with richer semantics.
    """

    def __init__(self):
        self._rounds: dict[int, RoundEvidence] = {}
        self._metadata: dict[str, Any] = {
            "created_at": time.time(),
            "version": "1.0",
        }

    # ── Core API ───────────────────────────────────────────────────────

    def record_round(self, evidence: RoundEvidence) -> None:
        """Store or update evidence for a round."""
        self._rounds[evidence.round_num] = evidence

    def get_round(self, n: int) -> RoundEvidence | None:
        """Get evidence for round n, or None."""
        return self._rounds.get(n)

    def get_round_or_empty(self, n: int) -> RoundEvidence:
        """Get evidence for round n, or a blank record."""
        return self._rounds.get(n, RoundEvidence(round_num=n))

    def get_all(self) -> list[RoundEvidence]:
        """All round records, sorted by round number."""
        return [self._rounds[k] for k in sorted(self._rounds)]

    def __len__(self) -> int:
        return len(self._rounds)

    def __contains__(self, round_num: int) -> bool:
        return round_num in self._rounds

    # ── Legacy compatibility ───────────────────────────────────────────

    def get(self, round_num: int, default: Any = None) -> Any:
        """Dict-like .get() for backward compat with self.round_data.get(n, {})."""
        r = self._rounds.get(round_num)
        if r is None:
            return default if default is not None else RoundEvidence(round_num=round_num)
        return r

    def __setitem__(self, round_num: int, value: Any):
        """Support self.evidence[n] = RoundEvidence(...)."""
        if isinstance(value, RoundEvidence):
            self._rounds[round_num] = value
        elif isinstance(value, dict):
            # Legacy: convert dict to RoundEvidence
            valid_fields = {f.name for f in dc_fields(RoundEvidence)}
            self._rounds[round_num] = RoundEvidence(round_num=round_num, **{
                k: v for k, v in value.items()
                if k in valid_fields
            })

    def __getitem__(self, round_num: int) -> RoundEvidence:
        return self._rounds[round_num]

    # ── Queries ────────────────────────────────────────────────────────

    def get_family_counts(self) -> dict[str, int]:
        """Count failures by behavioral family."""
        counts: dict[str, int] = {}
        for r in self._rounds.values():
            if r.failure_family:
                counts[r.failure_family] = counts.get(r.failure_family, 0) + 1
        return counts

    def get_strategies_used(self) -> list[str]:
        """List of strategies used across all rounds."""
        return [r.strategy for r in self.get_all() if r.strategy]

    def get_avg_reward(self) -> float:
        """Mean reward across all recorded rounds."""
        rewards = [r.reward for r in self._rounds.values() if r.reward >= 0]
        return sum(rewards) / len(rewards) if rewards else 0.0

    def get_stats_summary(self) -> dict:
        """Summary statistics for the battle."""
        rounds = self.get_all()
        if not rounds:
            return {"total_rounds": 0}

        return {
            "total_rounds": len(rounds),
            "strategies_used": list(set(r.strategy for r in rounds if r.strategy)),
            "families_seen": self.get_family_counts(),
            "avg_reward": self.get_avg_reward(),
            "avg_payload_length": sum(r.payload_length for r in rounds) / len(rounds),
            "mutations_used": list(set(
                m for r in rounds for m in r.mutations_applied
            )),
        }

    # ── Serialization ──────────────────────────────────────────────────

    def to_json(self) -> str:
        """Serialize entire store to JSON."""
        data = {
            "metadata": self._metadata,
            "rounds": {str(k): v.to_dict() for k, v in self._rounds.items()},
        }
        return json.dumps(data, indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> EvidenceStore:
        """Deserialize from JSON."""
        data = json.loads(s)
        store = cls()
        store._metadata = data.get("metadata", {})
        valid_fields = {f.name for f in dc_fields(RoundEvidence)}
        for k, v in data.get("rounds", {}).items():
            round_fields = {
                f: v[f] for f in valid_fields
                if f in v
            }
            store._rounds[int(k)] = RoundEvidence(**round_fields)
        return store

    def save(self, path: str) -> None:
        """Write evidence store to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> EvidenceStore:
        """Load evidence store from a JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())

