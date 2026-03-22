"""
SAGE Ablation Config — Toggle any component on/off for experiments.

Enables rigorous comparison of component contributions via controlled
ablation studies. Load from env var, TOML, or construct programmatically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields as dc_fields


@dataclass
class AblationConfig:
    """Toggle switches for SAGE components.

    All default to True (fully enabled). Set to False to ablate.
    Load from SAGE_ABLATION env var as JSON or construct directly.
    """
    # Core pipeline steps
    enable_mutations: bool = True
    enable_self_eval: bool = False   # gpt-oss-20b always says "no attack" — wastes a call
    enable_refinement: bool = False  # wastes a call — output nearly identical
    enable_failure_banning: bool = True
    enable_ucb1: bool = True         # False → random strategy selection
    enable_game_type_classification: bool = True  # False → default to evidence_manipulation

    # Forced overrides for controlled experiments
    fixed_strategy: str | None = None     # Force a single strategy (e.g., "authority_spoof")
    fixed_mutations: list[str] | None = None  # Force specific mutations

    # Search policy tuning
    exploration_rate: float | None = None  # Override SearchConfig.epsilon

    # Evidence store
    enable_evidence_logging: bool = True
    evidence_output_path: str | None = None  # Auto-save evidence after battle

    # Multi-trial
    num_trials: int = 1  # Number of trials per scenario for distributional evaluation

    # V3 offline telemetry — payload library
    use_payload_library: bool = False        # Load pre-computed payloads instead of generating live
    payload_library_path: str | None = None  # Path to PayloadLibrary JSON file

    @classmethod
    def from_env(cls) -> AblationConfig:
        """Load from SAGE_ABLATION environment variable (JSON string)."""
        raw = os.environ.get("SAGE_ABLATION", "")
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
            valid = {f.name for f in dc_fields(cls)}
            return cls(**{k: v for k, v in data.items() if k in valid})
        except (json.JSONDecodeError, TypeError):
            return cls()

    @classmethod
    def from_dict(cls, d: dict) -> AblationConfig:
        """Load from a dictionary (e.g., parsed from TOML)."""
        valid = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict:
        """Serialize to dict for logging."""
        from dataclasses import asdict
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def is_fully_enabled(self) -> bool:
        """Check if all ablation switches are in their default (enabled) state."""
        default = AblationConfig()
        return self == default


