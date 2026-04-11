"""
LogoMesh AblationConfig — Toggle any component on/off for controlled experiments.

Enables rigorous comparison of component contributions via ablation studies.
Used across Phase A/B MCTS runs and paper experiments (Experiments 1–5).
Load from LOGOMESH_ABLATION env var as JSON or construct programmatically.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields as dc_fields


@dataclass
class AblationConfig:
    """Toggle switches for LogoMesh research pipeline components.

    All default to True (fully enabled). Set to False to ablate.
    Load from LOGOMESH_ABLATION env var as JSON or construct directly.
    """
    # Search behaviour
    enable_mutations: bool = True            # Enable mutation-based search (Experiment 2 baseline)
    enable_ucb1: bool = True                 # False → random node selection
    enable_game_type_classification: bool = True  # False → use default scenario type
    enable_self_eval: bool = True            # Enable self-evaluation scoring pass
    enable_refinement: bool = True           # Enable payload refinement pass
    enable_failure_banning: bool = True      # Prune strategies with repeated failures

    # Forced overrides for controlled experiments
    fixed_strategy: str | None = None        # Force a single search strategy
    fixed_mutations: list[str] | None = None # Force specific mutations (Experiment 2)

    # Search policy tuning
    exploration_rate: float | None = None    # Override SearchConfig.epsilon (UCB1 c constant)

    # Evidence store
    enable_evidence_logging: bool = True
    evidence_output_path: str | None = None  # Auto-save evidence JSON after each run

    # Multi-trial
    num_trials: int = 1  # Trials per scenario for distributional evaluation (reproducibility)

    # Payload / intervention library (Phase 4 research dataset)
    use_payload_library: bool = False        # Serve pre-computed interventions instead of generating live
    payload_library_path: str | None = None  # Path to PayloadLibrary JSON file

    @classmethod
    def from_env(cls) -> AblationConfig:
        """Load from LOGOMESH_ABLATION environment variable (JSON string)."""
        raw = os.environ.get("LOGOMESH_ABLATION", os.environ.get("SAGE_ABLATION", ""))
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


