"""
SAGE — Scientific Adversarial Generation and Evaluation

Modular architecture for rigorous, reproducible adversarial testing.
Replaces ad-hoc heuristics with structured components:
  - ThreatModel: taxonomy-driven scenario encoding
  - EvidenceStore: structured per-round logging
  - SearchPolicy: decoupled bandit + pruning
  - Grader: pluggable outcome evaluation
  - AblationConfig: toggle any component on/off
  - WhiteBoxEvaluator: gradient-informed vulnerability mapping
"""

from .threat_model import ThreatModel, GoalTaxonomy, AttackSurface, TargetSpec
from .evidence_store import RoundEvidence, EvidenceStore
from .search_policy import SearchPolicy, SearchConfig
from .graders import Grader, GradeResult, PluginGrader, CompositeGrader
from .ablation import AblationConfig
from .whitebox import (
    WhiteBoxEvaluator, WhiteBoxEvalConfig,
    VulnerabilityReport, VulnerabilityFinding, VulnerabilityClass,
    ProbeType, WhiteBoxProbe,
    GradientSaliencyProbe, GCGAttackProbe, AttentionAnalysisProbe,
    LossLandscapeProbe, RepresentationEngineeringProbe,
    TokenSaliency, AttentionProbe, EmbeddingRegion, LossLandscapePoint,
)

__all__ = [
    "ThreatModel", "GoalTaxonomy", "AttackSurface", "TargetSpec",
    "RoundEvidence", "EvidenceStore",
    "SearchPolicy", "SearchConfig",
    "Grader", "GradeResult", "PluginGrader", "CompositeGrader",
    "AblationConfig",
    "WhiteBoxEvaluator", "WhiteBoxEvalConfig",
    "VulnerabilityReport", "VulnerabilityFinding", "VulnerabilityClass",
    "ProbeType", "WhiteBoxProbe",
    "GradientSaliencyProbe", "GCGAttackProbe", "AttentionAnalysisProbe",
    "LossLandscapeProbe", "RepresentationEngineeringProbe",
    "TokenSaliency", "AttentionProbe", "EmbeddingRegion", "LossLandscapePoint",
]

