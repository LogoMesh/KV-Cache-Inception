"""
LogoMesh — KV-Cache Inception Research Framework

Modular architecture for reversible MCTS in latent space and alignment faking detection.
Components:
  - BaseModelClient:  abstract LLM backend (local HF models, OpenAI API)
  - LocalLlamaOracle: HuggingFace transformers wrapper with hidden-state telemetry
  - HNeuronMonitor:   hallucination-associated neuron tracking (H-Neuron, bottom-up channel)
  - WhiteBoxEvaluator / RepresentationEngineeringProbe: RepE linear probes (top-down channel)
  - SearchPolicy:     UCB1 bandit (reused by Phase 2 KV-MCTS node selection)
  - PayloadLibrary:   pre-computed intervention storage (extended to research dataset in Phase 4)
  - EvidenceStore:    per-run structured logging
  - AblationConfig:   toggle switches for controlled experiments
"""

from .threat_model import ThreatModel, GoalTaxonomy, AttackSurface, TargetSpec
from .evidence_store import RoundEvidence, EvidenceStore
from .search_policy import SearchPolicy, SearchConfig
from .graders import Grader, GradeResult, PluginGrader, CompositeGrader
from .ablation import AblationConfig
from .oracle import BaseModelClient, OpenAIModelClient, AttackOracle, OpenAIOracle
from .local_model import LocalLlamaOracle
from .hneuron_monitor import HNeuronMonitor
from .payload_library import PayloadEntry, PayloadLibrary
from .library_loader import LibraryLoader
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
    "BaseModelClient", "OpenAIModelClient", "LocalLlamaOracle",
    # Backwards-compatible aliases
    "AttackOracle", "OpenAIOracle",
    "HNeuronMonitor",
    "PayloadEntry", "PayloadLibrary",
    "LibraryLoader",
    "WhiteBoxEvaluator", "WhiteBoxEvalConfig",
    "VulnerabilityReport", "VulnerabilityFinding", "VulnerabilityClass",
    "ProbeType", "WhiteBoxProbe",
    "GradientSaliencyProbe", "GCGAttackProbe", "AttentionAnalysisProbe",
    "LossLandscapeProbe", "RepresentationEngineeringProbe",
    "TokenSaliency", "AttentionProbe", "EmbeddingRegion", "LossLandscapePoint",
]

