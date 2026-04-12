"""
LogoMesh — KV-Cache Inception Research Framework

Modular architecture for reversible MCTS in latent space and alignment faking detection.
Components:
  - BaseModelClient:           abstract LLM backend (local HF models, OpenAI API)
  - LocalLlamaOracle:          HuggingFace transformers wrapper with hidden-state telemetry
  - HNeuronMonitor:            H-Neuron stress σ_H per layer (bottom-up channel)
  - PerLayerHonestyProjector:  RepE honesty ρ_R per layer + steering vectors (top-down channel)
  - TelemetryMatrix:           T_t ∈ ℝ^{2×L} per step, DiagnosticState, node reward (Eq. 8)
  - NullSpaceProjector:        Proj_Null(P)(h) for OEI computation (Eq. 10)
  - OEICalculator:             Orthogonal Escape Index
  - TDSCalculator:             Telemetry Divergence Score (JSD)
  - ReversibleMCTS:            KV-cache latent space MCTS with FP32 accumulator rollback
  - FP32Accumulator:           Theorem 1 exact reversibility accumulator
  - KVCacheNode / MCTSConfig:  MCTS tree node and hyperparameters
  - SearchPolicy:              UCB1 bandit (reused for MCTS node selection)
  - PayloadLibrary:            pre-computed intervention storage (Phase 4: research dataset)
  - EvidenceStore:             per-run structured logging
  - AblationConfig:            toggle switches for controlled experiments
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
    PerLayerHonestyProjector,
)
from .telemetry_matrix import TelemetryMatrix, DiagnosticState, classify, compute_node_reward
from .orthogonal_escape import NullSpaceProjector, OEICalculator, TDSCalculator
from .kv_mcts import FP32Accumulator, KVCacheNode, MCTSConfig, ReversibleMCTS

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
    # Phase 2 — per-layer telemetry
    "PerLayerHonestyProjector",
    # Phase 2 — telemetry matrix
    "TelemetryMatrix", "DiagnosticState", "classify", "compute_node_reward",
    # Phase 2 — orthogonal escape
    "NullSpaceProjector", "OEICalculator", "TDSCalculator",
    # Phase 2 — reversible MCTS
    "FP32Accumulator", "KVCacheNode", "MCTSConfig", "ReversibleMCTS",
]

