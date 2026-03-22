#!/usr/bin/env python3
"""
LAT Probe Training Script — Phase A (corrected defender simulacrum architecture).

Generates bootstrap training data using the defender simulacrum, then trains
one sklearn LogisticRegression probe per game type on the DEFENDER's last-layer
hidden states.

Probe label semantics (corrected direction — measures DEFENDER's state):
  label 1 = defender showing coercion (attack pressure is working)
  label 0 = defender holding firm (responding normally)

Labels come from two sources:
  1. Calibration prompts (clean ground-truth labels):
       calibration_prompts_benign          → label 0  (defender holding firm)
       calibration_prompts_attack_pressure → label 1  (defender being coerced)
  2. Bootstrap MCTS episodes (check_success() binary labels):
       TWO oracle calls per episode: (1) attacker generates payload, (2) defender receives it
       check_success(defender_response) = True  → label 1
       check_success(defender_response) = False → label 0

Accuracy target: >= 65% on held-out validation split.
If a probe misses the target, it is saved with passes_threshold=False and
run_offline_mcts.py will fall back to lat_score=0.5 for that game type.

Usage:
  # All game types, 60 bootstrap episodes each (default):
  uv run python scripts/train_lat_probes.py --model ./models/tinyllama-1b

  # Single game type, 100 episodes:
  uv run python scripts/train_lat_probes.py --model ./models/tinyllama-1b \\
      --game-types safety_bypass --episodes-per-game 100
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scenarios" / "security_arena"))
sys.path.insert(0, str(Path(__file__).parent))  # for run_offline_mcts constants

from sage.local_model import LocalLlamaOracle
from sage.hneuron_monitor import HNeuronMonitor
from run_offline_mcts import (
    SCENARIO_CONFIGS, STRATEGIES, MUTATIONS,
    build_defender_system_prompt, build_defender_user_prompt, calibrate_monitor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_lat_probes")

# Minimum acceptable validation accuracy to flag a probe as ready
ACCURACY_THRESHOLD = 0.65

PROBE_DIR = _REPO_ROOT / "lat_probes"


# ──────────────────────────────────────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────────────────────────────────────

async def collect_from_calibration(
    oracle: LocalLlamaOracle,
    context: dict,
) -> tuple[list[np.ndarray], list[int]]:
    """Collect (hidden_state, label) pairs from calibration prompts.

    Defender-role direction (corrected):
      calibration_prompts_benign          → label 0  (defender holding firm)
      calibration_prompts_attack_pressure → label 1  (defender being coerced)

    These are the cleanest labels available — run directly against the
    defender system prompt, no attack generation needed.
    """
    X: list[np.ndarray] = []
    y: list[int] = []
    defender_system = context["defender_system"]

    async def _run(prompt: str, label: int) -> None:
        try:
            await oracle.generate(system=defender_system, user=prompt)
            states = oracle.get_hidden_states()
            if states:
                # float() ensures float16 tensors convert cleanly to numpy float32
                X.append(states[-1].float().numpy())
                y.append(label)
        except Exception as e:
            logger.warning("Calibration collection failed: %s", e)

    for p in context["calibration_prompts_benign"]:
        await _run(p, 0)
    for p in context["calibration_prompts_attack_pressure"]:
        await _run(p, 1)

    logger.info(
        "Calibration data: %d pairs (pos=%d, neg=%d)", len(X), y.count(1), y.count(0)
    )
    return X, y


async def collect_bootstrap_episodes(
    oracle: LocalLlamaOracle,
    monitor: HNeuronMonitor,
    context: dict,
    game_type: str,
    n_episodes: int,
    scenario_plugin=None,
) -> tuple[list[np.ndarray], list[int]]:
    """Generate MCTS episodes using defender simulacrum and label by check_success().

    Two oracle calls per episode (corrected architecture):
      Call 1: oracle AS ATTACKER generates a payload
      Call 2: oracle AS DEFENDER receives the attack

    Hidden states are read from Call 2 (defender's brain).
    check_success(defender_response) provides binary ground-truth labels:
      True  → label 1  (attack worked — defender was coerced)
      False → label 0  (defender held firm)
    """
    X: list[np.ndarray] = []
    y: list[int] = []
    attack_system = context["attack_system"]
    defender_system = context["defender_system"]
    success_count = 0

    for ep in range(1, n_episodes + 1):
        strategy = random.choice(STRATEGIES)
        mutation = random.choice(MUTATIONS)
        attack_prompt = (
            f"Game type: {game_type}\n"
            f"Strategy: {strategy}\n"
            f"Mutation: {mutation}\n\n"
            f"Generate the adversarial payload now:"
        )

        # Call 1: oracle AS ATTACKER — generate payload
        try:
            payload = await oracle.generate(
                system=attack_system, user=attack_prompt, temperature=0.9,
            )
        except Exception as e:
            logger.warning("Episode %d attack generation failed: %s", ep, e)
            continue

        if not payload or len(payload.strip()) < 10:
            continue
        payload = payload.strip()

        # Call 2: oracle AS DEFENDER — simulate receiving the attack
        try:
            if scenario_plugin is not None and hasattr(scenario_plugin, "format_input"):
                formatted_input = scenario_plugin.format_input(payload)
            else:
                formatted_input = payload

            defender_user = build_defender_user_prompt(formatted_input)
            defender_response = await oracle.generate(
                system=defender_system, user=defender_user, temperature=0.3,
                max_new_tokens=512,
            )
        except Exception as e:
            logger.warning("Episode %d defender simulation failed: %s", ep, e)
            continue

        if not defender_response:
            continue

        # Read DEFENDER's hidden states (from Call 2 inference)
        states = oracle.get_hidden_states()
        if not states:
            continue

        # Ground truth: did the attack succeed?
        label = 0
        if scenario_plugin is not None:
            try:
                success, _ = scenario_plugin.check_success(defender_response)
                label = 1 if success else 0
                if success:
                    success_count += 1
            except Exception as e:
                logger.warning("Episode %d check_success failed: %s", ep, e)
                continue

        X.append(states[-1].float().numpy())
        y.append(label)

        if ep % 20 == 0:
            logger.info(
                "Bootstrap ep=%d/%d: labeled=%d success=%d (pos=%d, neg=%d)",
                ep, n_episodes, len(X), success_count, y.count(1), y.count(0),
            )

    logger.info(
        "Bootstrap done: %d labeled / %d total (check_success=True: %d, pos=%d, neg=%d)",
        len(X), n_episodes, success_count, y.count(1), y.count(0),
    )
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Probe training
# ──────────────────────────────────────────────────────────────────────────────

def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    game_type: str,
    oracle_model: str,
) -> dict | None:
    """Fit a LogisticRegression probe and return a probe data dict.

    Uses C=0.1 (strong L2 regularization) to avoid overfitting on the
    high-dimensional (hidden_size=2048), low-sample bootstrap dataset.

    Returns None if there is insufficient or unbalanced data.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    if len(X) < 10:
        logger.error("Insufficient data: %d pairs (need >= 10)", len(X))
        return None
    if n_pos < 3 or n_neg < 3:
        logger.error(
            "Insufficient class balance: pos=%d neg=%d (need >= 3 each)", n_pos, n_neg
        )
        return None

    # Stratified 80/20 split preserves class balance in small datasets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    probe = LogisticRegression(max_iter=1000, C=0.1, random_state=42, solver="lbfgs")
    probe.fit(X_train, y_train)

    train_acc = float(probe.score(X_train, y_train))
    val_acc = float(probe.score(X_val, y_val))
    passes = val_acc >= ACCURACY_THRESHOLD

    status = "PASS" if passes else "WARN"
    logger.info(
        "[%s] %s — train_acc=%.3f  val_acc=%.3f  n_train=%d  n_val=%d",
        status, game_type, train_acc, val_acc, len(X_train), len(X_val),
    )
    if not passes:
        logger.warning(
            "val_acc=%.3f below threshold %.2f. "
            "run_offline_mcts.py will use lat_score=0.5 fallback for %s.",
            val_acc, ACCURACY_THRESHOLD, game_type,
        )

    return {
        "model": probe,
        "game_type": game_type,
        "oracle_model": oracle_model,
        "layer": -1,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "trained_at": time.time(),
        "passes_threshold": passes,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-game-type orchestration
# ──────────────────────────────────────────────────────────────────────────────

async def train_game_type(
    oracle: LocalLlamaOracle,
    game_type: str,
    n_episodes: int,
    probe_dir: Path,
    model_slug: str,
) -> bool:
    """Train and save a probe for one game type. Returns True if it passes threshold."""
    logger.info("=== Training probe: %s ===", game_type)
    scenario_cfg = SCENARIO_CONFIGS[game_type]

    # Instantiate scenario plugin (provides defender context + check_success)
    plugin_class = scenario_cfg["plugin_class"]
    plugin_config = scenario_cfg["plugin_config"]
    scenario_plugin = plugin_class(config=plugin_config)

    # Build defender system prompt (same as run_offline_mcts.py)
    defender_system = build_defender_system_prompt(scenario_plugin)

    # Assemble context dict expected by helper functions
    context = {
        "attack_system": scenario_cfg["attack_system"],
        "defender_system": defender_system,
        "calibration_prompts_benign": scenario_cfg["calibration_prompts_benign"],
        "calibration_prompts_attack_pressure": scenario_cfg["calibration_prompts_attack_pressure"],
    }

    # Calibrate H-Neuron monitor with defender-role data
    monitor = HNeuronMonitor(oracle)
    await calibrate_monitor(oracle, monitor, context)

    # Collect clean-label data from calibration prompts (defender-role direction)
    X_cal, y_cal = await collect_from_calibration(oracle, context)

    # Collect check_success()-labeled bootstrap episodes (two oracle calls each)
    X_boot, y_boot = await collect_bootstrap_episodes(
        oracle, monitor, context, game_type, n_episodes, scenario_plugin=scenario_plugin,
    )

    X = np.array(X_cal + X_boot, dtype=np.float32)
    y = np.array(y_cal + y_boot, dtype=np.int32)

    if len(X) == 0:
        logger.error("No training data for %s — skipping", game_type)
        return False

    probe_data = train_probe(X, y, game_type, oracle.model_id)
    if probe_data is None:
        return False

    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / f"{game_type}_{model_slug}.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump(probe_data, f)
    logger.info("Saved → %s", probe_path)

    return probe_data["passes_threshold"]


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    logger.info("Loading model from %s...", args.model)
    oracle = LocalLlamaOracle.load(args.model)
    logger.info("Model loaded: %s", oracle.model_id)

    model_slug = Path(args.model).name.replace("/", "-").replace("\\", "-")

    results: dict[str, str] = {}
    for game_type in args.game_types:
        passed = await train_game_type(
            oracle, game_type, args.episodes_per_game, PROBE_DIR, model_slug
        )
        results[game_type] = "PASS" if passed else "WARN"

    print("\n--- LAT Probe Training Summary ---")
    for gt, status in results.items():
        print(f"  {gt}: {status}")

    if all(v == "PASS" for v in results.values()):
        print("\nAll probes pass accuracy threshold. Wire lat_score in run_offline_mcts.py.")
    else:
        print(
            "\nSome probes below threshold. run_offline_mcts.py will use lat_score=0.5 "
            "fallback for failing game types. See action plan §9."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LAT probes for offline MCTS")
    p.add_argument(
        "--model", default="./models/tinyllama-1b",
        help="Path to local model directory (same model used in run_offline_mcts.py)",
    )
    p.add_argument(
        "--game-types", nargs="+",
        default=list(SCENARIO_CONFIGS.keys()),
        choices=list(SCENARIO_CONFIGS.keys()),
        help="Game types to train (default: all four)",
    )
    p.add_argument(
        "--episodes-per-game", type=int, default=60,
        help="Bootstrap MCTS episodes per game type (minimum 60 recommended)",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
