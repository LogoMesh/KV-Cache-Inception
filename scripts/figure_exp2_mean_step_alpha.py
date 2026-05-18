"""Generate Figure 1 for §5 Experiment 2 (Latent Cartography MCTS).

Renders a grouped bar chart of mean-step alpha by prompt class (C1/C2/C3/C4) and
model scale (Llama 3.2-1B / Llama 3.2-3B), sourced directly from the Track G
MCTS raw artifacts. The script also re-derives every numeric claim in the
v10-exp2 draft (mean-step alpha + cross-scale Delta) from raw JSON and prints
an audit table so that any transcription mismatch against the v10-exp2 figure
spec or the Track G report is caught at figure-generation time rather than
propagating to canonical TeX at Day 7.

Trip-wire (per 2026-05-13 directive): if more than one independent transcription
error surfaces here (in addition to the two already filed against v10-exp2 from
Action 1), stop and run a full Track G report audit pass against the raw JSONs
before proceeding with any further use of Track G report numbers.

Output: docs/paper/figures/exp2-mean-step-alpha.pdf (vector-format figure for
the ACL submission).

Reproduction:
    uv run python scripts/figure_exp2_mean_step_alpha.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
FIGURES = REPO_ROOT / "docs" / "paper" / "figures"

SOURCES = {
    "1B": SCRIPTS / "_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json",
    "3B": SCRIPTS / "_track_g_mcts_results_meta-llama_Llama-3.2-3B-Instruct.json",
}

CLASSES = ["C1", "C2", "C3", "C4"]
CLASS_LABELS = {
    "C1": "C1 Factual",
    "C2": "C2 ARC-Easy",
    "C3": "C3 TruthfulQA",
    "C4": "C4 HellaSwag",
}

# v10-exp2 draft headline values (for audit cross-check)
DRAFT_BAR_ALPHA = {
    ("1B", "C1"): 0.34, ("1B", "C2"): 0.89, ("1B", "C3"): 0.83, ("1B", "C4"): 0.51,
    ("3B", "C1"): 0.61, ("3B", "C2"): 0.22, ("3B", "C3"): 0.28, ("3B", "C4"): 0.38,
}
DRAFT_DELTA = {"C1": +0.27, "C2": -0.67, "C3": -0.55, "C4": -0.13}


def derive_bar_alpha(path: Path) -> dict[str, float]:
    """Mean-step alpha per class from a Track G MCTS raw JSON.

    For each item with search_status=='ok', mean(best_path) is the
    per-item mean-step alpha; the per-class mean-step alpha bar_alpha is
    the mean over items.
    """
    with path.open() as f:
        data = json.load(f)
    bar = {}
    for cls in CLASSES:
        sub = [
            r for r in data["records"]
            if r["class"] == cls and r["search_status"] == "ok"
        ]
        if not sub:
            raise RuntimeError(f"No ok records for {cls} in {path}")
        per_item = [sum(r["best_path"]) / len(r["best_path"]) for r in sub]
        bar[cls] = sum(per_item) / len(per_item)
    return bar


def audit_table(raw_1b: dict[str, float], raw_3b: dict[str, float]) -> int:
    """Print the audit table and return the number of new discrepancies."""
    print("\n=== Audit: raw JSON vs v10-exp2 draft ===")
    print(f"{'Class':<8} {'Scale':<6} {'raw (3dp)':<12} {'draft (2dp)':<14} "
          f"{'rounded raw':<14} {'match?':<8}")
    n_disc = 0
    for cls in CLASSES:
        for scale, raw in [("1B", raw_1b[cls]), ("3B", raw_3b[cls])]:
            draft = DRAFT_BAR_ALPHA[(scale, cls)]
            rounded = round(raw, 2)
            match = abs(rounded - draft) < 1e-9
            mark = "OK" if match else "MISMATCH"
            print(f"{cls:<8} {scale:<6} {raw:<12.3f} {draft:<14.2f} "
                  f"{rounded:<14.2f} {mark:<8}")
            if not match:
                n_disc += 1

    print("\n=== Audit: cross-scale Delta (3B - 1B) ===")
    print(f"{'Class':<8} {'raw Delta':<14} {'draft Delta':<14} "
          f"{'round-then-subtract':<22} {'subtract-then-round':<22}")
    for cls in CLASSES:
        raw_delta = raw_3b[cls] - raw_1b[cls]
        draft_delta = DRAFT_DELTA[cls]
        round_then_sub = round(raw_3b[cls], 2) - round(raw_1b[cls], 2)
        sub_then_round = round(raw_delta, 2)
        print(f"{cls:<8} {raw_delta:<+14.3f} {draft_delta:<+14.2f} "
              f"{round_then_sub:<+22.2f} {sub_then_round:<+22.2f}")
    return n_disc


def render_figure(raw_1b: dict[str, float], raw_3b: dict[str, float],
                  out: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(CLASSES))
    width = 0.36

    vals_1b = [raw_1b[c] for c in CLASSES]
    vals_3b = [raw_3b[c] for c in CLASSES]

    bars_1b = ax.bar(x - width / 2, vals_1b, width,
                     label="Llama 3.2-1B", color="#4477AA", edgecolor="black",
                     linewidth=0.5)
    bars_3b = ax.bar(x + width / 2, vals_3b, width,
                     label="Llama 3.2-3B", color="#EE7733", edgecolor="black",
                     linewidth=0.5)

    for bars, vals in [(bars_1b, vals_1b), (bars_3b, vals_3b)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.018,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([CLASS_LABELS[c] for c in CLASSES])
    ax.set_ylabel(r"Mean-step $\bar{\alpha}$")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    raw_1b = derive_bar_alpha(SOURCES["1B"])
    raw_3b = derive_bar_alpha(SOURCES["3B"])
    n_disc = audit_table(raw_1b, raw_3b)
    out = FIGURES / "exp2-mean-step-alpha.pdf"
    render_figure(raw_1b, raw_3b, out)
    print(f"\nFigure written to: {out.relative_to(REPO_ROOT)}")
    print(f"Discrepancies vs v10-exp2 draft: {n_disc}")
    return n_disc


if __name__ == "__main__":
    raise SystemExit(0 if main() == 0 else 1)
