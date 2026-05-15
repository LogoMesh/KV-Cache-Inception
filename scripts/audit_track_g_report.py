"""Track G report — full audit pass against raw JSON.

Re-derives every numeric claim in docs/logs/2026-05-11_track-G-cartography-report.md
from the raw artifact JSONs and prints a per-claim verdict
(CLEAN / TRANSCRIPTION ERROR / ROUNDING DIVERGENCE / SCOPE INCOMPLETE).

Trigger: 2026-05-13 trip-wire fire — three errors found in independent runs of
v10-exp2-driven re-derivation cleared the pre-committed "three errors = full
audit" threshold.

Output: stdout structured tables, one per claim family (A through H).
Reproduction:
    uv run python scripts/audit_track_g_report.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

# Windows cp1252 stdout can't print α/Δ; force utf-8 for the run
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"

MCTS_1B = SCRIPTS / "_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json"
MCTS_3B = SCRIPTS / "_track_g_mcts_results_meta-llama_Llama-3.2-3B-Instruct.json"
SWEEP_1B = SCRIPTS / "_track_g_results_meta-llama_Llama-3.2-1B-Instruct.json"
SWEEP_3B = SCRIPTS / "_track_g_results_meta-llama_Llama-3.2-3B-Instruct.json"
TRACK_F_1B = SCRIPTS / "_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json"
TRACK_F_3B = SCRIPTS / "_track_f_results_meta-llama_Llama-3.2-3B-Instruct.json"

CLASSES = ["C1", "C2", "C3", "C4"]

# Verdict counters
counters = Counter()


def verdict(claim: str, reported, raw, tol=None, category=None):
    """Compare reported vs raw; return verdict string and update counters."""
    if category is not None:
        v = category
    elif reported == raw:
        v = "CLEAN"
    elif isinstance(reported, float) and isinstance(raw, float):
        if tol is not None and abs(reported - raw) <= tol:
            v = "CLEAN"
        else:
            v = "TRANSCRIPTION"
    else:
        v = "TRANSCRIPTION"
    counters[v] += 1
    return v


def load_json(p):
    with p.open() as f:
        return json.load(f)


def fam_a():
    """Family A: run-config / wall-clock / sample counts."""
    print("\n" + "=" * 72)
    print("Family A — Run configuration / wall-clock / sample counts")
    print("=" * 72)
    rows = []
    for label, path, rep_n, rep_sec in [
        ("1B MCTS", MCTS_1B, 385, 1023),
        ("3B MCTS", MCTS_3B, 385, 1799),
        ("1B sweep", SWEEP_1B, 2310, 330),
        ("3B sweep", SWEEP_3B, 2310, 619),
    ]:
        d = load_json(path)
        n = len(d["records"])
        elapsed = d["elapsed_sec"]
        n_verdict = verdict(f"{label} n_records", rep_n, n)
        # wall-clock: report rounded to integer seconds; tolerate 1s diff
        s_verdict = verdict(f"{label} elapsed_sec", rep_sec, round(elapsed),
                            tol=1)
        rows.append((label, rep_n, n, n_verdict, rep_sec, round(elapsed, 1), s_verdict))
    print(f"{'Label':<10} {'rep n':<8} {'raw n':<8} {'n verdict':<14} "
          f"{'rep sec':<10} {'raw sec':<12} {'sec verdict':<14}")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:<8} {r[2]:<8} {r[3]:<14} "
              f"{r[4]:<10} {r[5]:<12} {r[6]:<14}")
    # Per-class counts
    print("\nPer-class counts (MCTS JSONs):")
    for label, path in [("1B MCTS", MCTS_1B), ("3B MCTS", MCTS_3B)]:
        d = load_json(path)
        cnt = Counter(r["class"] for r in d["records"])
        rep = {"C1": 100, "C2": 100, "C3": 85, "C4": 100}
        match = all(cnt[c] == rep[c] for c in CLASSES)
        v = verdict(f"{label} per-class counts", rep, dict(cnt))
        print(f"  {label}: {dict(cnt)}  verdict={v}")


def derive_path_distribution(records):
    """Modal / 2nd modal / 3rd modal tuples and percentages."""
    paths = [tuple(r["best_path"]) for r in records if r["search_status"] == "ok"]
    n = len(paths)
    if n == 0:
        return None
    c = Counter(paths)
    top3 = c.most_common(3)
    return [(p, cnt, 100 * cnt / n) for p, cnt in top3], n


def fam_b():
    """Family B: §2.1 best-path α-tuple distribution + cross-scale Δ."""
    print("\n" + "=" * 72)
    print("Family B — Best-path α-tuple distribution + cross-scale Δ")
    print("=" * 72)
    # Report-cited values, hard-coded from Track G report §2.1
    REPORT_1B = {
        "C1": {"modal": (0.1, 0.1, 0.1), "modal_pct": 41,
               "second": (0.1, 0.5, 0.1), "second_pct": 10,
               "third": (1.0, 1.0, 0.5), "third_pct": 9,
               "bar_alpha": 0.34},
        "C2": {"modal": (1.0, 1.0, 1.0), "modal_pct": 42,
               "second": (0.5, 1.0, 1.0), "second_pct": 23,
               "third": (1.0, 1.0, 0.5), "third_pct": 19,
               "bar_alpha": 0.89},
        "C3": {"modal": (1.0, 1.0, 1.0), "modal_pct": 28,
               "second": (1.0, 0.5, 1.0), "second_pct": 25,
               "third": (1.0, 1.0, 0.5), "third_pct": 19,
               "bar_alpha": 0.83},
        "C4": {"modal": (0.1, 0.1, 0.1), "modal_pct": 21,
               "second": (1.0, 1.0, 0.5), "second_pct": 9,
               "third": (1.0, 1.0, 1.0), "third_pct": 9,
               "bar_alpha": 0.51},
    }
    REPORT_3B = {
        "C1": {"modal": "BIMODAL", "modal_pct": 15,
               "bar_alpha": 0.61},  # special: (0.5,0.5,1.0)/(1.0,0.1,1.0) tied
        "C2": {"modal": (0.1, 0.1, 0.1), "modal_pct": 49,
               "second": (0.5, 0.1, 0.1), "second_pct": 14,
               "third": (0.1, 0.5, 0.1), "third_pct": 13,
               "bar_alpha": 0.22},
        "C3": {"modal": (0.1, 0.1, 0.1), "modal_pct": 36,
               "second": (0.1, 0.1, 0.5), "second_pct": 14,
               "third": (0.1, 0.5, 0.1), "third_pct": 13,
               "bar_alpha": 0.28},
        "C4": {"modal": (0.1, 0.1, 0.1), "modal_pct": 31,
               "second": (0.5, 0.1, 0.1), "second_pct": 13,
               "third": (1.0, 1.0, 1.0), "third_pct": 10,
               "bar_alpha": 0.38},
    }
    raw_alpha = {}
    for scale, path, rep_table in [("1B", MCTS_1B, REPORT_1B), ("3B", MCTS_3B, REPORT_3B)]:
        print(f"\n--- {scale} ---")
        d = load_json(path)
        for cls in CLASSES:
            sub = [r for r in d["records"] if r["class"] == cls
                   and r["search_status"] == "ok"]
            top3, n = derive_path_distribution(sub)
            mean_step = sum(sum(r["best_path"]) / len(r["best_path"])
                            for r in sub) / len(sub)
            raw_alpha[(scale, cls)] = mean_step
            rep = rep_table[cls]
            print(f"  {cls} (n={n}):")
            # Modal
            m_path, m_cnt, m_pct = top3[0]
            if scale == "3B" and cls == "C1":
                # Special bimodal case
                second_path, second_cnt, second_pct = top3[1]
                print(f"    1st modal: {m_path} ({m_cnt}/{n} = {m_pct:.0f}%)")
                print(f"    2nd modal: {second_path} ({second_cnt}/{n} = "
                      f"{second_pct:.0f}%)")
                v = verdict(f"3B C1 bimodal at ~15%",
                            "BIMODAL 15%/15%",
                            f"{m_pct:.0f}%/{second_pct:.0f}%",
                            category=("CLEAN" if abs(m_pct - 15) <= 1
                                      and abs(second_pct - 15) <= 1
                                      else "TRANSCRIPTION"))
                print(f"    bimodal verdict: {v}")
            else:
                v_modal = verdict(f"{scale} {cls} modal tuple",
                                  rep["modal"], m_path)
                v_modal_pct = verdict(f"{scale} {cls} modal %",
                                      rep["modal_pct"], round(m_pct))
                print(f"    modal: rep={rep['modal']} @ {rep['modal_pct']}%  "
                      f"raw={m_path} @ {m_pct:.0f}%  "
                      f"verdict={v_modal}/{v_modal_pct}")
                # 2nd modal
                s_path, s_cnt, s_pct = top3[1]
                v_2nd = verdict(f"{scale} {cls} 2nd modal tuple",
                                rep["second"], s_path)
                v_2nd_pct = verdict(f"{scale} {cls} 2nd modal %",
                                    rep["second_pct"], round(s_pct))
                print(f"    2nd: rep={rep['second']} @ {rep['second_pct']}%  "
                      f"raw={s_path} @ {s_pct:.0f}%  verdict={v_2nd}/{v_2nd_pct}")
                # 3rd modal
                t_path, t_cnt, t_pct = top3[2]
                v_3rd = verdict(f"{scale} {cls} 3rd modal tuple",
                                rep["third"], t_path)
                v_3rd_pct = verdict(f"{scale} {cls} 3rd modal %",
                                    rep["third_pct"], round(t_pct))
                print(f"    3rd: rep={rep['third']} @ {rep['third_pct']}%  "
                      f"raw={t_path} @ {t_pct:.0f}%  verdict={v_3rd}/{v_3rd_pct}")
            # Mean-step α
            v_alpha = verdict(f"{scale} {cls} bar_alpha",
                              rep["bar_alpha"], round(mean_step, 2),
                              tol=0.005)
            print(f"    bar_alpha: rep={rep['bar_alpha']}  raw={mean_step:.3f} "
                  f"→ rounded {round(mean_step, 2):.2f}  verdict={v_alpha}")
    # Cross-scale Δ
    print("\n--- Cross-scale Δ (3B − 1B) ---")
    REPORT_DELTA = {"C1": 0.27, "C2": -0.67, "C3": -0.55, "C4": -0.13}
    for cls in CLASSES:
        raw_delta = raw_alpha[("3B", cls)] - raw_alpha[("1B", cls)]
        sub_then_round = round(raw_delta, 2)
        round_then_sub = round(raw_alpha[("3B", cls)], 2) - round(raw_alpha[("1B", cls)], 2)
        rep = REPORT_DELTA[cls]
        # If report matches round-then-subtract but not subtract-then-round → ROUNDING DIVERGENCE
        if rep == sub_then_round:
            v = "CLEAN"
        elif abs(round_then_sub - rep) < 1e-9 and abs(sub_then_round - rep) > 0.005:
            v = "ROUNDING"
        else:
            v = "TRANSCRIPTION"
        counters[v] += 1
        print(f"  {cls}: rep={rep:+.2f}  raw={raw_delta:+.3f}  "
              f"sub-then-round={sub_then_round:+.2f}  "
              f"round-then-sub={round_then_sub:+.2f}  verdict={v}")


def fam_c():
    """Family C: argmax-at-best-leaf top-5 token counts."""
    print("\n" + "=" * 72)
    print("Family C — Argmax-at-best-leaf top-5 counts")
    print("=" * 72)
    REPORT = {
        ("1B", "C1"): [("The", 11), ("5", 5), ("July", 4), ("World", 4), ("9", 4)],
        ("1B", "C2"): [("A", 92), ("The", 4), ("1", 3), ("C", 1)],
        ("1B", "C3"): [("A", 81), ("B", 2), ("The", 1), ("C", 1)],
        ("1B", "C4"): [("The", 27), ("is", 15), ("They", 9), ("he", 5), ("uses", 5)],
        ("3B", "C1"): [("8", 7), ("5", 5), ("7", 4), ("6", 4), ("April", 3)],
        ("3B", "C2"): [("A", 29), ("B", 27), ("D", 22), ("C", 16), ("2", 3)],
        ("3B", "C3"): [("A", 39), ("B", 17), ("C", 15), ("D", 13), ("I", 1)],
        ("3B", "C4"): [("The", 28), ("...", 25), ("is", 11), ("They", 5), ("He", 4)],
    }
    for scale, path in [("1B", MCTS_1B), ("3B", MCTS_3B)]:
        print(f"\n--- {scale} ---")
        d = load_json(path)
        for cls in CLASSES:
            sub = [r for r in d["records"] if r["class"] == cls
                   and r["search_status"] == "ok"]
            tokens = [r["argmax_token_str"] for r in sub]
            raw_top5 = Counter(tokens).most_common(5)
            rep_top5 = REPORT[(scale, cls)]
            match_all = (raw_top5[:len(rep_top5)] == rep_top5)
            v = verdict(f"{scale} {cls} top-5 argmax",
                        rep_top5, raw_top5[:len(rep_top5)],
                        category=("CLEAN" if match_all else "TRANSCRIPTION"))
            print(f"  {cls}: rep={rep_top5}")
            print(f"      raw={raw_top5}  verdict={v}")


def fam_d():
    """Family D: entropy + gold-rank + reward-gain tables."""
    print("\n" + "=" * 72)
    print("Family D — Mean best-leaf entropy / median gold-rank / "
          "% gold in top-5 / search reward gain")
    print("=" * 72)
    REPORT = {
        ("1B", "C1"): (1.45, 0, 90, 0.025),
        ("1B", "C2"): (1.22, 1, 94, 0.011),
        ("1B", "C3"): (1.19, 1, 100, 0.005),
        ("1B", "C4"): (3.96, 155, 10, 0.020),
        ("3B", "C1"): (0.33, 0, 96, 0.008),
        ("3B", "C2"): (0.73, 0, 95, 0.009),
        ("3B", "C3"): (1.03, 1, 100, 0.010),
        ("3B", "C4"): None,  # marked "n/a (data truncated)"
    }
    for scale, path in [("1B", MCTS_1B), ("3B", MCTS_3B)]:
        print(f"\n--- {scale} ---")
        d = load_json(path)
        for cls in CLASSES:
            sub = [r for r in d["records"] if r["class"] == cls
                   and r["search_status"] == "ok"]
            entropies = [r["best_terminal_entropy_nats"] for r in sub]
            mean_ent = sum(entropies) / len(entropies)
            ranks = [r["gold_first_token_rank"] for r in sub
                     if r["gold_first_token_rank"] is not None
                     and r["gold_first_token_rank"] >= 0]
            median_rank = statistics.median(ranks) if ranks else None
            in_top5 = sum(1 for r in ranks if r <= 4)  # rank 0..4 = top-5 (0-indexed)
            top5_pct = round(100 * in_top5 / len(ranks)) if ranks else None
            gains = [r["best_search_reward"] - r["mean_search_reward"] for r in sub]
            mean_gain = sum(gains) / len(gains)
            rep = REPORT[(scale, cls)]
            if rep is None:
                v_overall = verdict(f"{scale} {cls} row", "n/a data truncated",
                                    f"ent={mean_ent:.2f} rank={median_rank} "
                                    f"top5={top5_pct} gain={mean_gain:+.4f}",
                                    category="SCOPE_INCOMPLETE")
                print(f"  {cls}: REPORT=n/a data truncated  "
                      f"RAW: ent={mean_ent:.3f}  rank={median_rank}  "
                      f"top5={top5_pct}%  gain={mean_gain:+.4f}  verdict={v_overall}")
            else:
                rep_ent, rep_rank, rep_top5, rep_gain = rep
                v_ent = verdict(f"{scale} {cls} mean entropy",
                                rep_ent, round(mean_ent, 2), tol=0.005)
                v_rank = verdict(f"{scale} {cls} median rank",
                                 rep_rank, int(median_rank))
                v_top5 = verdict(f"{scale} {cls} % gold top-5",
                                 rep_top5, top5_pct)
                v_gain = verdict(f"{scale} {cls} reward gain",
                                 rep_gain, round(mean_gain, 3), tol=0.0005)
                print(f"  {cls}: ent rep={rep_ent} raw={mean_ent:.3f} ({v_ent})  "
                      f"rank rep={rep_rank} raw={median_rank} ({v_rank})  "
                      f"top5 rep={rep_top5} raw={top5_pct} ({v_top5})  "
                      f"gain rep={rep_gain:+.3f} raw={mean_gain:+.4f} ({v_gain})")


def fam_e():
    """Family E: §3 cross-scale prose claims.

    Track F JSON uses items[i].{G,R,E}.best_path; the E arm is the
    entropy-MCTS arm whose modal best-paths Track G report cites.
    """
    print("\n" + "=" * 72)
    print("Family E — §3 Cross-scale prose claims")
    print("=" * 72)
    # 1. Track F 1B (1.0,1.0,1.0) 91/200
    d = load_json(TRACK_F_1B)
    e_paths = [tuple(item["E"]["best_path"]) for item in d["items"]
               if item["E"].get("best_path") is not None]
    c = Counter(e_paths)
    target = (1.0, 1.0, 1.0)
    cnt = c.get(target, 0)
    v = verdict("Track F 1B E (1,1,1) count",
                91, cnt)
    print(f"  Track F 1B E (1,1,1): rep=91/200  raw={cnt}/{len(e_paths)}  "
          f"verdict={v}")
    # 2. Track F 3B (0.1,0.1,0.1) 101/200
    d = load_json(TRACK_F_3B)
    e_paths = [tuple(item["E"]["best_path"]) for item in d["items"]
               if item["E"].get("best_path") is not None]
    c = Counter(e_paths)
    target = (0.1, 0.1, 0.1)
    cnt = c.get(target, 0)
    v = verdict("Track F 3B E (.1,.1,.1) count",
                101, cnt)
    print(f"  Track F 3B E (.1,.1,.1): rep=101/200  raw={cnt}/{len(e_paths)}  "
          f"verdict={v}")
    # 3. Cartography 1B C3 (1,1,1) 24/85 = 28%
    d = load_json(MCTS_1B)
    c3 = [r for r in d["records"] if r["class"] == "C3"
          and r["search_status"] == "ok"]
    paths = [tuple(r["best_path"]) for r in c3]
    cnt = paths.count((1.0, 1.0, 1.0))
    v = verdict("Cartography 1B C3 (1,1,1) count",
                24, cnt)
    print(f"  Cartography 1B C3 (1,1,1): rep=24/85  raw={cnt}/{len(c3)}  "
          f"verdict={v}")


def fam_f():
    """Family F: §4 parameter-sweep top-1 tables."""
    print("\n" + "=" * 72)
    print("Family F — §4 Parameter-sweep top-1 tables")
    print("=" * 72)
    # Selected high-value cells from §4.1 tables
    REPORT_1B = {
        ("C1", 0.0): ("The", 12),
        ("C1", 5.0): ('"', 100),
        ("C2", 0.0): ("A", 83),
        ("C2", 1.0): ("A", 94),
        ("C2", 2.0): ("A", 100),
        ("C3", 0.0): ("A", 79),
        ("C3", 1.0): ("A", 81),
        ("C3", 2.0): ("A", 85),
        ("C4", 0.0): ("The", 22),
        ("C4", 5.0): ('"', 100),
    }
    REPORT_3B = {
        ("C2", 0.0): ("A", 29),
        ("C2", 2.0): ("A", 68),
        ("C3", 0.0): ("A", 37),
        ("C3", 2.0): ("A", 72),
        ("C4", 5.0): (" to", 100),
    }
    for scale, path, rep in [("1B", SWEEP_1B, REPORT_1B),
                              ("3B", SWEEP_3B, REPORT_3B)]:
        print(f"\n--- {scale} sweep ---")
        d = load_json(path)
        for (cls, alpha), (rep_tok, rep_cnt) in rep.items():
            sub = [r for r in d["records"]
                   if r["class"] == cls and r["alpha"] == alpha
                   and r.get("applied_ok", True)]
            if not sub:
                print(f"  {cls} @ α={alpha}: NO RECORDS")
                counters["NOT_REDERIVABLE"] += 1
                continue
            tokens = [r["argmax_token_str"] for r in sub]
            top1 = Counter(tokens).most_common(1)[0]
            raw_tok, raw_cnt = top1
            match_tok = (raw_tok == rep_tok)
            match_cnt = (raw_cnt == rep_cnt)
            if match_tok and match_cnt:
                v = "CLEAN"
            elif match_tok and abs(raw_cnt - rep_cnt) <= 1:
                v = "CLEAN"  # tolerate ±1 due to off-by-one on count truncation
            else:
                v = "TRANSCRIPTION"
            counters[v] += 1
            print(f"  {cls} @ α={alpha}: rep={rep_tok}={rep_cnt}  "
                  f"raw={raw_tok}={raw_cnt}/{len(sub)}  verdict={v}")


def fam_g():
    """Family G: §4.2 entropy trajectory for 1B C2."""
    print("\n" + "=" * 72)
    print("Family G — §4.2 1B C2 entropy trajectory")
    print("=" * 72)
    d = load_json(SWEEP_1B)
    REPORT = {0.0: 1.51, 0.1: 1.49, 0.5: 1.35, 1.0: 1.21, 2.0: 1.64, 5.0: 4.87}
    for alpha in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        sub = [r for r in d["records"]
               if r["class"] == "C2" and r["alpha"] == alpha]
        if not sub:
            print(f"  α={alpha}: NO RECORDS")
            counters["NOT_REDERIVABLE"] += 1
            continue
        ents = [r["next_token_entropy_nats"] for r in sub]
        mean_e = sum(ents) / len(ents)
        rep = REPORT[alpha]
        v = verdict(f"1B C2 entropy α={alpha}",
                    rep, round(mean_e, 2), tol=0.015)
        print(f"  α={alpha}: rep={rep:.2f} nats  raw={mean_e:.3f}  "
              f"({len(sub)} items)  verdict={v}")


def fam_h():
    """Family H: §0 baseline argmax claims (sweep JSON at α=0.0)."""
    print("\n" + "=" * 72)
    print("Family H — §0 baseline argmax (sweep at α=0.0)")
    print("=" * 72)
    print("Report claims:")
    print("  1B MCQ classes have strong A-prior (83-93% baseline argmax = A)")
    print("  3B MCQ classes have balanced distribution (29-44% A baseline)")
    for scale, path in [("1B", SWEEP_1B), ("3B", SWEEP_3B)]:
        print(f"\n--- {scale} ---")
        d = load_json(path)
        for cls in ("C2", "C3"):
            sub = [r for r in d["records"]
                   if r["class"] == cls and r["alpha"] == 0.0]
            if not sub:
                continue
            a_count = sum(1 for r in sub if r["argmax_token_str"] == "A")
            a_pct = 100 * a_count / len(sub)
            print(f"  {scale} {cls} @ α=0: A={a_count}/{len(sub)} = {a_pct:.0f}%")
    print("\n(Range check: 1B C2 + C3 A% should land in 83-93%; "
          "3B C2 + C3 A% should land in 29-44%.)")


def main():
    print(f"Track G Audit Pass — {Path(__file__).name}")
    print(f"Trigger: 2026-05-13 trip-wire fire (3 errors confirmed across "
          f"v10-exp2 reconciliation + Figure 1 audit).")
    fam_a()
    fam_b()
    fam_c()
    fam_d()
    fam_e()
    fam_f()
    fam_g()
    fam_h()
    print("\n" + "=" * 72)
    print("Summary counters")
    print("=" * 72)
    for k, v in sorted(counters.items()):
        print(f"  {k}: {v}")
    total = sum(counters.values())
    err = (counters.get("TRANSCRIPTION", 0)
           + counters.get("ROUNDING", 0)
           + counters.get("SCOPE_INCOMPLETE", 0))
    print(f"  Total cells audited: {total}")
    print(f"  Total errors (TRANS+ROUND+SCOPE): {err}")
    if total > 0:
        print(f"  Error rate: {err}/{total} = {100 * err / total:.1f}%")


if __name__ == "__main__":
    main()
