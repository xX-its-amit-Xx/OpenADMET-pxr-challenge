"""nb2161 -- CYCLE 121 TRAJECTORY SNAPSHOT over the full ~775-method,
121-cycle history of the PXR activity ladder.

PROTOCOL:
    Walks data/processed/*_summary.json files. Extracts honest cross-fit
    RAE estimates from every summary (under any of several known keys),
    counts methods / cycles / paradigms, computes the running honest
    floor and its gap to the current LB best (0.7655 placeholder).

    Records cycle 121 as the 5-method poll/verify/regularize/intersect/
    DART loop (nb2155 LB poll, nb2156 verify, nb2158 monotone, nb2159
    shap-intersect, nb2160 DART).

OUTPUTS:
    scripts/nb2161_trajectory.py
    data/processed/nb2161_summary.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pxr.paths import DATA_PROCESSED

TAG = "nb2161"

# Current LB best (PRE-unblind cap discovered 2026-06-02 / chemprop_aux 0.7655)
LB_CURRENT_BEST = 0.7655

# Candidate keys (in priority order) -- the first one present in a summary
# that yields a finite float is taken as that summary's honest cross-fit RAE.
RAE_KEYS_PRIORITY = [
    # trajectory / win-bag style
    "best_rae",
    "rae_pooled_median_120",
    "rae_pooled_mean_120",
    "rae_pooled_median",
    "rae_pooled_mean",
    "rae_median_bag",
    "rae_mean_bag",
    # generic
    "rae_cross_fit",
    "rae_oof",
    "rae",
    "oof_rae",
    "cross_fit_rae",
    "rae_honest",
    "honest_cross_fit_rae",
    "rae_final",
    "rae_best",
    # in-sample (less preferred, last resort) -- excluded from honest floor
]

# Keys that we treat as in-sample only -- recorded but NOT counted toward the
# honest floor (per the 2026-05-31 te-vs-pred_oof protocol memory).
IN_SAMPLE_KEYS = {
    "in_rae", "rae_in_sample", "rae_te_unb_idx",
}

# Paradigm classification (regex on method/tag strings)
PARADIGM_PATTERNS = [
    ("trajectory_bag",        r"trajectory|bag|win_bag|n_cycles"),
    ("shap_subset",           r"shap|top.?\d+|k_tuned"),
    ("slsqp_blend",           r"slsqp|blend|convex"),
    ("stack_lgbm",            r"stack|meta_learner|residual_stack"),
    ("rank_stretch_calib",    r"stretch|rank_calib|isotonic|conformal"),
    ("chemprop_gnn",          r"chemprop|mpnn|gnn|aux"),
    ("knn_tanimoto",          r"knn|tanimoto"),
    ("mmp_cliff",             r"mmp|cliff|matched_pair"),
    ("foundation_embed",      r"chembert|molt5|grover|foundation|pretrain|llm"),
    ("delta_ml",              r"delta|residual_ml|err_hat"),
    ("structure_pose",        r"boltz|pose|lddt|veto|holo|template"),
    ("dann_moe",              r"dann|mope|moe|router|domain_adv"),
    ("ssl_pretrain",          r"ssl|contrast|pretrain_ft|cep|sim"),
    ("ordinal_quantile",      r"quantile|pinball|tail_weight|pchip|cdf"),
    ("counter_assay",         r"counter|null|promiscuity|discount"),
    ("multitask_nr",          r"nr_family|multitask|tox21|broad_analogy"),
    ("active_learning",       r"active_learn|al_proxy"),
    ("persistence_homology",  r"persistence|homology|topology"),
    ("catboost_ordered",      r"catboost"),
    ("gp_tanimoto",           r"gp_tanimoto|gaussian_process"),
    ("schnet_3d",             r"schnet|3d_coord"),
    ("dart",                  r"dart"),
    ("monotone_constraint",   r"monotone"),
    ("smiles_aug_ttt",        r"smiles_aug|ttt|test_time"),
    ("huber_robust",          r"huber"),
    ("assay_decomposition",   r"assay_decomp"),
    ("submission_meta",       r"lb_poll|submission|leaderboard|verify"),
]


def _extract_cycle_num(tag: str) -> int | None:
    """Heuristic: nb<X> tag -- cycle ~= X // 10 (one cycle per 10-method block,
    roughly). nbABC patterns: A is era, BC is method-within-era. Not exact
    but used for grouping -- the exact cycle index is recorded separately."""
    m = re.match(r"nb(\d+)", tag)
    if m:
        n = int(m.group(1))
        # Buckets of 10: nb2150..nb2159 -> cycle ~ 215
        return n // 10
    return None


def _classify_paradigm(tag: str, method: str) -> str:
    text = f"{tag} {method}".lower()
    matches = []
    for name, pat in PARADIGM_PATTERNS:
        if re.search(pat, text):
            matches.append(name)
    if not matches:
        return "other"
    return matches[0]


def _extract_rae(summary: dict) -> tuple[float | None, str | None]:
    for k in RAE_KEYS_PRIORITY:
        if k in summary:
            v = summary[k]
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if vf > 0.0 and vf < 5.0:  # sanity: RAE between 0 and 5
                return vf, k
    return None, None


def _extract_in_sample_rae(summary: dict) -> float | None:
    for k in IN_SAMPLE_KEYS:
        if k in summary:
            try:
                vf = float(summary[k])
                if 0.0 < vf < 5.0:
                    return vf
            except (TypeError, ValueError):
                pass
    return None


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CYCLE 121 TRAJECTORY SNAPSHOT")
    print(f"          walk {DATA_PROCESSED}/*_summary.json")
    print(f"          LB_CURRENT_BEST = {LB_CURRENT_BEST}")
    print("=" * 78)

    summary_paths = sorted(DATA_PROCESSED.glob("*_summary.json"))
    print(f"[walk] found {len(summary_paths)} summary JSON files")

    records: list[dict] = []
    skipped = 0
    for p in summary_paths:
        try:
            with open(p) as f:
                s = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            skipped += 1
            continue
        if not isinstance(s, dict):
            skipped += 1
            continue
        tag = s.get("tag") or p.stem.replace("_summary", "")
        method = s.get("method") or s.get("notebook") or ""
        rae_val, rae_key = _extract_rae(s)
        in_rae = _extract_in_sample_rae(s)
        paradigm = _classify_paradigm(str(tag), str(method))
        cycle_bucket = _extract_cycle_num(str(tag))
        records.append({
            "path": str(p),
            "tag": tag,
            "method": method,
            "paradigm": paradigm,
            "cycle_bucket": cycle_bucket,
            "rae": rae_val,
            "rae_key": rae_key,
            "in_rae": in_rae,
            "pre_unblind_clean": s.get("pre_unblind_clean"),
        })

    n_records = len(records)
    n_with_rae = sum(1 for r in records if r["rae"] is not None)
    print(f"[walk] parsed {n_records}  honest_rae_present {n_with_rae}  "
          f"skipped {skipped}")

    # Method / cycle / paradigm counts
    distinct_tags = sorted({r["tag"] for r in records if r["tag"]})
    n_methods_distinct = len(distinct_tags)
    cycle_buckets = sorted({r["cycle_bucket"] for r in records
                            if r["cycle_bucket"] is not None})
    n_cycle_buckets = len(cycle_buckets)
    paradigms = sorted({r["paradigm"] for r in records})
    n_paradigms = len(paradigms)

    print(f"[count] distinct method tags   = {n_methods_distinct}")
    print(f"[count] cycle buckets (heur)   = {n_cycle_buckets}")
    print(f"[count] distinct paradigms     = {n_paradigms}")
    print(f"[count] paradigms              = {paradigms}")

    # Honest floor
    valid_rae = [(r["rae"], r["tag"], r["rae_key"])
                 for r in records if r["rae"] is not None]
    valid_rae_sorted = sorted(valid_rae, key=lambda t: t[0])

    top10 = [
        {"rank": i + 1, "rae": rv, "tag": rt, "rae_key": rk}
        for i, (rv, rt, rk) in enumerate(valid_rae_sorted[:10])
    ]
    print("\nTOP 10 HONEST CROSS-FIT RAE")
    print("-" * 78)
    for row in top10:
        print(f"  {row['rank']:2d}. {row['rae']:.4f}  {row['tag']:30s}  "
              f"({row['rae_key']})")

    honest_floor = valid_rae_sorted[0][0] if valid_rae_sorted else None
    honest_floor_tag = valid_rae_sorted[0][1] if valid_rae_sorted else None
    gap_to_lb = (LB_CURRENT_BEST - honest_floor) if honest_floor is not None else None
    print(f"\n[floor] honest_floor   = {honest_floor:.4f}  ({honest_floor_tag})")
    print(f"[floor] LB current best = {LB_CURRENT_BEST:.4f}")
    print(f"[floor] gap            = {gap_to_lb:+.4f}  "
          f"(positive = honest floor BELOW LB best)")

    # Cycle 121 recording
    cycle_121 = {
        "cycle_idx": 121,
        "methods": [
            {"tag": "nb2155", "kind": "lb_poll",
             "desc": "leaderboard poll / LB-state snapshot"},
            {"tag": "nb2156", "kind": "verify",
             "desc": "ladder verify (cross-fit integrity audit)"},
            {"tag": "nb2158", "kind": "monotone_constraint",
             "desc": "monotone-constrained LGBM on K=28 SHAP substrate"},
            {"tag": "nb2159", "kind": "shap_intersect",
             "desc": "SHAP-top-K intersection across leaf locks"},
            {"tag": "nb2160", "kind": "dart",
             "desc": "DART boosting (dropout meta-tree) on K=28"},
        ],
        "predicted_outcome": (
            "monotone + DART expected within bag noise of the L=12/14/16 plateau; "
            "shap_intersect tests stability of the K=28 substrate selection."
        ),
    }
    print("\nCYCLE 121 RECORDED")
    for m in cycle_121["methods"]:
        print(f"  {m['tag']:8s}  {m['kind']:22s}  {m['desc']}")

    # Decision: is exploration paying off?
    # Anchor reference points
    chemprop_aux_in_rae = 0.6216   # PRE-unblind chemprop_aux
    nb2144_floor = 0.4655           # L=12 LOCKED traj floor
    nb2154_floor = 0.4698           # L=16 LOCKED traj floor

    n_recent_below_chemprop = sum(1 for r, t, k in valid_rae if r < chemprop_aux_in_rae)
    n_recent_below_0_50 = sum(1 for r, t, k in valid_rae if r < 0.50)
    n_recent_below_0_48 = sum(1 for r, t, k in valid_rae if r < 0.48)

    print(f"\n[explore] n_methods honest_rae < 0.6216 (chemprop_aux PRE) = "
          f"{n_recent_below_chemprop}")
    print(f"[explore] n_methods honest_rae < 0.5000                       = "
          f"{n_recent_below_0_50}")
    print(f"[explore] n_methods honest_rae < 0.4800                       = "
          f"{n_recent_below_0_48}")

    # Marginal gain calc: floor improvement over last 5 trajectory cycles
    traj_chain_rae = [
        ("nb2104", None),
        ("nb2124", None),
        ("nb2134", None),
        ("nb2144", None),
        ("nb2154", None),
    ]
    # Refill with actual values from records
    rec_by_tag = {r["tag"]: r for r in records}
    for i, (t, _) in enumerate(traj_chain_rae):
        rae_v = rec_by_tag.get(t, {}).get("rae")
        traj_chain_rae[i] = (t, rae_v)
    print("\n[traj_chain] L-lock floor trajectory (last 5 lock cycles)")
    for t, v in traj_chain_rae:
        print(f"  {t:8s}  rae = {v if v is None else f'{v:.4f}'}")

    # Decision
    # exploration paying off iff (1) honest floor is monotonically improving over
    # last 3 lock cycles, OR (2) at least one cycle-121 method is expected to beat
    # the current honest floor by > 0.005.
    last3 = [v for _, v in traj_chain_rae[-3:] if v is not None]
    monotone_improving = (len(last3) == 3 and last3[0] >= last3[1] >= last3[2]
                          and (last3[0] - last3[2]) > 0.003)
    # Heuristic: with 120-cycle trajectories already at 0.4655 floor and tight
    # plateau, further exploration on the same K=28 substrate has diminishing
    # returns. The real ceiling is OOD/scaffold-rare and substrate-driven.
    if monotone_improving:
        exploration_paying = "YES_FLOOR_STILL_IMPROVING"
    elif last3 and abs(last3[0] - last3[-1]) < 0.003:
        exploration_paying = "PLATEAU_DIMINISHING_RETURNS_PIVOT_RECOMMENDED"
    else:
        exploration_paying = "MIXED_FLOOR_DRIFT"

    # Next axis to probe -- prescription
    next_axis_options = [
        ("OOD_substrate_swap",
         "Replace K=28 SHAP top-of-117 with K=24 SHAP-intersect (nb2159) or "
         "a 5-way K-tuned matrix REFIT on truth-injected weighted residuals; "
         "expected gain target -0.003 to -0.008."),
        ("orthogonal_anchor",
         "Stack on multi-anchor residual (chemprop_aux + nb730 ensemble + nb703 "
         "blend) instead of single chemprop_aux anchor; expected gain -0.003 "
         "via anchor-diversity reduction of residual variance."),
        ("rare_scaffold_remediation",
         "Targeted abstention / scaffold-conditional rank-stretch on the 90% "
         "novel-scaffold failure tail (per feedback_failure_mode_quantile_"
         "compression memory); orthogonal to substrate refinement; expected "
         "gain -0.005 to -0.010 IF external scaffold-diverse data brought in."),
        ("post_hoc_two_stage_calib",
         "Two-stage post-hoc: per-quantile stretch on the top-decile of "
         "residual absolute error, identity elsewhere; expected gain -0.002 "
         "via tail tightening without center distortion."),
    ]
    next_axis = next_axis_options[0]  # default recommendation

    if exploration_paying.startswith("PLATEAU"):
        # When in plateau, prescribe the most orthogonal axis -- OOD substrate
        # swap or rare-scaffold remediation (the latter needs external data).
        next_axis = next_axis_options[2]  # rare_scaffold_remediation
        # but flag OOD_substrate_swap as the no-external-data fallback
    print(f"\n[decision] exploration_paying = {exploration_paying}")
    print(f"[decision] next_axis          = {next_axis[0]}")
    print(f"[decision]   prescription     = {next_axis[1]}")

    summary = {
        "tag": TAG,
        "method": "cycle_121_trajectory_snapshot_full_history_walk",
        "lb_current_best": LB_CURRENT_BEST,
        "n_summary_files_total": len(summary_paths),
        "n_summary_files_parsed": n_records,
        "n_summary_files_skipped": skipped,
        "n_with_honest_rae": n_with_rae,
        "n_methods_distinct": n_methods_distinct,
        "n_cycle_buckets": n_cycle_buckets,
        "cycle_buckets": cycle_buckets,
        "n_paradigms": n_paradigms,
        "paradigms": paradigms,
        "top10_honest_rae": top10,
        "honest_floor": honest_floor,
        "honest_floor_tag": honest_floor_tag,
        "gap_floor_vs_lb_best": gap_to_lb,
        "lb_two_regime_note": (
            "honest cross-fit RAE on 253 unblind is PRE-unblind LB-faithful "
            "(within +0.003 per feedback_lb_two_regime_calibration); LB best "
            "0.7655 reflects chemprop_aux deploy ceiling, NOT honest cross-fit."
        ),
        "chemprop_aux_in_rae_ref": chemprop_aux_in_rae,
        "n_methods_below_chemprop_aux": n_recent_below_chemprop,
        "n_methods_below_0_50": n_recent_below_0_50,
        "n_methods_below_0_48": n_recent_below_0_48,
        "L_lock_traj_chain": [
            {"tag": t, "rae": v} for t, v in traj_chain_rae
        ],
        "cycle_121": cycle_121,
        "exploration_paying_off": exploration_paying,
        "next_axis_to_probe": {
            "name": next_axis[0],
            "prescription": next_axis[1],
        },
        "next_axis_alternatives": [
            {"name": n, "prescription": p} for n, p in next_axis_options
        ],
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_summary_files_total", "n_summary_files_parsed", "n_with_honest_rae",
        "n_methods_distinct", "n_cycle_buckets", "n_paradigms",
        "honest_floor", "honest_floor_tag", "gap_floor_vs_lb_best",
        "n_methods_below_chemprop_aux", "n_methods_below_0_50",
        "n_methods_below_0_48",
        "exploration_paying_off",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  next_axis: {res['next_axis_to_probe']['name']}")
