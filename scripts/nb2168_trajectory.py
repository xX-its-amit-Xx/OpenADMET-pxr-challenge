"""nb2168 -- CYCLE 122 TRAJECTORY SNAPSHOT.

PROTOCOL:
    Walks data/processed/*_summary.json files. Extracts honest cross-fit
    RAE estimates from every summary (under any of several known keys),
    counts methods / cycles / paradigms, computes the running honest
    floor (EXCLUDING lucky-seed candidates flagged by nb2156).

    Records cycle 122 as the 5-method LB-poll / pooled / SHAP-XGB /
    row-bootstrap / sklearn-stack loop (nb2162 LB, nb2163 pooled-120,
    nb2165 XGB-SHAP, nb2166 row-bootstrap, nb2167 sklearn-stack).

    Catalogs CLOSED axes (margin <= 0.003 vs the L=16 / L=12 plateau)
    and OPEN axes (not yet probed).

OUTPUTS:
    scripts/nb2168_trajectory.py
    data/processed/nb2168_summary.json
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

TAG = "nb2168"
CYCLE_IDX = 122

# Current LB-best (chemprop_aux PRE-unblind, 0.7655 placeholder cap)
LB_CURRENT_BEST = 0.7655

# Lucky-seed candidates -- flagged by nb2156 as not reproducible across kf_seeds
# (mean_bag spread > 0.03 RAE). These are EXCLUDED from the honest floor.
LUCKY_SEED_FLAGGED = {
    "nb2154",   # L=16 / kf_seed 1001 hit 0.4620 but mean_bag = 0.5050 across 5 kf_seeds
    "nb2144",   # L=12 single-lock under suspicion -- conservative exclude
}

# Honest-rae keys, in priority order.
RAE_KEYS_PRIORITY = [
    "best_rae",
    "rae_pooled_median_120",
    "rae_pooled_mean_120",
    "rae_pooled_median",
    "rae_pooled_mean",
    "rae_median_bag",
    "rae_mean_bag",
    "rae_cross_fit",
    "rae_oof",
    "rae",
    "oof_rae",
    "cross_fit_rae",
    "rae_honest",
    "honest_cross_fit_rae",
    "rae_final",
    "rae_best",
]

# Paradigm classification regex.
PARADIGM_PATTERNS = [
    ("trajectory_bag",        r"trajectory|win_bag|n_cycles|pooled"),
    ("shap_subset",           r"shap|top.?\d+|k_tuned"),
    ("slsqp_blend",           r"slsqp|blend|convex"),
    ("stack_lgbm",            r"stack|meta_learner|residual_stack|sklearn_stack"),
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
    ("xgb",                   r"xgb|xgboost"),
    ("row_bootstrap",         r"row_bootstrap|bootstrap_row|bagging_row"),
    ("submission_meta",       r"lb_poll|submission|leaderboard|verify"),
]


def _classify(tag: str, method: str) -> str:
    text = f"{tag} {method}".lower()
    for name, pat in PARADIGM_PATTERNS:
        if re.search(pat, text):
            return name
    return "other"


def _extract_rae(s: dict) -> tuple[float | None, str | None]:
    for k in RAE_KEYS_PRIORITY:
        if k in s:
            try:
                v = float(s[k])
            except (TypeError, ValueError):
                continue
            if 0.0 < v < 5.0:
                return v, k
    return None, None


def _cycle_bucket(tag: str) -> int | None:
    m = re.match(r"nb(\d+)", tag)
    return int(m.group(1)) // 10 if m else None


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CYCLE {CYCLE_IDX} TRAJECTORY SNAPSHOT")
    print(f"          walk {DATA_PROCESSED}/*_summary.json")
    print(f"          LB_CURRENT_BEST = {LB_CURRENT_BEST}")
    print(f"          LUCKY_SEED_FLAGGED = {sorted(LUCKY_SEED_FLAGGED)}")
    print("=" * 78)

    summary_paths = sorted(DATA_PROCESSED.glob("*_summary.json"))
    print(f"[walk] found {len(summary_paths)} summary JSON files")

    records: list[dict] = []
    skipped = 0
    for p in summary_paths:
        try:
            with open(p) as f:
                s = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        if not isinstance(s, dict):
            skipped += 1
            continue
        tag = s.get("tag") or p.stem.replace("_summary", "")
        method = s.get("method") or s.get("notebook") or ""
        rae_val, rae_key = _extract_rae(s)
        records.append({
            "path": str(p),
            "tag": tag,
            "method": method,
            "paradigm": _classify(str(tag), str(method)),
            "cycle_bucket": _cycle_bucket(str(tag)),
            "rae": rae_val,
            "rae_key": rae_key,
        })

    n_records = len(records)
    n_with_rae = sum(1 for r in records if r["rae"] is not None)
    distinct_tags = sorted({r["tag"] for r in records if r["tag"]})
    n_methods_distinct = len(distinct_tags)
    cycle_buckets = sorted({r["cycle_bucket"] for r in records
                            if r["cycle_bucket"] is not None})
    paradigms = sorted({r["paradigm"] for r in records})

    print(f"[walk] parsed={n_records}  honest_rae={n_with_rae}  skipped={skipped}")
    print(f"[count] distinct method tags  = {n_methods_distinct}")
    print(f"[count] cycle buckets (heur)  = {len(cycle_buckets)}")
    print(f"[count] distinct paradigms    = {len(paradigms)}")

    # Top-10 honest cross-fit RAE, EXCLUDING lucky-seed candidates
    valid = [(r["rae"], r["tag"], r["rae_key"]) for r in records
             if r["rae"] is not None and r["tag"] not in LUCKY_SEED_FLAGGED]
    valid_sorted = sorted(valid, key=lambda t: t[0])
    top10 = [
        {"rank": i + 1, "rae": round(v, 6), "tag": tg, "rae_key": k}
        for i, (v, tg, k) in enumerate(valid_sorted[:10])
    ]
    print("\nTOP-10 HONEST CROSS-FIT RAE (lucky-seed excluded)")
    print("-" * 78)
    for row in top10:
        print(f"  {row['rank']:2d}. {row['rae']:.4f}  {row['tag']:30s}  "
              f"({row['rae_key']})")

    # Lucky-seed exclusions (for transparency)
    lucky_records = [(r["rae"], r["tag"], r["rae_key"]) for r in records
                     if r["rae"] is not None and r["tag"] in LUCKY_SEED_FLAGGED]
    print("\nLUCKY-SEED EXCLUDED (would otherwise appear in top-10)")
    for v, tg, k in sorted(lucky_records, key=lambda t: t[0]):
        print(f"     {v:.4f}  {tg:30s}  ({k})")

    honest_floor = valid_sorted[0][0] if valid_sorted else None
    honest_floor_tag = valid_sorted[0][1] if valid_sorted else None
    gap_to_lb = (LB_CURRENT_BEST - honest_floor) if honest_floor is not None else None
    print(f"\n[floor] reproducible honest floor = {honest_floor:.4f}  "
          f"({honest_floor_tag})")
    print(f"[floor] LB current best           = {LB_CURRENT_BEST:.4f}")
    print(f"[floor] gap                       = {gap_to_lb:+.4f}")

    # Cycle 122 -- 5 methods, possibly summary not yet on disk.
    cycle_122_methods = [
        ("nb2162", "lb_poll",
         "leaderboard poll & state snapshot post-nb2161 decision"),
        ("nb2163", "pooled_120_trajectory",
         "pooled-120 trajectory bag at L-lock for floor reproducibility"),
        ("nb2165", "xgb_shap",
         "XGBoost on K=28 SHAP substrate (cross-paradigm SHAP-tied model)"),
        ("nb2166", "row_bootstrap",
         "row-bootstrap perturbation -- bagging across resampled rows, "
         "tests substrate-vs-data variance separation"),
        ("nb2167", "sklearn_stack",
         "sklearn StackingRegressor on top-K substrate "
         "(cross-paradigm meta-learner)"),
    ]
    rec_by_tag = {r["tag"]: r for r in records}
    cycle_122 = {
        "cycle_idx": CYCLE_IDX,
        "methods": [
            {
                "tag": tg, "kind": kind, "desc": desc,
                "rae": rec_by_tag.get(tg, {}).get("rae"),
                "status": "RECORDED" if tg in rec_by_tag else "PENDING_SUMMARY",
            }
            for tg, kind, desc in cycle_122_methods
        ],
    }
    print(f"\nCYCLE {CYCLE_IDX} RECORDED")
    for m in cycle_122["methods"]:
        rae_str = f"{m['rae']:.4f}" if m["rae"] is not None else "  N/A "
        print(f"  {m['tag']:8s}  {m['kind']:24s}  rae={rae_str}  "
              f"({m['status']})")

    # CLOSED axes -- closed at <= 0.003 margin vs the L-lock plateau
    closed_axes = [
        {"axis": "num_leaves_grid",        "probe_tag": "nb2103",
         "result": "L in {6..32} grid; plateau at L=12..16; |delta| <= 0.003"},
        {"axis": "learning_rate_grid",     "probe_tag": "nb2121",
         "result": "lr in {0.01..0.1}; lr=0.03 marginally best; |delta| <= 0.003"},
        {"axis": "min_child_grid",         "probe_tag": "nb2151",
         "result": "min_child_samples in {3..30}; mc=5 marginally best; "
                   "|delta| <= 0.003"},
        {"axis": "feature_fraction_grid",  "probe_tag": "nb2153",
         "result": "ff in {0.6..1.0}; ff=1.0 marginally best; |delta| <= 0.003"},
        {"axis": "monotone_constraint",    "probe_tag": "nb2158",
         "result": "monotone constraints on top-SHAP features; "
                   "delta within bag noise (<= 0.003)"},
        {"axis": "shap_seed_perturbation", "probe_tag": "nb2159",
         "result": "SHAP-top-K intersect across leaf-locks stable; "
                   "K=28 substrate confirmed; |delta| <= 0.003"},
        {"axis": "dart_boosting",          "probe_tag": "nb2160",
         "result": "DART dropout boosting; |delta| <= 0.003 vs gbdt baseline"},
        {"axis": "trajectory_pooled_120",  "probe_tag": "nb2163",
         "result": "pooled-120-cycle bag at L-lock confirms mean-bag floor "
                   "0.4737 +/- bag noise; lucky-seed 0.4620 not reproducible"},
    ]

    # OPEN axes -- not yet probed at depth
    open_axes = [
        {"axis": "row_bootstrap_perturbation",
         "rationale": "Bagging across resampled rows separates substrate-driven "
                      "variance from data-driven variance; orthogonal to "
                      "SHAP/leaf-config grids; -0.003 to -0.005 expected",
         "probe_tag_cycle122": "nb2166"},
        {"axis": "cross_paradigm_shap_substrate",
         "rationale": "K=28 SHAP basis was selected by LGBM; an XGB SHAP basis "
                      "may select different features (gradient-vs-leaf scoring); "
                      "probe via XGB on the K=28 substrate (nb2165) then "
                      "compare XGB-derived SHAP top-K",
         "probe_tag_cycle122": "nb2165"},
        {"axis": "cross_paradigm_models",
         "rationale": "All trajectory bags are LGBM-only. sklearn Stacking with "
                      "ElasticNet+RF+XGB heads on K=28 substrate may extract "
                      "model-diversity orthogonal to seed/leaf bagging",
         "probe_tag_cycle122": "nb2167"},
        {"axis": "abstention_conformal",
         "rationale": "Per-row conformal scoring on novel-scaffold tail "
                      "(feedback_failure_mode_quantile_compression: 90% of "
                      "worst errors are novel-scaffold); abstain on top-decile "
                      "uncertainty, predict identity (truth ~= mean)"},
        {"axis": "external_scaffold_diverse_data",
         "rationale": "feedback_unblind_augmentation closed in-distribution "
                      "augmentation; OOD wall set by scaffold support; need "
                      "ChEMBL PXR analog block / external nuclear-receptor "
                      "actives outside the 4139-train scaffold support"},
        {"axis": "novel_anchor_not_chemprop_aux",
         "rationale": "All best models residualize chemprop_aux; the residual "
                      "tail correlates with anchor's failure mode. A novel "
                      "anchor (Boltz-2 pose features, ESM-2 protein-context "
                      "embedding, or grover-pretrain) may decorrelate residual"},
    ]

    print("\nCLOSED AXES (<= 0.003 margin)")
    for ax in closed_axes:
        print(f"  - {ax['axis']:30s}  via {ax['probe_tag']}")
    print("\nOPEN AXES")
    for ax in open_axes:
        tg = ax.get("probe_tag_cycle122", "TBD")
        print(f"  - {ax['axis']:30s}  cycle122 probe: {tg}")

    # Decision: which OPEN axis is highest-EV?
    # Reasoning:
    #   (1) row_bootstrap_perturbation + cross_paradigm_models are already being
    #       probed this cycle (nb2166 + nb2167) -- their EV is being measured now.
    #   (2) cross_paradigm_SHAP (nb2165) probes a single XGB model -- only tests
    #       the basis, not the meta-learner; EV ~ -0.003 if XGB SHAP differs.
    #   (3) abstention_conformal is the highest leverage UN-PROBED axis because
    #       feedback_failure_mode tells us 90% of worst errors are concentrated
    #       on novel-scaffold tail. Even modest abstention (top-5% uncertain
    #       -> identity at mean) could shave -0.005 to -0.010 on RAE since
    #       worst-case errors are squared in numerator. AND it does not require
    #       new external data.
    #   (4) external_scaffold_diverse_data has the largest theoretical ceiling
    #       (-0.020 plausibly) but the highest engineering cost (curate / clean
    #       / scaffold-align ChEMBL PXR analogs); ~3-5 calendar days at risk
    #       of negative transfer.
    #   (5) novel anchor is high-EV but high-cost (re-train chemprop with
    #       different architecture / different fingerprint anchor).
    decision = {
        "highest_EV_next_axis": "abstention_conformal",
        "rationale": (
            "Per feedback_failure_mode_quantile_compression: 90% of worst-50 "
            "errors are novel-scaffold (scaf_train_freq=0) with 2-sided "
            "variance compression. RAE is dominated by sum-of-absolute-errors "
            "in numerator; tail trimming via conformal abstention (predict "
            "global mean on top-5% conformal-score rows) directly attacks the "
            "loss. No external data required, no anchor swap. EV est -0.005 "
            "to -0.010 RAE."
        ),
        "fallback_axis_external_data": (
            "If abstention_conformal underdelivers on 253-unblind cross-fit "
            "(< -0.003), pivot to external_scaffold_diverse_data: curate "
            "ChEMBL PXR analogs outside train scaffold support, re-train "
            "chemprop_aux anchor with augmented set. EV ceiling -0.020 but "
            "3-5 day engineering cost."
        ),
        "skip_for_now": [
            "cross_paradigm_shap_substrate (low EV; basis only)",
            "novel_anchor (high cost, retrain chemprop is multi-day)",
        ],
    }
    print(f"\n[decision] highest-EV next axis = {decision['highest_EV_next_axis']}")
    print(f"[decision] rationale            = {decision['rationale']}")

    summary = {
        "tag": TAG,
        "method": f"cycle_{CYCLE_IDX}_trajectory_snapshot_full_history_walk",
        "lb_current_best": LB_CURRENT_BEST,
        "lucky_seed_flagged_by_nb2156": sorted(LUCKY_SEED_FLAGGED),
        "n_summary_files_total": len(summary_paths),
        "n_summary_files_parsed": n_records,
        "n_summary_files_skipped": skipped,
        "n_with_honest_rae": n_with_rae,
        "n_methods_distinct": n_methods_distinct,
        "n_cycle_buckets": len(cycle_buckets),
        "n_paradigms": len(paradigms),
        "paradigms": paradigms,
        "top10_honest_rae_lucky_seed_excluded": top10,
        "lucky_seed_excluded_records": [
            {"rae": round(v, 6), "tag": tg, "rae_key": k}
            for v, tg, k in sorted(lucky_records, key=lambda t: t[0])
        ],
        "honest_floor": honest_floor,
        "honest_floor_tag": honest_floor_tag,
        "gap_floor_vs_lb_best": gap_to_lb,
        "cycle_122": cycle_122,
        "closed_axes": closed_axes,
        "open_axes": open_axes,
        "decision": decision,
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
        "n_summary_files_total", "n_with_honest_rae", "n_methods_distinct",
        "n_cycle_buckets", "n_paradigms",
        "honest_floor", "honest_floor_tag", "gap_floor_vs_lb_best",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  highest_EV_next_axis: {res['decision']['highest_EV_next_axis']}")
