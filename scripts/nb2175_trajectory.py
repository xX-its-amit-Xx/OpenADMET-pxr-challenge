"""nb2175 -- CYCLE 123 TRAJECTORY SNAPSHOT.

PROTOCOL:
    Walks data/processed/*_summary.json AND scripts/*.py / submissions/*.csv
    to compute total methods explored (~795 across 123 cycles). Extracts
    honest cross-fit RAE estimates from every summary (under any of several
    known keys), counts methods / cycles / paradigms, computes the running
    honest floor (EXCLUDING lucky-seed candidates flagged by nb2156).

    Records cycle 123 as the 5-method loop:
        LB poll, anchor swap (nb730 instead of chemprop_aux), family ablation,
        residual cascade, tanh target.

    Catalogs CLOSED axes after cycle 123 (adds: row_bootstrap, XGB SHAP,
    sklearn-stack, monotone, DART, feature_fraction, min_child, num_leaves,
    lr, pooled-120, SHAP-seed-perturbation -- all probed with |delta| <= 0.003
    vs L=16/L=12 plateau).

    Highlights STILL OPEN axes after cycle 123 (novel-anchor probed at
    nb730 in cycle 123 -- if marginal, mark NEAR-CLOSED; still genuinely open:
    external-data augmentation, abstention/conformal, NN model class,
    positive-unlabeled framing).

OUTPUTS:
    scripts/nb2175_trajectory.py
    data/processed/nb2175_summary.json
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

TAG = "nb2175"
CYCLE_IDX = 123

# Current LB-best (chemprop_aux PRE-unblind, 0.7655 placeholder cap)
LB_CURRENT_BEST = 0.7655

# Lucky-seed candidates -- flagged by nb2156 as not reproducible across kf_seeds
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
    ("stack_lgbm",            r"stack|meta_learner|residual_stack|sklearn_stack|cascade"),
    ("rank_stretch_calib",    r"stretch|rank_calib|isotonic|conformal|tanh"),
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
    ("multitask_nr",          r"nr_family|multitask|tox21|broad_analogy|family_ablation"),
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
    ("anchor_swap",           r"anchor_swap|nb730_anchor|novel_anchor"),
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


def _count_method_universe(root: Path) -> dict:
    """Count methods across summaries + scripts + submissions (~795 expected)."""
    n_summaries = len(list((root / "data/processed").glob("*_summary.json")))
    n_scripts = len(list((root / "scripts").glob("nb*.py")))
    n_submissions = len(list((root / "submissions").glob("*.csv")))
    # Method universe = unique nbXXX prefixes across all three pools
    tag_pat = re.compile(r"(nb\d+)")
    seen: set[str] = set()
    for p in (root / "data/processed").glob("*_summary.json"):
        m = tag_pat.match(p.stem)
        if m:
            seen.add(m.group(1))
    for p in (root / "scripts").glob("nb*.py"):
        m = tag_pat.match(p.stem)
        if m:
            seen.add(m.group(1))
    for p in (root / "submissions").glob("*.csv"):
        m = tag_pat.search(p.stem)
        if m:
            seen.add(m.group(1))
    return {
        "n_summaries": n_summaries,
        "n_scripts": n_scripts,
        "n_submissions": n_submissions,
        "n_distinct_nb_tags": len(seen),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CYCLE {CYCLE_IDX} TRAJECTORY SNAPSHOT")
    print(f"          walk {DATA_PROCESSED}/*_summary.json")
    print(f"          LB_CURRENT_BEST = {LB_CURRENT_BEST}")
    print(f"          LUCKY_SEED_FLAGGED = {sorted(LUCKY_SEED_FLAGGED)}")
    print("=" * 78)

    repo_root = Path(__file__).resolve().parents[1]
    universe = _count_method_universe(repo_root)
    print(f"[universe] n_summaries={universe['n_summaries']}  "
          f"n_scripts={universe['n_scripts']}  "
          f"n_submissions={universe['n_submissions']}  "
          f"n_distinct_nb_tags={universe['n_distinct_nb_tags']}")

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

    # Methods universe blends summary-tags + script-tags + submission-tags.
    # User-tracked "~795 methods" is the union across the three pools.
    methods_universe_total = universe["n_distinct_nb_tags"]

    print(f"[walk] parsed={n_records}  honest_rae={n_with_rae}  skipped={skipped}")
    print(f"[count] cycles                = {CYCLE_IDX}")
    print(f"[count] methods (universe)    = ~{methods_universe_total}")
    print(f"[count] methods (w/ summary)  = {n_methods_distinct}")
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

    lucky_records = [(r["rae"], r["tag"], r["rae_key"]) for r in records
                     if r["rae"] is not None and r["tag"] in LUCKY_SEED_FLAGGED]

    honest_floor = valid_sorted[0][0] if valid_sorted else None
    honest_floor_tag = valid_sorted[0][1] if valid_sorted else None
    gap_to_lb = (LB_CURRENT_BEST - honest_floor) if honest_floor is not None else None
    print(f"\n[floor] reproducible honest floor = {honest_floor:.4f}  "
          f"({honest_floor_tag})")
    print(f"[floor] LB current best           = {LB_CURRENT_BEST:.4f}")
    print(f"[floor] gap                       = {gap_to_lb:+.4f}")

    # Cycle 123 -- 5 methods this cycle.
    cycle_123_methods = [
        ("nb2170", "lb_poll",
         "leaderboard poll & state snapshot post-nb2168 decision; record "
         "current LB-best, latest submission scores, drift vs honest floor"),
        ("nb2171", "anchor_swap_nb730",
         "anchor swap -- residualize on nb730 (multi-seed null-ensemble, "
         "RAE 0.4603 cross-fit) instead of chemprop_aux; tests whether the "
         "residual tail decorrelates when the anchor is the strongest "
         "stand-alone honest model rather than the LB-leading PRE-unblind one"),
        ("nb2172", "family_ablation_nr",
         "nuclear-receptor family ablation -- drop NR-family features one at "
         "a time from the K=28 SHAP substrate, measure mean-bag delta; "
         "diagnoses which family carries the SHAP-substrate signal"),
        ("nb2173", "residual_cascade",
         "two-stage residual cascade: stage-1 anchor (nb730 or chemprop_aux), "
         "stage-2 LGBM on stage-1 residuals with K=28 substrate + counter-assay "
         "feature; tests cascade depth-2 vs single-stage stack"),
        ("nb2174", "tanh_target",
         "tanh-rescaled pEC50 target -- transform pEC50 via "
         "y' = tanh((y - mu) / sigma); tests whether bounded-target "
         "regression compresses novel-scaffold tail variance (operates on "
         "target, not features); compare against linear target at L-lock"),
    ]
    rec_by_tag = {r["tag"]: r for r in records}
    cycle_123 = {
        "cycle_idx": CYCLE_IDX,
        "methods": [
            {
                "tag": tg, "kind": kind, "desc": desc,
                "rae": rec_by_tag.get(tg, {}).get("rae"),
                "status": "RECORDED" if tg in rec_by_tag else "PENDING_SUMMARY",
            }
            for tg, kind, desc in cycle_123_methods
        ],
    }
    print(f"\nCYCLE {CYCLE_IDX} RECORDED")
    for m in cycle_123["methods"]:
        rae_str = f"{m['rae']:.4f}" if m["rae"] is not None else "  N/A "
        print(f"  {m['tag']:8s}  {m['kind']:24s}  rae={rae_str}  "
              f"({m['status']})")

    # CLOSED axes -- closed at <= 0.003 margin vs the L-lock plateau.
    # Cycle 123 adds row_bootstrap, XGB SHAP, sklearn-stack to the closed list
    # (probed cycle 122 -- all returned within bag noise of the plateau).
    closed_axes = [
        {"axis": "num_leaves_grid",        "probe_tag": "nb2103",
         "result": "L in {6..32} grid; plateau at L=12..16; |delta| <= 0.003"},
        {"axis": "learning_rate_grid",     "probe_tag": "nb2121",
         "result": "lr in {0.01..0.1}; lr=0.03 marginally best; "
                   "|delta| <= 0.003"},
        {"axis": "min_child_grid",         "probe_tag": "nb2151",
         "result": "min_child_samples in {3..30}; mc=5 marginally best; "
                   "|delta| <= 0.003"},
        {"axis": "feature_fraction_grid",  "probe_tag": "nb2153",
         "result": "ff in {0.6..1.0}; ff=1.0 marginally best; "
                   "|delta| <= 0.003"},
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
        {"axis": "row_bootstrap_perturbation", "probe_tag": "nb2166",
         "result": "row bagging on K=28 substrate; substrate-vs-data variance "
                   "separated; |delta| <= 0.003 vs leaf-only bag"},
        {"axis": "cross_paradigm_shap_substrate", "probe_tag": "nb2165",
         "result": "XGB on K=28 SHAP substrate; XGB-derived SHAP overlaps "
                   "LGBM-derived top-K by >85%; |delta| <= 0.003"},
        {"axis": "sklearn_stack_meta_learner", "probe_tag": "nb2167",
         "result": "sklearn StackingRegressor (ElasticNet + RF + XGB heads) "
                   "on K=28 substrate; meta-learner collapses to LGBM-like "
                   "weights; |delta| <= 0.003"},
    ]

    # OPEN axes after cycle 123. The novel-anchor axis WAS probed this cycle
    # via nb2171 (nb730 anchor); if its delta is within 0.003 of chemprop_aux
    # anchor, mark it NEAR-CLOSED. The remaining genuinely open axes are:
    open_axes = [
        {"axis": "novel_anchor_nb730",
         "status": "PROBED_CYCLE_123_PENDING_VERDICT",
         "rationale": "nb2171 swaps chemprop_aux for nb730 as the residualize "
                      "anchor. If residual tail decorrelates (delta -0.005+), "
                      "axis remains OPEN with EV; if delta within bag noise, "
                      "closes the anchor-swap dimension and forces external "
                      "data or NN model class as the next move"},
        {"axis": "external_scaffold_diverse_data",
         "status": "OPEN",
         "rationale": "feedback_unblind_augmentation closed in-distribution "
                      "augmentation; OOD wall set by scaffold support; "
                      "ChEMBL PXR analogs / external NR actives outside the "
                      "4139-train scaffold support are the highest theoretical "
                      "ceiling (-0.020 plausibly); 3-5 day engineering cost"},
        {"axis": "abstention_conformal_shrink",
         "status": "OPEN",
         "rationale": "Per feedback_failure_mode_quantile_compression: 90% of "
                      "worst-50 errors are novel-scaffold (scaf_train_freq=0). "
                      "Per-row conformal scoring + shrink-to-mean on top-5% "
                      "uncertain rows directly attacks RAE numerator; no "
                      "external data required. EV -0.005 to -0.010"},
        {"axis": "neural_network_model_class",
         "status": "OPEN",
         "rationale": "All trajectory bags + stacking heads are LGBM / XGB / "
                      "RF / ElasticNet. A small MLP or Transformer on K=28 "
                      "substrate or Morgan-2048 may extract non-axis-aligned "
                      "interactions tree ensembles miss; counter-evidence: "
                      "nb601/602 ChemBERTa-77M collapsed to 0-weight in SLSQP. "
                      "Worth probing on K=28 SHAP substrate (smaller surface, "
                      "less FT overfit) not raw Morgan"},
        {"axis": "positive_unlabeled_framing",
         "status": "OPEN",
         "rationale": "Single-concentration screen (21,003 rows, 8,126 unique "
                      "to SP) is currently used only as an unlabeled aux head. "
                      "Re-framing as positive-unlabeled (PU) learning -- hits "
                      "are positives, non-hits are unlabeled -- could extract "
                      "additional signal via PU bias correction (Elkan / "
                      "Kiryo nnPU); orthogonal to all tested axes"},
    ]

    print("\nCLOSED AXES (<= 0.003 margin)")
    for ax in closed_axes:
        print(f"  - {ax['axis']:32s}  via {ax['probe_tag']}")
    print("\nSTILL OPEN AXES (cycle 123 forward)")
    for ax in open_axes:
        print(f"  - {ax['axis']:32s}  [{ax['status']}]")

    # Decision: highest-EV next axis given what was learned in cycle 123.
    #
    # Reasoning chain after cycle 123 results:
    #
    #   (1) nb2171 anchor-swap to nb730 is THE biggest learn from cycle 123.
    #       - If nb2171 succeeds (delta <= -0.005), the residual-cascade
    #         nb2173 on the nb730 anchor inherits that gain -> double-up on
    #         cascade depth (nb730 anchor + 2-stage residual + K=28 substrate)
    #         as the next axis; EV -0.005 to -0.010.
    #       - If nb2171 fails (|delta| <= 0.003), then ALL within-train-manifold
    #         axes are now closed (leaf-config, SHAP basis, bagging surface,
    #         model class via stack, anchor identity, target shape). Only
    #         OUT-OF-MANIFOLD moves remain: external data, abstention, PU.
    #
    #   (2) Among out-of-manifold moves:
    #       - abstention_conformal_shrink: zero new data, 1-2 day engineering,
    #         EV -0.005 to -0.010 (directly attacks the novel-scaffold tail).
    #       - external_scaffold_diverse_data: 3-5 day engineering, EV -0.010
    #         to -0.020 ceiling but risk of negative transfer.
    #       - positive_unlabeled: 2-3 day engineering, EV -0.003 to -0.008,
    #         leverages single-conc 21k rows that have NOT been mined.
    #       - NN model class: 2-3 day engineering, mixed evidence (ChemBERTa
    #         already failed); worth retrying on K=28 substrate only.
    #
    #   (3) nb2174 tanh-target lesson: bounded-target compression is a TARGET-
    #       side move, orthogonal to feature/model axes. If it shows ANY signal
    #       (delta <= -0.003), it stacks additively with abstention.
    #
    # DECISION: abstention_conformal_shrink is highest-EV unconditionally
    # because (a) cheapest engineering (1-2 days), (b) directly attacks the
    # documented dominant failure mode (novel-scaffold tail = 90% of worst-50
    # errors), (c) stacks additively with whichever cycle-123 method wins
    # (anchor swap, cascade, or tanh target), (d) no external-data dependency,
    # (e) no negative-transfer risk.
    decision = {
        "highest_EV_next_axis": "abstention_conformal_shrink",
        "rationale": (
            "Cycle 123 probed the last within-manifold axis (anchor identity "
            "via nb2171 nb730 swap). Conditional on either outcome -- anchor "
            "swap helps (compound with abstention) or anchor swap is null "
            "(within-manifold closed) -- the highest-EV next axis is "
            "abstention/conformal-shrink: cheapest engineering (1-2 days), "
            "attacks the documented dominant failure mode (90% of worst-50 "
            "errors are novel-scaffold per "
            "feedback_failure_mode_quantile_compression), stacks additively "
            "with nb2171/nb2173/nb2174 wins, no external-data dependency, "
            "no negative-transfer risk. EV est -0.005 to -0.010 RAE."
        ),
        "secondary_axis_if_abstention_underdelivers": (
            "external_scaffold_diverse_data: curate ChEMBL PXR + NR actives "
            "outside train scaffold support, re-train chemprop_aux / nb730 "
            "anchor with augmented set. EV ceiling -0.010 to -0.020 but "
            "3-5 day engineering cost and negative-transfer risk."
        ),
        "tertiary_axis_if_external_data_blocked": (
            "positive_unlabeled framing on the 21k single-concentration screen "
            "(8126 SP-exclusive compounds). Re-frame hits as positives, "
            "non-hits as unlabeled; train PU classifier (Elkan / nnPU) and "
            "stack output as a 29th K-substrate feature. EV -0.003 to -0.008, "
            "orthogonal to all tested axes."
        ),
        "skip_for_now": [
            "neural_network_model_class: ChemBERTa already failed (nb601/602 "
            "0-weight); MLP retry on K=28 substrate is worth a single probe "
            "but not first-priority",
            "novel_anchor beyond nb730: anchor universe shrinks fast after "
            "nb730 swap; further anchor probes are 3rd-tier",
        ],
    }
    print(f"\n[decision] highest-EV next axis = "
          f"{decision['highest_EV_next_axis']}")
    print(f"[decision] rationale            = {decision['rationale']}")

    summary = {
        "tag": TAG,
        "method": f"cycle_{CYCLE_IDX}_trajectory_snapshot_full_history_walk",
        "lb_current_best": LB_CURRENT_BEST,
        "lucky_seed_flagged_by_nb2156": sorted(LUCKY_SEED_FLAGGED),
        "n_cycles_total": CYCLE_IDX,
        "method_universe": universe,
        "method_universe_total": methods_universe_total,
        "n_summary_files_total": len(summary_paths),
        "n_summary_files_parsed": n_records,
        "n_summary_files_skipped": skipped,
        "n_with_honest_rae": n_with_rae,
        "n_methods_distinct_summary": n_methods_distinct,
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
        "cycle_123": cycle_123,
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
        "n_cycles_total", "method_universe_total",
        "n_summary_files_total", "n_with_honest_rae",
        "n_methods_distinct_summary",
        "n_cycle_buckets", "n_paradigms",
        "honest_floor", "honest_floor_tag", "gap_floor_vs_lb_best",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  highest_EV_next_axis: {res['decision']['highest_EV_next_axis']}")
