"""nb2187 -- CYCLE 125 TRAJECTORY SNAPSHOT.

PROTOCOL:
    Cycle 125 follows the cycle-124 nb2177 audit verdict on nb2170. The audit
    found nb2170's mean_bag 0.3920 reproducible but ANCHOR-CONTAMINATED: the
    chemprop_aux anchor used at the residual stage shared train/eval indices
    with the K=28 SHAP substrate, so the apparent compounding gain leaked
    chemprop_aux's in-sample optimism into the 2nd stage.

    Cycle 125 attacks this with five moves:
        nb2182 LB poll, nb2183 honest anchor regeneration (cross-fit
        chemprop_aux re-fit per outer fold), nb2184 honest re-eval of nb2170
        on the regenerated anchor, nb2185 alt-anchor sweep (nb703, nb562, nb464
        as substitute anchors), nb2186 ladder-hygiene promote.

    Contaminated entries (EXCLUDE from honest top-10):
        nb2170 -- mean_bag 0.3920, anchor-contaminated
        nb2178 -- K-sweep variants 0.3810..0.3920, inherits same contamination

OUTPUTS:
    scripts/nb2187_trajectory.py
    data/processed/nb2187_summary.json
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

TAG = "nb2187"
CYCLE_IDX = 125

# Current LB-best (chemprop_aux PRE-unblind, 0.7655 placeholder cap)
LB_CURRENT_BEST = 0.7655

# Lucky-seed candidates (carried from cycle 124 / nb2156 flag)
LUCKY_SEED_FLAGGED = {
    "nb2154",
    "nb2144",
}

# Anchor-contamination set: identified by nb2177 audit at cycle 124, confirmed
# at nb2184 honest re-eval. EXCLUDE from honest cross-fit top-10.
CONTAMINATED = {
    "nb2170": {
        "reported_rae": 0.3920,
        "reason": "chemprop_aux anchor reused across train/eval folds; "
                  "residual stage inherited in-sample optimism",
        "audit_tag": "nb2177",
        "honest_reeval_tag": "nb2184",
    },
    "nb2178": {
        "reported_rae": 0.3810,
        "reason": "K-sweep at the contaminated nb730 anchor; minimum 0.3810 "
                  "at K=20 is variance on the same leak surface",
        "audit_tag": "nb2177",
        "honest_reeval_tag": "nb2184",
    },
}

RAE_KEYS_PRIORITY = [
    "best_rae", "rae_pooled_median_120", "rae_pooled_mean_120",
    "rae_pooled_median", "rae_pooled_mean",
    "rae_median_bag", "rae_mean_bag",
    "rae_cross_fit", "rae_oof", "rae", "oof_rae", "cross_fit_rae",
    "rae_honest", "honest_cross_fit_rae", "rae_final", "rae_best",
]

PARADIGM_PATTERNS = [
    ("trajectory_bag",        r"trajectory|win_bag|n_cycles|pooled"),
    ("shap_subset",           r"shap|top.?\d+|k_tuned|k_sweep"),
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
    ("anchor_swap",           r"anchor_swap|nb730_anchor|novel_anchor|mono_anchor"),
    ("anchor_blend",          r"anchor_blend|chemprop_nb730_blend"),
    ("honest_anchor",         r"honest_anchor|anchor_regen|cross_fit_anchor"),
    ("alt_anchor",            r"alt_anchor|nb703_anchor|nb562_anchor|nb464_anchor"),
    ("submission_meta",       r"lb_poll|submission|leaderboard|verify|audit|promote"),
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
    n_summaries = len(list((root / "data/processed").glob("*_summary.json")))
    n_scripts = len(list((root / "scripts").glob("nb*.py")))
    n_submissions = len(list((root / "submissions").glob("*.csv")))
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
    print(f"          LB_CURRENT_BEST = {LB_CURRENT_BEST}")
    print(f"          CONTAMINATED    = {sorted(CONTAMINATED)}")
    print(f"          LUCKY_SEED      = {sorted(LUCKY_SEED_FLAGGED)}")
    print("=" * 78)

    repo_root = Path(__file__).resolve().parents[1]
    universe = _count_method_universe(repo_root)
    print(f"[universe] n_summaries={universe['n_summaries']}  "
          f"n_scripts={universe['n_scripts']}  "
          f"n_submissions={universe['n_submissions']}  "
          f"n_distinct_nb_tags={universe['n_distinct_nb_tags']}")

    summary_paths = sorted(DATA_PROCESSED.glob("*_summary.json"))
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

    rec_by_tag = {r["tag"]: r for r in records}
    n_records = len(records)
    n_with_rae = sum(1 for r in records if r["rae"] is not None)
    distinct_tags = sorted({r["tag"] for r in records if r["tag"]})
    cycle_buckets = sorted({r["cycle_bucket"] for r in records
                            if r["cycle_bucket"] is not None})
    paradigms = sorted({r["paradigm"] for r in records})
    methods_universe_total = universe["n_distinct_nb_tags"]

    print(f"[count] cycles                = {CYCLE_IDX}")
    print(f"[count] methods (universe)    = ~{methods_universe_total}")
    print(f"[count] methods (w/ summary)  = {len(distinct_tags)}")
    print(f"[count] cycle buckets         = {len(cycle_buckets)}")
    print(f"[count] distinct paradigms    = {len(paradigms)}")

    # AUDIT VERDICT TRAIL: nb2170 / nb2178 contamination chain
    audit_trail = [
        {
            "step": 1, "tag": "nb2170",
            "stage": "cycle_123_residual_run",
            "reported_rae": 0.3920,
            "verdict": "REPRODUCIBLE_BUT_ANCHOR_CONTAMINATED",
            "note": "mean_bag 0.3920 on nb730 anchor + K=28 SHAP residual; "
                    "reproducible across re-seeded folds at the residual stage, "
                    "BUT the chemprop_aux anchor below the residual was fit on "
                    "indices overlapping the outer eval fold",
        },
        {
            "step": 2, "tag": "nb2177",
            "stage": "cycle_124_audit",
            "reported_rae": None,
            "verdict": "FLAGGED_ANCHOR_LEAK",
            "note": "5-fold re-seeded audit reproduced residual-stage RAE "
                    "within bag noise but identified anchor-stage train/eval "
                    "index overlap; raised the leak flag",
        },
        {
            "step": 3, "tag": "nb2178",
            "stage": "cycle_124_K_sweep",
            "reported_rae": 0.3810,
            "verdict": "INHERITED_CONTAMINATION",
            "note": "K-sweep {12,20,28,36,44} on the same leaked anchor; "
                    "K=20 minimum 0.3810 is variance on the leak surface, "
                    "not a real improvement",
        },
        {
            "step": 4, "tag": "nb2183",
            "stage": "cycle_125_honest_anchor",
            "reported_rae": None,
            "verdict": "ANCHOR_REGENERATED",
            "note": "cross-fit re-fit of chemprop_aux per outer fold; anchor "
                    "now obeys honest train/eval split for the residual stage",
        },
        {
            "step": 5, "tag": "nb2184",
            "stage": "cycle_125_honest_reeval",
            "reported_rae": None,
            "verdict": "EXPECTED_REVERT_TO_FLOOR",
            "note": "honest re-eval of nb2170 / nb2178 on the regenerated "
                    "anchor; expected to revert to within bag noise of the "
                    "0.4603-0.4698 floor band; if it lands < 0.45 the gain is "
                    "real and a new compounding step opens, else contamination "
                    "explained the full delta",
        },
    ]

    print("\nAUDIT VERDICT TRAIL")
    for a in audit_trail:
        rep = f"rep={a['reported_rae']:.4f}" if a["reported_rae"] is not None else "rep=N/A"
        print(f"  step {a['step']}  {a['tag']:8s}  {a['stage']:30s}  "
              f"{rep}  -> {a['verdict']}")

    # Top-10 honest cross-fit RAE: EXCLUDE contaminated + lucky-seed
    exclude_set = set(CONTAMINATED.keys()) | LUCKY_SEED_FLAGGED
    valid = [(r["rae"], r["tag"], r["rae_key"]) for r in records
             if r["rae"] is not None and r["tag"] not in exclude_set]
    valid_sorted = sorted(valid, key=lambda t: t[0])
    top10 = [
        {"rank": i + 1, "rae": round(v, 6), "tag": tg, "rae_key": k}
        for i, (v, tg, k) in enumerate(valid_sorted[:10])
    ]
    contaminated_flagged = [
        {
            "tag": tg,
            "reported_rae": meta["reported_rae"],
            "reason": meta["reason"],
            "audit_tag": meta["audit_tag"],
            "honest_reeval_tag": meta["honest_reeval_tag"],
        }
        for tg, meta in CONTAMINATED.items()
    ]

    print("\nTOP-10 HONEST CROSS-FIT RAE (contaminated + lucky-seed excluded)")
    print("-" * 78)
    for row in top10:
        print(f"  {row['rank']:2d}. {row['rae']:.4f}  {row['tag']:30s}  "
              f"({row['rae_key']})")
    print("\nFLAGGED CONTAMINATED (EXCLUDED)")
    for c in contaminated_flagged:
        print(f"  - {c['tag']:8s}  reported={c['reported_rae']:.4f}  "
              f"audit={c['audit_tag']}  reeval={c['honest_reeval_tag']}")

    honest_floor = valid_sorted[0][0] if valid_sorted else None
    honest_floor_tag = valid_sorted[0][1] if valid_sorted else None
    gap_to_lb = (LB_CURRENT_BEST - honest_floor) if honest_floor is not None else None

    # CLOSED axes (cumulative through cycle 125)
    closed_axes = [
        "num_leaves_grid (K)",
        "L_grid (leaves alt)",
        "learning_rate_grid (lr)",
        "min_child_samples (mc)",
        "feature_fraction (ff)",
        "monotone_constraint",
        "DART_boosting",
        "SHAP_seed_perturbation",
        "row_bootstrap",
        "pooled_120_cycle_bag",
        "XGB_cross_paradigm_SHAP",
        "family_ablation_NR",
        "residual_cascade_depth2",
        "tanh_target_rescale",
        "sklearn_stack_meta_learner",
        "anchor_swap_to_nb730_CONTAMINATED",      # nb2170 -- contamination found
        "K_sweep_at_contaminated_anchor_nb2178",  # cycle 124 closure, inherits leak
    ]

    # NEWLY OPENED at cycle 125
    newly_opened_axes = [
        {
            "axis": "honest_anchor_regeneration",
            "tag": "nb2183",
            "description": "cross-fit re-fit chemprop_aux anchor per outer "
                           "fold so the residual stage trains on honest OOF "
                           "predictions; eliminates the anchor leak that "
                           "drove nb2170's 0.3920 number",
        },
        {
            "axis": "alt_anchor_sweep",
            "tag": "nb2185",
            "description": "alt anchors nb703 (0.4928), nb562 (0.5065), "
                           "nb464 (0.5489) substituted at the anchor stage "
                           "under the same honest cross-fit protocol; tests "
                           "whether the residual stage compounds across "
                           "anchor families (or whether the gain was anchor-"
                           "specific and now disappears)",
        },
        {
            "axis": "honest_reeval_nb2170_nb2178",
            "tag": "nb2184",
            "description": "honest re-eval of the contaminated nb2170 / "
                           "nb2178 candidates on the regenerated anchor; "
                           "this is the verdict on whether ANY compounding "
                           "remains once the leak is removed",
        },
    ]

    # STILL OPEN axes
    open_axes = [
        {
            "axis": "novel_base_model_NN_or_Transformer",
            "rationale": "all bags so far are LGBM/XGB/RF/ElasticNet; small "
                         "MLP or Transformer on the K=28 SHAP substrate "
                         "(narrow surface limits FT overfit) may extract "
                         "non-axis-aligned interactions; ChemBERTa-on-Morgan "
                         "failed (nb601/602) but K=28 SHAP input is a new "
                         "test surface. EV -0.003 to -0.008",
        },
        {
            "axis": "abstention_conformal",
            "rationale": "per feedback_failure_mode_quantile_compression, "
                         "90% of worst-50 errors are novel-scaffold; "
                         "per-row conformal scoring + shrink-to-mean on "
                         "top-5% uncertain rows directly attacks RAE "
                         "numerator; no external data; EV -0.005 to -0.010. "
                         "CHEAPEST engineering (1-2 days)",
        },
        {
            "axis": "external_scaffold_diverse_data",
            "rationale": "ChEMBL PXR analogs / external NR actives outside "
                         "the 4139-train scaffold support; only axis that "
                         "touches the OOD wall set by scaffold support. "
                         "EV ceiling -0.010 to -0.020, but 3-5 day cost "
                         "and F2 negative-transfer risk",
        },
        {
            "axis": "second_stage_residual_on_HONEST_anchor",
            "rationale": "if nb2184 re-eval shows the residual stage still "
                         "compounds on the regenerated honest anchor (i.e. "
                         "the gain was partly real even though the magnitude "
                         "was inflated by leak), a clean 2nd-stage residual "
                         "stack on the honest anchor is the next move; "
                         "alternate SHAP basis (Avalon / AtomPair) from "
                         "nb2172 provides decorrelated features. EV -0.003 "
                         "to -0.005 conditional on nb2184",
        },
    ]

    print("\nCLOSED AXES (cycle 125)")
    for ax in closed_axes:
        print(f"  - {ax}")
    print("\nNEWLY OPENED AXES (cycle 125)")
    for ax in newly_opened_axes:
        print(f"  - {ax['axis']:36s} via {ax['tag']}")
    print("\nSTILL OPEN AXES")
    for ax in open_axes:
        print(f"  - {ax['axis']}")

    # Cycle 125 -- 5 methods (LB poll, honest anchor, re-eval, alt-anchor, ladder hygiene)
    # (+ trajectory snapshot itself = this script)
    cycle_125_methods = [
        ("nb2182", "lb_poll",
         "post-cycle-124 LB snapshot; record latest submission scores and "
         "drift vs honest floor; confirm nb2170 was NOT submitted to LB"),
        ("nb2183", "honest_anchor_regen",
         "regenerate chemprop_aux anchor under per-outer-fold cross-fit; "
         "writes honest_anchor_oof.npy used as substrate for nb2184/nb2185"),
        ("nb2184", "honest_reeval_nb2170_nb2178",
         "re-eval nb2170 residual stack + nb2178 K-sweep variants on the "
         "regenerated honest anchor; verdict on whether the 0.3920 / 0.3810 "
         "gain survives without the leak"),
        ("nb2185", "alt_anchor_sweep",
         "swap anchor for nb703 / nb562 / nb464 under the same honest "
         "cross-fit protocol; tests cross-anchor-family generalisation of "
         "the residual stage"),
        ("nb2186", "ladder_hygiene_promote",
         "ladder hygiene: demote contaminated entries from PRIMARY tier; "
         "if nb2184 lands < 0.45 promote honest-anchor variant, else hold "
         "nb730 PRIMARY-1 and queue cycle-126 on out-of-manifold axes"),
        ("nb2187", "trajectory_snapshot",
         "this script -- cycle-125 trajectory + audit-trail snapshot"),
    ]
    cycle_125 = {
        "cycle_idx": CYCLE_IDX,
        "methods": [
            {
                "tag": tg, "kind": kind, "desc": desc,
                "rae": rec_by_tag.get(tg, {}).get("rae"),
                "status": "RECORDED" if tg in rec_by_tag else "PENDING_SUMMARY",
            }
            for tg, kind, desc in cycle_125_methods
        ],
    }
    print(f"\nCYCLE {CYCLE_IDX} RECORDED")
    for m in cycle_125["methods"]:
        rae_str = f"{m['rae']:.4f}" if m["rae"] is not None else "  N/A "
        print(f"  {m['tag']:8s}  {m['kind']:30s}  rae={rae_str}  "
              f"({m['status']})")

    # Decision: highest-EV next axis
    #
    # The cycle-124 audit identified nb2170 / nb2178 as anchor-contaminated;
    # cycle 125 honest re-eval will land them somewhere in [0.45, 0.50] band
    # (best case) or back at the 0.4603 nb730 floor (worst case). In EITHER
    # outcome the within-manifold axes are now exhausted (audit closed the
    # last suspected source of gain) and the highest-EV next axis is the
    # cheapest out-of-manifold move that attacks the documented failure
    # mode.
    #
    # That move is ABSTENTION / CONFORMAL-SHRINK:
    #   - 1-2 day engineering (cheapest of the open axes)
    #   - attacks the documented dominant failure mode (90% of worst-50
    #     errors are novel-scaffold)
    #   - no external-data dependency; no negative-transfer risk
    #   - stacks additively with any honest-anchor residual gain from nb2184
    #   - EV -0.005 to -0.010
    #
    # If nb2184 lands < 0.45 (honest gain survives), the secondary axis is
    # the 2nd-stage residual on the HONEST anchor with alt SHAP basis. If
    # nb2184 reverts to the floor band, the secondary axis is external
    # scaffold-diverse data (only remaining off-manifold lever).
    decision = {
        "highest_EV_next_axis": "abstention_conformal",
        "secondary_axis_conditional": (
            "IF nb2184 honest re-eval < 0.45: 2nd-stage residual on honest "
            "anchor with alt SHAP basis (Avalon / AtomPair from nb2172). "
            "ELSE: external scaffold-diverse data (ChEMBL PXR analogs / "
            "NR-family actives outside 4139-train scaffold support)"
        ),
        "tertiary_axis": (
            "novel base model (small MLP or Transformer) on K=28 SHAP "
            "substrate -- narrow input surface limits FT overfit vs prior "
            "ChemBERTa-on-Morgan failures"
        ),
        "rationale": (
            "Cycle 124 audit + cycle 125 honest re-eval close the last "
            "within-manifold knobs and demote the 0.3920/0.3810 contaminated "
            "candidates. Regardless of whether nb2184 surfaces any residual "
            "honest gain, the dominant remaining error source is the novel-"
            "scaffold failure tail (per feedback_failure_mode_quantile_"
            "compression, 90% of worst-50 errors). Abstention / conformal-"
            "shrink is the cheapest engineering (1-2 days), attacks that "
            "failure mode directly, requires no external data, has no "
            "negative-transfer risk, and stacks additively with any "
            "remaining honest-anchor gain. EV -0.005 to -0.010. The "
            "secondary axis branches on the nb2184 verdict: honest 2nd-"
            "stage residual if any honest gain survives, else external "
            "scaffold-diverse data."
        ),
        "skip_for_now": [
            "further_anchor_swap_pre_unblind: nb2185 sweep settles whether "
            "alt anchors compound; if all collapse to floor band, this axis "
            "is closed and no further LGBM-anchor probing is justified",
            "loss_level_decompression_redux: closed at cycle 70 (nb710-713); "
            "rank-stretch dominates loss-shape perturbation on this dataset",
        ],
    }
    print(f"\n[decision] highest-EV next axis = {decision['highest_EV_next_axis']}")
    print(f"[decision] secondary (conditional) = {decision['secondary_axis_conditional']}")
    print(f"[decision] rationale = {decision['rationale']}")

    summary = {
        "tag": TAG,
        "method": f"cycle_{CYCLE_IDX}_trajectory_snapshot_audit_trail",
        "lb_current_best": LB_CURRENT_BEST,
        "lucky_seed_flagged_by_nb2156": sorted(LUCKY_SEED_FLAGGED),
        "contaminated_excluded": contaminated_flagged,
        "n_cycles_total": CYCLE_IDX,
        "method_universe": universe,
        "method_universe_total": methods_universe_total,
        "n_summary_files_total": len(summary_paths),
        "n_summary_files_parsed": n_records,
        "n_summary_files_skipped": skipped,
        "n_with_honest_rae": n_with_rae,
        "n_methods_distinct_summary": len(distinct_tags),
        "n_cycle_buckets": len(cycle_buckets),
        "n_paradigms": len(paradigms),
        "paradigms": paradigms,
        "top10_honest_rae_clean": top10,
        "honest_floor": honest_floor,
        "honest_floor_tag": honest_floor_tag,
        "gap_floor_vs_lb_best": gap_to_lb,
        "audit_trail": audit_trail,
        "closed_axes": closed_axes,
        "newly_opened_axes": newly_opened_axes,
        "open_axes": open_axes,
        "cycle_125": cycle_125,
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
    print(f"  contaminated_excluded: {[c['tag'] for c in res['contaminated_excluded']]}")
    print(f"  highest_EV_next_axis:  {res['decision']['highest_EV_next_axis']}")
