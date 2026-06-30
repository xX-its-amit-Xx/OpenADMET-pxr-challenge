"""nb1564 — Compile final LB candidates ranking.

For each PRE-unblind deploy CSV / te_*.npy referenced by nb14XX / nb15XX summaries:
  - honest_RAE: cross-fit / BoB-pooled / per-seed-mean (PRE-unblind anchor)
  - in_RAE on 253 unblind: te[unb_idx] vs y_true
  - predicted_LB = honest_RAE + 0.003 (PRE-unblind calibration)
  - confidence: BoB-validated / multi-seed / single-seed / sub-margin

Output:
  data/processed/final_lb_candidates.json
  data/processed/final_lb_report.md
"""
from __future__ import annotations
import json
import os
import glob
from pathlib import Path

import numpy as np

PROC = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/processed")
SUBS = Path(r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/submissions")
OUT_JSON = PROC / "final_lb_candidates.json"
OUT_MD = PROC / "final_lb_report.md"

MARGIN = 0.003  # PRE-unblind calibration shift

# ------------------------------------------------------------------------
# 1. Load 253-unblind index + truth
# ------------------------------------------------------------------------
unb_idx = np.load(PROC / "_audit_unblind_idx.npy")
y_unb = np.load(PROC / "_audit_unblind_y.npy")
assert unb_idx.shape == y_unb.shape == (253,), f"unblind audit shapes wrong: {unb_idx.shape}, {y_unb.shape}"

def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = float(np.abs(y_true - y_pred).sum())
    den = float(np.abs(y_true - y_true.mean()).sum())
    if den == 0.0:
        return float("nan")
    return num / den

def in_rae_from_te(te_path: Path) -> float | None:
    """Slice te[unb_idx] and compute RAE against y_unb."""
    try:
        te = np.load(te_path)
    except Exception:
        return None
    if te.shape[0] != 513:
        return None
    pred = te[unb_idx]
    return rae(y_unb, pred)

# ------------------------------------------------------------------------
# 2. Honest-RAE extraction priority list
# ------------------------------------------------------------------------
HONEST_KEYS_PRIORITY = [
    # Cross-fit / pooled (most honest)
    "honest_crossfit_RAE_nb1484_slsqp",
    "honest_crossfit_RAE_nb1484_naive",
    "honest_crossfit_RAE_nb1472",
    "honest_crossfit_RAE_nb1471",
    "honest_crossfit_RAE_nb1460",
    "slsqp_crossfit_rae",
    "slsqp_cross_fit_rae",
    "rae_slsqp_crossfit",
    "rae_slsqp_cross_fit",
    "rae_cross_fit_slsqp",
    "crossfit_rae",
    "rae_cross_fit",
    "rae_grid_crossfit",
    # BoB-pooled per-outer-seed
    "rae_bob_mean",
    "rae_bob_median",
    "rae_bob_nb1484_mean",
    "rae_bob_nb1501_mean",
    "bob_mean_pooled_rae",
    "bob_median_pooled_rae",
    "best_bob_rae",
    "per_outer_mean_rae",
    "per_outer_rae_mean",
    # mean-bag / per-seed mean (single-seed grade)
    "rae_mean_bag",
    "best_mean_bag_rae",
    "rae_per_seed_mean",
    "rae_per_seed_median",
    "best_rae_mean_bag",
    "best_rae_median_bag",
    # honest_lb_anchor (already computed)
    "honest_lb_anchor",
    "honest_lb_anchor_mean",
    "honest_lb_anchor_median",
    # last resort
    "rae_best",
    "best_rae",
    "pick_rae",
]

CSV_PATH_KEYS = [
    "csv_path",
    "csv_path_mean",
    "csv_path_median",
    "csv_mean_path",
    "csv_median_path",
    "mean_csv_path",
    "median_csv_path",
]

TE_PATH_KEYS = [
    "te_path",
    "te_npy_path",
    "te_path_mean",
    "te_path_median",
    "te_mean_path",
    "te_mean_npy_path",
    "te_median_path",
    "te_median_npy_path",
    "te_nb1450_path",
    "te_nb1441_path",
    "te_nb1462_path",
    "te_nb1512_bob_mean_path",
    "te_nb1533_path",
    "chemprop_te_path",
]

def first_existing(d: dict, keys: list) -> tuple[str | None, float | None]:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v)):
            return k, float(v)
        if isinstance(v, str):
            return k, v
    return None, None

def first_existing_path(d: dict, keys: list) -> tuple[str | None, str | None]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and len(v) > 0:
            p = Path(v)
            if p.exists():
                return k, str(p)
    return None, None

def glob_deploy_for_tag(tag: str) -> tuple[str | None, str | None]:
    """Look for canonical deploy te_<tag>.npy + <tag>_*.csv pair."""
    te_candidates = [
        PROC / f"te_{tag}.npy",
        PROC / f"te_{tag}_mean.npy",
        PROC / f"te_{tag}_blend.npy",
        PROC / f"te_{tag}_deploy.npy",
    ]
    te_path = None
    for p in te_candidates:
        if p.exists():
            arr = np.load(p)
            if arr.shape == (513,):
                te_path = str(p)
                break
    csv_path = None
    for p in sorted(SUBS.glob(f"{tag}_*.csv")):
        csv_path = str(p)
        break
    return te_path, csv_path

def glob_oof_for_tag(tag: str) -> str | None:
    """Look for *_oof.npy (size 253) honest cross-fit prediction on unblind."""
    oof_candidates = sorted(PROC.glob(f"{tag}_*oof*.npy")) + sorted(PROC.glob(f"{tag}_*bob*oof*.npy"))
    for p in oof_candidates:
        try:
            arr = np.load(p)
        except Exception:
            continue
        if arr.shape == (253,):
            return str(p)
    return None

def rae_from_oof_path(p: str | None) -> float | None:
    if p is None:
        return None
    try:
        a = np.load(p)
    except Exception:
        return None
    if a.shape != (253,):
        return None
    return rae(y_unb, a)

# ------------------------------------------------------------------------
# 3. Confidence labels
# ------------------------------------------------------------------------
def confidence_label(summary: dict, honest_key: str | None) -> str:
    if honest_key and "bob" in honest_key.lower():
        return "BoB-validated"
    if honest_key and ("crossfit" in honest_key.lower() or "cross_fit" in honest_key.lower()):
        return "cross-fit"
    if honest_key and "per_outer" in honest_key.lower():
        return "BoB-validated"
    if honest_key and "mean_bag" in honest_key.lower():
        n_seeds = len(summary.get("per_seed_rae", []) or summary.get("resid_seeds", []) or [])
        if n_seeds >= 5:
            return "multi-seed"
        return "single-seed"
    if honest_key and "per_seed" in honest_key.lower():
        return "multi-seed"
    if honest_key and "honest_lb_anchor" in honest_key.lower():
        return "honest-anchor"
    return "single-seed"

# ------------------------------------------------------------------------
# 4. Walk summaries
# ------------------------------------------------------------------------
summary_files = sorted(
    glob.glob(str(PROC / "nb14*_summary.json")) + glob.glob(str(PROC / "nb15*_summary.json"))
)

records = []
skipped = []
for fp in summary_files:
    tag = Path(fp).stem.replace("_summary", "")
    try:
        with open(fp) as fh:
            d = json.load(fh)
    except Exception as e:
        skipped.append({"tag": tag, "reason": f"json_load_fail: {e}"})
        continue

    honest_key, honest_rae = first_existing(d, HONEST_KEYS_PRIORITY)
    if honest_rae is None or not isinstance(honest_rae, (int, float)):
        skipped.append({"tag": tag, "reason": "no_honest_rae_key"})
        continue
    if honest_rae != honest_rae:  # NaN
        skipped.append({"tag": tag, "reason": "honest_rae_nan"})
        continue

    csv_key, csv_path = first_existing_path(d, CSV_PATH_KEYS)
    te_key, te_path = first_existing_path(d, TE_PATH_KEYS)

    # Fallback: glob conventional te_<tag>.npy + <tag>_*.csv pair
    if te_path is None or csv_path is None:
        g_te, g_csv = glob_deploy_for_tag(tag)
        if te_path is None:
            te_path = g_te
        if csv_path is None:
            csv_path = g_csv

    # If still no 513-deploy te, search for OOF (size 253) for honest-RAE diagnostic
    oof_path = None
    if te_path is None:
        oof_path = glob_oof_for_tag(tag)

    # Resolve in_RAE from te (513 deploy)
    in_rae = None
    if te_path is not None:
        in_rae = in_rae_from_te(Path(te_path))

    # If OOF available, recompute honest RAE directly (sanity)
    oof_rae = rae_from_oof_path(oof_path)

    # Fallback to in-summary value for in_RAE
    if in_rae is None:
        for k in ("in_rae_unb_blend", "in_RAE_unb_blend_in_sample",
                  "in_RAE_unb_deploy_in_sample", "in_RAE_unb_deploy_in_sample_mean",
                  "in_RAE_unb_deploy_in_sample_median", "in_rae_unb_mean", "in_rae_unb_median"):
            v = d.get(k)
            if isinstance(v, (int, float)) and v == v:
                in_rae = float(v)
                break

    deployable = (csv_path is not None) and (te_path is not None)

    predicted_lb = honest_rae + MARGIN
    conf = confidence_label(d, honest_key)
    if not deployable:
        conf = conf + " (OOF-only)"

    records.append({
        "tag": tag,
        "summary_path": fp,
        "csv_path": csv_path,
        "te_path": te_path,
        "oof_path": oof_path,
        "oof_rae_recomputed": oof_rae,
        "honest_rae": honest_rae,
        "honest_rae_key": honest_key,
        "in_rae_253": in_rae,
        "predicted_lb": predicted_lb,
        "confidence": conf,
        "deployable": deployable,
        "margin": MARGIN,
        "n_unb": d.get("n_unb", 253),
        "anchor": d.get("anchor"),
        "model_family": d.get("model_family"),
    })

# ------------------------------------------------------------------------
# 5. Also include direct PRE-unblind candidates from
#    data/processed/pre_unblind_lb_candidates_all.csv (legacy ladder)
# ------------------------------------------------------------------------
import csv as csvmod
legacy_path = PROC / "pre_unblind_lb_candidates_all.csv"
if legacy_path.exists():
    seen_te = {Path(r["te_path"]).name for r in records if r["te_path"]}
    with open(legacy_path, newline="") as fh:
        reader = csvmod.DictReader(fh)
        for row in reader:
            te_stem = row["file"]
            te_path = PROC / f"{te_stem}.npy"
            if not te_path.exists():
                continue
            if te_path.name in seen_te:
                continue
            try:
                in_rae = float(row["in_RAE"])
            except Exception:
                continue
            if in_rae <= 0.001:
                # zero placeholder, skip
                continue
            # Single-fold OOF-only legacy candidate; treat in_RAE as both honest and in
            # but apply +0.003 margin only (these are PRE-unblind nb<320 trained on 4139)
            nb_num = row.get("nb_num", "")
            try:
                n = float(nb_num)
            except Exception:
                n = None
            # Only keep nb < 320 PRE-unblind (per LB two-regime calibration)
            if n is not None and n >= 320:
                continue
            # Guess corresponding submission CSV
            cand_csv = SUBS / f"{te_stem.replace('te_', '')}.csv"
            csv_path = str(cand_csv) if cand_csv.exists() else None
            records.append({
                "tag": te_stem,
                "summary_path": str(legacy_path),
                "csv_path": csv_path,
                "te_path": str(te_path),
                "oof_path": None,
                "oof_rae_recomputed": None,
                "honest_rae": in_rae,
                "honest_rae_key": "legacy_pre_unblind_in_RAE",
                "in_rae_253": in_rae,
                "predicted_lb": in_rae + MARGIN,
                "confidence": "legacy-PRE-unblind",
                "deployable": csv_path is not None,
                "margin": MARGIN,
                "n_unb": 253,
                "anchor": None,
                "model_family": None,
            })

# ------------------------------------------------------------------------
# 6. Rank ascending by predicted_lb
# ------------------------------------------------------------------------
records.sort(key=lambda r: r["predicted_lb"])
for i, r in enumerate(records, start=1):
    r["rank"] = i

# ------------------------------------------------------------------------
# 7. Write JSON
# ------------------------------------------------------------------------
out = {
    "tag": "nb1564",
    "method": "final_lb_candidates_ranking",
    "calibration_margin": MARGIN,
    "calibration_note": "predicted_LB = honest_RAE + 0.003 (PRE-unblind two-regime calibration)",
    "n_candidates": len(records),
    "n_skipped": len(skipped),
    "skipped_examples": skipped[:20],
    "ranking": records,
}
with open(OUT_JSON, "w") as fh:
    json.dump(out, fh, indent=2)

# ------------------------------------------------------------------------
# 8. Write markdown report
# ------------------------------------------------------------------------
top_n = 15
deployable_records = [r for r in records if r["deployable"]]
# re-rank within deployable
for i, r in enumerate(deployable_records, start=1):
    r["rank_deployable"] = i

md_lines = []
md_lines.append("# nb1564 - Final LB Candidates Report")
md_lines.append("")
md_lines.append(f"- Total candidates analyzed: **{len(records)}**")
md_lines.append(f"- Submission-ready (have CSV + te_513): **{len(deployable_records)}**")
md_lines.append(f"- Skipped (no honest_RAE key): {len(skipped)}")
md_lines.append(f"- Calibration margin: **predicted_LB = honest_RAE + {MARGIN}** (PRE-unblind regime)")
md_lines.append("")
md_lines.append("## A. Top-15 SUBMISSION-READY candidates (have deploy CSV)")
md_lines.append("")
md_lines.append("| Rank | Tag | honest_RAE | in_RAE(253) | predicted_LB | Confidence | Submission CSV |")
md_lines.append("|---:|---|---:|---:|---:|---|---|")
for r in deployable_records[:top_n]:
    csv_display = Path(r["csv_path"]).name
    in_rae_str = f"{r['in_rae_253']:.4f}" if r["in_rae_253"] is not None else "-"
    md_lines.append(
        f"| {r['rank_deployable']} | {r['tag']} | {r['honest_rae']:.4f} | {in_rae_str} | {r['predicted_lb']:.4f} | {r['confidence']} | {csv_display} |"
    )
md_lines.append("")
md_lines.append("### Absolute submission paths")
md_lines.append("")
for r in deployable_records[:top_n]:
    md_lines.append(f"- **#{r['rank_deployable']}** `{r['tag']}` -> `{r['csv_path']}` (predicted LB **{r['predicted_lb']:.4f}**)")
md_lines.append("")

md_lines.append("## B. Top-15 ALL candidates (including OOF-only diagnostics)")
md_lines.append("")
md_lines.append("| Rank | Tag | honest_RAE | in_RAE(253) | predicted_LB | Confidence | Deployable |")
md_lines.append("|---:|---|---:|---:|---:|---|---|")
for r in records[:top_n]:
    in_rae_str = f"{r['in_rae_253']:.4f}" if r["in_rae_253"] is not None else "-"
    md_lines.append(
        f"| {r['rank']} | {r['tag']} | {r['honest_rae']:.4f} | {in_rae_str} | {r['predicted_lb']:.4f} | {r['confidence']} | {'yes' if r['deployable'] else 'no'} |"
    )
md_lines.append("")

md_lines.append("## C. Source-key distribution (top-15 deployable)")
md_lines.append("")
from collections import Counter
ck = Counter(r["honest_rae_key"] for r in deployable_records[:top_n])
for k, c in ck.most_common():
    md_lines.append(f"- `{k}`: {c}")
md_lines.append("")
with open(OUT_MD, "w", encoding="utf-8") as fh:
    fh.write("\n".join(md_lines))

# ------------------------------------------------------------------------
# 9. Print top-15 to stdout
# ------------------------------------------------------------------------
print(f"WROTE: {OUT_JSON}")
print(f"WROTE: {OUT_MD}")
print()
print("=" * 140)
print("SUBMISSION-READY TOP-15 (have deploy CSV + te_513)")
print("=" * 140)
print(f"{'rank':>4}  {'tag':<35}  {'honest_RAE':>10}  {'in_RAE_253':>10}  {'pred_LB':>8}  {'confidence':<32}  csv")
print("-" * 160)
for r in deployable_records[:top_n]:
    csv = Path(r["csv_path"]).name
    in_rae_str = f"{r['in_rae_253']:.4f}" if r["in_rae_253"] is not None else "      -"
    print(f"{r['rank_deployable']:>4}  {r['tag']:<35}  {r['honest_rae']:>10.4f}  {in_rae_str:>10}  {r['predicted_lb']:>8.4f}  {r['confidence']:<32}  {csv}")

print()
print("=" * 140)
print("ALL CANDIDATES TOP-15 (includes OOF-only diagnostics)")
print("=" * 140)
print(f"{'rank':>4}  {'tag':<35}  {'honest_RAE':>10}  {'in_RAE_253':>10}  {'pred_LB':>8}  {'confidence':<32}  deployable")
print("-" * 160)
for r in records[:top_n]:
    in_rae_str = f"{r['in_rae_253']:.4f}" if r["in_rae_253"] is not None else "      -"
    print(f"{r['rank']:>4}  {r['tag']:<35}  {r['honest_rae']:>10.4f}  {in_rae_str:>10}  {r['predicted_lb']:>8.4f}  {r['confidence']:<32}  {'yes' if r['deployable'] else 'no'}")
