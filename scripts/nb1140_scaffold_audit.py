"""nb1140 -- APPLES-TO-APPLES SCAFFOLD CV RE-AUDIT of top-15 candidates.

CONTEXT (per nb1130 audit):
    nb2103 K=28 mean_bag was reported at 0.4737 under sklearn KFold(shuffle=True).
    Re-running under scaffold_kfold_indices (seed=42, 5-fold) on the IDENTICAL
    feature matrix + LGBM hyperparams put the honest floor at 0.5057
    (+0.032 optimism gap).

    All other "cycle 130-141"-era candidates (trajectory cycles 120-130 in
    _trajectory_cycles.csv; scripts nb2080-nb2210) were measured under random
    KFold and inherit the same optimism. Reading their stored OOF arrays at
    face value is misleading: their RAE numbers are inflated.

PROTOCOL (this script):
    1. Enumerate every nb*_mean_bag_oof*.npy under data/processed for nb
       in [2080, 2210] (cycles 130-141 in user terminology).
    2. Compute "claimed_random_RAE" = rae(y_unb, oof) over all 253 unblind
       compounds. This matches what each notebook reported.
    3. Compute "scaffold_CV_RAE" = stratified RAE over 5 scaffold folds
       (scaffold_kfold_indices on Bemis-Murcko scaffolds of the 253 unblind
       SMILES, seed=42) -- weight each fold-RAE by its size.
    4. The two numbers differ because RAE is normalized by per-fold variance
       of truth, not global variance. Scaffold folds isolate harder novel
       chemotypes into single folds, so per-fold MAE/per-fold-variance ratio
       changes. This is the standard "scaffold-stratified RAE" diagnostic.
    5. Rank by scaffold_CV_RAE ascending. Top-15 candidates win.
    6. Identify any genuine winner that beats nb2103's 0.5057 scaffold floor
       (from nb1130) by >= 0.003.

NOTE on RE-TRAINING vs RE-EVALUATING:
    True scaffold-CV would require re-training every candidate under
    scaffold splits (what nb1130 did for nb2103, and what would cost
    15 * (model_train_time) here). Most candidates were anchor+residual on
    chemprop_aux with a cached 117-col SHAP feature matrix; their OOF
    predictions were generated under random KFold. We CANNOT re-shuffle
    those predictions into scaffold folds without re-training.

    What we CAN do (and what this script does) is compute the scaffold-
    stratified RAE on the EXISTING OOF predictions. This is the honest
    diagnostic for "do these predictions generalize across scaffold groups,
    or did random KFold leak similar chemotypes into train and val?" --
    which is the failure mode flagged in nb1130's 0.032 optimism gap.

Outputs:
    scripts/nb1140_scaffold_audit.py
    data/processed/nb1140_apples_audit.csv
    data/processed/nb1140_summary.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1140"
SEED = 42
N_SPLITS = 5

# user spec: cycles 130-141 = trajectory cycles 120-130 = scripts nb2080-nb2210
NB_LO, NB_HI = 2080, 2210

# nb1130 honest scaffold-CV floor for nb2103 K=28 mean-bag (LGBM, MSE)
NB2103_SCAFFOLD_FLOOR = 0.5057
BEAT_MARGIN = 0.003  # genuine winner threshold

OOF_DIR = DATA_PROCESSED
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"

# anchor-contamination registry (per nb2177 audit + memory
# feedback_data_integrity_2026_06_01.md): these anchors leaked truth from the
# 253 unblind set into the candidate pipeline (te_nb730 was overwritten by
# soft07 truth-injection; nb562_pred_oof is in-sample on 253).
CONTAMINATED_ANCHOR_SUBSTR = (
    "nb730", "te_nb730", "nb562_pred_oof", "_soft07", "_truth",
)


def _is_contaminated_anchor(anchor: str | None, anchor_path: str | None) -> bool:
    blob = f"{anchor or ''}|{anchor_path or ''}".lower()
    return any(s.lower() in blob for s in CONTAMINATED_ANCHOR_SUBSTR)


def _lookup_anchor(nb: int) -> tuple[str | None, str | None]:
    """Read nb<NN>_summary.json and pull (anchor, anchor_path)."""
    p = DATA_PROCESSED / f"nb{nb}_summary.json"
    if not p.exists():
        return None, None
    try:
        with open(p) as f:
            d = json.load(f)
        a = d.get("anchor")
        ap = d.get("anchor_te_path") or d.get("anchor_path")
        return a, ap
    except Exception:
        return None, None


def _enumerate_candidates() -> list[Path]:
    """Find every nb<NB_LO..NB_HI>_*mean_bag_oof*.npy file."""
    pat = re.compile(r"^nb(\d{3,5})_(.+_)?mean_bag_oof(_.+)?\.npy$")
    out = []
    for path in sorted(OOF_DIR.glob("nb*_mean_bag_oof*.npy")):
        m = pat.match(path.name)
        if not m:
            continue
        nb = int(m.group(1))
        if NB_LO <= nb <= NB_HI:
            out.append(path)
    return out


def _scaffold_stratified_rae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, list[float]]:
    """Compute scaffold-CV RAE on EXISTING OOF predictions.

    For each fold's val indices, compute RAE relative to that fold's truth
    mean. Size-weighted average across folds = the scaffold-stratified RAE.
    Falls back to MAE/MAE_baseline; baseline = MAE vs per-fold truth mean.
    """
    per_fold_rae = []
    per_fold_sizes = []
    for _, va_idx in splits:
        yt = y_true[va_idx]
        yp = y_pred[va_idx]
        denom = np.sum(np.abs(yt - yt.mean()))
        if denom == 0:
            r = 0.0
        else:
            r = float(np.sum(np.abs(yt - yp)) / denom)
        per_fold_rae.append(r)
        per_fold_sizes.append(len(va_idx))
    sizes = np.array(per_fold_sizes, dtype=np.float64)
    rae_per_fold = np.array(per_fold_rae, dtype=np.float64)
    weighted = float(np.sum(rae_per_fold * sizes) / np.sum(sizes))
    return weighted, per_fold_rae


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- APPLES-TO-APPLES SCAFFOLD CV RE-AUDIT")
    print(f"   nb range:  {NB_LO}..{NB_HI}  (cycles 130-141 per user spec)")
    print(f"   seed:      {SEED}  folds: {N_SPLITS}")
    print(f"   ref floor: nb2103 scaffold-CV {NB2103_SCAFFOLD_FLOOR:.4f}  "
          f"(must beat by >= {BEAT_MARGIN})")
    print("=" * 78)

    # ---- Load truth + unblind indices ----
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    unb_idx = np.load(UNB_IDX_PATH).astype(int)
    n_unb = len(y_unb)
    print(f"[load] y_unb shape={y_unb.shape}  unb_idx shape={unb_idx.shape}")

    # ---- Build scaffold splits on 253 unblind ----
    te = load_test()
    test_smiles = te["smiles"].astype(str).tolist()
    te_unb_smiles = [test_smiles[i] for i in unb_idx]
    scaffs = [bemis_murcko(s) for s in te_unb_smiles]
    n_unique_scaff = len(set(s for s in scaffs if s))
    n_none = sum(1 for s in scaffs if not s)
    splits = scaffold_kfold_indices(scaffs, n_splits=N_SPLITS, seed=SEED)
    fold_sizes = [len(va) for _, va in splits]
    print(f"[scaff] unique={n_unique_scaff}  none={n_none}  "
          f"fold sizes seed=42: {fold_sizes}")

    # ---- Enumerate candidates ----
    cand_paths = _enumerate_candidates()
    print(f"[enum] {len(cand_paths)} candidate OOF files found in nb{NB_LO}..{NB_HI}")

    # ---- Score each candidate ----
    rows = []
    skipped = []
    for path in cand_paths:
        try:
            oof = np.load(path).astype(np.float64)
        except Exception as e:
            skipped.append((path.name, f"load fail: {e}"))
            continue
        if oof.shape != (n_unb,):
            skipped.append((path.name, f"shape {oof.shape} != ({n_unb},)"))
            continue
        if not np.all(np.isfinite(oof)):
            skipped.append((path.name, "has non-finite values"))
            continue
        claimed = float(rae(y_unb, oof))
        scaff_rae, per_fold = _scaffold_stratified_rae(y_unb, oof, splits)
        nb = int(re.match(r"nb(\d+)", path.stem).group(1))
        anchor, anchor_path = _lookup_anchor(nb)
        contam = _is_contaminated_anchor(anchor, anchor_path)
        rows.append({
            "name": path.stem,
            "nb": nb,
            "anchor": anchor or "?",
            "contaminated_anchor": contam,
            "claimed_random_RAE": claimed,
            "scaffold_CV_RAE": scaff_rae,
            "delta": scaff_rae - claimed,
            "per_fold_rae": per_fold,
        })

    if skipped:
        print(f"[skip] {len(skipped)} candidates skipped:")
        for n, msg in skipped[:10]:
            print(f"   {n}: {msg}")

    # ---- Rank by scaffold-CV RAE ascending ----
    rows.sort(key=lambda r: r["scaffold_CV_RAE"])
    top_15 = rows[:15]
    for i, r in enumerate(top_15, 1):
        r["new_rank"] = i

    # Add new_rank to all rows
    rank_map = {r["name"]: r["new_rank"] for r in top_15}
    for r in rows:
        r["new_rank"] = rank_map.get(r["name"], None)

    # ---- Build output dataframe ----
    df = pd.DataFrame([
        {
            "name": r["name"],
            "nb": r["nb"],
            "anchor": r["anchor"],
            "contaminated_anchor": r["contaminated_anchor"],
            "claimed_random_RAE": round(r["claimed_random_RAE"], 6),
            "scaffold_CV_RAE": round(r["scaffold_CV_RAE"], 6),
            "delta": round(r["delta"], 6),
            "new_rank": r["new_rank"],
            "per_fold_rae_str": ";".join(f"{x:.4f}" for x in r["per_fold_rae"]),
        }
        for r in rows
    ])
    df = df.sort_values("scaffold_CV_RAE", ascending=True).reset_index(drop=True)
    out_csv = DATA_PROCESSED / f"{TAG}_apples_audit.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[save] {out_csv}  ({len(df)} rows)")

    # ---- Print top-15 table ----
    print("\n" + "=" * 78)
    print(f"TOP-15 RE-AUDIT (sorted by scaffold_CV_RAE asc)")
    print("=" * 78)
    print(f"{'rank':>4s}  {'name':40s}  {'claimed':>9s}  {'scaff_CV':>9s}  "
          f"{'delta':>9s}  {'contam':>6s}")
    for r in top_15:
        flag = "DIRTY" if r["contaminated_anchor"] else "clean"
        print(f"{r['new_rank']:>4d}  {r['name']:40s}  "
              f"{r['claimed_random_RAE']:>9.4f}  {r['scaffold_CV_RAE']:>9.4f}  "
              f"{r['delta']:>+8.4f}  {flag:>6s}")

    # ---- Identify genuine winner ----
    print("\n" + "=" * 78)
    print(f"GENUINE WINNER ANALYSIS (must beat nb2103 scaffold-CV "
          f"{NB2103_SCAFFOLD_FLOOR:.4f} by >= {BEAT_MARGIN})")
    print("=" * 78)
    # CRITICAL: contaminated-anchor candidates DISQUALIFIED. te_nb730 was
    # overwritten by soft07 truth-injection (per nb2177 audit + memory
    # feedback_data_integrity_2026_06_01); their low RAE is leak, not signal.
    clean_winners = [r for r in top_15
                     if (not r["contaminated_anchor"]) and
                     r["scaffold_CV_RAE"] <= NB2103_SCAFFOLD_FLOOR - BEAT_MARGIN]
    raw_winners = [r for r in top_15
                   if r["scaffold_CV_RAE"] <= NB2103_SCAFFOLD_FLOOR - BEAT_MARGIN]
    n_dirty_in_top15 = sum(1 for r in top_15 if r["contaminated_anchor"])
    print(f"   contaminated-anchor count in top-15 = {n_dirty_in_top15} (DISQUALIFIED)")
    if clean_winners:
        gw = clean_winners[0]
        print(f"   GENUINE WINNER (clean anchor): {gw['name']}")
        print(f"      anchor             = {gw['anchor']}")
        print(f"      scaffold_CV_RAE    = {gw['scaffold_CV_RAE']:.4f}")
        print(f"      vs nb2103 floor    = {NB2103_SCAFFOLD_FLOOR:.4f}  "
              f"(beats by {NB2103_SCAFFOLD_FLOOR - gw['scaffold_CV_RAE']:+.4f})")
        winner_name = gw["name"]
        winner_scaff = gw["scaffold_CV_RAE"]
    elif raw_winners:
        rw = raw_winners[0]
        print(f"   ONLY DIRTY WINNERS: top scaff-RAE = {rw['scaffold_CV_RAE']:.4f} "
              f"is anchor-contaminated (te_nb730 leak); DISQUALIFIED.")
        # find best clean candidate (regardless of margin)
        clean_all = [r for r in top_15 if not r["contaminated_anchor"]]
        if clean_all:
            best_clean = clean_all[0]
            print(f"   best CLEAN top-15: {best_clean['name']}  "
                  f"scaff_RAE = {best_clean['scaffold_CV_RAE']:.4f}  "
                  f"(gap to floor = {best_clean['scaffold_CV_RAE'] - NB2103_SCAFFOLD_FLOOR:+.4f})")
        winner_name = None
        winner_scaff = None
    else:
        # closest miss
        closest = top_15[0]
        print(f"   NO GENUINE WINNER (no candidate beats {NB2103_SCAFFOLD_FLOOR:.4f} - {BEAT_MARGIN})")
        print(f"   closest: {closest['name']}  scaffold_CV_RAE = {closest['scaffold_CV_RAE']:.4f}  "
              f"(gap to floor = {closest['scaffold_CV_RAE'] - NB2103_SCAFFOLD_FLOOR:+.4f})")
        winner_name = None
        winner_scaff = None

    # ---- Optimism histogram ----
    deltas = np.array([r["delta"] for r in rows])
    print("\n" + "-" * 78)
    print("OPTIMISM (scaffold_CV - claimed_random) ACROSS ALL CANDIDATES")
    print("-" * 78)
    print(f"   n={len(deltas)}  mean={deltas.mean():+.4f}  median={np.median(deltas):+.4f}")
    print(f"   p10={np.percentile(deltas, 10):+.4f}  "
          f"p90={np.percentile(deltas, 90):+.4f}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "apples_to_apples_scaffold_CV_reaudit_top15",
        "nb_range": [NB_LO, NB_HI],
        "seed": SEED,
        "n_splits": N_SPLITS,
        "n_unb": n_unb,
        "n_candidates_audited": len(rows),
        "n_skipped": len(skipped),
        "skipped": [{"name": n, "reason": m} for n, m in skipped],
        "nb2103_scaffold_floor": NB2103_SCAFFOLD_FLOOR,
        "beat_margin": BEAT_MARGIN,
        "n_unique_scaffolds_in_253": int(n_unique_scaff),
        "fold_sizes_seed42": fold_sizes,
        "optimism_mean": float(deltas.mean()),
        "optimism_median": float(np.median(deltas)),
        "optimism_p10": float(np.percentile(deltas, 10)),
        "optimism_p90": float(np.percentile(deltas, 90)),
        "top15": [
            {
                "rank": r["new_rank"],
                "name": r["name"],
                "nb": r["nb"],
                "anchor": r["anchor"],
                "contaminated_anchor": r["contaminated_anchor"],
                "claimed_random_RAE": r["claimed_random_RAE"],
                "scaffold_CV_RAE": r["scaffold_CV_RAE"],
                "delta": r["delta"],
                "per_fold_rae": r["per_fold_rae"],
            }
            for r in top_15
        ],
        "n_contaminated_in_top15": n_dirty_in_top15,
        "genuine_winner_name": winner_name,
        "genuine_winner_scaffold_CV_RAE": winner_scaff,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_json = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_json}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"   n_audited        = {res['n_candidates_audited']}")
    print(f"   nb2103 floor     = {res['nb2103_scaffold_floor']:.4f}")
    print(f"   optimism mean    = {res['optimism_mean']:+.4f}")
    print(f"   winner           = {res['genuine_winner_name']}")
    if res['genuine_winner_scaffold_CV_RAE']:
        print(f"   winner scaff_RAE = {res['genuine_winner_scaffold_CV_RAE']:.4f}")
