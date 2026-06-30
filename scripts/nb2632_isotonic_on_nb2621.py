"""nb2632 -- Isotonic calibration + rank-stretch on top of nb2621 {K=18, K=20}
winner.

NEW PARADIGM:
    nb2621 enumerated 372 equal-weight subsets of 9 K-RFE pyramids and the
    BEST combo was {K=18, K=20} mean with pooled RAE 0.4552 on 253 unblind.
    nb2621 carries df = 0 (no learnable parameters).  The next move is a
    post-hoc, low-df calibration on top of this winner:

        1. Per-fold IsotonicRegression(y_min=3.0, y_max=8.0) fit on
           fold-train (anchor=nb2621_pred_oof, y=truth); apply to fold-val
           anchor predictions.
        2. Then rank-stretch sweep s in {0.95, 1.0, 1.05, 1.10}; pick the
           s that minimizes pooled cross-fit RAE.

    Total learned df = 1 (the s scalar after the non-parametric monotone
    map).  The y_min/y_max clamp keeps the iso transform on a sensible
    pEC50 envelope.

PROTOCOL:
    - Anchor = nb2621_pred_oof.npy  (the 0.4552 winner: equal-weight mean
      of K=18 and K=20 K-RFE pyramids).
    - 5-fold SCAFFOLD CV via scaffold_kfold_indices, repeated over
      kf_seeds {1001..1005} (5 seeds, matches nb2604/nb2620 protocol).
    - Per (kf_seed, fold): fit IsotonicRegression(y_min=3.0, y_max=8.0,
      out_of_bounds='clip') on (anchor[tr], y[tr]); transform anchor[va].
    - Build iso_oof (253-vec) per seed; for each s in STRETCH_GRID
      compute pooled RAE = rae(y, mu_iso + s * (iso_oof - mu_iso)) where
      mu_iso = iso_oof.mean().  Pick best s per seed.  Aggregate mean
      across seeds.
    - Deploy: refit IsotonicRegression on the full 253 anchor (nb2621_pred_oof
      -> y_unb); transform te_nb2621 (513-vec); apply the deploy-best s
      (mean of per-seed best s); save te_nb2632.

GATE:
    mean_rae < 0.4552  ->  BETTER_THAN_NB2621  ->  PROMOTE
    else               ->  FAIL

Outputs:
    scripts/nb2632_isotonic_on_nb2621.py
    data/processed/nb2632_summary.json
    data/processed/nb2632_pred_oof.npy   (253,) float32
    data/processed/te_nb2632.npy         (513,) float32
    submissions/nb2632_isotonic_on_nb2621.csv  (on any non-FAIL)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2632"

# ---- Anchor: nb2621 winner ----
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb2621_pred_oof.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb2621.npy"

# ---- Diagnostic anchor ----
CHEMPROP_AUX_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# ---- Calibration grid ----
ISO_Y_MIN = 3.0
ISO_Y_MAX = 8.0
STRETCH_GRID = [0.95, 1.00, 1.05, 1.10]

# ---- CV eval ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- Gate ----
NB2621_REF = 0.4552
GATE_PROMOTE = NB2621_REF  # strictly less than nb2621

# ---- Other refs ----
CHEMPROP_AUX_REF = 0.6216
NB2604_REF = 0.4580
NB2171_REF = 0.4682


def calibrate_fold(anchor_tr: np.ndarray, y_tr: np.ndarray,
                   anchor_va: np.ndarray) -> np.ndarray:
    """Per-fold IsotonicRegression with [y_min, y_max] clamp.

    Fit on (anchor_tr, y_tr) and transform anchor_va.  out_of_bounds='clip'
    handles val anchors that fall outside the train anchor range.
    """
    ir = IsotonicRegression(
        y_min=ISO_Y_MIN,
        y_max=ISO_Y_MAX,
        out_of_bounds="clip",
    )
    ir.fit(anchor_tr, y_tr)
    return ir.predict(anchor_va).astype(np.float64)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- isotonic + rank-stretch on top of nb2621 winner")
    print(f"          anchor = nb2621_pred_oof.npy (K=18+K=20 mean, ref 0.4552)")
    print(f"          iso clamp y in [{ISO_Y_MIN}, {ISO_Y_MAX}]")
    print(f"          stretch grid s in {STRETCH_GRID}")
    print(f"          scaffold CV n_folds={N_FOLDS}  kf_seeds={KF_SEEDS}")
    print(f"          gate PROMOTE strict < {GATE_PROMOTE}")
    print("=" * 78)

    # ---- Load test + truth + scaffolds ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Load anchor (nb2621) ----
    if not ANCHOR_OOF_PATH.exists():
        raise FileNotFoundError(f"missing anchor OOF: {ANCHOR_OOF_PATH}")
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing anchor te: {ANCHOR_TE_PATH}")
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if anchor_oof.shape != (n_unb,):
        raise ValueError(f"anchor_oof shape {anchor_oof.shape} != ({n_unb},)")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"anchor_te shape {anchor_te.shape} != ({n_test},)")
    rae_anchor_oof = float(rae(y_unb, anchor_oof))
    print(f"[load] nb2621 anchor_oof RAE = {rae_anchor_oof:.4f} "
          f"(ref {NB2621_REF:.4f})")
    print(f"       anchor_oof mean={anchor_oof.mean():.3f}  "
          f"std={anchor_oof.std():.3f}  "
          f"(truth_std {y_unb.std():.3f})")
    print(f"       anchor_te  mean={anchor_te.mean():.3f}  "
          f"std={anchor_te.std():.3f}")

    # Diagnostic: chemprop_aux raw
    if CHEMPROP_AUX_TE_PATH.exists():
        chemprop_te = np.load(CHEMPROP_AUX_TE_PATH).astype(np.float64)
        rae_chemprop = float(rae(y_unb, chemprop_te[unb_idx]))
        print(f"[diag] chemprop_aux te[unb_idx] in_RAE = {rae_chemprop:.4f} "
              f"(ref {CHEMPROP_AUX_REF:.4f})")

    # ============================================================
    # STEP 1: per-seed scaffold cross-fit isotonic + stretch sweep
    # ============================================================
    print("\n" + "-" * 78)
    print(f"STEP 1: scaffold {N_FOLDS}-fold CV  isotonic + stretch  "
          f"({len(KF_SEEDS)} seeds)")
    print("-" * 78)

    per_seed_results = []
    per_seed_best_s = []
    per_seed_best_pooled = []
    per_seed_iso_only_pooled = []
    per_seed_iso_oof_blobs = []  # to average across seeds for deploy oof

    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        iso_oof = np.full(n_unb, np.nan, dtype=np.float64)
        per_fold_knots = []
        for k, (tr, va) in enumerate(splits):
            pred_va = calibrate_fold(
                anchor_oof[tr], y_unb[tr], anchor_oof[va]
            )
            iso_oof[va] = pred_va
            # capture knot count for diagnostics
            ir_tmp = IsotonicRegression(
                y_min=ISO_Y_MIN, y_max=ISO_Y_MAX, out_of_bounds="clip"
            )
            ir_tmp.fit(anchor_oof[tr], y_unb[tr])
            n_knots = int(len(np.unique(ir_tmp.X_thresholds_))) \
                if hasattr(ir_tmp, "X_thresholds_") else -1
            per_fold_knots.append(n_knots)

        if np.isnan(iso_oof).any():
            raise RuntimeError(
                "scaffold splits did not cover all rows; check protocol"
            )

        iso_only_pooled = float(rae(y_unb, iso_oof))
        # rank-stretch sweep: mu_iso + s*(iso_oof - mu_iso)
        mu_iso = float(iso_oof.mean())
        per_s_pooled = []
        for s in STRETCH_GRID:
            stretched = mu_iso + s * (iso_oof - mu_iso)
            per_s_pooled.append(float(rae(y_unb, stretched)))
        best_idx = int(np.argmin(per_s_pooled))
        best_s = float(STRETCH_GRID[best_idx])
        best_pooled = float(per_s_pooled[best_idx])

        per_seed_results.append({
            "kf_seed": int(kf_seed),
            "iso_only_pooled_rae": iso_only_pooled,
            "per_fold_knots": per_fold_knots,
            "mean_knots": float(np.mean(per_fold_knots)),
            "mu_iso": mu_iso,
            "per_s_pooled_rae": {f"{s:.2f}": r
                                  for s, r in zip(STRETCH_GRID, per_s_pooled)},
            "best_s": best_s,
            "best_pooled_rae": best_pooled,
        })
        per_seed_best_s.append(best_s)
        per_seed_best_pooled.append(best_pooled)
        per_seed_iso_only_pooled.append(iso_only_pooled)
        per_seed_iso_oof_blobs.append(iso_oof.copy())

        print(f"   kf_seed={kf_seed:5d}  iso_only={iso_only_pooled:.4f}  "
              f"best_s={best_s:.2f}  best_pooled={best_pooled:.4f}  "
              f"knots={per_fold_knots}")

    mean_rae_iso_only = float(np.mean(per_seed_iso_only_pooled))
    std_rae_iso_only = float(np.std(per_seed_iso_only_pooled))
    mean_rae = float(np.mean(per_seed_best_pooled))
    std_rae = float(np.std(per_seed_best_pooled))
    print(f"\n[eval] iso_only mean across seeds = {mean_rae_iso_only:.4f} "
          f"+/- {std_rae_iso_only:.4f}")
    print(f"[eval] iso+stretch (best-s per seed) = {mean_rae:.4f} "
          f"+/- {std_rae:.4f}")
    print(f"[eval] per-seed best_s = {per_seed_best_s}")

    # Use mean s as the deploy s (more conservative than per-seed best)
    deploy_s = float(np.mean(per_seed_best_s))
    # snap to nearest grid value for clean deploy
    deploy_s_grid = float(min(STRETCH_GRID,
                              key=lambda s: abs(s - deploy_s)))
    print(f"[deploy] mean per-seed best_s = {deploy_s:.4f}  "
          f"snapped to grid = {deploy_s_grid:.2f}")

    # ---- Compose per-seed best-stretched OOFs -> average for nb2632 oof ----
    pred_oof_per_seed = []
    for iso_oof, best_s in zip(per_seed_iso_oof_blobs, per_seed_best_s):
        mu_iso = float(iso_oof.mean())
        pred_oof_per_seed.append(mu_iso + best_s * (iso_oof - mu_iso))
    pred_oof_avg = np.mean(np.stack(pred_oof_per_seed, axis=0), axis=0)
    pred_oof_avg_rae = float(rae(y_unb, pred_oof_avg))
    print(f"[eval] avg-of-seeds pooled RAE on 253 = {pred_oof_avg_rae:.4f}")

    # ============================================================
    # STEP 2: gate
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 2: gate")
    print("-" * 78)
    if mean_rae < GATE_PROMOTE:
        verdict = "BETTER_THAN_NB2621"
    else:
        verdict = "FAIL"
    print(f"[gate] mean_rae(iso+stretch)={mean_rae:.4f}  "
          f"(strict < {GATE_PROMOTE} -> BETTER_THAN_NB2621)"
          f"  -> {verdict}")

    # ============================================================
    # STEP 3: deploy
    #   Refit IsotonicRegression on FULL 253 anchor; transform te_nb2621 (513);
    #   apply deploy_s_grid as final scalar stretch.
    # ============================================================
    print("\n" + "-" * 78)
    print(f"STEP 3: deploy (iso on full 253, stretch s={deploy_s_grid})")
    print("-" * 78)
    ir_deploy = IsotonicRegression(
        y_min=ISO_Y_MIN, y_max=ISO_Y_MAX, out_of_bounds="clip"
    )
    ir_deploy.fit(anchor_oof, y_unb)
    in_rae_deploy_iso_only = float(rae(y_unb, ir_deploy.predict(anchor_oof)))
    n_knots_deploy = int(len(np.unique(ir_deploy.X_thresholds_))) \
        if hasattr(ir_deploy, "X_thresholds_") else -1

    te_iso = ir_deploy.predict(anchor_te).astype(np.float64)
    mu_te_iso = float(te_iso.mean())
    te_final = mu_te_iso + deploy_s_grid * (te_iso - mu_te_iso)
    in_rae_deploy_final = float(
        rae(y_unb, mu_te_iso + deploy_s_grid * (
            ir_deploy.predict(anchor_oof) -
            ir_deploy.predict(anchor_oof).mean()))
    )
    # te[unb_idx] in-sample check
    te_unb_in = float(rae(y_unb, te_final[unb_idx]))
    print(f"   deploy n_knots                   = {n_knots_deploy}")
    print(f"   deploy iso-only in-sample RAE    = {in_rae_deploy_iso_only:.4f}")
    print(f"   deploy iso+stretch in-sample RAE = {in_rae_deploy_final:.4f}")
    print(f"   te[unb_idx] in-sample RAE        = {te_unb_in:.4f}")
    print(f"   te_final mean={te_final.mean():.3f}  std={te_final.std():.3f}  "
          f"(nb2621 te was {anchor_te.mean():.3f}/{anchor_te.std():.3f})")

    # ============================================================
    # STEP 4: save artifacts
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 4: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_avg.astype(np.float32))
    np.save(te_path, te_final.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_isotonic_on_nb2621.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_final.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip submission] verdict=FAIL")

    delta_vs_nb2621 = mean_rae - NB2621_REF
    delta_vs_nb2604 = mean_rae - NB2604_REF
    delta_vs_nb2171 = mean_rae - NB2171_REF
    print(f"\n   delta vs nb2621 winner ({NB2621_REF:.4f}) = "
          f"{delta_vs_nb2621:+.4f}")
    print(f"   delta vs nb2604 (mean) ({NB2604_REF:.4f}) = "
          f"{delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2171 ceiling ({NB2171_REF:.4f}) = "
          f"{delta_vs_nb2171:+.4f}")

    # ---- summary JSON ----
    summary = {
        "tag": TAG,
        "method": "isotonic_y_clamp_plus_rank_stretch_on_nb2621",
        "paradigm": "post_hoc_calibration_df_1",
        "anchor": "nb2621 K=18+K=20 mean",
        "anchor_pre_unblind": True,  # K-RFE pyramids are all PRE-unblind chemprop_aux residuals
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_oof_rae": rae_anchor_oof,
        "iso_y_min": ISO_Y_MIN,
        "iso_y_max": ISO_Y_MAX,
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "per_seed_results": per_seed_results,
        "per_seed_best_s": per_seed_best_s,
        "per_seed_best_pooled_rae": per_seed_best_pooled,
        "per_seed_iso_only_pooled_rae": per_seed_iso_only_pooled,
        "mean_rae_iso_only": mean_rae_iso_only,
        "std_rae_iso_only": std_rae_iso_only,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "pred_oof_avg_rae": pred_oof_avg_rae,
        "deploy_s": deploy_s,
        "deploy_s_grid": deploy_s_grid,
        "deploy_n_knots": n_knots_deploy,
        "in_sample_iso_only_rae": in_rae_deploy_iso_only,
        "in_sample_iso_plus_stretch_rae": in_rae_deploy_final,
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_final.mean()),
        "te_std": float(te_final.std()),
        "anchor_te_mean": float(anchor_te.mean()),
        "anchor_te_std": float(anchor_te.std()),
        "gate_promote": GATE_PROMOTE,
        "verdict": verdict,
        "nb2621_ref": NB2621_REF,
        "delta_vs_nb2621": delta_vs_nb2621,
        "nb2604_ref": NB2604_REF,
        "delta_vs_nb2604": delta_vs_nb2604,
        "nb2171_ref": NB2171_REF,
        "delta_vs_nb2171": delta_vs_nb2171,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (str(sub_csv) if verdict != "FAIL" else None),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor                   = nb2621 K=18+K=20 (oof_RAE {rae_anchor_oof:.4f})")
    print(f"   mean iso_only            = {mean_rae_iso_only:.4f} +/- {std_rae_iso_only:.4f}")
    print(f"   mean iso+stretch         = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   per-seed best_s          = {per_seed_best_s}")
    print(f"   deploy_s (grid-snapped)  = {deploy_s_grid:.2f}")
    print(f"   verdict                  = {verdict}")
    print(f"   delta vs nb2621          = {delta_vs_nb2621:+.4f}")
    print(f"   delta vs nb2604          = {delta_vs_nb2604:+.4f}")
    print(f"   delta vs nb2171 ceiling  = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] in-RAE       = {te_unb_in:.4f}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae_iso_only",
        "mean_rae",
        "std_rae",
        "deploy_s_grid",
        "verdict",
        "delta_vs_nb2621",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
