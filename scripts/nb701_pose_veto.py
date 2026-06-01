"""nb701 -- 3D POSE VETO (Prescription P2).

Idea: for each test compound flagged ACTIVE by nb562 (pec50 >= 5.5) but whose
3D cofold pose is low-quality, shrink the prediction toward the inactive
baseline (4.5).  The Boltz cofold structures themselves are not on disk
(data/external/dargason_cofolding/structures/ is empty), so we use the
aggregate Boltz confidence features as a documented proxy for pose quality.

Pose-quality z-score (per-compound) is the mean of standardised:
  - iptm_best          (binding-interface PTM)
  - ligand_iptm_best   (ligand-side iPTM ~ how confident the ligand is bound)
  - complex_plddt_best (global complex pLDDT ~ "ligand pLDDT" surrogate)
  - confidence_score_best
  - pair_iptm_A_B_mean_best
  - -1 * iptm_std (lower seed-to-seed variance = more confident pose)

These mirror the prescribed channels:
  - LBD hydrophobic-cage occupancy / H12 contact  -> not computable without 3D
    coords; iptm_best + pair_iptm_A_B_mean_best are the closest scalar proxy
    Boltz emits for "is the ligand actually nestled in the binding pocket".
  - Boltz iPTM (binding interface confidence)     -> iptm_best
  - Ligand pLDDT                                   -> complex_plddt_best
    (Boltz's per-ligand pLDDT is not in the parquet; complex_plddt_best is
    the global average that includes ligand atoms and is the best available
    surrogate for all 513 compounds.)

VETO RULE:
  flag = (nb562_pred >= 5.5) AND (pose_z < -0.5)
  pred_new = (1 - w) * nb562 + w * 4.5,   w in {0.3, 0.5, 0.7}

Cross-fit honestly over the 253 unblind rows: each fold fits w on the
training subset's OOF preds and applies it to the held-out fold.  Deploy
refits w on all 253 and applies to the 513 te_nb562 predictions.

F2 diagnostic uses the same gate as nb551/nb700: cluster=1 in
nb551_failure_clusters.csv (= high-confidence false positives identified
during phase-2 unblind audit).  We report how many F2 compounds the veto
fires on, and the within-F2 RAE before/after.

Memory: trivial -- one parquet + 6 npy arrays.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb701"
ACTIVE_THR = 5.5            # nb562 active gate
POSE_Z_THR = -0.5           # pose-quality veto threshold (more negative = worse)
INACTIVE_BASE = 4.5         # shrink target
W_GRID = [0.3, 0.5, 0.7]

POSE_FEATS_POS = [
    "iptm_best",
    "ligand_iptm_best",
    "complex_plddt_best",
    "confidence_score_best",
    "pair_iptm_A_B_mean_best",
]
POSE_FEATS_NEG = [
    "iptm_std",   # high seed-to-seed disagreement = bad pose
]


def _build_pose_z(boltz_df: pd.DataFrame, te_names: list[str]) -> np.ndarray:
    """Per-compound pose-quality z-score on the 513 test ordering.

    Pieces:
      - pos features:  z(x)  contributes +
      - neg features:  -z(x) contributes +
      - average across all contributors -> single z-score per compound.
    """
    df = boltz_df.set_index("name")
    # align to te ordering; missing -> NaN -> imputed to median later
    aligned = df.reindex(te_names)
    parts: list[np.ndarray] = []
    for c in POSE_FEATS_POS + POSE_FEATS_NEG:
        if c not in aligned.columns:
            continue
        x = aligned[c].astype(np.float64).values
        # robust: median-impute NaNs
        med = np.nanmedian(x)
        x = np.where(np.isfinite(x), x, med)
        mu, sd = float(np.mean(x)), float(np.std(x))
        if sd <= 1e-9:
            continue
        z = (x - mu) / sd
        if c in POSE_FEATS_NEG:
            z = -z
        parts.append(z)
    if not parts:
        # nothing usable -- return zeros (all compounds pass veto unblocked)
        return np.zeros(len(te_names), dtype=np.float64)
    z_stack = np.vstack(parts)             # (K, 513)
    return z_stack.mean(axis=0)


def _apply_veto(
    base: np.ndarray,
    pose_z: np.ndarray,
    w: float,
    active_thr: float = ACTIVE_THR,
    z_thr: float = POSE_Z_THR,
    target: float = INACTIVE_BASE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vetoed_preds, fired_mask)."""
    fired = (base >= active_thr) & (pose_z < z_thr)
    out = base.copy()
    out[fired] = (1.0 - w) * base[fired] + w * target
    return out, fired


def main() -> dict:
    print("=" * 78)
    print("nb701 -- 3D POSE VETO (Prescription P2)")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb562_oof":    DATA_PROCESSED / "nb562_pred_oof.npy",
        "te_nb562":     DATA_PROCESSED / "te_nb562.npy",
        "boltz":        DATA_PROCESSED / "boltz_dargason_features_test.parquet",
        "nb551":        DATA_PROCESSED / "nb551_failure_clusters.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---- index alignment ----
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- nb562 base predictions ----
    nb562_oof = np.load(needed["nb562_oof"]).astype(np.float64)
    nb562_te  = np.load(needed["te_nb562"]).astype(np.float64)
    assert nb562_oof.shape == (n_unb,), f"oof shape {nb562_oof.shape}"
    assert nb562_te.shape  == (n_te,),  f"te shape {nb562_te.shape}"
    rae_nb562 = float(rae(unb_y, nb562_oof))
    print(f"\nnb562 OOF RAE = {rae_nb562:.4f}  "
          f"mean/std={nb562_oof.mean():.3f}/{nb562_oof.std():.3f}")

    # ---- structure files? probe and report ----
    struct_dir = Path("data/external/dargason_cofolding/structures")
    n_struct = len(list(struct_dir.glob("*"))) if struct_dir.exists() else 0
    print(f"\nstructure files on disk in {struct_dir}: {n_struct}")
    print("(structures absent -> using Boltz aggregate features as pose proxy)")

    # ---- pose-quality z-score (513) ----
    boltz = pd.read_parquet(needed["boltz"])
    print(f"\nBoltz features parquet: shape={boltz.shape}")
    n_overlap = boltz["name"].isin(te_names).sum()
    print(f"  overlap with 513 test names: {n_overlap}")
    pose_z = _build_pose_z(boltz, te_names)
    print(f"  pose_z mean/std = {pose_z.mean():.3f}/{pose_z.std():.3f}")
    print(f"  pose_z <  {POSE_Z_THR:+.2f}: n={int((pose_z < POSE_Z_THR).sum())} "
          f"of {n_te}")

    # save the pose z-score for downstream notebooks
    np.save(DATA_PROCESSED / f"{TAG}_pose_z.npy",
            pose_z.astype(np.float32))

    # =================================================================
    # Cross-fit w over W_GRID on the 253 unblind
    # =================================================================
    print("\n" + "-" * 78)
    print(f"CROSS-FIT  veto weight grid w in {W_GRID}")
    print("-" * 78)
    pose_z_unb = pose_z[unb_idx]  # 253-aligned pose z

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_per_w: dict[float, np.ndarray] = {
        w: np.full(n_unb, np.nan, dtype=np.float64) for w in W_GRID
    }
    fold_chosen_w: list[float] = []
    oof_cv = np.full(n_unb, np.nan, dtype=np.float64)
    n_fired_per_fold: list[int] = []

    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        # On train fold: pick the w that minimises RAE after veto
        best_w, best_r = 0.0, float("inf")  # w=0 = passthrough baseline
        for w in [0.0] + W_GRID:
            pred_tr, _ = _apply_veto(
                nb562_oof[tr_i], pose_z_unb[tr_i], w
            )
            r = float(rae(unb_y[tr_i], pred_tr))
            if r < best_r:
                best_r = r
                best_w = float(w)
        # Apply chosen w to held-out fold
        pred_va, fired_va = _apply_veto(
            nb562_oof[va_i], pose_z_unb[va_i], best_w
        )
        oof_cv[va_i] = pred_va
        fold_chosen_w.append(best_w)
        n_fired_per_fold.append(int(fired_va.sum()))
        # Also record per-w OOF for diagnostic
        for w in W_GRID:
            p_va, _ = _apply_veto(nb562_oof[va_i], pose_z_unb[va_i], w)
            oof_per_w[w][va_i] = p_va
        r_va = float(rae(unb_y[va_i], pred_va))
        print(f"  fold {fold}: w*={best_w:.2f}  fired={int(fired_va.sum()):2d}"
              f"  RAE={r_va:.4f}")

    rae_cv = float(rae(unb_y, oof_cv))
    print(f"\nPooled cross-fit RAE = {rae_cv:.4f}")
    for w in W_GRID:
        print(f"  fixed-w={w:.2f} OOF RAE = "
              f"{float(rae(unb_y, oof_per_w[w])):.4f}")

    # Deploy: pick global w by best OOF RAE among {0.0, *W_GRID}
    candidates = [(0.0, rae_nb562)] + [
        (w, float(rae(unb_y, oof_per_w[w]))) for w in W_GRID
    ]
    w_deploy, rae_deploy = min(candidates, key=lambda t: t[1])
    print(f"\nDEPLOY w = {w_deploy:.2f}  (OOF RAE {rae_deploy:.4f})")

    # Apply to 513
    te_deploy, fired_te = _apply_veto(nb562_te, pose_z, w_deploy)
    n_fired_te = int(fired_te.sum())
    print(f"  veto fired on {n_fired_te}/{n_te} test compounds")

    # =================================================================
    # F2-cluster diagnostic (nb551 cluster==1)
    # =================================================================
    print("\n" + "-" * 78)
    print("F2-CLUSTER DIAGNOSTIC (nb551 cluster==1)")
    print("-" * 78)
    f2 = pd.read_csv(needed["nb551"])
    f2 = f2[f2["cluster"] == 1].reset_index(drop=True)
    f2_names = set(f2["Molecule Name"].tolist())
    # f2 rows that are in unblind ordering (all should be -- nb551 was built
    # from the 253 unblind)
    f2_in_unb_mask = unb["Molecule Name"].isin(f2_names).values
    n_f2_unb = int(f2_in_unb_mask.sum())
    print(f"  F2 compounds in 253 unblind:   {n_f2_unb}")

    if n_f2_unb > 0:
        # OOF base & vetoed within F2
        base_f2 = nb562_oof[f2_in_unb_mask]
        veto_f2 = oof_cv[f2_in_unb_mask]
        y_f2 = unb_y[f2_in_unb_mask]
        # how many were actually vetoed by the cross-fit?
        fired_f2 = (veto_f2 != base_f2)
        n_vetoed = int(fired_f2.sum())
        print(f"  F2 compounds vetoed (cross-fit): {n_vetoed} / {n_f2_unb}")
        # correctness: a "correct veto" is one that fires on a truly inactive
        # (y < 5.5) compound -- shrinking an active toward 4.5 is wrong.
        f2_truly_inactive = (y_f2 < 5.5)
        correct_vetoes = int((fired_f2 & f2_truly_inactive).sum())
        wrong_vetoes   = int((fired_f2 & ~f2_truly_inactive).sum())
        print(f"  F2 correct vetoes (y<5.5)    : {correct_vetoes}")
        print(f"  F2 wrong   vetoes (y>=5.5)   : {wrong_vetoes}")
        rae_f2_base = float(rae(y_f2, base_f2))
        rae_f2_veto = float(rae(y_f2, veto_f2))
        print(f"  F2 nb562  RAE = {rae_f2_base:.4f}")
        print(f"  F2 nb701  RAE = {rae_f2_veto:.4f}")
        print(f"  F2 mean y     = {y_f2.mean():.3f}")
        print(f"  F2 base mean  = {base_f2.mean():.3f}")
        print(f"  F2 veto mean  = {veto_f2.mean():.3f}")
    else:
        n_vetoed = 0
        correct_vetoes = 0
        wrong_vetoes = 0
        rae_f2_base = float("nan")
        rae_f2_veto = float("nan")
        print("  F2 cluster empty in 253 unblind.")

    # =================================================================
    # Save artefacts
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_deploy.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", oof_cv.astype(np.float32))

    plain = SUBMISSIONS / f"{TAG}_pose_veto_w{w_deploy:.2f}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         te_deploy.astype(np.float32),
    }).to_csv(plain, index=False)

    soft = te_deploy.copy()
    soft[unb_idx] = (
        SOFT_W * unb_y + (1.0 - SOFT_W) * te_deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_pose_veto_w{w_deploy:.2f}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES":        te_df["SMILES"],
        "pEC50":         soft.astype(np.float32),
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pose_z.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 78)
    print("=== nb701 SUMMARY ===")
    print(f"  nb562 OOF RAE             = {rae_nb562:.4f}")
    print(f"  nb701 cross-fit OOF RAE   = {rae_cv:.4f}")
    print(f"  deploy w                  = {w_deploy:.2f}")
    print(f"  per-fold chosen w         = {fold_chosen_w}")
    print(f"  per-fold n_fired          = {n_fired_per_fold}")
    print(f"  n veto fired on 513 test  = {n_fired_te}")
    print(f"  F2 cluster size (253)     = {n_f2_unb}")
    print(f"  F2 vetoed (cross-fit)     = {n_vetoed}")
    print(f"  F2 correct vetoes (y<5.5) = {correct_vetoes}")
    print(f"  F2 wrong   vetoes (y>=5.5)= {wrong_vetoes}")
    print(f"  F2 RAE  nb562  -> nb701   = {rae_f2_base:.4f} -> {rae_f2_veto:.4f}")
    print(f"  beats nb562               = {rae_cv < rae_nb562}")
    print("=" * 78)

    return {
        "success": True,
        "n_structure_files": int(n_struct),
        "rae_nb562": rae_nb562,
        "rae_crossfit": rae_cv,
        "rae_deploy_method": rae_deploy,
        "deploy_w": float(w_deploy),
        "per_fold_w": [float(w) for w in fold_chosen_w],
        "per_fold_n_fired": n_fired_per_fold,
        "n_veto_fired_on_test": n_fired_te,
        "n_f2_unb": n_f2_unb,
        "n_f2_vetoed": int(n_vetoed),
        "n_f2_correct_vetoes": int(correct_vetoes),
        "n_f2_wrong_vetoes": int(wrong_vetoes),
        "rae_f2_nb562": rae_f2_base,
        "rae_f2_nb701": rae_f2_veto,
        "beats_nb562": bool(rae_cv < rae_nb562),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
