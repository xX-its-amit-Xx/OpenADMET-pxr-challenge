"""nb2932 -- Isotonic calibration on top of nb2920 best 2-anchor blend.

NEW PARADIGM:
    nb2920 swept w on the {nb2240_K20, nb1191} 2-anchor blend and the
    BEST mix was w=0.7 (pred = 0.7 * nb2240_oof + 0.3 * nb1191_oof) with
    pooled RAE = 0.4596 on 253 unblind (MARGINAL_BEAT vs 0.4598 ref).
    nb2920 carries df = 0 (no learnable parameters).

    This script tests whether adding a per-fold IsotonicRegression on top
    of this marginal-beat blend extracts further calibration gain (or
    confirms the post-hoc-iso axis is closed at this anchor).

PROTOCOL:
    - pred_base = 0.7 * nb2240_K20_oof + 0.3 * nb1191_oof
    - 5-fold SCAFFOLD CV via scaffold_kfold_indices, repeated over 5
      kf_seeds {1001..1005}.
    - Per (kf_seed, fold): fit IsotonicRegression(y_min=3.0, y_max=8.0,
      out_of_bounds='clip') on (pred_base[tr], y[tr]); transform
      pred_base[va] to produce iso_oof.
    - Pooled RAE per seed = rae(y, iso_oof); aggregate mean across seeds.
    - Deploy: refit IsotonicRegression on the full 253 (pred_base -> y_unb);
      transform pred_base on the 513 deploy te (te_base = 0.7 * te_nb2240_K20
      + 0.3 * te_nb1191); save te_nb2932.

GATES:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    otherwise          -> "FAIL"

Outputs:
    scripts/nb2932_iso_on_nb2920.py
    data/processed/nb2932_summary.json
    data/processed/nb2932_pred_oof.npy   (253,) float32
    data/processed/te_nb2932.npy         (513,) float32
    submissions/nb2932_iso_on_nb2920.csv  (on any non-FAIL)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

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

TAG = "nb2932"

# ---- Anchors (nb2920 best w=0.7 blend) ----
ANCHOR_A_NAME = "nb2240_K20"
ANCHOR_A_OOF = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
ANCHOR_A_TE = DATA_PROCESSED / "te_nb2240_K20.npy"

ANCHOR_B_NAME = "nb1191"
ANCHOR_B_OOF = DATA_PROCESSED / "nb1191_pred_oof.npy"
ANCHOR_B_TE = DATA_PROCESSED / "te_nb1191.npy"

W_A = 0.7  # weight on nb2240_K20
W_B = 0.3  # weight on nb1191

# ---- Isotonic clamp ----
ISO_Y_MIN = 3.0
ISO_Y_MAX = 8.0

# ---- CV protocol ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- Gates ----
PROMOTE_THR = 0.4570
MARGINAL_THR = 0.4598

# ---- Reference points ----
NB2920_REF = 0.4596  # nb2920 best pooled RAE (w=0.7)


def calibrate_fold(base_tr: np.ndarray, y_tr: np.ndarray,
                   base_va: np.ndarray) -> np.ndarray:
    """Per-fold IsotonicRegression with [y_min, y_max] clamp.

    Fit on (base_tr, y_tr) and transform base_va.  out_of_bounds='clip'
    handles val anchors that fall outside the train anchor range.
    """
    ir = IsotonicRegression(
        y_min=ISO_Y_MIN,
        y_max=ISO_Y_MAX,
        out_of_bounds="clip",
    )
    ir.fit(base_tr, y_tr)
    return ir.predict(base_va).astype(np.float64)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- isotonic calibration on nb2920 best 2-anchor blend")
    print(f"          pred_base = {W_A} * {ANCHOR_A_NAME} + {W_B} * {ANCHOR_B_NAME}")
    print(f"          iso clamp y in [{ISO_Y_MIN}, {ISO_Y_MAX}]")
    print(f"          scaffold CV n_folds={N_FOLDS}  kf_seeds={KF_SEEDS}")
    print(f"          gates: <{PROMOTE_THR} PROMOTE  <{MARGINAL_THR} MARGINAL_BEAT")
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

    # ---- Load anchors ----
    for p in (ANCHOR_A_OOF, ANCHOR_A_TE, ANCHOR_B_OOF, ANCHOR_B_TE):
        if not p.exists():
            raise FileNotFoundError(f"missing anchor artefact: {p}")

    oof_a = np.load(ANCHOR_A_OOF).astype(np.float64)
    oof_b = np.load(ANCHOR_B_OOF).astype(np.float64)
    te_a = np.load(ANCHOR_A_TE).astype(np.float64)
    te_b = np.load(ANCHOR_B_TE).astype(np.float64)
    assert oof_a.shape == (n_unb,), f"{ANCHOR_A_NAME} oof {oof_a.shape}"
    assert oof_b.shape == (n_unb,), f"{ANCHOR_B_NAME} oof {oof_b.shape}"
    assert te_a.shape == (n_test,), f"{ANCHOR_A_NAME} te {te_a.shape}"
    assert te_b.shape == (n_test,), f"{ANCHOR_B_NAME} te {te_b.shape}"

    rae_a = float(rae(y_unb, oof_a))
    rae_b = float(rae(y_unb, oof_b))
    print(f"[anchor] {ANCHOR_A_NAME:12s} oof_RAE={rae_a:.4f}")
    print(f"[anchor] {ANCHOR_B_NAME:12s} oof_RAE={rae_b:.4f}")

    # ---- Build pred_base (w=0.7 blend) ----
    pred_base_oof = W_A * oof_a + W_B * oof_b
    pred_base_te = W_A * te_a + W_B * te_b
    rae_base = float(rae(y_unb, pred_base_oof))
    print(f"[base]   pred_base = {W_A}*{ANCHOR_A_NAME} + {W_B}*{ANCHOR_B_NAME}  "
          f"oof_RAE={rae_base:.4f}  (nb2920 ref {NB2920_REF:.4f})")
    print(f"         pred_base_oof mean={pred_base_oof.mean():.3f}  "
          f"std={pred_base_oof.std():.3f}  (truth_std {y_unb.std():.3f})")
    print(f"         pred_base_te  mean={pred_base_te.mean():.3f}  "
          f"std={pred_base_te.std():.3f}")

    # ============================================================
    # STEP 1: per-seed scaffold cross-fit isotonic
    # ============================================================
    print("\n" + "-" * 78)
    print(f"STEP 1: scaffold {N_FOLDS}-fold CV  isotonic  "
          f"({len(KF_SEEDS)} seeds)")
    print("-" * 78)

    per_seed_results = []
    per_seed_pooled = []
    per_seed_iso_oof_blobs = []

    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        iso_oof = np.full(n_unb, np.nan, dtype=np.float64)
        per_fold_knots = []
        per_fold_raes = []
        for k, (tr, va) in enumerate(splits):
            pred_va = calibrate_fold(
                pred_base_oof[tr], y_unb[tr], pred_base_oof[va]
            )
            iso_oof[va] = pred_va
            per_fold_raes.append(float(rae(y_unb[va], pred_va)))
            ir_tmp = IsotonicRegression(
                y_min=ISO_Y_MIN, y_max=ISO_Y_MAX, out_of_bounds="clip"
            )
            ir_tmp.fit(pred_base_oof[tr], y_unb[tr])
            n_knots = int(len(np.unique(ir_tmp.X_thresholds_))) \
                if hasattr(ir_tmp, "X_thresholds_") else -1
            per_fold_knots.append(n_knots)

        if np.isnan(iso_oof).any():
            raise RuntimeError(
                "scaffold splits did not cover all rows; check protocol"
            )

        pooled = float(rae(y_unb, iso_oof))
        fold_mean = float(np.mean(per_fold_raes))
        fold_std = float(np.std(per_fold_raes))
        per_seed_results.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "fold_rae_mean": fold_mean,
            "fold_rae_std": fold_std,
            "fold_raes": per_fold_raes,
            "per_fold_knots": per_fold_knots,
            "mean_knots": float(np.mean(per_fold_knots)),
        })
        per_seed_pooled.append(pooled)
        per_seed_iso_oof_blobs.append(iso_oof.copy())

        print(f"   kf_seed={kf_seed:5d}  pooled={pooled:.4f}  "
              f"fold_mean={fold_mean:.4f}  fold_std={fold_std:.4f}  "
              f"knots={per_fold_knots}")

    mean_rae = float(np.mean(per_seed_pooled))
    std_rae = float(np.std(per_seed_pooled))
    print(f"\n[eval] iso mean across seeds = {mean_rae:.4f} +/- {std_rae:.4f}")

    # ---- Average per-seed iso OOFs -> deploy pred_oof ----
    pred_oof_avg = np.mean(np.stack(per_seed_iso_oof_blobs, axis=0), axis=0)
    pred_oof_avg_rae = float(rae(y_unb, pred_oof_avg))
    print(f"[eval] avg-of-seeds pooled RAE on 253 = {pred_oof_avg_rae:.4f}")

    # ============================================================
    # STEP 2: gate
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 2: gate")
    print("-" * 78)
    if mean_rae < PROMOTE_THR:
        verdict = "PROMOTE"
    elif mean_rae < MARGINAL_THR:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] mean_rae={mean_rae:.4f}  "
          f"(<{PROMOTE_THR} PROMOTE, <{MARGINAL_THR} MARGINAL_BEAT)"
          f"  ->  {verdict}")

    # ============================================================
    # STEP 3: deploy (refit iso on full 253 -> transform te_base 513)
    # ============================================================
    print("\n" + "-" * 78)
    print("STEP 3: deploy")
    print("-" * 78)
    ir_deploy = IsotonicRegression(
        y_min=ISO_Y_MIN, y_max=ISO_Y_MAX, out_of_bounds="clip"
    )
    ir_deploy.fit(pred_base_oof, y_unb)
    n_knots_deploy = int(len(np.unique(ir_deploy.X_thresholds_))) \
        if hasattr(ir_deploy, "X_thresholds_") else -1
    in_rae_deploy = float(rae(y_unb, ir_deploy.predict(pred_base_oof)))
    te_final = ir_deploy.predict(pred_base_te).astype(np.float64)
    te_unb_in = float(rae(y_unb, te_final[unb_idx]))
    print(f"   deploy n_knots         = {n_knots_deploy}")
    print(f"   deploy iso in-sample   = {in_rae_deploy:.4f}")
    print(f"   te[unb_idx] in-sample  = {te_unb_in:.4f}")
    print(f"   te_final mean={te_final.mean():.3f}  std={te_final.std():.3f}  "
          f"(pred_base_te was {pred_base_te.mean():.3f}/{pred_base_te.std():.3f})")

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

    sub_csv = SUBMISSIONS / f"{TAG}_iso_on_nb2920.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_final.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip submission] verdict=FAIL")

    delta_vs_nb2920 = mean_rae - NB2920_REF
    print(f"\n   delta vs nb2920 ({NB2920_REF:.4f}) = {delta_vs_nb2920:+.4f}")

    # ---- summary JSON ----
    summary = {
        "tag": TAG,
        "method": "isotonic_y_clamp_on_nb2920_best_2anchor_blend",
        "paradigm": "post_hoc_iso_on_marginal_beat_2anchor",
        "anchors": [
            {"name": ANCHOR_A_NAME, "weight": W_A,
             "oof_path": str(ANCHOR_A_OOF), "te_path": str(ANCHOR_A_TE),
             "oof_rae_unb": rae_a},
            {"name": ANCHOR_B_NAME, "weight": W_B,
             "oof_path": str(ANCHOR_B_OOF), "te_path": str(ANCHOR_B_TE),
             "oof_rae_unb": rae_b},
        ],
        "anchor_pre_unblind": True,  # both nb2240_K20 and nb1191 are PRE-unblind clean
        "w_A": W_A,
        "w_B": W_B,
        "pred_base_oof_rae": rae_base,
        "iso_y_min": ISO_Y_MIN,
        "iso_y_max": ISO_Y_MAX,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "per_seed_results": per_seed_results,
        "per_seed_pooled_rae": per_seed_pooled,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "pred_oof_avg_rae": pred_oof_avg_rae,
        "deploy_n_knots": n_knots_deploy,
        "in_sample_iso_rae": in_rae_deploy,
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_final.mean()),
        "te_std": float(te_final.std()),
        "pred_base_te_mean": float(pred_base_te.mean()),
        "pred_base_te_std": float(pred_base_te.std()),
        "promote_thr": PROMOTE_THR,
        "marginal_thr": MARGINAL_THR,
        "verdict": verdict,
        "nb2920_ref": NB2920_REF,
        "delta_vs_nb2920": delta_vs_nb2920,
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
    print(f"   pred_base                = {W_A}*{ANCHOR_A_NAME} + {W_B}*{ANCHOR_B_NAME}  "
          f"(oof_RAE {rae_base:.4f})")
    print(f"   mean iso (across seeds)  = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   avg-of-seeds pooled RAE  = {pred_oof_avg_rae:.4f}")
    print(f"   delta vs nb2920          = {delta_vs_nb2920:+.4f}")
    print(f"   te[unb_idx] in-RAE       = {te_unb_in:.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "std_rae",
        "pred_oof_avg_rae",
        "verdict",
        "delta_vs_nb2920",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
