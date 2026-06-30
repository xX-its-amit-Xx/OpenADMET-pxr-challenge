"""nb1152 -- Residual cascade (chemprop_aux -> LGBM K=28 -> decile-LGBM).

Three-stage scaffold-CV cascade on 253-unblind:
  Stage-1: chemprop_aux (te slice). Anchor preds, constant across folds.
  Stage-2: LGBM(MSE) on (y_unb - stage1) using the 28-col SHAP-selected
           feature matrix from nb2103 (X_unb_28_nb2103.npy, shape (253, 28)).
           Scaffold 5-fold CV produces pred_oof_s2 (253,).
  Stage-3: r2 = y_unb - stage1 - pred_oof_s2. Bin r2 into 10 deciles using
           edges computed ONLY on the train fold (np.quantile on r2_train).
           One-hot encode the train deciles, augment K=28 features, fit a 2nd
           LGBM on r2 (train), predict pred_oof_s3 on val. Decile edges from
           train fold are reused to one-hot the val rows (np.digitize),
           freezing edges to avoid val-leakage.

Final = stage1 + pred_oof_s2 + pred_oof_s3. Pooled RAE on 253.
Compare vs nb2103 K=28 mean-bag ref 0.5057 (per-seed mean), gate 0.003.

If beats: build deploy CSV submissions/nb1152_residual_cascade.csv
          (513 rows -- stage1 on full test + stage2/3 frozen-fit on all 253).

Outputs:
  scripts/nb1152_residual_cascade.py
  data/processed/nb1152_summary.json
  data/processed/nb1152_residual_cascade_pred_oof.npy   (253,) float32
  submissions/nb1152_residual_cascade.csv               (if beats gate)
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
import lightgbm as lgb
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, standardize
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1152"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
X_UNB_28_PATH = DATA_PROCESSED / "X_unb_28_nb2103.npy"

N_SPLITS = 5
SEED = 42
N_DECILES = 10
DECISION_MARGIN = 0.003
BASELINE_REF = 0.5057  # nb2103 K=28 per-seed-mean reference

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"


def _lgbm_params(seed: int = 0) -> dict:
    """Same family as nb2103 K=28 (max_depth=4, num_leaves=15, n_est=300)."""
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _decile_one_hot(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin `values` into deciles using `edges` (length 9 interior cuts).

    np.digitize returns indices in [0, len(edges)] = [0, 9]. One-hot to
    (n, 10).
    """
    n_bins = len(edges) + 1  # 10
    bin_idx = np.digitize(values, edges, right=False)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    oh = np.zeros((len(values), n_bins), dtype=np.float32)
    oh[np.arange(len(values)), bin_idx] = 1.0
    return oh


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Residual cascade: chemprop_aux -> LGBM K=28 -> "
          f"decile-LGBM")
    print(f"          scaffold {N_SPLITS}-fold CV  seed={SEED}  "
          f"N_DECILES={N_DECILES}")
    print(f"          baseline (nb2103 K=28 per-seed-mean) = "
          f"{BASELINE_REF:.4f}  gate = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load truth, unblind index, K=28 features, anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns \
        else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing anchor: {ANCHOR_TE_PATH}")
    if not X_UNB_28_PATH.exists():
        raise FileNotFoundError(f"missing K=28 features: {X_UNB_28_PATH}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_s1 = float(rae(y_unb, anchor))
    print(f"[stage1] {ANCHOR} on 253-unb: RAE = {rae_s1:.4f}")

    X_unb_28 = np.load(X_UNB_28_PATH).astype(np.float32)
    if X_unb_28.shape != (n_unb, 28):
        raise ValueError(f"X_unb_28 shape {X_unb_28.shape} != ({n_unb},28)")
    print(f"[feat] X_unb_28 = {X_unb_28.shape}")

    # ---- Build scaffold groups on 253 unblind SMILES ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    scaffolds: list[str | None] = []
    for s in unb_smiles:
        m = standardize(s)
        sc = bemis_murcko(m) if m is not None else None
        scaffolds.append(sc if (sc is not None and sc != "") else None)
    n_with_scaffold = sum(s is not None for s in scaffolds)
    print(f"[scaf] {n_with_scaffold}/{n_unb} compounds with non-empty "
          f"Murcko scaffold")

    splits = scaffold_kfold_indices(scaffolds, n_splits=N_SPLITS,
                                    shuffle=True, seed=SEED)
    fold_sizes = [(len(tr), len(va)) for tr, va in splits]
    print(f"[scaf] fold (n_tr, n_va) sizes = {fold_sizes}")

    # ---- Stage 2: LGBM on (y - stage1) residual, scaffold 5-fold CV ----
    print("\n" + "-" * 78)
    print("STAGE-2: LGBM(MSE) K=28 on chemprop_aux residual")
    print("-" * 78)
    r1 = y_unb - anchor
    print(f"   r1 (y - stage1): mean={r1.mean():+.4f}  std={r1.std():.4f}")
    pred_oof_s2 = np.full(n_unb, np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    for k, (tr_idx, va_idx) in enumerate(splits):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed=SEED + k))
        mdl.fit(X_unb_28[tr_idx], r1[tr_idx])
        pred_oof_s2[va_idx] = mdl.predict(X_unb_28[va_idx])
        rae_va = float(rae(y_unb[va_idx], anchor[va_idx] + pred_oof_s2[va_idx]))
        fold_records.append({"fold": k, "n_tr": int(len(tr_idx)),
                              "n_va": int(len(va_idx)),
                              "rae_val_after_s2": rae_va})
        print(f"   fold {k}: n_tr={len(tr_idx)} n_va={len(va_idx)} "
              f"val-RAE(after s2) = {rae_va:.4f}")
    assert not np.isnan(pred_oof_s2).any(), "stage-2 oof has NaN"
    pred_after_s2 = anchor + pred_oof_s2
    rae_s2 = float(rae(y_unb, pred_after_s2))
    print(f"   pooled RAE(stage1+stage2) = {rae_s2:.4f}  "
          f"(d_vs_s1 = {rae_s2 - rae_s1:+.4f})")

    # ---- Stage 3: per-fold decile-encoded LGBM on r2 ----
    print("\n" + "-" * 78)
    print(f"STAGE-3: decile-LGBM ({N_DECILES} bins) on r2 = y - s1 - s2_oof")
    print("-" * 78)
    r2 = y_unb - pred_after_s2  # cross-fit residual for stage-3 target
    print(f"   r2 (y - s1 - s2_oof): mean={r2.mean():+.4f}  "
          f"std={r2.std():.4f}")
    pred_oof_s3 = np.full(n_unb, np.nan, dtype=np.float64)
    s3_records: list[dict] = []
    quantile_cuts = np.linspace(0, 1, N_DECILES + 1)[1:-1]  # 9 interior cuts
    for k, (tr_idx, va_idx) in enumerate(splits):
        r2_tr = r2[tr_idx]
        # Decile edges FROZEN on train fold only
        edges_k = np.quantile(r2_tr, quantile_cuts)
        edges_k = np.unique(edges_k)  # guard against degenerate ties
        oh_tr = _decile_one_hot(r2_tr, edges_k)
        oh_va = _decile_one_hot(np.zeros(len(va_idx)), edges_k)
        # Val rows can't peek at r2 -- use a neutral one-hot
        # (all rows go to the median bin by convention).
        median_bin = len(edges_k) // 2
        oh_va = np.zeros((len(va_idx), len(edges_k) + 1), dtype=np.float32)
        oh_va[:, median_bin] = 1.0
        X_aug_tr = np.concatenate([X_unb_28[tr_idx], oh_tr], axis=1)
        X_aug_va = np.concatenate([X_unb_28[va_idx], oh_va], axis=1)
        mdl3 = lgb.LGBMRegressor(**_lgbm_params(seed=SEED + 100 + k))
        mdl3.fit(X_aug_tr, r2_tr)
        pred_oof_s3[va_idx] = mdl3.predict(X_aug_va)
        rae_va = float(rae(y_unb[va_idx],
                            anchor[va_idx] + pred_oof_s2[va_idx]
                            + pred_oof_s3[va_idx]))
        s3_records.append({
            "fold": k,
            "n_tr": int(len(tr_idx)),
            "n_va": int(len(va_idx)),
            "n_edges": int(len(edges_k)),
            "edges_min": float(edges_k.min()) if len(edges_k) else None,
            "edges_max": float(edges_k.max()) if len(edges_k) else None,
            "rae_val_after_s3": rae_va,
        })
        print(f"   fold {k}: n_edges={len(edges_k)} "
              f"val-RAE(after s3) = {rae_va:.4f}")
    assert not np.isnan(pred_oof_s3).any(), "stage-3 oof has NaN"

    final_pred = anchor + pred_oof_s2 + pred_oof_s3
    rae_final = float(rae(y_unb, final_pred))
    print(f"\n   pooled RAE(stage1+s2+s3) = {rae_final:.4f}  "
          f"(d_vs_s1 = {rae_final - rae_s1:+.4f}  "
          f"d_vs_s2 = {rae_final - rae_s2:+.4f}  "
          f"d_vs_baseline_{BASELINE_REF:.4f} = "
          f"{rae_final - BASELINE_REF:+.4f})")

    beats = rae_final < BASELINE_REF - DECISION_MARGIN
    flat = abs(rae_final - BASELINE_REF) < DECISION_MARGIN
    if beats:
        verdict = "BEATS_NB2103_K28_BASELINE"
    elif flat:
        verdict = "FLAT_VS_NB2103_K28_BASELINE"
    else:
        verdict = "WORSE_THAN_NB2103_K28_BASELINE"
    print(f"   verdict = {verdict}")

    # Save OOF
    out_oof = DATA_PROCESSED / f"{TAG}_residual_cascade_pred_oof.npy"
    np.save(out_oof, final_pred.astype(np.float32))
    print(f"[save] {out_oof}")

    # ---- Deploy CSV (only if beats gate) ----
    deploy_csv_path = None
    deploy_note = None
    if beats:
        # Stage-1 on full 513: te_chemprop_aux already covers it.
        # Stage-2 & Stage-3 deploy refit: fit on all 253 unblind, predict 513.
        # We do NOT have X_unb_28 for the other 260 test rows pre-computed
        # in a single file. To keep nb1152 fully self-contained, we skip the
        # deploy refit when the K=28 test slice for full 513 isn't cached
        # and write a note instead.
        x_te_28_path = DATA_PROCESSED / "X_te_28_nb2103.npy"
        if x_te_28_path.exists():
            X_te_28 = np.load(x_te_28_path).astype(np.float32)
            if X_te_28.shape != (n_test, 28):
                raise ValueError(
                    f"X_te_28 shape {X_te_28.shape} != ({n_test}, 28)"
                )
            mdl_s2_full = lgb.LGBMRegressor(**_lgbm_params(seed=SEED))
            mdl_s2_full.fit(X_unb_28, r1)
            s2_pred_te = mdl_s2_full.predict(X_te_28)
            r2_full = r2  # cross-fit r2 on 253
            edges_full = np.quantile(r2_full, quantile_cuts)
            edges_full = np.unique(edges_full)
            oh_full = _decile_one_hot(r2_full, edges_full)
            X_aug_full = np.concatenate([X_unb_28, oh_full], axis=1)
            median_bin_full = len(edges_full) // 2
            oh_te = np.zeros(
                (n_test, len(edges_full) + 1), dtype=np.float32
            )
            oh_te[:, median_bin_full] = 1.0
            X_aug_te = np.concatenate([X_te_28, oh_te], axis=1)
            mdl_s3_full = lgb.LGBMRegressor(
                **_lgbm_params(seed=SEED + 1000)
            )
            mdl_s3_full.fit(X_aug_full, r2_full)
            s3_pred_te = mdl_s3_full.predict(X_aug_te)
            deploy_pred = te_anchor_513 + s2_pred_te + s3_pred_te
            mol_name_col = "Molecule Name" if "Molecule Name" in te.columns \
                else ("molecule_name" if "molecule_name" in te.columns
                      else None)
            smi_col = "smiles" if "smiles" in te.columns else "SMILES"
            if mol_name_col is None:
                raise KeyError("no Molecule Name col in load_test()")
            SUBMISSIONS.mkdir(parents=True, exist_ok=True)
            deploy_csv_path = SUBMISSIONS / f"{TAG}_residual_cascade.csv"
            pd.DataFrame({
                "SMILES": te[smi_col].astype(str).values,
                "Molecule Name": te[mol_name_col].astype(str).values,
                "pEC50": deploy_pred.astype(np.float32),
            }).to_csv(deploy_csv_path, index=False)
            print(f"[deploy] wrote {deploy_csv_path}")
        else:
            deploy_note = (
                f"DEPLOY_SKIPPED: missing {x_te_28_path}; can't refit "
                f"K=28 features on full 513 test set."
            )
            print(f"[deploy] {deploy_note}")
    else:
        deploy_note = "GATE_NOT_PASSED: no deploy CSV written."
        print(f"[deploy] {deploy_note}")

    summary = {
        "tag": TAG,
        "method": "residual_cascade_chemprop_aux_LGBM_K28_decileLGBM",
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "x_unb_28_path": str(X_UNB_28_PATH),
        "n_unb": n_unb,
        "n_splits": N_SPLITS,
        "seed": SEED,
        "n_deciles": N_DECILES,
        "decision_margin": DECISION_MARGIN,
        "baseline_ref": BASELINE_REF,
        "rae_stage1": rae_s1,
        "rae_stage1_plus_stage2": rae_s2,
        "rae_stage1_plus_s2_plus_s3": rae_final,
        "delta_final_vs_stage1": rae_final - rae_s1,
        "delta_final_vs_stage2": rae_final - rae_s2,
        "delta_final_vs_baseline": rae_final - BASELINE_REF,
        "beats_baseline": bool(beats),
        "flat_vs_baseline": bool(flat),
        "verdict": verdict,
        "stage2_fold_records": fold_records,
        "stage3_fold_records": s3_records,
        "fold_sizes_tr_va": [list(t) for t in fold_sizes],
        "n_with_scaffold": int(n_with_scaffold),
        "deploy_csv_path": str(deploy_csv_path) if deploy_csv_path else None,
        "deploy_note": deploy_note,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_stage1",
        "rae_stage1_plus_stage2",
        "rae_stage1_plus_s2_plus_s3",
        "delta_final_vs_baseline",
        "beats_baseline",
        "verdict",
        "deploy_csv_path",
        "deploy_note",
    ):
        print(f"  {k}: {res.get(k)}")
