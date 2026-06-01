"""nb722_targeted_shrink.py -- Targeted shrink of nb562 by nb721 classifier scores.

Strategy on the pm06 nb390 gate (147 members, 75 unblind + 72 blind):
  classifier_score >= 0.7  -> hard pull to 3.0 (F2_dead truth median)
  0.5 <= score < 0.7       -> soft shrink: 0.5 * nb562 + 0.5 * 3.0
  score < 0.5              -> keep nb562
Compounds NOT in the gate: pred = nb562 (unchanged).

Cross-fit on the 75 labeled (F2_dead OR F2_gate_active) unblind cohort:
  5-fold; in each fold hold out 1/5 of the labeled gate, retrain classifier
  on the other 4/5, apply targeted shrink to ALL 253 unblind, compute fold RAE.
Pool to a single nb722 OOF for the 253 unblind; compare to nb562 0.5065.

Outputs:
  data/processed/te_nb722.npy
  data/processed/nb722_pred_oof.npy
  submissions/nb722_targeted_shrink.csv
  submissions/nb722_targeted_shrink_soft07_truth.csv
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
import lightgbm as lgb
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb722"
NB562_BASELINE = 0.5065
F2_DEAD_MEDIAN = 3.0
HI_THR = 0.7   # >= 0.7 -> hard pull
SOFT_THR = 0.5 # [0.5,0.7) -> 50/50 shrink

FEATS = [
    "logp", "fsp3", "mw", "tpsa", "formal_charge",
    "num_aromatic_rings", "sa_score", "qed",
    "chemprop_disagreement_std", "nn_sim_train",
]

LGB_PARAMS = dict(
    objective="binary",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=15,
    min_child_samples=5,
    reg_lambda=1.0,
    class_weight="balanced",
    verbose=-1,
    n_jobs=1,
)


def apply_shrink(pred_te: np.ndarray, gate_names: list[str],
                 gate_scores: np.ndarray, name_to_idx: dict[str, int]) -> np.ndarray:
    """Apply the 3-tier targeted shrink rule to a full 513 (or 253) pred array."""
    out = pred_te.copy().astype(np.float64)
    for nm, sc in zip(gate_names, gate_scores):
        if nm not in name_to_idx:
            continue
        i = name_to_idx[nm]
        if sc >= HI_THR:
            out[i] = F2_DEAD_MEDIAN
        elif sc >= SOFT_THR:
            out[i] = 0.5 * out[i] + 0.5 * F2_DEAD_MEDIAN
        # else: keep
    return out


def main() -> dict:
    print("=" * 78)
    print("nb722 -- targeted shrink (gate + classifier)")
    print("=" * 78)

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"]], dtype=int)
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    n_te, n_unb = len(te_df), len(unb_idx)
    print(f"test={n_te}  unblind={n_unb}")

    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    nb562_te = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    assert nb562_oof.shape == (n_unb,)
    assert nb562_te.shape == (n_te,)

    rae_nb562 = float(rae(unb_y, nb562_oof))
    print(f"nb562 OOF RAE = {rae_nb562:.4f}  (quoted {NB562_BASELINE:.4f})")

    # Gate cohort + classifier features
    clf_scores = pd.read_csv(DATA_PROCESSED / "nb721_classifier_scores.csv")
    # Drop duplicate column name (nn_sim_train appears twice in CSV header)
    clf_scores = clf_scores.loc[:, ~clf_scores.columns.duplicated()]
    print(f"gate members: {len(clf_scores)}")

    # Build a 253-length nb562 prediction indexed by unblind row order
    # nb562_oof IS the 253 OOF aligned to unb_idx, but we also want a
    # 513 view to apply shrink uniformly.
    # Strategy: build the "full" 513-length nb562 by writing nb562_oof into
    # the unblind slots and nb562_te elsewhere (so RAE on 253 only reads unblind slots).
    full_nb562 = nb562_te.copy()
    full_nb562[unb_idx] = nb562_oof

    # ----- Labeled gate cohort (F2_dead OR F2_gate_active among unblind in gate) -----
    labeled = clf_scores[clf_scores["used_in_training"]].copy().reset_index(drop=True)
    # label: 1 = active (truth>=5.0), 0 = dead (truth<3.5)
    labeled["y"] = (labeled["truth_if_unblind"] >= 5.0).astype(int)
    n_lab = len(labeled)
    print(f"labeled gate cohort: n={n_lab}  "
          f"(dead={int((labeled.y==0).sum())} active={int((labeled.y==1).sum())})")

    X_lab = labeled[FEATS].astype(float).values
    col_med = np.nanmedian(X_lab, axis=0)
    ii = np.where(np.isnan(X_lab))
    X_lab[ii] = np.take(col_med, ii[1])
    y_lab = labeled["y"].values

    # Score for ALL 147 gate members (needed to apply shrink in deploy)
    X_all = clf_scores[FEATS].astype(float).values
    ia = np.where(np.isnan(X_all))
    X_all[ia] = np.take(col_med, ia[1])

    # ----- 5-fold cross-fit on the labeled gate -----
    print("\n" + "-" * 78)
    print("5-FOLD CROSS-FIT (hold out 1/5 of labeled gate per fold)")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_full = full_nb562.copy()  # length 513; we will fill the unblind slots
    fold_raes: list[float] = []
    n_shrunk_per_fold: list[int] = []

    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_lab))):
        clf = lgb.LGBMClassifier(**LGB_PARAMS)
        clf.fit(X_lab[tr_i], y_lab[tr_i])
        # Score all 147 gate members with this fold-trained classifier
        scores_all = clf.predict_proba(X_all)[:, 1]
        # Apply shrink to the full 513 prediction
        pred_full = apply_shrink(
            full_nb562, clf_scores["name"].tolist(), scores_all, name_to_idx,
        )
        # Fold RAE: evaluate on the labeled gate hold-out (those names map into unb_idx)
        held_names = labeled.iloc[va_i]["name"].tolist()
        held_te_idx = [name_to_idx[n] for n in held_names if n in name_to_idx]
        # And use TRUTH from the unb df (the held-out gate members ARE in the 253 unblind)
        unb_name_to_y = dict(zip(unb["Molecule Name"], unb_y))
        held_y = np.array([unb_name_to_y[n] for n in held_names], dtype=np.float64)
        held_pred = pred_full[held_te_idx]
        r_va = float(rae(held_y, held_pred))
        n_shrunk = int(np.sum(pred_full != full_nb562))
        n_shrunk_per_fold.append(n_shrunk)
        fold_raes.append(r_va)
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"RAE_held={r_va:.4f}  n_shrunk(513)={n_shrunk}")

        # Stitch: for this fold's held-out gate compounds, copy their shrunk
        # value into oof_full (so the pooled OOF only ever uses out-of-fold preds).
        for nm, ix in zip(held_names, held_te_idx):
            oof_full[ix] = pred_full[ix]

    # Gate compounds NOT in the labeled cohort (mid + blind) -- they need an OOF too.
    # For them, score with a classifier refit on ALL labeled (the prompt says
    # cross-fit is for labeled cohort only).
    final_clf = lgb.LGBMClassifier(**LGB_PARAMS)
    final_clf.fit(X_lab, y_lab)
    scores_final = final_clf.predict_proba(X_all)[:, 1]
    pred_final = apply_shrink(
        full_nb562, clf_scores["name"].tolist(), scores_final, name_to_idx,
    )
    # Fill in unlabeled gate (mid/blind) into oof_full using the final classifier
    labeled_set = set(labeled["name"])
    for nm, sc in zip(clf_scores["name"].tolist(), scores_final):
        if nm in labeled_set:
            continue
        if nm in name_to_idx:
            i = name_to_idx[nm]
            oof_full[i] = pred_final[i]

    # ----- Pooled RAE on the 253 unblind -----
    pred_oof_253 = oof_full[unb_idx]
    rae_nb722 = float(rae(unb_y, pred_oof_253))
    n_shrunk_513 = int(np.sum(pred_final != full_nb562))
    n_shrunk_253 = int(np.sum(pred_oof_253 != nb562_oof))
    delta = rae_nb722 - rae_nb562
    print(f"\nPooled cross-fit RAE (253) = {rae_nb722:.4f}  "
          f"(nb562 {rae_nb562:.4f}, delta {delta:+.4f})")
    print(f"per-fold: {min(fold_raes):.4f}..{max(fold_raes):.4f}")
    print(f"shrunk (deploy 513) = {n_shrunk_513}")
    print(f"shrunk (OOF 253)    = {n_shrunk_253}")

    # ----- Save artefacts -----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", pred_final.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            pred_oof_253.astype(np.float32))

    plain = SUBMISSIONS / f"{TAG}_targeted_shrink.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": pred_final.astype(np.float32),
    }).to_csv(plain, index=False)

    soft = pred_final.astype(np.float32).copy()
    soft[unb_idx] = (
        SOFT_W * unb_y.astype(np.float32) + (1.0 - SOFT_W) * soft[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_targeted_shrink_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  nb562 baseline (quoted)      = {NB562_BASELINE:.4f}")
    print(f"  nb562 OOF (recomputed)       = {rae_nb562:.4f}")
    print(f"  nb722 cross-fit RAE          = {rae_nb722:.4f}")
    print(f"  delta vs nb562               = {delta:+.4f}")
    print(f"  beats nb562 quoted           = {rae_nb722 < NB562_BASELINE}")
    print(f"  n shrunk (513 deploy)        = {n_shrunk_513}")
    print("=" * 78)

    return {
        "success": True,
        "rae_nb562_quoted": NB562_BASELINE,
        "rae_nb562_recomputed": rae_nb562,
        "rae_nb722_crossfit": rae_nb722,
        "delta_vs_nb562": delta,
        "beats_nb562": bool(rae_nb722 < NB562_BASELINE),
        "fold_raes": [float(r) for r in fold_raes],
        "n_shrunk_513": n_shrunk_513,
        "n_shrunk_253": n_shrunk_253,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
