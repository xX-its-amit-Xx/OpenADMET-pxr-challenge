"""nb1113 -- Meta MLP stacker on 253 unblind.

Hypothesis: a small MLP (8 -> 16 -> 1, ReLU, Dropout 0.3) can capture
non-linear interactions between heterogeneous predictors (chemprop,
nb972 long-train, ridge, kNN, Mordred LGBM) that linear/tree stackers
miss, while three physchem context features (logP, MW, TPSA) let the
network gate predictor trust by chemistry regime.

Risk: 8 -> 16 -> 1 has roughly 8*16 + 16 + 16 + 1 = 161 weights, fitting
on ~200 training-fold rows. We mitigate with Huber loss, L2 1e-4, dropout
0.3, and early stopping on a held-out validation slice within each fold.

Procedure:
  1. Build 8-column feature matrix on 513 (5 predictors + logP/MW/TPSA).
  2. Slice to 253 unblind rows.
  3. 5-fold KFold (shuffle=True, random_state=0) cross-fit on the 253.
     Within each fold: split train 80/20, train MLP with early stopping
     (patience 20, max 300 epochs), predict the validation fold.
  4. Pooled RAE on the assembled OOF vector.

Outputs:
  data/processed/nb1113_summary.json
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
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from pxr.chem import compute_physchem
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1113_meta_mlp"
N_FOLDS = 5
SEED = 0
HIDDEN = 16
DROPOUT = 0.3
L2 = 1e-4
LR = 1e-3
MAX_EPOCHS = 300
PATIENCE = 20
VAL_FRAC = 0.20
HUBER_DELTA = 1.0
NB1070_REFERENCE = 0.5771


class MetaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HIDDEN, dropout: float = DROPOUT) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x).squeeze(-1)


def build_features() -> tuple[np.ndarray, list[str]]:
    """Stack 5 predictors + 3 physchem -> (513, 8)."""
    te = load_test()
    smiles = te["smiles"].tolist()

    chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    ridge = np.load(DATA_PROCESSED / "te_nb1101.npy").astype(np.float64)
    knn = np.load(DATA_PROCESSED / "te_nb1104.npy").astype(np.float64)
    mordred = np.load(DATA_PROCESSED / "te_nb1030.npy").astype(np.float64)

    logp = np.zeros(len(smiles), dtype=np.float64)
    mw = np.zeros(len(smiles), dtype=np.float64)
    tpsa = np.zeros(len(smiles), dtype=np.float64)
    for i, smi in enumerate(smiles):
        pc = compute_physchem(smi)
        if pc is None:
            continue
        logp[i] = float(pc["logp"])
        mw[i] = float(pc["mw"])
        tpsa[i] = float(pc["tpsa"])

    X = np.stack([chemprop, nb972, ridge, knn, mordred, logp, mw, tpsa], axis=1)
    cols = ["chemprop_aux", "nb972", "ridge_nb1101", "knn_nb1104",
            "mordred_nb1030", "logp", "mw", "tpsa"]
    return X.astype(np.float32), cols


def train_one_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_va: np.ndarray, seed: int) -> np.ndarray:
    """Train MLP on X_tr with internal 80/20 early-stopping split, predict X_va."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_tr = len(X_tr)
    n_val_internal = max(2, int(round(VAL_FRAC * n_tr)))
    perm = np.random.permutation(n_tr)
    val_idx = perm[:n_val_internal]
    fit_idx = perm[n_val_internal:]

    Xf = torch.from_numpy(X_tr[fit_idx]).float()
    yf = torch.from_numpy(y_tr[fit_idx]).float()
    Xv = torch.from_numpy(X_tr[val_idx]).float()
    yv = torch.from_numpy(y_tr[val_idx]).float()
    Xq = torch.from_numpy(X_va).float()

    model = MetaMLP(in_dim=X_tr.shape[1])
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    loss_fn = nn.HuberLoss(delta=HUBER_DELTA)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    bad = 0
    for _ in range(MAX_EPOCHS):
        model.train()
        optim.zero_grad()
        pred = model(Xf)
        loss = loss_fn(pred, yf)
        loss.backward()
        optim.step()

        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(Xv), yv).item())
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = model(Xq).cpu().numpy().astype(np.float64)
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- small MLP stacker (8 -> 16 -> 1, dropout 0.3, Huber)")
    print("=" * 78)

    X, cols = build_features()
    print(f"\n[feat] X shape = {X.shape}  cols = {cols}")
    for j, c in enumerate(cols):
        v = X[:, j]
        print(f"  [{j}] {c:>20s}  mean={v.mean():.3f}  std={v.std():.3f}  "
              f"min={v.min():.3f}  max={v.max():.3f}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    X_unb = X[unb_idx]
    n = len(y_unb)
    print(f"\n[load] unb n={n}  X_unb shape={X_unb.shape}")

    # Standardize features using unblind-fold-train stats per fold (no leakage).
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT  (KFold shuffle, seed={SEED})")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n, np.nan, dtype=np.float64)
    per_fold = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        X_tr_raw = X_unb[tr_loc]
        y_tr = y_unb[tr_loc].astype(np.float32)
        X_va_raw = X_unb[va_loc]

        mu = X_tr_raw.mean(axis=0)
        sd = X_tr_raw.std(axis=0)
        sd[sd < 1e-8] = 1.0
        X_tr = ((X_tr_raw - mu) / sd).astype(np.float32)
        X_va = ((X_va_raw - mu) / sd).astype(np.float32)

        pred_va = train_one_fold(X_tr, y_tr, X_va, seed=SEED + k)
        oof[va_loc] = pred_va
        fold_rae = float(rae(y_unb[va_loc], pred_va))
        per_fold.append(fold_rae)
        print(f"  fold {k}: n_tr={len(tr_loc):3d}  n_va={len(va_loc):3d}  "
              f"RAE={fold_rae:.4f}")

    pooled = float(rae(y_unb, oof))
    print("\n" + "-" * 78)
    print(f"[result] pooled cross-fit RAE = {pooled:.4f}")
    print(f"[result] per-fold mean        = {np.mean(per_fold):.4f}  "
          f"std={np.std(per_fold):.4f}")

    # Reference comparisons
    chemprop_in = float(rae(y_unb, X_unb[:, 0].astype(np.float64)))
    nb972_in = float(rae(y_unb, X_unb[:, 1].astype(np.float64)))
    print(f"\n[ref] chemprop_aux in-sample 253 RAE = {chemprop_in:.4f}")
    print(f"[ref] nb972         in-sample 253 RAE = {nb972_in:.4f}")
    print(f"[ref] nb1070 reference (cross-fit)    = {NB1070_REFERENCE:.4f}")

    delta_vs_nb1070 = pooled - NB1070_REFERENCE
    beats_nb1070 = bool(pooled < NB1070_REFERENCE)
    if delta_vs_nb1070 < -0.005:
        verdict = "BEATS_NB1070"
    elif abs(delta_vs_nb1070) <= 0.005:
        verdict = "TIES_NB1070"
    else:
        verdict = "WORSE_THAN_NB1070"
    print(f"[verdict] delta vs nb1070 = {delta_vs_nb1070:+.4f}  -> {verdict}")

    summary = {
        "tag": TAG,
        "feature_cols": cols,
        "n_features": int(X.shape[1]),
        "hidden": HIDDEN,
        "dropout": DROPOUT,
        "l2": L2,
        "lr": LR,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "val_frac": VAL_FRAC,
        "huber_delta": HUBER_DELTA,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "n_unb": int(n),
        "per_fold_rae": per_fold,
        "per_fold_mean": float(np.mean(per_fold)),
        "per_fold_std": float(np.std(per_fold)),
        "pooled_rae": pooled,
        "chemprop_in_rae": chemprop_in,
        "nb972_in_rae": nb972_in,
        "nb1070_reference": NB1070_REFERENCE,
        "delta_vs_nb1070": delta_vs_nb1070,
        "beats_nb1070": beats_nb1070,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / "nb1113_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[wall] {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("n_features", "per_fold_rae", "per_fold_mean", "pooled_rae",
              "chemprop_in_rae", "nb972_in_rae", "delta_vs_nb1070",
              "beats_nb1070", "verdict"):
        print(f"  {k}: {res.get(k)}")
