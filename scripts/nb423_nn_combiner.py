"""nb423 -- Small MLP combiner for weak-but-orthogonal predictors.

Input features per compound (11 dims):
  4 chemprop family : nb93, nb130, nb264, chemprop_aux
  3 strong orthog.  : nb417 (Tox21), nb418 (BindingDB), nb411 (counter)
  2 weak orthog.    : nb412 (USRCAT), nb413 (NR chrono)
  1 anchor          : nb320
  1 disagreement    : chemprop_disagreement_std (from nb422)

Architecture:
  11 -> Linear(32) -> ReLU -> Dropout(0.3) -> Linear(16) -> ReLU -> Linear(1)
  L2 weight decay 1e-3.  MSE loss.  Adam.

Honest scoring:
  5-fold cross-fit on 253 unblind labels -> pooled RAE.

Deployment:
  Retrain on ALL 253 unblind, apply to full 513.
  Average across a small bag of seeds to reduce init variance.

Outputs:
  data/processed/te_nb423.npy
  submissions/nb423_nn_combiner.csv             (rules-safe)
  submissions/nb423_nn_combiner_soft07_truth.csv (truth-inject on 253 unblind)
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
PREDICTOR_FILES = [
    ("nb93_chemprop_large_gpu", "te_nb93_chemprop_large_gpu.npy"),
    ("nb130_external_pxr",      "te_nb130_external_pxr.npy"),
    ("nb264_chemprop_mt",       "te_nb264_chemprop_mt.npy"),
    ("chemprop_aux",            "te_chemprop_aux.npy"),
    ("nb417_tox21",             "te_nb417_tox21.npy"),
    ("nb418_bindingdb",         "te_nb418_external.npy"),
    ("nb411_counter_residual",  "te_nb411.npy"),
    ("nb412_usrcat",            "te_nb412.npy"),
    ("nb413_nr_chrono",         "te_nb413.npy"),
    ("nb320_anchor",            "te_nb320_phase2_top20.npy"),
]
DISAGREE_CSV = "nb422_compound_uncertainty.csv"

N_FOLDS = 5
N_SEEDS = 5            # bag for variance reduction
EPOCHS = 400
BATCH = 64
LR = 1e-3
WD = 1e-3              # L2 weight decay
HIDDEN1 = 32
HIDDEN2 = 16
DROPOUT = 0.3
SOFT_W = 0.7

NB400_RAE = 0.5698
NB420_RAE = 0.5759
TARGET = 0.55


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class Combiner(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Linear(HIDDEN2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(X_tr, y_tr, in_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Combiner(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict(model, X):
    with torch.no_grad():
        return model(torch.from_numpy(X).float()).numpy()


def bagged_predict(X_tr, y_tr, X_eval, in_dim, seeds):
    preds = np.zeros((len(seeds), X_eval.shape[0]), dtype=float)
    for i, s in enumerate(seeds):
        m = train_one(X_tr, y_tr, in_dim, s)
        preds[i] = predict(m, X_eval)
    return preds.mean(axis=0)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("nb423 -- NN COMBINER (weak-but-orthogonal MLP)")
    print("=" * 78)

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_te_idx = np.array([name_to_idx[n] for n in unb["Molecule Name"]
                           if n in name_to_idx])
    unb_y = unb["pEC50"].values.astype(float)
    print(f"unblind={len(unb_te_idx)}  still-blind={513 - len(unb_te_idx)}\n")

    # Load 10 predictors
    cols = []
    feat_names = []
    for name, fname in PREDICTOR_FILES:
        path = DATA_PROCESSED / fname
        if not path.exists():
            raise FileNotFoundError(path)
        arr = np.load(path).astype(float)
        assert arr.shape == (513,), (name, arr.shape)
        cols.append(arr)
        feat_names.append(name)

    # Load disagreement (11th feature)
    dis_df = pd.read_csv(DATA_PROCESSED / DISAGREE_CSV)
    dis_lookup = dict(zip(dis_df["compound_name"],
                          dis_df["chemprop_disagreement_std"]))
    dis_col = np.array([dis_lookup[n] for n in te_df["Molecule Name"]],
                       dtype=float)
    cols.append(dis_col)
    feat_names.append("chemprop_disagreement_std")

    X_full = np.column_stack(cols)        # (513, 11)
    in_dim = X_full.shape[1]
    print(f"Feature matrix: {X_full.shape}  features={feat_names}\n")

    # Standardize per-feature (fit on unblind only -> avoid leakage from blind)
    X_unb = X_full[unb_te_idx]
    mu = X_unb.mean(axis=0)
    sd = X_unb.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    X_full_n = (X_full - mu) / sd
    X_unb_n = X_full_n[unb_te_idx]

    # Per-predictor honest unblind RAE (sanity)
    print("Per-feature honest unblind RAE:")
    for i, n in enumerate(feat_names):
        if n == "chemprop_disagreement_std":
            print(f"  {n:32s} (uncertainty input; no RAE)")
            continue
        r = rae(unb_y, X_full[unb_te_idx, i])
        print(f"  {n:32s} RAE={r:.4f}")
    print()

    # ---------------- 5-fold cross-fit ----------------
    n = len(unb_y)
    rng = np.random.RandomState(0)
    idx = rng.permutation(n)
    folds = np.array_split(idx, N_FOLDS)
    held = np.full(n, np.nan)
    fold_raes = []
    seeds = list(range(N_SEEDS))
    for k in range(N_FOLDS):
        val = folds[k]
        trn = np.concatenate([folds[j] for j in range(N_FOLDS) if j != k])
        p_val = bagged_predict(X_unb_n[trn], unb_y[trn],
                               X_unb_n[val], in_dim, seeds)
        held[val] = p_val
        fr = rae(unb_y[val], p_val)
        fold_raes.append(fr)
        print(f"  fold {k}: n_val={len(val)}  RAE={fr:.4f}")
    crossfit_rae = rae(unb_y, held)
    print(f"\nPer-fold RAE: {np.round(fold_raes, 4).tolist()}")
    print(f"Cross-fitted (pooled) RAE = {crossfit_rae:.4f}")

    # ---------------- Refit on ALL 253 unblind, predict full 513 ----------------
    print("\nRefitting on full 253 unblind (bag of seeds) ...")
    deploy = bagged_predict(X_unb_n, unb_y, X_full_n, in_dim, seeds)
    insample_pred = deploy[unb_te_idx]
    insample_rae = rae(unb_y, insample_pred)
    gap = crossfit_rae - insample_rae
    print(f"In-sample (unblind) RAE = {insample_rae:.4f}")
    print(f"Gap (crossfit - insample) = {gap:+.4f}")

    # ---------------- Write outputs ----------------
    out_safe = SUBMISSIONS / "nb423_nn_combiner.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb423_nn_combiner_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(out_soft, index=False)

    np.save(DATA_PROCESSED / "te_nb423.npy", deploy)

    print(f"\nWrote {out_safe}")
    print(f"Wrote {out_soft}")
    print(f"Wrote te_nb423.npy  std={deploy.std():.3f}  "
          f"min={deploy.min():.3f}  max={deploy.max():.3f}")

    print(f"\n=== CROSSFIT RAE: {crossfit_rae:.4f}  "
          f"(nb400={NB400_RAE}, nb420={NB420_RAE}, target<{TARGET}) ===")
    print(f"=== GAP         : {gap:+.4f}  (sane if <0.03) ===")
    beats_nb400 = crossfit_rae < NB400_RAE
    if crossfit_rae < TARGET:
        print("\n*** FRONTIER BROKEN ***")
    elif beats_nb400:
        print("\nBeats nb400 baseline but did not break <0.55 frontier.")
    else:
        print("\nDid not beat nb400 baseline.")

    return {
        "crossfit_rae": float(crossfit_rae),
        "insample_rae": float(insample_rae),
        "gap": float(gap),
        "beats_nb400": bool(beats_nb400),
        "frontier_broken": bool(crossfit_rae < TARGET),
    }


if __name__ == "__main__":
    main()
