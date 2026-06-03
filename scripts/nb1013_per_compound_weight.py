"""nb1013 -- Per-compound MLP blend weight between chemprop_aux and nb972.

Hypothesis: a global SLSQP weight (e.g., w_chem~=0.76 from nb1001) is a
single scalar applied to every compound.  Local chemistry may want a
different mix --- some scaffolds may favour chemprop_aux's GNN inductive
bias while others may favour nb972's gradient-boosted descriptors.

Procedure on the 253 unblind:
  - Featurize 253 SMILES with Morgan(2048) + RDKit(~217) = ~2265 dims.
  - Standardize features (fit on train fold only).
  - Train a small MLP (D -> 64 -> 16 -> 1, sigmoid) per fold, that
    outputs w_chem in [0, 1] per compound.
  - Loss: |y - (w*chemprop_aux + (1-w)*nb972)|   (MAE-on-blend).
  - 5-fold cross-fit.  Held-out predictions get the held-out fold's w.
  - Pooled cross-fit RAE on 253.
  - Compare to nb1001 (global SLSQP + stretch) honest cross-fit baseline.

Deploy: refit MLP on all 253, apply to 513 te files.
Outputs:
  data/processed/te_nb1013.npy
  data/processed/nb1013_summary.json
  submissions/nb1013_per_compound_weight.csv
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

from pxr.data import load_test
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1013"
CANDIDATES = ["chemprop_aux", "nb972_long_train"]  # [chem, nb972]
N_FOLDS = 5
SEED = 42
EPOCHS = 200
LR = 3e-3
WD = 1e-4
HIDDEN1 = 64
HIDDEN2 = 16
NB1001_CROSSFIT = 0.5594  # reference; updated from nb1001 summary if present


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


class WeightMLP(nn.Module):
    def __init__(self, d_in: int, h1: int = HIDDEN1, h2: int = HIDDEN2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)  # (N,) in (0,1)


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu, sd


def train_one_fold(
    X_tr: np.ndarray, p_chem_tr: np.ndarray, p_nb972_tr: np.ndarray,
    y_tr: np.ndarray, epochs: int, seed: int,
) -> WeightMLP:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = WeightMLP(d_in=X_tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    Xt = torch.from_numpy(X_tr.astype(np.float32))
    pc = torch.from_numpy(p_chem_tr.astype(np.float32))
    pn = torch.from_numpy(p_nb972_tr.astype(np.float32))
    yt = torch.from_numpy(y_tr.astype(np.float32))
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        w = model(Xt)
        blend = w * pc + (1.0 - w) * pn
        loss = torch.mean(torch.abs(yt - blend))
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def predict_w(model: WeightMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(X.astype(np.float32))).cpu().numpy()


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-compound MLP blend weight (chemprop_aux vs nb972)")
    print("=" * 78)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].tolist()
    preds_513 = np.column_stack([load_te(c, te_names) for c in CANDIDATES])
    print(f"[load] preds_513 shape = {preds_513.shape}")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] P_unb shape = {P_unb.shape}  y shape = {y_unb.shape}")

    # ---- Featurize all 513 (so we can also predict deploy w on 513) ----
    print("[feat] computing combined features for 513 ...")
    X_513 = impute(combined(te_smiles))
    X_unb = X_513[unb_idx]
    print(f"[feat] X_513 = {X_513.shape}  X_unb = {X_unb.shape}")

    # ---- Individual sanity ----
    indiv_rae = {c: float(rae(y_unb, P_unb[:, j]))
                 for j, c in enumerate(CANDIDATES)}
    for c, r in indiv_rae.items():
        print(f"   {c:30s}: in_RAE = {r:.4f}")

    # =================================================================
    # 5-fold cross-fit
    # =================================================================
    print("\n" + "-" * 78)
    print(f"5-FOLD CROSS-FIT  (MLP D->64->16->1 sigmoid, "
          f"epochs={EPOCHS}, lr={LR})")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_w = np.full(n_unb, np.nan)
    oof_blend = np.full(n_unb, np.nan)
    fold_rows = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        # standardize on train fold
        mu, sd = standardize_fit(X_unb[tr_loc])
        X_tr = (X_unb[tr_loc] - mu) / sd
        X_va = (X_unb[va_loc] - mu) / sd
        model = train_one_fold(
            X_tr, P_unb[tr_loc, 0], P_unb[tr_loc, 1], y_unb[tr_loc],
            epochs=EPOCHS, seed=SEED + k,
        )
        w_va = predict_w(model, X_va)
        blend_va = w_va * P_unb[va_loc, 0] + (1.0 - w_va) * P_unb[va_loc, 1]
        oof_w[va_loc] = w_va
        oof_blend[va_loc] = blend_va
        rae_va = float(rae(y_unb[va_loc], blend_va))
        fold_rows.append({
            "fold": k, "n_va": int(len(va_loc)),
            "w_mean": float(w_va.mean()), "w_std": float(w_va.std()),
            "w_min": float(w_va.min()), "w_max": float(w_va.max()),
            "val_rae": rae_va,
        })
        print(f"   fold {k}: w_mean={w_va.mean():.3f} "
              f"(min={w_va.min():.2f}, max={w_va.max():.2f})  "
              f"val_RAE={rae_va:.4f}")

    pooled_rae = float(rae(y_unb, oof_blend))
    print(f"\n[cv] pooled honest cross-fit RAE on 253 = {pooled_rae:.4f}")
    print(f"   per-compound w_chem: mean={oof_w.mean():.3f}  "
          f"std={oof_w.std():.3f}  range=[{oof_w.min():.2f},"
          f"{oof_w.max():.2f}]")

    # =================================================================
    # Comparison
    # =================================================================
    delta = pooled_rae - NB1001_CROSSFIT
    print("\n" + "-" * 78)
    print(f"COMPARISON vs nb1001 honest cross-fit (~{NB1001_CROSSFIT:.4f})")
    print("-" * 78)
    print(f"   nb1013 per-compound MLP   = {pooled_rae:.4f}")
    print(f"   nb1001 global SLSQP+s     = {NB1001_CROSSFIT:.4f}")
    print(f"   delta                     = {delta:+.4f}")
    if delta < -0.005:
        verdict = "WIN"
    elif delta < 0.005:
        verdict = "TIE"
    else:
        verdict = "LOSE_TO_GLOBAL"
    print(f"   verdict                   = {verdict}")

    # =================================================================
    # Deploy: refit on all 253, predict w on all 513
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (refit MLP on all 253, predict w on 513)")
    print("-" * 78)
    mu, sd = standardize_fit(X_unb)
    X_unb_n = (X_unb - mu) / sd
    X_513_n = (X_513 - mu) / sd
    model = train_one_fold(
        X_unb_n, P_unb[:, 0], P_unb[:, 1], y_unb,
        epochs=EPOCHS, seed=SEED,
    )
    w_513 = predict_w(model, X_513_n)
    deploy_513 = (w_513 * preds_513[:, 0]
                  + (1.0 - w_513) * preds_513[:, 1]).astype(np.float32)
    w_unb_deploy = predict_w(model, X_unb_n)
    in_rae = float(rae(
        y_unb,
        w_unb_deploy * P_unb[:, 0] + (1.0 - w_unb_deploy) * P_unb[:, 1],
    ))
    print(f"   deploy w_513 mean/std   = {w_513.mean():.3f} / {w_513.std():.3f}")
    print(f"   deploy w_513 range      = [{w_513.min():.3f}, {w_513.max():.3f}]")
    print(f"   te(513) mean/std        = "
          f"{deploy_513.mean():.3f} / {deploy_513.std():.3f}")
    print(f"   in-sample RAE 253       = {in_rae:.4f}  (overfit lower bound)")

    # =================================================================
    # Save
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_per_compound_weight.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    summary = {
        "tag": TAG,
        "candidates": CANDIDATES,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "epochs": EPOCHS,
        "lr": LR,
        "wd": WD,
        "hidden": [HIDDEN1, HIDDEN2],
        "indiv_in_rae": indiv_rae,
        "fold_results": fold_rows,
        "pooled_cv_rae_253": pooled_rae,
        "oof_w_mean": float(oof_w.mean()),
        "oof_w_std": float(oof_w.std()),
        "oof_w_min": float(oof_w.min()),
        "oof_w_max": float(oof_w.max()),
        "nb1001_crossfit_ref": NB1001_CROSSFIT,
        "delta_vs_nb1001": float(delta),
        "verdict": verdict,
        "deploy_w_513_mean": float(w_513.mean()),
        "deploy_w_513_std": float(w_513.std()),
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "in_sample_rae_overfit_bound": in_rae,
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pool                     = {CANDIDATES}")
    print(f"   honest cross-fit RAE 253 = {pooled_rae:.4f}")
    print(f"   nb1001 ref               = {NB1001_CROSSFIT:.4f}")
    print(f"   delta                    = {delta:+.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   per-compound w mean/std  = "
          f"{oof_w.mean():.3f} / {oof_w.std():.3f}")
    print(f"   wall                     = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("pooled_cv_rae_253", "delta_vs_nb1001", "verdict",
              "oof_w_mean", "oof_w_std", "in_sample_rae_overfit_bound",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
