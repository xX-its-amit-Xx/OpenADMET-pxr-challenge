"""nb494 -- MLP RESIDUAL ROUTER.

Hypothesis: nb481's shallow LGBM (max_depth=3, n_est=80) may miss non-linear
feature interactions in the 33-feat x 253-row regime. A small MLP with two
hidden layers can in principle capture these — provided we regularize hard
(Dropout 0.3, weight_decay 1e-3) and standardize inputs.

Pipeline (mirrors nb481 except the residual learner):
  1. Reuse nb481's 33-column extended feature matrix (verbatim builder).
  2. Target: residual = truth - nb432  on 253 unblind rows.
  3. MLP: 33 -> 64 -> 32 -> 1, ReLU + Dropout 0.3.
     AdamW(lr=1e-3, weight_decay=1e-3), 100 epochs, early stop patience=15
     on an internal 20% holdout (per fold).
  4. 5-fold KFold cross-fit -> honest residual_oof (253,).
  5. Sigmoid err_hat gate identical to nb472/nb481:
       alpha_i = sigmoid((err_hat_i - median(err_hat)) * 4.0)
  6. pred_oof = nb432[unb_idx] + alpha_i * resid_oof  -> unblind RAE.
  7. Deploy: 5-seed average MLP refit on all 253 -> resid_hat_513.
  8. Save te_nb494.npy + pred_oof + plain + soft07_truth submissions.

Target: cross-fit unblind RAE < 0.5349 (nb481).

CPU-only torch is fine — model is tiny (~2.4k params).
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
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Reuse nb481's feature matrix builder verbatim.
sys.path.insert(0, os.path.dirname(__file__))
from nb481_residual_router_extended import build_feature_matrix, MISSING_LOG  # noqa: E402

from pxr.eval import rae  # noqa: E402
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS  # noqa: E402

SEED = 0
N_FOLDS = 5
SCALE = 4.0
SOFT_W = 0.7

# MLP / training config
EPOCHS = 100
PATIENCE = 15
LR = 1e-3
WD = 1e-3
DROPOUT = 0.3
HIDDEN1, HIDDEN2 = 64, 32
INTERNAL_VAL_FRAC = 0.20
N_DEPLOY_SEEDS = 5

torch.set_num_threads(2)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class ResidualMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN1),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int,
) -> tuple[ResidualMLP, StandardScaler]:
    """Standardize, split internal val, train with early stopping, return best."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_tr))
    n_val = max(2, int(round(len(X_tr) * INTERNAL_VAL_FRAC)))
    val_idx = idx[:n_val]
    fit_idx = idx[n_val:]

    scaler = StandardScaler()
    Xs_fit = scaler.fit_transform(X_tr[fit_idx]).astype(np.float32)
    Xs_val = scaler.transform(X_tr[val_idx]).astype(np.float32)
    y_fit = y_tr[fit_idx].astype(np.float32)
    y_val = y_tr[val_idx].astype(np.float32)

    torch.manual_seed(seed)
    mdl = ResidualMLP(in_dim=X_tr.shape[1])
    opt = torch.optim.AdamW(mdl.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()

    Xf_t = torch.from_numpy(Xs_fit)
    yf_t = torch.from_numpy(y_fit)
    Xv_t = torch.from_numpy(Xs_val)
    yv_t = torch.from_numpy(y_val)

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in mdl.state_dict().items()}
    bad = 0
    for ep in range(EPOCHS):
        mdl.train()
        opt.zero_grad()
        pred = mdl(Xf_t)
        loss = loss_fn(pred, yf_t)
        loss.backward()
        opt.step()
        mdl.eval()
        with torch.no_grad():
            vloss = float(loss_fn(mdl(Xv_t), yv_t))
        if vloss < best_val - 1e-6:
            best_val = vloss
            best_state = {k: v.detach().clone() for k, v in mdl.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    mdl.load_state_dict(best_state)
    mdl.eval()
    return mdl, scaler


def predict_mlp(mdl: ResidualMLP, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        return mdl(torch.from_numpy(Xs)).numpy().astype(np.float32)


def main() -> dict:
    print("=" * 78)
    print("nb494 -- MLP RESIDUAL ROUTER  (33 -> 64 -> 32 -> 1, Dropout 0.3)")
    print("=" * 78)

    needed = {
        "te_nb432.npy": DATA_PROCESSED / "te_nb432.npy",
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING (required):", missing)
        return {"success": False, "missing": missing}

    nb432 = np.load(needed["te_nb432.npy"]).astype(np.float32)
    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    n_te = nb432.shape[0]

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}

    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_unb = len(unb_te_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---------- Reuse nb481's 33-col feature matrix ----------
    print("\nBuilding feature matrix (reusing nb481 builder):")
    X, feat_names = build_feature_matrix(te_df, nb432)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    print(f"  X shape: {X.shape}  cols={len(feat_names)}")
    if MISSING_LOG:
        print("  Missing feature sources (filled 0):")
        for f in MISSING_LOG:
            print(f"    - {f}")

    X_unb = X[unb_te_idx]
    y_resid = (unb_y - nb432[unb_te_idx]).astype(np.float32)
    print(f"\nResidual: mean={y_resid.mean():+.3f} std={y_resid.std():.3f} "
          f"|mean|={np.abs(y_resid).mean():.3f}")

    # ---------- 5-fold cross-fit MLP ----------
    print(f"\n{N_FOLDS}-fold cross-fit MLP (epochs<= {EPOCHS}, patience={PATIENCE}):")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    resid_oof = np.zeros(n_unb, dtype=np.float32)
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mdl, sc = train_mlp(X_unb[tr_i], y_resid[tr_i], seed=SEED + fold)
        resid_oof[va_i] = predict_mlp(mdl, sc, X_unb[va_i])
        print(f"  fold {fold}: n_tr={len(tr_i)} n_va={len(va_i)}  "
              f"resid_oof mean={resid_oof[va_i].mean():+.3f} "
              f"std={resid_oof[va_i].std():.3f}")

    rho_resid, _ = spearmanr(resid_oof, y_resid)
    rho_abs, _ = spearmanr(np.abs(resid_oof), np.abs(y_resid))
    rho_resid = float(rho_resid) if np.isfinite(rho_resid) else 0.0
    rho_abs = float(rho_abs) if np.isfinite(rho_abs) else 0.0
    print(f"\nSpearman(resid_oof, true resid)       = {rho_resid:.4f}")
    print(f"Spearman(|resid_oof|, |true resid|)  = {rho_abs:.4f}")

    # ---------- Deploy: 5-seed avg MLP refit on all 253 ----------
    print(f"\nDeploy: {N_DEPLOY_SEEDS}-seed average MLP refit on all 253:")
    deploy_preds = []
    for s in range(N_DEPLOY_SEEDS):
        mdl, sc = train_mlp(X_unb, y_resid, seed=100 + s)
        p = predict_mlp(mdl, sc, X)
        deploy_preds.append(p)
        print(f"  seed {s}: deploy resid mean={p.mean():+.3f}  std={p.std():.3f}")
    resid_hat_513 = np.mean(deploy_preds, axis=0).astype(np.float32)
    print(f"  ENSEMBLE: mean={resid_hat_513.mean():+.3f}  "
          f"std={resid_hat_513.std():.3f}")

    # ---------- Soft gate ----------
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)
    print(f"\nGate alpha (513): min={alpha_513.min():.3f}  "
          f"med={np.median(alpha_513):.3f}  max={alpha_513.max():.3f}  "
          f"mean={alpha_513.mean():.3f}")

    deploy = (nb432 + alpha_513 * resid_hat_513).astype(np.float32)
    print(f"\nDeploy nb494: mean={deploy.mean():.3f} std={deploy.std():.3f}  "
          f"|delta vs nb432|.mean = {np.abs(deploy - nb432).mean():.3f}")

    # ---------- Honest cross-fit unblind RAE ----------
    nb494_unb_oof = (nb432[unb_te_idx] + alpha_513[unb_te_idx] * resid_oof
                     ).astype(np.float32)
    rae_nb432 = float(rae(unb_y, nb432[unb_te_idx]))
    rae_nb494_oof = float(rae(unb_y, nb494_unb_oof))
    rae_nb494_insample = float(rae(unb_y, deploy[unb_te_idx]))

    NB481_TARGET = 0.5349
    beats_nb481 = rae_nb494_oof < NB481_TARGET
    beats_nb432 = rae_nb494_oof < rae_nb432

    print("\nUnblind RAE (n=253):")
    print(f"  nb432 baseline             = {rae_nb432:.4f}")
    print(f"  nb494 cross-fit (honest)   = {rae_nb494_oof:.4f}")
    print(f"  nb494 in-sample (refit)    = {rae_nb494_insample:.4f}")
    print(f"  beats nb481 (<0.5349)      = {beats_nb481}")
    print(f"  beats nb432                = {beats_nb432}")

    # ---------- Save ----------
    np.save(DATA_PROCESSED / "te_nb494.npy", deploy)
    np.save(DATA_PROCESSED / "te_nb494_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / "te_nb494_alpha.npy", alpha_513)
    np.save(DATA_PROCESSED / "nb494_pred_oof.npy", nb494_unb_oof)
    np.save(DATA_PROCESSED / "nb494_resid_oof.npy", resid_oof)

    plain = SUBMISSIONS / "nb494_mlp_residual_router.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    soft_path = SUBMISSIONS / "nb494_mlp_residual_router_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb494.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb494_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb494 SUMMARY ===")
    print(f"  n_features                          = {len(feat_names)}")
    print(f"  spearman(resid_oof, true)           = {rho_resid:.4f}")
    print(f"  spearman(|resid_oof|, |true|)       = {rho_abs:.4f}")
    print(f"  unblind RAE nb432                    = {rae_nb432:.4f}")
    print(f"  unblind RAE nb494 cross-fit          = {rae_nb494_oof:.4f}")
    print(f"  unblind RAE nb494 in-sample          = {rae_nb494_insample:.4f}")
    print(f"  beats nb481 (<0.5349)                = {beats_nb481}")
    print("=" * 78)

    return {
        "success": True,
        "n_features": len(feat_names),
        "spearman_resid": rho_resid,
        "spearman_abs_resid": rho_abs,
        "rae_nb432_unb": rae_nb432,
        "unblind_rae_nb494_oof": rae_nb494_oof,
        "unblind_rae_nb494_insample": rae_nb494_insample,
        "beats_nb481": bool(beats_nb481),
        "beats_nb432": bool(beats_nb432),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
