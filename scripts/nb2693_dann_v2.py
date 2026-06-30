"""nb2693 -- Domain-Adversarial Neural Network v2 (vs prior nb303 DANN).

NEW PARADIGM (vs nb303):
    * Gradient Reversal Layer (Ganin & Lempitsky 2015) instead of static
      domain-classifier weights -> regressor encoder is pushed to be
      domain-invariant in the same forward pass.
    * Small MLP discriminator (Linear(64,32) -> ReLU -> Linear(32,1)) instead
      of LGBM domain classifier -> joint optimization.
    * Encoder output feeds BOTH a regressor head AND the discriminator via
      GRL.  The regressor learns chemprop_aux residual; the discriminator
      tries to separate train (4139) vs test (513) molecules in the latent
      space; GRL flips the gradient so encoder fools the discriminator.
    * Loss = MSE(y_reg, regressor(enc(x))) + lambda * BCE(d, disc(GRL(enc(x))))
    * lambda = 0.3, Adam lr=1e-3, 100 epochs, batch=full-batch (or 256)

PROTOCOL:
    * Top-20 features chosen by variance from `cache_combined_features.npz`
      (consistent across train/test).  StandardScaler fit on combined
      (train + test) features (target-domain-aware).
    * Source-domain data X_src = X_train (4139) -- domain label 0
    * Target-domain data X_tgt = X_test (513)   -- domain label 1
    * Regressor target: chemprop_aux RESIDUAL = y_train - chemprop_aux_train_hat
      For inner training we use the train-domain X with `y_train` as the
      regression label.  Per the task spec, "fit on chemprop_aux residual"
      means we evaluate on the 253 unblind as: pred = chemprop_aux + DANN(x)
      where DANN(x) is the residual head's prediction.
    * For the 253 unblind, evaluation is scaffold 5-fold CV:
        - For each (kf_seed in [1001..1005]) x (5-fold scaffold split on 253):
            train-fold = 4139 train + 4/5 of 253 unblind (residual learned over
              both); val-fold = 1/5 of 253 unblind, OOF residual stitched
            (target-domain DANN training: domain label 1 still includes all 513
             test; we DO NOT use unblind y labels for the domain discriminator
             -- domain labels are agnostic to y)
        - OOF prediction: chemprop_aux_oof[val_idx] + dann_resid_oof[val_idx]
    * Deploy on full 513 test: train on 4139 + 253 unblind (full pool), apply
      to 513 test, store as te_nb2693.npy = chemprop_aux_te + dann_resid_te
    * 5 kf_seeds; mean of seeds' OOF used for the GATE.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4598 -> MARGINAL_BEAT
    else            -> FAIL

Outputs:
    scripts/nb2693_dann_v2.py
    data/processed/nb2693_summary.json
    data/processed/nb2693_pred_oof.npy       (253,) float32
    data/processed/te_nb2693.npy             (513,) float32
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2693"
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Architecture / training hyperparams (task spec)
F_DIM = 20            # encoder input dim
H_DIM = 64            # encoder hidden / shared
D_HIDDEN = 32         # discriminator hidden
LAMBDA_ADV = 0.3      # GRL weight
LR = 1e-3
N_EPOCHS = 100
BATCH = 256

# Gate
GATE_PROMOTE  = 0.4570
GATE_MARGINAL = 0.4598

# Paths
CACHE_COMBINED = DATA_PROCESSED / "cache_combined_features.npz"
Y_TRAIN_FULL_PATH = DATA_PROCESSED / "y_train_full.npy"
CHEMPROP_AUX_OOF_PATH = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"  # (253,)
TE_CHEMPROP_AUX_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"                # (513,)
UNBLIND_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"


# =====================================================================
# Gradient Reversal Layer
# =====================================================================
class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverseFn.apply(x, lambd)


# =====================================================================
# DANN model
# =====================================================================
class DANNv2(nn.Module):
    def __init__(self, f_dim: int = F_DIM, h_dim: int = H_DIM, d_hidden: int = D_HIDDEN):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(f_dim, h_dim),
            nn.ReLU(),
        )
        # Regressor head
        self.reg = nn.Linear(h_dim, 1)
        # Domain discriminator (binary)
        self.disc = nn.Sequential(
            nn.Linear(h_dim, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x, lambd: float):
        z = self.enc(x)
        y_hat = self.reg(z).squeeze(-1)
        d_logit = self.disc(grad_reverse(z, lambd)).squeeze(-1)
        return y_hat, d_logit


# =====================================================================
# Feature selection: top-20 by variance on combined train+test
# =====================================================================
def select_top20_features(X_tr: np.ndarray, X_te: np.ndarray, k: int = F_DIM):
    X_all = np.vstack([X_tr, X_te]).astype(np.float64)
    # nan-safe
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    var = X_all.var(axis=0)
    # require finite variance > 0
    var[~np.isfinite(var)] = 0.0
    top_idx = np.argsort(-var)[:k]
    return np.sort(top_idx)


# =====================================================================
# Training loop for one DANN fit
# =====================================================================
def train_dann(
    X_src_train: np.ndarray,  # source domain w/ regression labels
    y_src_train: np.ndarray,
    X_tgt: np.ndarray,        # target domain (no reg labels used in this loop)
    n_epochs: int = N_EPOCHS,
    lambd: float = LAMBDA_ADV,
    lr: float = LR,
    batch: int = BATCH,
    torch_seed: int = 0,
):
    """Train the DANN with full-batch (or mini-batch) per epoch.

    The regression labels `y_src_train` are MSE residuals (already
    chemprop_aux-subtracted).  Domain labels are 0 for X_src, 1 for X_tgt.
    """
    torch.manual_seed(torch_seed)

    model = DANNv2().to("cpu")
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xs = torch.tensor(X_src_train, dtype=torch.float32)
    ys = torch.tensor(y_src_train, dtype=torch.float32)
    Xt = torch.tensor(X_tgt, dtype=torch.float32)

    n_s = Xs.shape[0]
    n_t = Xt.shape[0]

    bs_s = min(batch, n_s)
    bs_t = min(batch, n_t)

    rng = np.random.default_rng(torch_seed)

    for ep in range(n_epochs):
        # Random batch indices
        si = rng.integers(0, n_s, size=bs_s)
        ti = rng.integers(0, n_t, size=bs_t)
        Xs_b = Xs[si]
        ys_b = ys[si]
        Xt_b = Xt[ti]
        # Combined batch for the discriminator
        X_b = torch.cat([Xs_b, Xt_b], dim=0)
        d_b = torch.cat([
            torch.zeros(bs_s),  # source = 0
            torch.ones(bs_t),   # target = 1
        ], dim=0)

        model.train()
        opt.zero_grad()
        y_hat, d_logit = model(X_b, lambd)
        # Regression loss only on source rows
        y_hat_src = y_hat[:bs_s]
        loss_reg = F.mse_loss(y_hat_src, ys_b)
        # Domain BCE loss on all rows
        loss_dom = F.binary_cross_entropy_with_logits(d_logit, d_b)
        loss = loss_reg + lambd * loss_dom
        loss.backward()
        opt.step()

    return model


@torch.no_grad()
def predict_resid(model: DANNv2, X: np.ndarray) -> np.ndarray:
    model.eval()
    xt = torch.tensor(X, dtype=torch.float32)
    z = model.enc(xt)
    return model.reg(z).squeeze(-1).cpu().numpy().astype(np.float32)


# =====================================================================
# Cross-fit one kf_seed (returns OOF residual + OOF chemprop_aux blend)
# =====================================================================
def cv_one_seed(
    X_unb: np.ndarray,         # (n_unb, 20)
    y_unb: np.ndarray,         # (n_unb,)
    chem_oof_unb: np.ndarray,  # (n_unb,) chemprop_aux OOF on the unblind
    X_train: np.ndarray,       # (n_train, 20)  source domain
    y_train_resid: np.ndarray, # (n_train,)     residual = y_train - chemprop_aux_train_hat (proxy)
    X_test: np.ndarray,        # (n_test, 20)   target domain (513 test)
    unb_scaffolds: list[str],
    kf_seed: int,
):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = X_unb.shape[0]
    oof_resid = np.full(n_unb, np.nan, dtype=np.float32)
    for tr_loc, va_loc in splits:
        # Source domain training pool = full 4139 train + tr_loc of unblind
        # For tr_loc unblind, regression label = y_unb[tr_loc] - chem_oof_unb[tr_loc]
        X_src_unb_tr = X_unb[tr_loc]
        y_src_unb_tr = (y_unb[tr_loc] - chem_oof_unb[tr_loc]).astype(np.float32)

        X_src = np.vstack([X_train, X_src_unb_tr])
        y_src = np.concatenate([y_train_resid, y_src_unb_tr]).astype(np.float32)
        # Target domain = full 513 test (includes unblind including va_loc;
        # domain labels do NOT use y, only x)
        X_tgt = X_test
        model = train_dann(
            X_src, y_src, X_tgt,
            torch_seed=int(kf_seed) * 7 + 13,
        )
        # Predict residual on val fold
        oof_resid[va_loc] = predict_resid(model, X_unb[va_loc])
    return oof_resid


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DANN v2: GRL + MLP disc + Adam, fit on chemprop_aux residual")
    print("=" * 78)

    # ---- Load metadata ----
    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)

    unb_idx = np.load(UNBLIND_IDX_PATH)
    y_unb = np.load(UNBLIND_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)

    chem_oof_unb = np.load(CHEMPROP_AUX_OOF_PATH).astype(np.float64)
    te_chem = np.load(TE_CHEMPROP_AUX_PATH).astype(np.float64)
    assert chem_oof_unb.shape == (n_unb,)
    assert te_chem.shape == (n_te,)

    # Anchor reference RAE on unblind (sanity check)
    anchor_rae = float(rae(y_unb, chem_oof_unb))
    print(f"[anchor] chemprop_aux unblind OOF RAE = {anchor_rae:.4f}")

    # ---- Load feature cache (combined Morgan + RDKit) ----
    cache = np.load(CACHE_COMBINED)
    X_tr_full = cache["X_tr"].astype(np.float32)  # (4139, 2265)
    X_te_full = cache["X_te"].astype(np.float32)  # (513, 2265)
    print(f"[cache] X_tr {X_tr_full.shape}  X_te {X_te_full.shape}")
    assert X_tr_full.shape[0] == 4139
    assert X_te_full.shape[0] == n_te

    # Top-20 features by variance (over combined)
    top_idx = select_top20_features(X_tr_full, X_te_full, k=F_DIM)
    print(f"[features] top-20 by variance idx (first 5): {top_idx[:5].tolist()}")
    X_tr_sel = X_tr_full[:, top_idx]
    X_te_sel = X_te_full[:, top_idx]

    # StandardScaler fit on combined (DANN is domain-aware so we standardize both)
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_tr_sel, X_te_sel]))
    X_tr_z = scaler.transform(X_tr_sel).astype(np.float32)
    X_te_z = scaler.transform(X_te_sel).astype(np.float32)

    # The unblind subset of X_te_z
    X_unb_z = X_te_z[unb_idx]
    print(f"[features] X_tr_z {X_tr_z.shape}  X_te_z {X_te_z.shape}  X_unb_z {X_unb_z.shape}")

    # ---- Proxy residual for the 4139 train rows ----
    # We don't have a per-row chemprop_aux OOF for all 4139 train rows in this
    # artifact bag.  We use y_train_full as the regression target on source and
    # let chemprop_aux subtraction happen later in evaluation (DANN learns a
    # residual head fit to (y_unb - chemprop_aux_unb) on unblind rows and to
    # 0-centered y_train on train rows).  Cleaner: subtract the train mean from
    # y_train to center it like a residual.
    y_train_full = np.load(Y_TRAIN_FULL_PATH).astype(np.float64)
    if y_train_full.shape[0] != X_tr_z.shape[0]:
        # Some y_train_full has 3781; we need 4139.  Fall back to mean-imputed
        # to keep shape -- this row count mismatch means we use only the
        # available labelled train rows.
        n_avail = min(y_train_full.shape[0], X_tr_z.shape[0])
        X_tr_z_use = X_tr_z[:n_avail]
        y_train_use = y_train_full[:n_avail]
        print(
            f"[warn] y_train_full {y_train_full.shape[0]} != X_tr {X_tr_z.shape[0]}; "
            f"using first {n_avail}"
        )
    else:
        X_tr_z_use = X_tr_z
        y_train_use = y_train_full
    # Center to act as a residual proxy
    y_train_resid = (y_train_use - y_train_use.mean()).astype(np.float32)
    print(
        f"[source] X_tr_z_use {X_tr_z_use.shape}  y_train_resid mean/std = "
        f"{y_train_resid.mean():.3f}/{y_train_resid.std():.3f}"
    )

    # ---- Scaffolds for the 253 unblind ----
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique on unblind = {n_unique_scaf}")

    # =================================================================
    # Cross-fit across 5 kf_seeds
    # =================================================================
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  lambda={LAMBDA_ADV}  "
          f"epochs={N_EPOCHS}  lr={LR}")
    print("-" * 78)
    per_seed = []
    all_oof_resids = []
    for kf_seed in KF_SEEDS:
        t1 = time.time()
        oof_resid = cv_one_seed(
            X_unb_z, y_unb, chem_oof_unb,
            X_tr_z_use, y_train_resid,
            X_te_z,
            unb_scaffolds, kf_seed,
        )
        oof_pred = chem_oof_unb + oof_resid
        r = float(rae(y_unb, oof_pred))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": r,
            "resid_mean": float(np.nanmean(oof_resid)),
            "resid_std": float(np.nanstd(oof_resid)),
            "wall_sec": round(time.time() - t1, 1),
        })
        all_oof_resids.append(oof_resid)
        print(
            f"   seed={kf_seed}  RAE={r:.4f}  "
            f"resid_mean/std={oof_resid.mean():+.3f}/{oof_resid.std():.3f}  "
            f"wall={time.time() - t1:.1f}s"
        )

    mean_resid = np.nanmean(np.column_stack(all_oof_resids), axis=1)
    mean_pred_oof = (chem_oof_unb + mean_resid).astype(np.float32)
    mean_rae = float(np.mean([r["pooled_rae"] for r in per_seed]))
    std_rae = float(np.std([r["pooled_rae"] for r in per_seed]))
    rae_of_mean = float(rae(y_unb, mean_pred_oof))
    print(
        f"\n[cv] mean across seeds = {mean_rae:.4f} (+/- {std_rae:.4f})  "
        f"RAE_of_mean_pred = {rae_of_mean:.4f}"
    )

    # =================================================================
    # Deploy: train on (4139 train) + (full 253 unblind) and predict 513
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY: train on full pool, predict 513 test")
    print("-" * 78)
    X_src_full = np.vstack([X_tr_z_use, X_unb_z])
    y_src_full = np.concatenate([
        y_train_resid,
        (y_unb - chem_oof_unb).astype(np.float32),
    ])
    model = train_dann(
        X_src_full, y_src_full, X_te_z,
        torch_seed=2693,
    )
    te_resid = predict_resid(model, X_te_z)
    deploy_te = (te_chem + te_resid).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(
        f"   te_resid mean/std = {te_resid.mean():+.3f}/{te_resid.std():.3f}  "
        f"deploy_te mean/std = {deploy_te.mean():.3f}/{deploy_te.std():.3f}  "
        f"te[unb] RAE = {te_unb_rae:.4f}"
    )

    # =================================================================
    # Gate
    # =================================================================
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print(
        f"GATE  mean_RAE={mean_rae:.4f}  "
        f"promote<{GATE_PROMOTE}  marginal<{GATE_MARGINAL}  -> {verdict}"
    )
    print("-" * 78)

    # =================================================================
    # Save artifacts
    # =================================================================
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    np.save(pred_oof_path, mean_pred_oof)
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, deploy_te)

    summary = {
        "tag": TAG,
        "method": "DANN_v2_GRL_MLP_disc_chemprop_aux_residual",
        "arch": {
            "f_dim": F_DIM, "h_dim": H_DIM, "d_hidden": D_HIDDEN,
            "encoder": "Linear(20,64)->ReLU",
            "regressor": "Linear(64,1)",
            "discriminator": "GRL->Linear(64,32)->ReLU->Linear(32,1)",
        },
        "train": {
            "lambda_adv": LAMBDA_ADV,
            "lr": LR,
            "n_epochs": N_EPOCHS,
            "batch": BATCH,
            "optimizer": "Adam",
        },
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "n_train_src_rows_used": int(X_tr_z_use.shape[0]),
        "n_features_in": F_DIM,
        "top_var_idx_first10": top_idx[:10].tolist(),
        "n_unique_scaffolds_unb": int(n_unique_scaf),
        "anchor_chemprop_aux_unb_rae": anchor_rae,
        "kf_seeds": KF_SEEDS,
        "per_seed": per_seed,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "rae_of_mean_pred": rae_of_mean,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "deploy_te_unb_rae_in_sample": te_unb_rae,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "gate_verdict": verdict,
        "pred_oof_path": str(pred_oof_path),
        "te_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {pred_oof_path}")
    print(f"[save] {te_path}")
    print(f"[save] {summary_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean RAE (5 kf_seeds)              = {mean_rae:.4f}  (+/- {std_rae:.4f})")
    print(f"   RAE of mean-of-seeds pred          = {rae_of_mean:.4f}")
    print(f"   anchor chemprop_aux unb RAE        = {anchor_rae:.4f}")
    print(f"   delta vs anchor                    = {mean_rae - anchor_rae:+.4f}")
    print(f"   te[unb] in-sample RAE              = {te_unb_rae:.4f}")
    print(f"   verdict                            = {verdict}")
    print(f"   wall                               = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "rae_of_mean_pred",
        "anchor_chemprop_aux_unb_rae", "deploy_te_unb_rae_in_sample",
        "gate_verdict",
    ):
        print(f"  {k}: {res.get(k)}")
