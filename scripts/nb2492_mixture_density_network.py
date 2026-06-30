"""nb2492 -- Mixture Density Network (3-component Gaussian) on chemprop_aux residual.

CONTEXT:
    Cycle 167+ post-hoc-blend ceiling on chemprop_aux anchor is 0.4682 (nb2171
    deep-30) and the K=20-anchored cousin nb2240 sits at OOF RAE 0.4630 on the
    253 unblind. This script tests whether a 3-component Gaussian Mixture
    Density Network on the X_117 feature set (per-component routing via
    pi_k softmax) can extract residual signal beyond the nb2240 anchor.

PROTOCOL:
    1. Load X_117 features + y_unb + nb2240 mean-bag OOF anchor.
    2. Residual = y_unb - nb2240_oof.
    3. 5-fold scaffold CV on 253 unb.
    4. MLP encoder Linear(117,256)->ReLU->Dropout(0.2)->Linear(256,128)->ReLU.
       Three heads (pi, mu, sigma) each Linear(128, 3). pi via softmax,
       sigma via softplus + eps.
    5. Loss = mean NLL of mixture density (logsumexp trick).
    6. Train 200 epochs, batch_size=32, Adam lr=1e-3 per fold.
    7. Inference: weighted-mean prediction = sum_k pi_k * mu_k (the residual).
    8. Final pred = nb2240 anchor + MDN residual.
    9. Gate vs 0.4570 (PROMOTE) / 0.4601 (MARGINAL_BEAT).
   10. Save nb2492_summary.json + pred_oof + te (513).

Outputs:
    scripts/nb2492_mixture_density_network.py
    data/processed/nb2492_summary.json
    data/processed/nb2492_pred_oof.npy   (253,) float32
    data/processed/te_nb2492.npy         (513,) float32
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2492"

# -----------------------------
# Config
# -----------------------------
N_FOLDS = 5
KF_SEED = 1001
N_EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3
N_MIX = 3
HID1 = 256
HID2 = 128
DROPOUT = 0.2
DEVICE = torch.device("cpu")
SIGMA_EPS = 1e-3

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

ANCHOR_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb2240.npy"
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"


# -----------------------------
# Model
# -----------------------------
class MDN(nn.Module):
    def __init__(self, in_dim: int, hid1: int, hid2: int, n_mix: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hid1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid1, hid2),
            nn.ReLU(),
        )
        self.head_pi = nn.Linear(hid2, n_mix)
        self.head_mu = nn.Linear(hid2, n_mix)
        self.head_sigma = nn.Linear(hid2, n_mix)

    def forward(self, x):
        h = self.encoder(x)
        pi_logits = self.head_pi(h)
        log_pi = F.log_softmax(pi_logits, dim=-1)
        mu = self.head_mu(h)
        sigma = F.softplus(self.head_sigma(h)) + SIGMA_EPS
        return log_pi, mu, sigma


def mdn_nll(log_pi, mu, sigma, y):
    # y shape (B,) -> (B, 1) broadcast across n_mix
    y_b = y.unsqueeze(-1)
    log_norm = -0.5 * torch.log(torch.tensor(2.0 * np.pi)) - torch.log(sigma)
    log_gauss = log_norm - 0.5 * ((y_b - mu) / sigma) ** 2
    log_mix = log_pi + log_gauss  # (B, K)
    nll = -torch.logsumexp(log_mix, dim=-1)
    return nll.mean()


def mdn_mean(log_pi, mu):
    pi = torch.exp(log_pi)
    return (pi * mu).sum(dim=-1)


def train_one_fold(X_tr, y_tr, X_va, X_te_full, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MDN(in_dim=X_tr.shape[1], hid1=HID1, hid2=HID2, n_mix=N_MIX, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    X_tr_t = torch.from_numpy(X_tr.astype(np.float32))
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
    X_va_t = torch.from_numpy(X_va.astype(np.float32))
    X_te_t = torch.from_numpy(X_te_full.astype(np.float32))

    ds = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(N_EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            log_pi, mu, sigma = model(xb)
            loss = mdn_nll(log_pi, mu, sigma, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        log_pi_va, mu_va, _ = model(X_va_t)
        pred_va = mdn_mean(log_pi_va, mu_va).cpu().numpy().astype(np.float64)
        log_pi_te, mu_te, _ = model(X_te_t)
        pred_te = mdn_mean(log_pi_te, mu_te).cpu().numpy().astype(np.float64)
    return pred_va, pred_te


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 3-comp MDN on chemprop_aux/nb2240 residual")
    print("=" * 78)

    # --- Load test set, scaffolds, truth, anchor ---
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_test={n_test}  n_unb={n_unb}  unique_scaf={n_uniq_scaf}")

    X_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X_te = np.load(X117_TE_PATH).astype(np.float32)
    print(f"[feat] X_unb={X_unb.shape}  X_te={X_te.shape}")
    assert X_unb.shape == (n_unb, 117) and X_te.shape == (n_test, 117)

    # Anchor (nb2240) on 253 + full 513
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
    assert anchor_oof.shape == (n_unb,) and anchor_te.shape == (n_test,)
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[anchor] nb2240 oof RAE = {rae_anchor:.4f}")

    residual = (y_unb - anchor_oof).astype(np.float64)
    print(f"[resid] mean={residual.mean():.4f}  std={residual.std():.4f}")

    # Per-fold feature standardisation (fit on train only)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )

    pred_resid_oof = np.zeros(n_unb, dtype=np.float64)
    pred_resid_te_per_fold = np.zeros((N_FOLDS, n_test), dtype=np.float64)
    per_fold_log = []
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV (seed={KF_SEED})")
    print("-" * 78)
    for f_i, (tr_loc, va_loc) in enumerate(splits):
        ts = time.time()
        mu_tr = X_unb[tr_loc].mean(axis=0)
        sd_tr = X_unb[tr_loc].std(axis=0)
        sd_tr = np.where(sd_tr < 1e-6, 1.0, sd_tr)
        X_tr_n = (X_unb[tr_loc] - mu_tr) / sd_tr
        X_va_n = (X_unb[va_loc] - mu_tr) / sd_tr
        X_te_n = (X_te - mu_tr) / sd_tr

        y_tr = residual[tr_loc]
        pred_va, pred_te = train_one_fold(
            X_tr_n, y_tr, X_va_n, X_te_n, seed=KF_SEED + f_i,
        )
        pred_resid_oof[va_loc] = pred_va
        pred_resid_te_per_fold[f_i] = pred_te
        rae_fold = float(rae(y_unb[va_loc], anchor_oof[va_loc] + pred_va))
        per_fold_log.append({
            "fold": int(f_i),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "rae_corrected_va": rae_fold,
            "resid_pred_mean": float(pred_va.mean()),
            "resid_pred_std": float(pred_va.std()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(
            f"   fold {f_i}  n_tr={len(tr_loc):3d}  n_va={len(va_loc):3d}  "
            f"rae_corr={rae_fold:.4f}  wall={time.time()-ts:.1f}s"
        )

    final_oof = (anchor_oof + pred_resid_oof).astype(np.float64)
    mean_rae = float(rae(y_unb, final_oof))
    print(f"\n[oof] final RAE (anchor + MDN residual) = {mean_rae:.4f}")
    print(f"[oof] anchor alone                     = {rae_anchor:.4f}")
    print(f"[oof] delta                            = {mean_rae - rae_anchor:+.4f}")

    # Deploy te = anchor_te + mean of per-fold te residuals
    mean_resid_te = pred_resid_te_per_fold.mean(axis=0)
    te_deploy = (anchor_te + mean_resid_te).astype(np.float32)
    te_unb_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"[te] te_deploy mean={te_deploy.mean():.3f}  std={te_deploy.std():.3f}")
    print(f"[te] te[unb_idx] RAE = {te_unb_rae:.4f}")

    # Gate decision
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_rae       = {mean_rae:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # Save artefacts
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, final_oof.astype(np.float32))
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "3-comp MDN on chemprop_aux/nb2240 residual; weighted-mean inference",
        "anchor_name": "nb2240_mean_bag_oof_K20",
        "rae_anchor": rae_anchor,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim": int(X_unb.shape[1]),
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "n_epochs": N_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "n_mix": N_MIX,
        "hid1": HID1,
        "hid2": HID2,
        "dropout": DROPOUT,
        "per_fold": per_fold_log,
        "mean_rae": mean_rae,
        "delta_vs_anchor": mean_rae - rae_anchor,
        "te_unb_rae_in_sample": te_unb_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae       = {mean_rae:.4f}")
    print(f"   anchor_rae     = {rae_anchor:.4f}")
    print(f"   delta          = {mean_rae - rae_anchor:+.4f}")
    print(f"   te_unb_rae     = {te_unb_rae:.4f}")
    print(f"   verdict        = {verdict}")
    print(f"   wall           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "rae_anchor",
        "delta_vs_anchor",
        "te_unb_rae_in_sample",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
