"""nb2561 -- L0 sparse network (Hard Concrete) linear meta-learner on K=20.

NEW PARADIGM:
    Stochastic L0 gates on linear weights via the Hard Concrete relaxation
    (Louizos et al., 2018 "Learning Sparse Neural Networks through L0
    Regularization").  Per-weight learnable log_alpha controls a stretched
    Concrete distribution which is hard-clipped to [0, 1]; the gate's
    expected-active probability gives a *differentiable* L0 surrogate that
    can be added directly to the MSE objective.

    Sits on the same fixed substrate tuple as nb2550 (chemprop_aux anchor +
    K=20 from nb2240's RFE).  Compared to ElasticNetCV's L1+L2 shrinkage,
    L0 Hard Concrete chooses *which* coefficients to keep non-zero (rather
    than shrinking all of them toward zero), which on the (20, 253)
    residual problem may be the right inductive bias if only a small subset
    of K=20 features carries signal in the chemprop_aux residual axis.

ARCHITECTURE:
    nn.Linear(20 -> 1) with per-weight Hard Concrete gates.
    Gate sample (training):
        u ~ Uniform(eps, 1 - eps)
        s = sigmoid((log(u) - log(1 - u) + log_alpha) / beta)
        s_bar = s * (zeta - gamma) + gamma   # stretch from [0,1] -> [gamma, zeta]
        z = clip(s_bar, 0, 1)                # hard concrete
        w_eff = w * z
    Gate sample (eval): z = clip(sigmoid(log_alpha / beta) * (zeta-gamma) + gamma, 0, 1)
    L0 regulariser:
        P(gate active) = sigmoid(log_alpha - beta * log(-gamma / zeta))
        L0 = lambda * sum(P_active)
    Loss = MSE(residual, w_eff @ x + b) + L0

TRAIN:
    Adam(lr=1e-3), 1000 epochs, full-batch, lambda_L0 = 0.01
    beta=2/3, gamma=-0.1, zeta=1.1, eps=1e-6 (Hard Concrete defaults)

PROTOCOL:
    1. Load X_117 (test 513) -> slice to K=20 indices from nb2240.
    2. residual = y_unb - chemprop_aux[unb_idx]   (only PRE-clean anchor)
    3. Per scaffold-fold: StandardScaler fit on train fold -> transform val.
    4. Train L0 net 1000 epochs on (X_tr, residual_tr); predict val with
       eval-time deterministic gate.
    5. 5-fold scaffold CV on 253 unblind, 5 kf_seeds {1001..1005}.
       (scaffold_kfold_indices per cv-protocol-audit memo.)
    6. Aggregate per-seed corrected RAE -> mean-bag.
    7. Deploy: refit L0 net on full 253 per seed, predict 513 residual,
       mean-bag.

GATE (on mean-bag corrected RAE):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4601 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2561_l0_sparse_network.py
    data/processed/nb2561_summary.json
    data/processed/nb2561_pred_oof.npy   (253,) float32 mean-bag CORRECTED
    data/processed/te_nb2561.npy         (513,) float32 deploy refit
"""
from __future__ import annotations

import json
import math
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
from sklearn.preprocessing import StandardScaler

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2561"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

# Training hyperparameters (per task spec)
LR = 1e-3
N_EPOCHS = 1000
LAMBDA_L0 = 0.01

# Hard Concrete defaults (Louizos et al., 2018)
BETA = 2.0 / 3.0
GAMMA = -0.1
ZETA = 1.1
EPS = 1e-6
# log alpha initialisation: weakly-active gates so net starts with most
# weights "on" and the L0 term gradually deactivates them.
LOG_ALPHA_INIT_MEAN = 0.0
LOG_ALPHA_INIT_STD = 0.01

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # LGBM K=20 mean-bag reference

DEVICE = torch.device("cpu")  # CPU-only per pyproject default
TORCH_SEED_BASE = 7919


# ============================================================================
# Hard Concrete L0 linear layer
# ============================================================================
class L0Linear(nn.Module):
    """nn.Linear with per-weight Hard Concrete L0 gates.

    weight: (out_features, in_features)
    log_alpha: same shape as weight, trainable
    bias: optional (no gate -- always-on)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        beta: float = BETA,
        gamma: float = GAMMA,
        zeta: float = ZETA,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.eps = eps

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32)
        )
        nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))

        self.log_alpha = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32)
        )
        nn.init.normal_(
            self.log_alpha,
            mean=LOG_ALPHA_INIT_MEAN,
            std=LOG_ALPHA_INIT_STD,
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        # Precompute the constant log(-gamma/zeta) used by the L0 closed form.
        self._log_neg_gamma_over_zeta = math.log(-gamma / zeta)

    def _sample_gate(self) -> torch.Tensor:
        """Sample Hard Concrete gates (training-time stochastic path)."""
        u = torch.empty_like(self.log_alpha).uniform_(self.eps, 1.0 - self.eps)
        # logits of Concrete: log(u/(1-u))
        logit_u = torch.log(u) - torch.log1p(-u)
        s = torch.sigmoid((logit_u + self.log_alpha) / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, 0.0, 1.0)
        return z

    def _deterministic_gate(self) -> torch.Tensor:
        """Deterministic eval-time gate (no noise)."""
        s = torch.sigmoid(self.log_alpha / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        z = torch.clamp(s_bar, 0.0, 1.0)
        return z

    def l0_regulariser(self) -> torch.Tensor:
        """Expected number of active gates (differentiable L0 surrogate)."""
        p_active = torch.sigmoid(
            self.log_alpha - self.beta * self._log_neg_gamma_over_zeta
        )
        return p_active.sum()

    def expected_n_active(self) -> float:
        with torch.no_grad():
            p_active = torch.sigmoid(
                self.log_alpha - self.beta * self._log_neg_gamma_over_zeta
            )
            return float(p_active.sum().item())

    def n_active_deterministic(self, tol: float = 1e-3) -> int:
        with torch.no_grad():
            z = self._deterministic_gate()
            return int((z > tol).sum().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            z = self._sample_gate()
        else:
            z = self._deterministic_gate()
        w_eff = self.weight * z
        return F.linear(x, w_eff, self.bias)


# ============================================================================
# Train / predict helpers
# ============================================================================
def _train_l0(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    feat_dim: int,
    seed: int,
) -> L0Linear:
    """Train an L0Linear(20 -> 1) on residual with full-batch Adam."""
    torch.manual_seed(seed + TORCH_SEED_BASE)
    np.random.seed(seed + TORCH_SEED_BASE)

    model = L0Linear(feat_dim, 1, bias=True).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    xt = torch.from_numpy(X_tr.astype(np.float32)).to(DEVICE)
    yt = torch.from_numpy(y_tr.astype(np.float32)).reshape(-1, 1).to(DEVICE)

    model.train()
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        pred = model(xt)
        mse = F.mse_loss(pred, yt)
        l0 = model.l0_regulariser()
        loss = mse + LAMBDA_L0 * l0
        loss.backward()
        opt.step()

    return model


def _predict_l0(model: L0Linear, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        out = model(xt).cpu().numpy().reshape(-1)
    return out.astype(np.float64)


def _scaffold_cv_one_seed(
    X: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """One scaffold-CV pass: fit per-fold StandardScaler + L0 net."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_diags = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_loc]).astype(np.float32)
        X_va = scaler.transform(X[va_loc]).astype(np.float32)
        mdl = _train_l0(X_tr, residual[tr_loc], X_tr.shape[1], seed=kf_seed)
        oof[va_loc] = _predict_l0(mdl, X_va)
        fold_diags.append({
            "fold": fold_i,
            "expected_n_active": mdl.expected_n_active(),
            "n_active_det": mdl.n_active_deterministic(),
            "bias": float(mdl.bias.detach().cpu().item()),
        })
    return oof, fold_diags


def _deploy_te_one_seed(
    X_unb: np.ndarray,
    residual: np.ndarray,
    X_te: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Fit StandardScaler + L0 net on full 253, predict 513 residual."""
    scaler = StandardScaler()
    X_unb_s = scaler.fit_transform(X_unb).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)
    mdl = _train_l0(X_unb_s, residual, X_unb_s.shape[1], seed=seed)
    pred = _predict_l0(mdl, X_te_s).astype(np.float32)
    diag = {
        "expected_n_active": mdl.expected_n_active(),
        "n_active_det": mdl.n_active_deterministic(),
        "bias": float(mdl.bias.detach().cpu().item()),
    }
    return pred, diag


# ============================================================================
# MAIN
# ============================================================================
def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- L0 sparse network (Hard Concrete) on K=20 substrate")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold")
    print(f"        kf_seeds = {KF_SEEDS}")
    print(f"        L0Linear(20 -> 1) + bias; Adam(lr={LR}); "
          f"epochs={N_EPOCHS}; lambda_L0={LAMBDA_L0}")
    print(f"        HC params: beta={BETA:.4f} gamma={GAMMA} zeta={ZETA}")
    print(f"        per-fold StandardScaler (fit-on-train-only)")
    print(f"        GATE: mean_rae < {GATE_PROMOTE} PROMOTE; "
          f"< {GATE_MARGINAL} MARGINAL_BEAT; else FAIL")
    print("=" * 78)

    # ---- Load truth + anchor ----
    te = load_test()
    n_test = len(te)
    test_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"anchor te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor te shape {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load X_117 substrate + slice to K=20 ----
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"X_117 cache missing: {X117_UNB_PATH} or {X117_TE_PATH}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_unb.shape != (n_unb, 117):
        raise ValueError(f"X117_unb shape {X117_unb.shape}")
    if X117_te.shape != (n_test, 117):
        raise ValueError(f"X117_te shape {X117_te.shape}")
    X117_unb = np.where(np.isfinite(X117_unb), X117_unb, 0.0).astype(np.float32)
    X117_te = np.where(np.isfinite(X117_te), X117_te, 0.0).astype(np.float32)
    print(f"[feat] X117_unb = {X117_unb.shape}  X117_te = {X117_te.shape}")

    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"nb2240 summary missing: {NB2240_SUMMARY}")
    with open(NB2240_SUMMARY) as f:
        nb2240 = json.load(f)
    k20_idx = list(nb2240["k20_surviving_idx_in_117"])
    k20_names = list(nb2240["k20_surviving_names"])
    assert len(k20_idx) == 20, f"expected 20 K=20 indices, got {len(k20_idx)}"
    print(f"[K20] loaded {len(k20_idx)} surviving indices from nb2240")

    X_unb = X117_unb[:, k20_idx].astype(np.float32)
    X_te = X117_te[:, k20_idx].astype(np.float32)
    feat_dim = X_unb.shape[1]
    assert feat_dim == 20, f"feat_dim {feat_dim} != 20"
    print(f"[feat] X_unb (K=20) = {X_unb.shape}  X_te = {X_te.shape}")

    # ---- Scaffolds ----
    unb_smiles = [test_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"5-SEED SCAFFOLD-CV  seeds={KF_SEEDS}  folds={N_FOLDS}  dim={feat_dim}")
    print("-" * 78)
    per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_fold_diags: list[list[dict]] = []
    per_seed_deploy_diag: list[dict] = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        resid_oof, fold_diags = _scaffold_cv_one_seed(
            X_unb, residual, unb_scaffolds, seed,
        )
        per_seed_oof_resid[i] = resid_oof
        per_seed_fold_diags.append(fold_diags)
        te_resid, deploy_diag = _deploy_te_one_seed(X_unb, residual, X_te, seed)
        per_seed_te_resid[i] = te_resid
        per_seed_deploy_diag.append(deploy_diag)
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        mean_exp_nact = float(np.mean([d["expected_n_active"] for d in fold_diags]))
        mean_det_nact = float(np.mean([d["n_active_det"] for d in fold_diags]))
        print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
              f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
              f"exp_act~{mean_exp_nact:.1f}/{feat_dim}  "
              f"det_act~{mean_det_nact:.1f}/{feat_dim}  "
              f"deploy_det_act={deploy_diag['n_active_det']}/{feat_dim}  "
              f"wall={time.time() - ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_resid = per_seed_oof_resid.mean(axis=0)
    median_bag_resid = np.median(per_seed_oof_resid, axis=0)
    rae_mean_bag = float(rae(y_unb, anchor + mean_bag_resid))
    rae_median_bag = float(rae(y_unb, anchor + median_bag_resid))

    print("\n[cv] per_seed_mean RAE = "
          f"{per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag   RAE = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE = {rae_median_bag:.4f}")
    print(f"[cv] anchor     RAE = {rae_anchor:.4f}  "
          f"(d_mean_bag = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"[cv] reference  nb2240 K=20 LGBM mean-bag = {NB2240_K20_REF:.4f}  "
          f"(d = {rae_mean_bag - NB2240_K20_REF:+.4f})")

    # ---- Deploy te (mean-bag corrected) ----
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_deploy = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}  "
          f"(deploy refit, in-sample optimism expected)")

    # ---- Save artefacts ----
    pred_oof_corrected = (anchor + mean_bag_resid).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    # ---- Gate ----
    if rae_mean_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_mean_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_bag_rae        = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = "
          f"{rae_mean_bag < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{rae_mean_bag < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": ("l0_sparse_network_hard_concrete_linear_on_X117_K20_"
                   "residual_on_chemprop_aux"),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "x117_unb_path": str(X117_UNB_PATH),
        "x117_te_path": str(X117_TE_PATH),
        "k20_idx_source": str(NB2240_SUMMARY),
        "k20_surviving_idx_in_117": [int(j) for j in k20_idx],
        "k20_surviving_names": k20_names,
        "n_unb": n_unb,
        "n_test": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "feat_dim": int(feat_dim),
        "model_class": "L0Linear (Hard Concrete) -- nn.Linear(20 -> 1) with L0 gates",
        "l0_params": {
            "lr": LR,
            "n_epochs": N_EPOCHS,
            "lambda_l0": LAMBDA_L0,
            "beta": BETA,
            "gamma": GAMMA,
            "zeta": ZETA,
            "eps": EPS,
            "log_alpha_init_mean": LOG_ALPHA_INIT_MEAN,
            "log_alpha_init_std": LOG_ALPHA_INIT_STD,
            "scaler": "StandardScaler (per-fold fit-on-train)",
            "device": str(DEVICE),
            "torch_seed_base": TORCH_SEED_BASE,
        },
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,  # alias for gate consumers
        "delta_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_vs_nb2240_K20_lgbm": rae_mean_bag - NB2240_K20_REF,
        "nb2240_K20_lgbm_ref": NB2240_K20_REF,
        "per_seed_fold_diagnostics": per_seed_fold_diags,
        "per_seed_deploy_diagnostics": per_seed_deploy_diag,
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "pre_unblind_clean_anchor": True,
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
        "per_seed_rae",
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20_lgbm",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
