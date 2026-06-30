"""nb2821 -- Denoising autoencoder for K=20 feature smoothing on chemprop_aux residual.

NEW PARADIGM:
    Denoising autoencoders (Vincent et al. 2008) learn a robust low-dimensional
    encoding by reconstructing CLEAN inputs from a corrupted (noisy) version.
    The training objective forces the bottleneck to capture only the signal
    structure that survives random Gaussian perturbation -- effectively a
    learned, non-linear feature smoother.

    Distinction from prior tabular DL attacks on this anchor:
      - MLP/NN heads (cycle-134): supervised end-to-end, every input neuron
        feeds every output; no unsupervised denoising objective.
      - TabNet (nb2811): supervised sparse attention masks, no reconstruction.
      - LGBM K=20 (nb2103/nb2112/nb2240): hard splits on RAW features, never
        any latent smoothing.
      - DAE here: UNSUPERVISED denoising pretraining on pooled (4139+513=4652)
        feature rows; bottleneck encoding then fed to downstream LGBM on the
        253 unblind residual. The AE only ever sees X, never sees y.

    Hypothesis: high-frequency feature noise (bit-flips, Mordred outliers)
    drives part of the K=20 LGBM variance ceiling at n=253. Smoothing K=20 ->
    8-d denoised latent stabilises the downstream tree splits.

PROTOCOL:
    1. Load X_117_unb (253, 117), X_117_te (513, 117), X_117_train (4139, 117).
       Slice first K=20 cols on all three.
    2. Pool train+test features (4139+513 = 4652, 20) for AE training; per-col
       standardise to zero-mean / unit-var.
    3. AE arch: Encoder Linear(20,16)->ReLU->Linear(16,8), Decoder
       Linear(8,16)->ReLU->Linear(16,20). Adam lr=1e-3, MSE reconstruction.
       Per training step add Gaussian noise sigma=0.1 to input, target is the
       clean (un-noised) standardised input. 200 epochs, batch 64.
    4. Encode all 253 + 513 rows through bottleneck -> 8-d smoothed features.
    5. Anchor: chemprop_aux (PRE-unblind, verified clean). Residual = y - anchor.
    6. LGBM(max_depth=4, num_leaves=15, n_estimators=300, lr=0.03) on 8-d
       latent, fit residual under 5-fold scaffold CV across 5 kf_seeds
       {1001..1005}. Final per-row pred = anchor + resid_hat (clip 3.0-8.0).
    7. Deploy: refit AE+LGBM on all 253 -> predict 513.

GATE:
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

Outputs:
    scripts/nb2821_autoencoder_denoise.py
    data/processed/nb2821_summary.json
    data/processed/nb2821_pred_oof.npy   (253,) float32
    data/processed/te_nb2821.npy         (513,) float32
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
import lightgbm as lgb

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as e:  # noqa: BLE001
    print("INSTALL_FAILED")
    print(f"reason: {type(e).__name__}: {e}")
    sys.exit(0)

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2821"

# --------------------------------------------------------------------------
# Autoencoder hyperparameters (per spec)
# --------------------------------------------------------------------------
AE_IN_DIM = 20
AE_HID = 16
AE_BOTTLE = 8
AE_NOISE_SIGMA = 0.1
AE_LR = 1e-3
AE_EPOCHS = 200
AE_BATCH = 64
AE_SEED = 42

# Downstream LGBM
LGBM_PARAMS = dict(
    objective="regression",
    max_depth=4,
    num_leaves=15,
    n_estimators=300,
    learning_rate=0.03,
    min_child_samples=5,
    reg_lambda=2.0,
    n_jobs=2,
    verbosity=-1,
)

# CV protocol
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Paths
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
X117_TRAIN_PATH = DATA_PROCESSED / "pyramid" / "X_117_train.npy"
TE_CHEM_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

K_SLICE = 20


class DenoisingAE(nn.Module):
    def __init__(self, in_dim: int = AE_IN_DIM, hid: int = AE_HID,
                 bottle: int = AE_BOTTLE):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, bottle),
        )
        self.dec = nn.Sequential(
            nn.Linear(bottle, hid),
            nn.ReLU(),
            nn.Linear(hid, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def _standardise(X_train: np.ndarray, *others: np.ndarray):
    """Per-column standardisation using mean/std from X_train (pooled fit-set).
    Returns (X_train_std, *others_std, mu, sigma). NaN-safe."""
    Xt = np.where(np.isfinite(X_train), X_train, np.nan).astype(np.float64)
    mu = np.nanmean(Xt, axis=0)
    sd = np.nanstd(Xt, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)

    def _apply(X):
        Y = np.where(np.isfinite(X), X, mu).astype(np.float64)
        return ((Y - mu) / sd).astype(np.float32)

    return (_apply(X_train),) + tuple(_apply(o) for o in others) + (mu, sd)


def _train_dae(X_pool_std: np.ndarray, seed: int = AE_SEED) -> DenoisingAE:
    """Train denoising AE on pooled standardised features.
    Per-step: add iid N(0, sigma^2) noise to input, reconstruct CLEAN input.
    Returns trained model (eval mode)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = DenoisingAE()
    opt = torch.optim.Adam(model.parameters(), lr=AE_LR)
    loss_fn = nn.MSELoss()

    X_t = torch.from_numpy(X_pool_std.astype(np.float32))
    ds = TensorDataset(X_t)
    loader = DataLoader(ds, batch_size=AE_BATCH, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(AE_EPOCHS):
        epoch_loss = 0.0
        n_batches = 0
        for (xb,) in loader:
            xb_clean = xb
            xb_noisy = xb_clean + AE_NOISE_SIGMA * torch.randn_like(xb_clean)
            opt.zero_grad()
            recon = model(xb_noisy)
            loss = loss_fn(recon, xb_clean)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        if (epoch + 1) % 50 == 0:
            print(f"   AE epoch {epoch+1:3d}/{AE_EPOCHS}  "
                  f"avg_mse={epoch_loss / max(n_batches, 1):.5f}")

    model.eval()
    return model


def _encode(model: DenoisingAE, X_std: np.ndarray) -> np.ndarray:
    """Run encoder on standardised input -> (n, 8) latent."""
    model.eval()
    with torch.no_grad():
        z = model.encode(torch.from_numpy(X_std.astype(np.float32)))
    return z.numpy().astype(np.float32)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Denoising AE bottleneck features on chemprop_aux residual")
    print("=" * 78)

    # ---- Load test set + scaffolds + truth ----
    te = load_test()
    n_test = len(te)
    smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    te_smiles = te[smi_col].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[load] n_test={n_test}  n_unb={n_unb}  unique_scaf={n_uniq_scaf}")

    # ---- Load X_117 (train + unb + te), slice K=20 ----
    for p in (X117_UNB_PATH, X117_TE_PATH, X117_TRAIN_PATH):
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")
    X_unb_117 = np.load(X117_UNB_PATH).astype(np.float32)
    X_te_117 = np.load(X117_TE_PATH).astype(np.float32)
    X_tr_117 = np.load(X117_TRAIN_PATH).astype(np.float32)
    assert X_unb_117.shape == (n_unb, 117), f"X_unb shape {X_unb_117.shape}"
    assert X_te_117.shape == (n_test, 117), f"X_te shape {X_te_117.shape}"
    print(f"[feat] X_tr_117={X_tr_117.shape}  X_unb_117={X_unb_117.shape}  "
          f"X_te_117={X_te_117.shape}")

    X_unb_K = X_unb_117[:, :K_SLICE].astype(np.float32)
    X_te_K = X_te_117[:, :K_SLICE].astype(np.float32)
    X_tr_K = X_tr_117[:, :K_SLICE].astype(np.float32)

    # ---- Standardise using POOLED train+test stats (AE sees only X) ----
    X_pool_K = np.concatenate([X_tr_K, X_te_K], axis=0).astype(np.float32)
    X_pool_std, X_unb_std, X_te_std, mu_pool, sd_pool = _standardise(
        X_pool_K, X_unb_K, X_te_K,
    )
    print(f"[std] pool fit={X_pool_K.shape}  unb_std={X_unb_std.shape}  "
          f"te_std={X_te_std.shape}")

    # ---- Train denoising AE on pooled (train+test) features ----
    print("\n" + "-" * 78)
    print(f"DENOISING AE  in_dim={AE_IN_DIM}  hid={AE_HID}  bottle={AE_BOTTLE}  "
          f"sigma={AE_NOISE_SIGMA}  lr={AE_LR}  epochs={AE_EPOCHS}  "
          f"batch={AE_BATCH}")
    print("-" * 78)
    ae = _train_dae(X_pool_std, seed=AE_SEED)

    # Encode all 253 + 513 rows
    Z_unb = _encode(ae, X_unb_std)  # (253, 8)
    Z_te = _encode(ae, X_te_std)    # (513, 8)
    print(f"[encode] Z_unb={Z_unb.shape}  Z_te={Z_te.shape}")
    print(f"[encode] Z_unb mean={Z_unb.mean():+.3f}  std={Z_unb.std():.3f}")
    print(f"[encode] Z_te  mean={Z_te.mean():+.3f}  std={Z_te.std():.3f}")

    # ---- Anchor (chemprop_aux, PRE-unblind verified-clean) ----
    if not TE_CHEM_PATH.exists():
        raise FileNotFoundError(f"missing test anchor: {TE_CHEM_PATH}")
    te_chem = np.load(TE_CHEM_PATH).astype(np.float64)
    assert te_chem.shape == (n_test,), f"te_chem shape {te_chem.shape}"
    anchor_unb = te_chem[unb_idx]
    anchor_te = te_chem.copy()
    rae_anchor_unb = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux te[unb_idx] RAE = {rae_anchor_unb:.4f}")

    # ---- Residual ----
    resid_unb = y_unb - anchor_unb
    print(f"[resid] mean={resid_unb.mean():+.3f}  std={resid_unb.std():.3f}  "
          f"min={resid_unb.min():+.2f}  max={resid_unb.max():+.2f}")

    # ---- Scaffold 5-fold CV across 5 kf_seeds (LGBM on 8-d bottleneck) ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  seeds={KF_SEEDS}\n"
          f"LGBM on Z (8-d): max_depth=4 num_leaves=15 n_est=300 lr=0.03")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
        fold_info = []
        for fi, (tr_loc, va_loc) in enumerate(splits):
            mdl = lgb.LGBMRegressor(random_state=kf_seed + fi, **LGBM_PARAMS)
            mdl.fit(Z_unb[tr_loc], resid_unb[tr_loc])
            pred = mdl.predict(Z_unb[va_loc]).astype(np.float64)
            oof_resid[va_loc] = pred
            fold_info.append({
                "fold": fi,
                "n_tr": int(len(tr_loc)),
                "n_va": int(len(va_loc)),
                "resid_pred_mean": float(pred.mean()),
                "resid_pred_std": float(pred.std()),
            })
        assert not np.isnan(oof_resid).any(), "oof_resid has NaN -- fold cover incomplete"
        oof_pred = anchor_unb + oof_resid
        oof_pred = np.clip(oof_pred, 3.0, 8.0)
        pooled = float(rae(y_unb, oof_pred))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "folds": fold_info,
            "wall_sec": round(time.time() - ts, 2),
        })
        all_oofs.append(oof_pred)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  wall={time.time()-ts:.1f}s")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] pooled RAE mean across seeds = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs      = {final_oof_rae:.4f}")
    print(f"[cv] delta vs anchor (chemprop_aux)= {pooled_rae_mean - rae_anchor_unb:+.4f}")

    # ---- Deploy: refit on ALL 253 -> predict on 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit LGBM on all 253 (Z_unb -> residual) -> apply to Z_te")
    print("-" * 78)
    mdl_deploy = lgb.LGBMRegressor(random_state=AE_SEED, **LGBM_PARAMS)
    mdl_deploy.fit(Z_unb, resid_unb)
    deploy_resid_te = mdl_deploy.predict(Z_te).astype(np.float64)
    deploy_te = anchor_te + deploy_resid_te
    deploy_te = np.clip(deploy_te, 3.0, 8.0).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, deploy_te[unb_idx].astype(np.float64)))
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_in_rae:.4f}  (in-sample)")

    # ---- Gate ----
    if pooled_rae_mean < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   mean_rae       = {pooled_rae_mean:.4f}")
    print(f"   gate PROMOTE   = < {GATE_PROMOTE}")
    print(f"   gate MARGINAL  = < {GATE_MARGINAL}")
    print(f"   verdict        = {verdict}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": (
            "Denoising autoencoder (Vincent et al. 2008) for K=20 feature "
            "smoothing: Encoder Linear(20,16)->ReLU->Linear(16,8) + Decoder "
            "Linear(8,16)->ReLU->Linear(16,20); Adam lr=1e-3 MSE reconstruction "
            "with Gaussian noise sigma=0.1 on input, 200 epochs, batch 64. "
            "Bottleneck encoding (8-d) fed to LGBM on chemprop_aux residual."
        ),
        "paradigm": (
            "unsupervised_denoising_autoencoder_bottleneck_smoothing_"
            "vincent_2008_then_lgbm_on_anchor_residual"
        ),
        "anchor_name": "chemprop_aux (PRE-unblind, verified clean)",
        "rae_anchor_unb": rae_anchor_unb,
        "ae_in_dim": AE_IN_DIM,
        "ae_hid": AE_HID,
        "ae_bottle": AE_BOTTLE,
        "ae_noise_sigma": AE_NOISE_SIGMA,
        "ae_lr": AE_LR,
        "ae_epochs": AE_EPOCHS,
        "ae_batch": AE_BATCH,
        "ae_seed": AE_SEED,
        "ae_train_pool_size": int(X_pool_K.shape[0]),
        "lgbm_params": LGBM_PARAMS,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": n_uniq_scaf,
        "feat_dim_raw_K20": K_SLICE,
        "feat_dim_bottleneck": AE_BOTTLE,
        "per_seed": per_seed,
        "mean_rae": pooled_rae_mean,
        "pooled_rae_std_seeds": pooled_rae_std,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "delta_vs_anchor": pooled_rae_mean - rae_anchor_unb,
        "te_unb_rae_in_sample": te_unb_in_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
        "z_unb_mean": float(Z_unb.mean()),
        "z_unb_std": float(Z_unb.std()),
        "z_te_mean": float(Z_te.mean()),
        "z_te_std": float(Z_te.std()),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "promote": bool(verdict == "PROMOTE"),
        "marginal_beat": bool(verdict == "MARGINAL_BEAT"),
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
    print(f"   mean_rae (5 seeds)            = {pooled_rae_mean:.4f} "
          f"(+/- {pooled_rae_std:.4f})")
    print(f"   delta vs anchor (chemprop_aux)= {pooled_rae_mean - rae_anchor_unb:+.4f}")
    print(f"   verdict                       = {verdict}")
    print(f"   wall                          = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "pooled_rae_std_seeds",
        "rae_of_mean_of_seed_oofs",
        "rae_anchor_unb",
        "delta_vs_anchor",
        "verdict",
        "te_unb_rae_in_sample",
    ):
        print(f"  {k}: {res.get(k)}")
