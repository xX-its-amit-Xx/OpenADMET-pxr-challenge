"""nb2683 -- Cross-attention between Morgan FP and 9-d physchem features.

NEW PARADIGM (cycle 170):
    Most prior NN attempts on the (chemprop_aux anchor, n=253) substrate
    treated Morgan FP and physchem as a single concatenated vector (e.g.
    the K=28 SHAP feature set or the combined 2265-d featurizer).  That
    concat geometry forces every layer to learn substructure x physchem
    interaction terms *implicitly* through dense FF layers.

    This script makes the interaction *explicit*: physchem rows are
    queries that *attend* over a learnable projection of the 2048-bit
    Morgan fingerprint, so the model can learn which substructures
    matter for which physchem property (MW / logP / TPSA / etc).  The
    attention mask gives a different inductive bias than every
    fingerprint+LGBM or concat-FF NN the ladder has tried — the only
    lever cycle 169 left open.

ARCHITECTURE (per task spec):
    morgan_embed   : Linear(2048, 64)               -> (B, 64)
    physchem_embed : Linear(9, 64)                   -> (B, 64)
    cross_attn     : MultiheadAttention(d=64, h=4)
                       Q = physchem_embed.unsqueeze(1)   -> (B, 1, 64)
                       K = V = morgan_embed.unsqueeze(1) -> (B, 1, 64)
    head           : Linear(64, 1)                   -> (B,)
    Trained on residual = y_unb - anchor[unb_idx] (chemprop_aux).

PROTOCOL (per spec):
    - 5-fold scaffold CV on 253 (5 kf_seeds, pooled across seeds)
    - Per fold: Adam lr=1e-3, MSE loss, batch_size=32, 100 epochs.
    - Predict residual on held-out fold and on all 513 test compounds.
    - pred_oof_unb (253,) and te_513 (513,) saved.

GATES:
    pooled RAE < 0.4570  -> "PROMOTE"
    pooled RAE < 0.4598  -> "MARGINAL_BEAT"
    else                 -> "FAIL"

OUTPUTS:
    data/processed/nb2683_summary.json
    data/processed/nb2683_pred_oof.npy   (253,) float32
    data/processed/te_nb2683.npy         (513,) float32

REFERENCES:
    nb2171 PRIMARY-1 deep-30 pooled RAE = 0.4682
    nb2095 deep-30                       = 0.4720
    nb1191 deep-30 PRE-pyramid           = 0.4718
    chemprop_aux te[unb_idx]             = 0.6216  (anchor RAE)
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
import torch.optim as optim
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, compute_physchem, morgan_fp_batch, standardize
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2683"
PARENT_TAG = "cycle170_substrate_change"

# -- Hyperparams (per task spec) --------------------------------------------
D_MODEL = 64
N_HEADS = 4
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
TRAIN_SEED = 1009
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
N_FOLDS = 5

MORGAN_BITS = 2048
PHYSCHEM_DIM = 9  # MW, LogP, TPSA, HBD, HBA, RotBonds, fsp3, rings, charge

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# -- Gates (per task spec) --------------------------------------------------
GATE_PROMOTE = 0.4570
GATE_MARGINAL_BEAT = 0.4598

# -- References --------------------------------------------------------------
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216

PHYSCHEM_KEYS = [
    "mw",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotbonds",
    "fsp3",
    "n_rings",
    "formal_charge",
]


# ============================================================================
# featurization (Morgan 2048 + 9-d physchem)
# ============================================================================

def compute_physchem_matrix(smiles):
    """(N, 9) physchem feature matrix in PHYSCHEM_KEYS order."""
    N = len(smiles)
    X = np.zeros((N, PHYSCHEM_DIM), dtype=np.float32)
    for i, s in enumerate(smiles):
        d = compute_physchem(s)
        if d is None:
            continue
        for j, k in enumerate(PHYSCHEM_KEYS):
            v = d.get(k, 0.0)
            X[i, j] = float(v) if v is not None else 0.0
    return X


def standardize_features(X_train, X_test):
    """Per-feature mean/std fit on train, applied to both. Avoid div-by-zero."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Xtr = (X_train - mu) / sd
    Xte = (X_test - mu) / sd
    return Xtr.astype(np.float32), Xte.astype(np.float32), mu, sd


# ============================================================================
# Cross-attention model
# ============================================================================

class CrossAttnFPPhyschem(nn.Module):
    """Physchem (Q) cross-attends over Morgan FP (K=V) projections."""

    def __init__(self, morgan_bits=MORGAN_BITS, physchem_dim=PHYSCHEM_DIM,
                 d_model=D_MODEL, num_heads=N_HEADS):
        super().__init__()
        self.morgan_embed = nn.Linear(morgan_bits, d_model)
        self.physchem_embed = nn.Linear(physchem_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.head = nn.Linear(d_model, 1)

    def forward(self, morgan, physchem):
        # morgan   : (B, 2048) float
        # physchem : (B, 9)    float (standardized)
        m_e = self.morgan_embed(morgan).unsqueeze(1)      # (B, 1, 64)
        p_e = self.physchem_embed(physchem).unsqueeze(1)  # (B, 1, 64)
        # Q = physchem, K = V = morgan
        attn_out, _ = self.cross_attn(p_e, m_e, m_e, need_weights=False)  # (B, 1, 64)
        z = attn_out.squeeze(1)                            # (B, 64)
        out = self.head(z).squeeze(-1)                     # (B,)
        return out


# ============================================================================
# train one fold
# ============================================================================

def train_one_fold(M_tr, P_tr, y_tr, M_va, P_va, M_te, P_te_arr,
                   epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                   device="cpu", seed=TRAIN_SEED):
    """Train cross-attention model on residual; return va_pred, te_pred."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CrossAttnFPPhyschem().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Mt = torch.tensor(M_tr, dtype=torch.float32, device=device)
    Pt = torch.tensor(P_tr, dtype=torch.float32, device=device)
    Yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    n_tr = Mt.shape[0]

    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n_tr)
        for i in range(0, n_tr, batch_size):
            b = idx[i:i + batch_size]
            b_t = torch.tensor(b, dtype=torch.long, device=device)
            opt.zero_grad()
            pred = model(Mt.index_select(0, b_t), Pt.index_select(0, b_t))
            loss = loss_fn(pred, Yt.index_select(0, b_t))
            loss.backward()
            opt.step()

    model.eval()

    def _predict(M, P):
        with torch.no_grad():
            Mx = torch.tensor(M, dtype=torch.float32, device=device)
            Px = torch.tensor(P, dtype=torch.float32, device=device)
            n = Mx.shape[0]
            out = np.zeros(n, dtype=np.float32)
            bs = 64
            for i in range(0, n, bs):
                p = model(Mx[i:i + bs], Px[i:i + bs]).cpu().numpy()
                out[i:i + bs] = p
            return out

    return _predict(M_va, P_va), _predict(M_te, P_te_arr)


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Cross-attention(Morgan FP, physchem) residual / chemprop_aux")
    print(f"   d_model={D_MODEL}  heads={N_HEADS}  morgan_bits={MORGAN_BITS}")
    print(f"   physchem_dim={PHYSCHEM_DIM}")
    print(f"   epochs={EPOCHS}  batch_size={BATCH_SIZE}  lr={LR}")
    print(f"   kf_seeds={KF_SEEDS}")
    print(f"   gate PROMOTE       < {GATE_PROMOTE}")
    print(f"   gate MARGINAL_BEAT < {GATE_MARGINAL_BEAT}")
    print("=" * 78)

    # ---- load truth, anchor, scaffolds ------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = (y_unb - anchor).astype(np.float32)

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- featurize: Morgan FP + physchem ---------------------------------
    print(f"\n[feat] Morgan FP (2048 bits) ...")
    fp_unb = morgan_fp_batch(unb_smiles, radius=2, n_bits=MORGAN_BITS).astype(np.float32)
    fp_te = morgan_fp_batch(te_smiles, radius=2, n_bits=MORGAN_BITS).astype(np.float32)
    print(f"   fp_unb {fp_unb.shape}  density={fp_unb.mean():.4f}")
    print(f"   fp_te  {fp_te.shape}   density={fp_te.mean():.4f}")

    print(f"[feat] physchem (9-d) ...")
    phys_unb = compute_physchem_matrix(unb_smiles)
    phys_te = compute_physchem_matrix(te_smiles)
    print(f"   phys_unb {phys_unb.shape}  mean={phys_unb.mean(axis=0).round(2)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # ---- aggregate across kf_seeds ----------------------------------------
    seed_pooled_rae = []
    seed_oof = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float32)
    seed_te = np.zeros((len(KF_SEEDS), n_test), dtype=np.float32)

    for s_i, kf_seed in enumerate(KF_SEEDS):
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        pred_resid_oof = np.zeros(n_unb, dtype=np.float32)
        pred_resid_te_folds = np.zeros((N_FOLDS, n_test), dtype=np.float32)
        fold_rae = []
        for f_i, (tr_loc, va_loc) in enumerate(splits):
            M_tr_raw, P_tr_raw = fp_unb[tr_loc], phys_unb[tr_loc]
            M_va_raw, P_va_raw = fp_unb[va_loc], phys_unb[va_loc]
            # standardize physchem on TRAIN fold only
            P_tr, P_va, mu, sd = standardize_features(P_tr_raw, P_va_raw)
            P_te = ((phys_te - mu) / sd).astype(np.float32)
            # Morgan is binary -> no standardization needed
            va_pred, te_pred = train_one_fold(
                M_tr_raw, P_tr, residual[tr_loc],
                M_va_raw, P_va, fp_te, P_te,
                epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE,
                device=device, seed=TRAIN_SEED + s_i,
            )
            pred_resid_oof[va_loc] = va_pred
            pred_resid_te_folds[f_i] = te_pred
            fold_pred_full = anchor[va_loc] + va_pred.astype(np.float64)
            r_fold = float(rae(y_unb[va_loc], fold_pred_full))
            fold_rae.append(r_fold)
        pred_full_oof = anchor + pred_resid_oof.astype(np.float64)
        pooled = float(rae(y_unb, pred_full_oof))
        seed_pooled_rae.append(pooled)
        seed_oof[s_i] = pred_full_oof.astype(np.float32)
        seed_te[s_i] = (te_anchor_513 + pred_resid_te_folds.mean(axis=0)
                        .astype(np.float64)).astype(np.float32)
        print(f"   kf_seed {kf_seed}: pooled RAE = {pooled:.4f}  "
              f"fold mean = {np.mean(fold_rae):.4f}  "
              f"wall = {time.time()-ts:.1f}s")

    # ---- pooled across seeds ----------------------------------------------
    mean_rae = float(np.mean(seed_pooled_rae))
    std_rae = float(np.std(seed_pooled_rae, ddof=1)) if len(KF_SEEDS) > 1 else 0.0
    pred_full_oof_seedmean = seed_oof.mean(axis=0).astype(np.float64)
    pooled_seedmean_rae = float(rae(y_unb, pred_full_oof_seedmean))
    pred_te_513_seedmean = seed_te.mean(axis=0).astype(np.float64)
    te_unb_in_rae = float(rae(y_unb, pred_te_513_seedmean[unb_idx]))

    print("\n" + "-" * 78)
    print("RESULT (5-kf_seed aggregate)")
    print("-" * 78)
    print(f"   per-seed pooled RAE = {[round(x, 4) for x in seed_pooled_rae]}")
    print(f"   mean +/- std        = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   pooled (seedmean)   = {pooled_seedmean_rae:.4f}")
    print(f"   te[unb_idx] RAE     = {te_unb_in_rae:.4f}")
    print(f"   pred_oof_unb std    = {pred_full_oof_seedmean.std():.3f}  "
          f"(truth_std {y_unb.std():.3f})")
    print(f"   pred_te_513 m/s     = {pred_te_513_seedmean.mean():.3f} / "
          f"{pred_te_513_seedmean.std():.3f}")

    # ---- gate -------------------------------------------------------------
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL_BEAT:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"

    delta_vs_nb2171 = mean_rae - NB2171_REF
    delta_vs_anchor = mean_rae - rae_anchor

    print(f"\n   GATE: mean_rae = {mean_rae:.4f}")
    print(f"     <{GATE_PROMOTE} -> PROMOTE")
    print(f"     <{GATE_MARGINAL_BEAT} -> MARGINAL_BEAT")
    print(f"     else            -> FAIL")
    print(f"   -> {verdict}")
    print(f"   delta vs nb2171 ({NB2171_REF}) = {delta_vs_nb2171:+.4f}")
    print(f"   delta vs anchor ({rae_anchor:.4f}) = {delta_vs_anchor:+.4f}")

    # ---- save artifacts ---------------------------------------------------
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_full_oof_seedmean.astype(np.float32))
    np.save(te_path, pred_te_513_seedmean.astype(np.float32))
    print(f"\n   [save] {oof_path}")
    print(f"   [save] {te_path}")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "cross_attn_morgan2048_physchem9_residual_chemprop_aux",
        "architecture": {
            "morgan_bits": MORGAN_BITS,
            "physchem_dim": PHYSCHEM_DIM,
            "physchem_keys": PHYSCHEM_KEYS,
            "d_model": D_MODEL,
            "num_heads": N_HEADS,
            "blocks": [
                "morgan_embed: Linear(2048, 64)",
                "physchem_embed: Linear(9, 64)",
                "cross_attn: MultiheadAttention(d=64, h=4) Q=phys, K=V=morgan",
                "head: Linear(64, 1)",
            ],
        },
        "hparams": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "loss": "MSE",
            "optimizer": "Adam",
            "kf_seeds": KF_SEEDS,
            "n_folds": N_FOLDS,
            "train_seed_base": TRAIN_SEED,
        },
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_seed_pooled_rae": seed_pooled_rae,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "pooled_seedmean_rae": pooled_seedmean_rae,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "pred_oof_std": float(pred_full_oof_seedmean.std()),
        "te_mean": float(pred_te_513_seedmean.mean()),
        "te_std": float(pred_te_513_seedmean.std()),
        "truth_std": float(y_unb.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "nb2171_ref": NB2171_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_nb2171": delta_vs_nb2171,
        "delta_vs_anchor": delta_vs_anchor,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal_beat": GATE_MARGINAL_BEAT,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae (5 kf_seeds) = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   te[unb_idx] RAE       = {te_unb_in_rae:.4f}")
    print(f"   delta vs nb2171       = {delta_vs_nb2171:+.4f}")
    print(f"   -> {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "std_rae",
        "pooled_seedmean_rae",
        "te_unb_in_sample_rae",
        "delta_vs_nb2171",
        "delta_vs_anchor",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
