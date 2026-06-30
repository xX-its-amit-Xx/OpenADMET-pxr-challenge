"""nb2673 -- Set Transformer on bag-of-atoms (residual-on-chemprop_aux).

NEW PARADIGM (cycle 170):
    Most prior NN attempts (nb1086, nb266, nb267 meta-stacks, K-pyramid NN
    heads) used fixed-length fingerprint / descriptor vectors as input.
    The substrate fixed by cycle-134 paradigm exhaustion was (anchor =
    chemprop_aux, feature set = K=28 SHAP-LGBM, n=253).  Cycle 169 closed
    the post-hoc-blend axis on that substrate.

    This script changes the SUBSTRATE: each molecule is treated as an
    unordered SET of atom-feature vectors and the model is a
    permutation-invariant Set Transformer (Lee et al. 2019, "Set
    Transformer: A Framework for Attention-based Permutation-Invariant
    Neural Networks").  The expectation is NOT that the raw Set
    Transformer matches a tuned LGBM in 200 epochs at n=253; the
    expectation is that it learns a *different* error mode than every
    fingerprint+LGBM the ladder has tried, which is the only lever cycle
    169 left open.

ARCHITECTURE:
    Per atom feature vector (16 dims):
        element one-hot (C, N, O, F, P, S, Cl, Br, I, other)  = 10 dims
        degree (int, 0..5+ -> /6)                              =  1 dim
        formal charge (clipped -2..2, /4 + 0.5)                =  1 dim
        explicit H count (0..4, /4)                            =  1 dim
        is_aromatic (bool)                                     =  1 dim
        is_in_ring (bool)                                      =  1 dim
        hybridization sp3 indicator (bool)                     =  1 dim
    -> 16-dim per-atom feature, projected to d=64 by Linear(16, 64).

    Encoder stack:
        x = Linear(16, 64)(atom_feats)           # (B, n_atoms, 64)
        x = ISAB(d=64, num_heads=4, num_inducing=8)(x)
        z = PMA(d=64, num_heads=4, num_seeds=1)(x)        # (B, 1, 64)
        out = Linear(64, 1)(z.squeeze(1))                  # (B,)

    Trained on residual = y_unb - anchor[unb_idx] (chemprop_aux).

PROTOCOL (per spec):
    - 5-fold scaffold CV on 253 (kf_seed=1001 -- matches nb2661 etc.)
    - Per fold: train Set Transformer for 200 epochs, Adam lr=1e-3,
      MSE loss, batch_size=16.  Predict residual on held-out fold and
      on all 513 test atoms.
    - pred_oof_unb (253,) and te_513 (513,) saved.
    - Pooled scaffold-CV RAE is reported.

GATES:
    pooled RAE < 0.4570  -> "PROMOTE"
    pooled RAE < 0.4598  -> "MARGINAL_BEAT"
    else                 -> "FAIL"

OUTPUTS:
    data/processed/nb2673_summary.json
    data/processed/nb2673_pred_oof.npy   (253,) float32
    data/processed/te_nb2673.npy         (513,) float32

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

from pxr.chem import bemis_murcko, standardize
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2673"
PARENT_TAG = "cycle170_substrate_change"

# -- Hyperparams (per task spec) --------------------------------------------
D_MODEL = 64
N_HEADS = 4
N_INDUCING = 8
N_SEEDS_PMA = 1
EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-3
TRAIN_SEED = 1009            # match nb1191 / cycle-160 convention
KF_SEED = 1001               # scaffold-CV split seed (per task spec match)
N_FOLDS = 5

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# -- Gates (per task spec) --------------------------------------------------
GATE_PROMOTE = 0.4570
GATE_MARGINAL_BEAT = 0.4598

# -- References --------------------------------------------------------------
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216

# -- Atom-feature columns ----------------------------------------------------
ELEMENT_SYMBOLS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
ATOM_FEAT_DIM = len(ELEMENT_SYMBOLS) + 1 + 1 + 1 + 1 + 1 + 1 + 1  # = 16


# ============================================================================
# featurization (per-atom feature extraction)
# ============================================================================

def _atom_feats(atom):
    """16-dim feature vector for a single RDKit atom."""
    feats = np.zeros(ATOM_FEAT_DIM, dtype=np.float32)
    sym = atom.GetSymbol()
    if sym in ELEMENT_SYMBOLS:
        feats[ELEMENT_SYMBOLS.index(sym)] = 1.0
    else:
        feats[len(ELEMENT_SYMBOLS)] = 1.0   # "other"
    off = len(ELEMENT_SYMBOLS) + 1
    feats[off + 0] = float(atom.GetDegree()) / 6.0
    fc = max(-2, min(2, int(atom.GetFormalCharge())))
    feats[off + 1] = (fc + 2.0) / 4.0
    feats[off + 2] = float(atom.GetTotalNumHs()) / 4.0
    feats[off + 3] = 1.0 if atom.GetIsAromatic() else 0.0
    feats[off + 4] = 1.0 if atom.IsInRing() else 0.0
    feats[off + 5] = 1.0 if atom.GetHybridization() == Chem.HybridizationType.SP3 else 0.0
    return feats


def smiles_to_atom_set(smi):
    """Return (n_atoms, 16) np.float32; empty (0, 16) if parsing fails."""
    mol = standardize(smi)
    if mol is None:
        return np.zeros((1, ATOM_FEAT_DIM), dtype=np.float32)
    rows = [_atom_feats(a) for a in mol.GetAtoms()]
    if not rows:
        return np.zeros((1, ATOM_FEAT_DIM), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def pad_atom_sets(atom_sets):
    """Pad list of (n_i, 16) arrays to a (B, max_n, 16) tensor + mask."""
    B = len(atom_sets)
    max_n = max(a.shape[0] for a in atom_sets)
    X = np.zeros((B, max_n, ATOM_FEAT_DIM), dtype=np.float32)
    mask = np.zeros((B, max_n), dtype=np.float32)
    for i, a in enumerate(atom_sets):
        n = a.shape[0]
        X[i, :n] = a
        mask[i, :n] = 1.0
    return X, mask


# ============================================================================
# Set Transformer modules (Lee et al. 2019, condensed)
# ============================================================================

class MAB(nn.Module):
    """Multi-Head Attention Block: Q attends over K=V, residual+FFN."""

    def __init__(self, d, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.ln2 = nn.LayerNorm(d)

    def forward(self, Q, K, mask_K=None):
        # mask_K: (B, n_K) with 1 = valid, 0 = pad.  MHA expects key_padding_mask=True on PAD.
        kpm = None
        if mask_K is not None:
            kpm = (mask_K < 0.5)
        h, _ = self.attn(Q, K, K, key_padding_mask=kpm, need_weights=False)
        Q1 = self.ln1(Q + h)
        Q2 = self.ln2(Q1 + self.ff(Q1))
        return Q2


class ISAB(nn.Module):
    """Induced Set Attention Block: ISAB_m(X) = MAB(X, MAB(I, X))."""

    def __init__(self, d, num_heads, num_inducing):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, d) * 0.1)
        self.mab1 = MAB(d, num_heads)
        self.mab2 = MAB(d, num_heads)

    def forward(self, X, mask=None):
        B = X.shape[0]
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, X, mask_K=mask)
        return self.mab2(X, H, mask_K=None)


class PMA(nn.Module):
    """Pooling by Multi-head Attention: PMA(X) = MAB(S, X)."""

    def __init__(self, d, num_heads, num_seeds):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, d) * 0.1)
        self.mab = MAB(d, num_heads)

    def forward(self, X, mask=None):
        B = X.shape[0]
        S = self.S.expand(B, -1, -1)
        return self.mab(S, X, mask_K=mask)


class SetTransformer(nn.Module):
    def __init__(self, atom_dim=ATOM_FEAT_DIM, d=D_MODEL, num_heads=N_HEADS,
                 num_inducing=N_INDUCING, num_seeds=N_SEEDS_PMA):
        super().__init__()
        self.embed = nn.Linear(atom_dim, d)
        self.isab = ISAB(d, num_heads, num_inducing)
        self.pma = PMA(d, num_heads, num_seeds)
        self.head = nn.Linear(d, 1)

    def forward(self, X, mask):
        x = self.embed(X)
        x = self.isab(x, mask=mask)
        z = self.pma(x, mask=mask)
        out = self.head(z.squeeze(1)).squeeze(-1)
        return out


# ============================================================================
# train one fold
# ============================================================================

def train_one_fold(X_tr_sets, y_tr, X_va_sets, y_va, X_te_sets,
                   epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, device="cpu"):
    """Train one fold on residual; return va_pred, te_pred (numpy float32)."""
    torch.manual_seed(TRAIN_SEED)
    np.random.seed(TRAIN_SEED)
    model = SetTransformer().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    n_tr = len(X_tr_sets)
    for ep in range(epochs):
        model.train()
        idx = np.random.permutation(n_tr)
        for i in range(0, n_tr, batch_size):
            b = idx[i:i + batch_size]
            batch_sets = [X_tr_sets[j] for j in b]
            Xb, Mb = pad_atom_sets(batch_sets)
            Xb_t = torch.tensor(Xb, device=device)
            Mb_t = torch.tensor(Mb, device=device)
            Yb_t = Yt[b]
            opt.zero_grad()
            pred = model(Xb_t, Mb_t)
            loss = loss_fn(pred, Yb_t)
            loss.backward()
            opt.step()

    model.eval()
    def _batched_predict(sets, bs=32):
        n = len(sets)
        out = np.zeros(n, dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n, bs):
                chunk = sets[i:i + bs]
                Xb, Mb = pad_atom_sets(chunk)
                Xb_t = torch.tensor(Xb, device=device)
                Mb_t = torch.tensor(Mb, device=device)
                p = model(Xb_t, Mb_t).cpu().numpy()
                out[i:i + bs] = p
        return out

    va_pred = _batched_predict(X_va_sets)
    te_pred = _batched_predict(X_te_sets)
    return va_pred, te_pred


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Set Transformer on bag-of-atoms (residual / chemprop_aux)")
    print(f"   d={D_MODEL}  heads={N_HEADS}  num_inducing={N_INDUCING}")
    print(f"   epochs={EPOCHS}  batch_size={BATCH_SIZE}  lr={LR}")
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

    # ---- featurize as atom sets -------------------------------------------
    print(f"\n[feat] extracting per-atom features...")
    unb_atom_sets = [smiles_to_atom_set(s) for s in unb_smiles]
    te_atom_sets = [smiles_to_atom_set(s) for s in te_smiles]
    n_atoms_unb = np.array([a.shape[0] for a in unb_atom_sets])
    n_atoms_te = np.array([a.shape[0] for a in te_atom_sets])
    print(f"   unb n_atoms: mean={n_atoms_unb.mean():.1f} "
          f"min={n_atoms_unb.min()} max={n_atoms_unb.max()}")
    print(f"   te  n_atoms: mean={n_atoms_te.mean():.1f} "
          f"min={n_atoms_te.min()} max={n_atoms_te.max()}")

    # ---- 5-fold scaffold CV ------------------------------------------------
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=KF_SEED,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    pred_resid_oof = np.zeros(n_unb, dtype=np.float32)
    pred_resid_te_folds = np.zeros((N_FOLDS, n_test), dtype=np.float32)
    fold_rae = []

    for f_i, (tr_loc, va_loc) in enumerate(splits):
        ts = time.time()
        X_tr_sets = [unb_atom_sets[i] for i in tr_loc]
        y_tr = residual[tr_loc]
        X_va_sets = [unb_atom_sets[i] for i in va_loc]
        y_va = residual[va_loc]
        va_pred, te_pred = train_one_fold(
            X_tr_sets, y_tr, X_va_sets, y_va, te_atom_sets,
            epochs=EPOCHS, lr=LR, batch_size=BATCH_SIZE, device=device,
        )
        pred_resid_oof[va_loc] = va_pred
        pred_resid_te_folds[f_i] = te_pred
        # pooled fold RAE on FULL prediction (anchor + residual)
        fold_pred_full = anchor[va_loc] + va_pred.astype(np.float64)
        r_fold = float(rae(y_unb[va_loc], fold_pred_full))
        fold_rae.append(r_fold)
        print(f"   fold {f_i}: n_tr={len(tr_loc):3d}  n_va={len(va_loc):3d}  "
              f"rae={r_fold:.4f}  wall={time.time()-ts:.1f}s")

    # ---- assemble full predictions ----------------------------------------
    pred_full_oof = anchor + pred_resid_oof.astype(np.float64)
    pooled_rae = float(rae(y_unb, pred_full_oof))
    fold_mean = float(np.mean(fold_rae))
    fold_std = float(np.std(fold_rae, ddof=1))

    pred_resid_te = pred_resid_te_folds.mean(axis=0)
    pred_te_513 = te_anchor_513 + pred_resid_te.astype(np.float64)
    te_unb_in_rae = float(rae(y_unb, pred_te_513[unb_idx]))

    print("\n" + "-" * 78)
    print("RESULT")
    print("-" * 78)
    print(f"   pooled scaffold-CV RAE = {pooled_rae:.4f}")
    print(f"   per-fold mean +/- std  = {fold_mean:.4f} +/- {fold_std:.4f}")
    print(f"   per-fold min / max     = {min(fold_rae):.4f} / {max(fold_rae):.4f}")
    print(f"   te[unb_idx] RAE        = {te_unb_in_rae:.4f}")
    print(f"   pred_oof_unb std       = {pred_full_oof.std():.3f}  "
          f"(truth_std {y_unb.std():.3f})")
    print(f"   pred_te_513 mean/std   = {pred_te_513.mean():.3f} / "
          f"{pred_te_513.std():.3f}")

    # ---- gate -------------------------------------------------------------
    if pooled_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae < GATE_MARGINAL_BEAT:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"

    delta_vs_nb2171 = pooled_rae - NB2171_REF
    delta_vs_anchor = pooled_rae - rae_anchor

    print(f"\n   GATE: pooled = {pooled_rae:.4f}")
    print(f"     <{GATE_PROMOTE} -> PROMOTE")
    print(f"     <{GATE_MARGINAL_BEAT} -> MARGINAL_BEAT")
    print(f"     else            -> FAIL")
    print(f"   -> {verdict}")
    print(f"   delta vs nb2171 ({NB2171_REF}) = {delta_vs_nb2171:+.4f}")
    print(f"   delta vs anchor ({rae_anchor:.4f}) = {delta_vs_anchor:+.4f}")

    # ---- save artifacts ---------------------------------------------------
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_full_oof.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"\n   [save] {oof_path}")
    print(f"   [save] {te_path}")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "set_transformer_isab_pma_bag_of_atoms_residual_chemprop_aux",
        "architecture": {
            "atom_feat_dim": ATOM_FEAT_DIM,
            "d_model": D_MODEL,
            "num_heads": N_HEADS,
            "num_inducing": N_INDUCING,
            "num_seeds_pma": N_SEEDS_PMA,
            "blocks": ["Linear(16,64)", "ISAB(64,h=4,m=8)",
                       "PMA(64,h=4,s=1)", "Linear(64,1)"],
        },
        "hparams": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "loss": "MSE",
            "optimizer": "Adam",
        },
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "kf_seed": KF_SEED,
        "train_seed": TRAIN_SEED,
        "n_folds": N_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "n_atoms_unb_mean": float(n_atoms_unb.mean()),
        "n_atoms_unb_min": int(n_atoms_unb.min()),
        "n_atoms_unb_max": int(n_atoms_unb.max()),
        "n_atoms_te_mean": float(n_atoms_te.mean()),
        "n_atoms_te_min": int(n_atoms_te.min()),
        "n_atoms_te_max": int(n_atoms_te.max()),
        "scaffold_cv_pooled_rae": pooled_rae,
        "scaffold_cv_fold_rae": fold_rae,
        "scaffold_cv_fold_mean": fold_mean,
        "scaffold_cv_fold_std": fold_std,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "pred_oof_std": float(pred_full_oof.std()),
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
    print(f"   pooled scaffold-CV RAE = {pooled_rae:.4f}")
    print(f"   per-fold mean +/- std  = {fold_mean:.4f} +/- {fold_std:.4f}")
    print(f"   te[unb_idx] RAE        = {te_unb_in_rae:.4f}")
    print(f"   delta vs nb2171        = {delta_vs_nb2171:+.4f}")
    print(f"   -> {verdict}")
    print(f"   wall                   = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "scaffold_cv_pooled_rae",
        "scaffold_cv_fold_mean",
        "scaffold_cv_fold_std",
        "te_unb_in_sample_rae",
        "delta_vs_nb2171",
        "delta_vs_anchor",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
