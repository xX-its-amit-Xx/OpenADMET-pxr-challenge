"""nb2801 -- Fixed NN meta-stacker v2 (vs nb2792 collapse).

CONTEXT:
    nb2792 NN meta-stack collapsed because (i) z-scored anchors that are
    already on the pEC50 scale, distorting the natural identity mapping the
    meta-learner needs to land near; (ii) tiny capacity (3->16->1, 81 params)
    + Dropout(0.2) at n_train ~200 starved the model; (iii) 200 epochs of
    full-batch Adam at lr=1e-3 did not converge from a noisy init.

    NEW PARADIGM (nb2801):
    - DEEPER torch NN: Linear(3,32) -> ReLU -> Linear(32,16) -> ReLU -> Linear(16,1)
    - NO z-scoring: anchors already on pEC50 scale, identity is the desired
      near-optimum.
    - NO dropout (3 features only -- dropout starves not regularizes here).
    - LONGER training: 1000 epochs, lr=5e-3 Adam, MSE.

SUBSTRATE (PRE-clean only, 3 anchors):
    - nb2240_K20      (K=20 residual stack on chemprop_aux)
    - chemprop_aux    (nb1133, 4139 PRE-unblind only)
    - counter_clean   (nb2490 counter-assay residual, nb730-free)

PROTOCOL:
    - 5-fold scaffold CV on 253 unblind, 5 kf_seeds {42, 1, 7, 137, 1009}
    - Meta-NN per fold: Adam lr=5e-3, MSE, 1000 epochs, batch=full-train
    - raw inputs (no z-score) for both train and inference
    - Deploy: refit on all 253, predict te 513
    - torch seeds set per (kf_seed, fold) for reproducibility

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4598  -> MARGINAL_BEAT
    else                -> FAIL
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
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2801"
N_FOLDS = 5
KF_SEEDS = [42, 1, 7, 137, 1009]
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Meta-NN hyperparams (v2: deeper, no dropout, longer training, no z-score)
HIDDEN1 = 32
HIDDEN2 = 16
LR = 5e-3
EPOCHS = 1000

# ---- PRE-clean anchors only (3 required) ----
CANDIDATE_ANCHORS = [
    ("nb2240_K20",   DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
                     DATA_PROCESSED / "te_nb2240_K20.npy",
                     "PRE-clean (K=20 residual stack on chemprop_aux)"),
    ("chemprop_aux", DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
                     DATA_PROCESSED / "te_chemprop_aux.npy",
                     "PRE-clean (4139 PRE-unblind only)"),
    ("counter_clean",DATA_PROCESSED / "nb2490_pred_oof.npy",
                     DATA_PROCESSED / "te_nb2490.npy",
                     "PRE-clean counter-assay residual on chemprop_aux (nb730-free)"),
]


class MetaNNv2(nn.Module):
    """3 -> 32 -> 16 -> 1, ReLU, no dropout."""
    def __init__(self, n_in: int, h1: int = HIDDEN1, h2: int = HIDDEN2):
        super().__init__()
        self.fc1 = nn.Linear(n_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        out = self.fc3(h)
        return out.squeeze(-1)


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_meta(X_tr: np.ndarray, y_tr: np.ndarray, n_in: int, seed: int) -> MetaNNv2:
    _set_seed(seed)
    model = MetaNNv2(n_in)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X_tr.astype(np.float32))
    yt = torch.from_numpy(y_tr.astype(np.float32))
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred = model(Xt)
        loss = loss_fn(pred, yt)
        loss.backward()
        opt.step()
    return model


def _predict_meta(model: MetaNNv2, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(X.astype(np.float32))).numpy()
    return out.astype(np.float64)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Fixed NN meta-stacker v2 (deeper, raw-scale, longer)")
    print("=" * 78)

    # ---- Load ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test} n_unb={n_unb}")

    # ---- Resolve anchors (strict: need all 3) ----
    anchor_names = []
    anchor_provenance = {}
    oof_cols, te_cols = [], []
    anchor_skipped = {}
    for name, oof_path, te_path, prov in CANDIDATE_ANCHORS:
        if not oof_path.exists():
            anchor_skipped[name] = f"pred_oof missing at {oof_path}"
            print(f"   SKIP {name}: pred_oof missing")
            continue
        if not te_path.exists():
            anchor_skipped[name] = f"te missing at {te_path}"
            print(f"   SKIP {name}: te missing")
            continue
        oof = np.load(oof_path).astype(np.float64)
        te_v = np.load(te_path).astype(np.float64)
        if oof.shape[0] != n_unb:
            anchor_skipped[name] = f"shape mismatch oof={oof.shape} expected ({n_unb},)"
            print(f"   SKIP {name}: shape mismatch")
            continue
        if te_v.shape[0] != n_test:
            anchor_skipped[name] = f"shape mismatch te={te_v.shape} expected ({n_test},)"
            print(f"   SKIP {name}: te shape mismatch")
            continue
        anchor_names.append(name)
        anchor_provenance[name] = prov
        oof_cols.append(oof)
        te_cols.append(te_v)
    K = len(anchor_names)
    if K < 3:
        raise RuntimeError(f"Need 3 PRE-clean anchors, got {K}: {anchor_names}")

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    rae_anchors = {k: float(rae(y_unb, P_unb[:, i]))
                   for i, k in enumerate(anchor_names)}
    print(f"[anchors] K={K}")
    for k in anchor_names:
        print(f"   {k:14s}  unb_RAE={rae_anchors[k]:.4f}  [{anchor_provenance[k]}]")
    if anchor_skipped:
        print(f"[skipped] {anchor_skipped}")

    # ---- Scaffold folds ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}  n_folds={N_FOLDS}  kf_seeds={KF_SEEDS}")
    print(f"[meta-nn-v2] Linear({K},{HIDDEN1}) -> ReLU -> "
          f"Linear({HIDDEN1},{HIDDEN2}) -> ReLU -> Linear({HIDDEN2},1)  "
          f"raw-scale  lr={LR}  epochs={EPOCHS}  no_dropout")

    # ---- NN meta CV ----
    print("\n" + "-" * 78)
    print("NN meta-stack v2 CV (5 kf_seeds x 5-fold scaffold)")
    print("-" * 78)
    per_seed_pooled = []
    per_seed_mean_fold = []
    oof_seed_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)

    for s_idx, kf_seed in enumerate(KF_SEEDS):
        splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                        shuffle=True, seed=kf_seed)
        oof_nn = np.full(n_unb, np.nan, dtype=np.float64)
        fold_rae = []
        for f_idx, (tr_loc, va_loc) in enumerate(splits):
            # NO z-scoring: anchors already on pEC50 scale
            Xtr = P_unb[tr_loc]
            Xva = P_unb[va_loc]
            torch_seed = int(kf_seed * 1000 + f_idx)
            model = _train_meta(Xtr, y_unb[tr_loc], K, torch_seed)
            pred = _predict_meta(model, Xva)
            oof_nn[va_loc] = pred
            r = float(rae(y_unb[va_loc], pred))
            fold_rae.append(r)
        pooled = float(rae(y_unb, oof_nn))
        mean_fold = float(np.mean(fold_rae))
        per_seed_pooled.append(pooled)
        per_seed_mean_fold.append(mean_fold)
        oof_seed_stack[s_idx] = oof_nn
        print(f"   kf_seed={kf_seed:5d}  pooled={pooled:.4f}  mean_fold={mean_fold:.4f}")

    mean_pooled = float(np.mean(per_seed_pooled))
    std_pooled = float(np.std(per_seed_pooled))
    oof_final = oof_seed_stack.mean(axis=0)
    final_pooled_on_seed_mean = float(rae(y_unb, oof_final))

    print(f"\n[wide-seed] mean pooled = {mean_pooled:.4f} +/- {std_pooled:.4f} "
          f"(n_seeds={len(KF_SEEDS)})")
    print(f"[wide-seed] pooled on seed-mean OOF = {final_pooled_on_seed_mean:.4f}")

    # ---- Gate ----
    if mean_pooled < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_pooled < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_pooled {mean_pooled:.4f}  "
          f"(< {GATE_PROMOTE} PROMOTE / < {GATE_MARGINAL} MARGINAL)  ->  {verdict}")

    # ---- Deploy: refit on all 253 ----
    print("\n" + "-" * 78)
    print("DEPLOY: refit NN on all 253, predict 513")
    print("-" * 78)
    # No z-scoring on deploy either -- raw scale throughout
    model_deploy = _train_meta(P_unb, y_unb, K, seed=42)
    te_pred = _predict_meta(model_deploy, P_te).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    n_params = sum(p.numel() for p in model_deploy.parameters())
    print(f"   te mean={te_pred.mean():.3f} std={te_pred.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}  (expected << pooled)")
    print(f"   deploy n_params={n_params}")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_final.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_nn_meta_v2.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "nn_meta_stack_v2_preclean_raw_scale",
        "architecture": f"Linear({K},{HIDDEN1})->ReLU->Linear({HIDDEN1},{HIDDEN2})->ReLU->Linear({HIDDEN2},1)",
        "fix_vs_nb2792": "deeper(3->32->16->1) + no_zscore + no_dropout + lr5e-3 + 1000ep",
        "anchor_pool": anchor_names,
        "anchor_provenance": anchor_provenance,
        "anchor_skipped": anchor_skipped,
        "anchor_in_rae": rae_anchors,
        "nn_hidden1": HIDDEN1,
        "nn_hidden2": HIDDEN2,
        "nn_dropout": 0.0,
        "nn_zscore": False,
        "nn_lr": LR,
        "nn_epochs": EPOCHS,
        "nn_n_params": int(n_params),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_mean_fold_rae": per_seed_mean_fold,
        "mean_pooled_rae": mean_pooled,
        "std_pooled_rae": std_pooled,
        "pooled_on_seed_mean_oof": final_pooled_on_seed_mean,
        "mean_rae": mean_pooled,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean pooled RAE     = {mean_pooled:.4f} +/- {std_pooled:.4f}  ({verdict})")
    print(f"   K anchors used      = {K}  ({anchor_names})")
    print(f"   nn n_params         = {n_params}  (h1={HIDDEN1}, h2={HIDDEN2}, no_dropout)")
    print(f"   wall                = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("mean_pooled_rae", "std_pooled_rae", "verdict",
              "K_anchors", "te_unb_in_sample_rae", "submission_csv"):
        print(f"  {k}: {res.get(k)}")
