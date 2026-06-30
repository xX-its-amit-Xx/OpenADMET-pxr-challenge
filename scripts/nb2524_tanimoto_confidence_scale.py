"""nb2524 -- Tanimoto-confidence-scaled prediction shrinkage on nb2240.

NEW PARADIGM:
    For each test row, shrink (nb2240_pred - train_mean) toward 0 by a
    confidence factor derived from the row's max Tanimoto similarity to
    any of the 4139 train Morgan fingerprints. Low-similarity (novel-
    scaffold / off-manifold) rows are pulled back toward the fold-train
    pEC50 mean; high-similarity rows are left near the nb2240 prediction.

PROTOCOL:
    1. Compute max-Tanimoto per test row to all 4139 train Morgan FPs.
    2. shrink_factor = sigmoid(10 * (max_tan - 0.35))  (smooth ramp).
    3. Final = train_mean + shrink_factor * (nb2240_pred - train_mean).
    4. 5-fold scaffold-CV on the 253 unblind: per-fold train_mean from
       the fold-train labels (NOT global), evaluated with 5 kf_seeds
       {1001..1005}.
    5. Gate: mean_rae < 0.4570 -> PROMOTE; < 0.4601 -> MARGINAL_BEAT;
       else FAIL.
    6. Save nb2524_summary.json + pred_oof + te.

Outputs:
    scripts/nb2524_tanimoto_confidence_scale.py
    data/processed/nb2524_summary.json
    data/processed/nb2524_pred_oof.npy   (253,) float32
    data/processed/te_nb2524.npy         (513,) float32
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
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, morgan_fp_batch, standardize
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2524"

# ------------------------- shrinkage hyperparameters ------------------------
SIM_PIVOT = 0.35       # smooth ramp center
SIM_SLOPE = 10.0       # sigmoid sharpness
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def max_tanimoto_to_pool(fp_query: np.ndarray, fp_pool: np.ndarray) -> np.ndarray:
    """Return max-Tanimoto per query row to any pool row (uint8 ECFP4 inputs)."""
    a = fp_query.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    out = np.zeros(n_q, dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T  # (block, n_pool)
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        out[s:e] = sim.max(axis=1)
    return out


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Tanimoto-confidence shrinkage on nb2240")
    print("=" * 78)

    # ---- Load anchor predictions + truth ----
    nb2240_te = np.load(DATA_PROCESSED / "te_nb2240.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_test = nb2240_te.shape[0]
    n_unb = len(y_unb)
    nb2240_unb = nb2240_te[unb_idx]
    print(f"[load] n_test={n_test}  n_unb={n_unb}")
    rae_nb2240_unb = float(rae(y_unb, nb2240_unb))
    print(f"[load] nb2240 te[unb_idx] in_RAE = {rae_nb2240_unb:.4f}")

    # ---- Load train + test SMILES, standardize, compute Morgan FPs ----
    train = load_train()
    train = train[train["pec50"].notna()].copy().reset_index(drop=True)
    train_std = [standardize(s) for s in train["smiles"].astype(str).tolist()]
    train_smi = []
    for m in train_std:
        if m is None:
            train_smi.append("")
        else:
            from rdkit import Chem as _Chem
            train_smi.append(_Chem.MolToSmiles(m))
    fp_train = morgan_fp_batch(train_smi)
    keep = fp_train.sum(axis=1) > 0
    fp_train = fp_train[keep]
    train_pec50 = train["pec50"].to_numpy(dtype=np.float64)[keep]
    print(f"[train] n={len(train_pec50)}  global pEC50 mean={train_pec50.mean():.4f}")

    te = load_test()
    te_smi_col = "smiles" if "smiles" in te.columns else "SMILES"
    te_smi = te[te_smi_col].astype(str).tolist()
    te_std = [standardize(s) for s in te_smi]
    te_std_smi = []
    for m in te_std:
        if m is None:
            te_std_smi.append("")
        else:
            from rdkit import Chem as _Chem
            te_std_smi.append(_Chem.MolToSmiles(m))
    fp_te = morgan_fp_batch(te_std_smi)
    print(f"[test] n={n_test}  fp shape={fp_te.shape}")

    # ---- Max-Tanimoto per test row to train pool ----
    print(f"[tan] computing max-Tanimoto per test row...")
    ts = time.time()
    max_tan_te = max_tanimoto_to_pool(fp_te, fp_train)  # (513,)
    print(f"[tan] done wall={time.time()-ts:.1f}s  "
          f"mean={max_tan_te.mean():.3f}  median={np.median(max_tan_te):.3f}  "
          f"q10={np.quantile(max_tan_te, 0.10):.3f}  "
          f"q90={np.quantile(max_tan_te, 0.90):.3f}")

    # shrink_factor depends only on test row -> precompute
    shrink_te = sigmoid(SIM_SLOPE * (max_tan_te - SIM_PIVOT)).astype(np.float64)
    print(f"[shrink] sigmoid(10*(s-0.35)) "
          f"mean={shrink_te.mean():.3f}  median={np.median(shrink_te):.3f}  "
          f"min={shrink_te.min():.3f}  max={shrink_te.max():.3f}")
    shrink_unb = shrink_te[unb_idx]

    # ---- 5-fold scaffold CV on 253, 5 seeds, per-fold train_mean ----
    unb_smiles = [te_smi[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaf] n_unique={n_unique_scaf}")

    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  pivot={SIM_PIVOT}  slope={SIM_SLOPE}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        fold_means = []
        for tr_loc, va_loc in splits:
            mu_tr = float(y_unb[tr_loc].mean())
            fold_means.append(mu_tr)
            oof[va_loc] = mu_tr + shrink_unb[va_loc] * (nb2240_unb[va_loc] - mu_tr)
        pooled = float(rae(y_unb, oof))
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "fold_train_means": fold_means,
        })
        all_oofs.append(oof)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"fold_means={[f'{m:.2f}' for m in fold_means]}")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] pooled_RAE mean = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs = {final_oof_rae:.4f}")

    # ---- Deploy: use FULL 253 mean for train_mean on the 513 ----
    mu_full = float(y_unb.mean())
    deploy_te = (mu_full + shrink_te * (nb2240_te - mu_full)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"\n[deploy] mu_full(253)={mu_full:.4f}")
    print(f"[deploy] te(513) mean={deploy_te.mean():.3f}  std={deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in_RAE={te_unb_rae:.4f}")

    # ---- Gate ----
    if pooled_rae_mean_seeds < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean_seeds < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    print(f"   pooled_RAE_mean   = {pooled_rae_mean_seeds:.4f}")
    print(f"   GATE_PROMOTE       < {GATE_PROMOTE:.4f}")
    print(f"   GATE_MARGINAL_BEAT < {GATE_MARGINAL:.4f}")
    print(f"   verdict           = {verdict}")
    print(f"   delta vs nb2240 in_RAE = {pooled_rae_mean_seeds - rae_nb2240_unb:+.4f}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "tanimoto_confidence_scaled_shrinkage_on_nb2240",
        "anchor": "nb2240",
        "sim_pivot": SIM_PIVOT,
        "sim_slope": SIM_SLOPE,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_train_pool": int(len(train_pec50)),
        "n_unique_scaffolds": n_unique_scaf,
        "rae_nb2240_unb": rae_nb2240_unb,
        "max_tan_stats": {
            "mean": float(max_tan_te.mean()),
            "median": float(np.median(max_tan_te)),
            "q10": float(np.quantile(max_tan_te, 0.10)),
            "q25": float(np.quantile(max_tan_te, 0.25)),
            "q75": float(np.quantile(max_tan_te, 0.75)),
            "q90": float(np.quantile(max_tan_te, 0.90)),
            "min": float(max_tan_te.min()),
            "max": float(max_tan_te.max()),
        },
        "shrink_factor_stats": {
            "mean": float(shrink_te.mean()),
            "median": float(np.median(shrink_te)),
            "q10": float(np.quantile(shrink_te, 0.10)),
            "q90": float(np.quantile(shrink_te, 0.90)),
            "min": float(shrink_te.min()),
            "max": float(shrink_te.max()),
        },
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "per_seed": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "deploy_mu_full": mu_full,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_unb_in_rae": te_unb_rae,
        "delta_vs_nb2240_unb": pooled_rae_mean_seeds - rae_nb2240_unb,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled_RAE_mean        = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta vs nb2240 in_RAE = {pooled_rae_mean_seeds - rae_nb2240_unb:+.4f}")
    print(f"   verdict                = {verdict}")
    print(f"   wall                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds",
        "rae_nb2240_unb",
        "delta_vs_nb2240_unb",
        "verdict",
        "te_unb_in_rae",
        "max_tan_stats",
        "shrink_factor_stats",
    ):
        print(f"  {k}: {res.get(k)}")
