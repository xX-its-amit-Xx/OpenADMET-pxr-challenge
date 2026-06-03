"""nb1100 -- TabPFN regressor on PCA-50 of combined (Morgan + RDKit).

Protocol per request:
  1. Load TabPFNRegressor (ICL transformer).
  2. PCA-reduce combined features (Morgan 2048 + RDKit ~217 = 2265) to 50 dims.
  3. Fit on 4139 CRC pEC50; predict on 513.
  4. Compute in_RAE on 253 unblind index. Pearson vs nb972.
  5. If Pearson < 0.95, bag 3 seeds.

Wall-time target: <15 min.
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1100"
PCA_DIM = 50
SEEDS = [42, 0, 137]
NB1070_REF = 0.5715  # in_RAE on 253 (deploy median)


def in_rae(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    return float(np.mean(np.abs(yt - yp)) / np.mean(np.abs(yt - yt.mean())))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- TabPFNRegressor on PCA-{PCA_DIM} of combined features")
    print("=" * 78)

    # Truth/unblind
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    assert len(unb_idx) == len(y_unb) == 253

    # Data
    tr = load_train()
    te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    print(f"[data] train={len(tr)}  test={len(te)}")

    # Features
    print("[feat] computing combined Morgan+RDKit ...")
    X_tr_raw = impute(combined(tr["smiles"].tolist()))
    X_te_raw = impute(combined(te["smiles"].tolist()))
    print(f"   X_tr_raw={X_tr_raw.shape}  X_te_raw={X_te_raw.shape}")

    # Scale + PCA
    print(f"[pca] standardize + PCA to {PCA_DIM} dims ...")
    sc = StandardScaler(with_mean=True, with_std=True)
    X_tr_s = sc.fit_transform(X_tr_raw)
    X_te_s = sc.transform(X_te_raw)
    pca = PCA(n_components=PCA_DIM, random_state=42)
    X_tr = pca.fit_transform(X_tr_s).astype(np.float32)
    X_te = pca.transform(X_te_s).astype(np.float32)
    var_ratio = float(pca.explained_variance_ratio_.sum())
    print(f"   PCA shapes: X_tr={X_tr.shape}  X_te={X_te.shape}  "
          f"var_ratio_sum={var_ratio:.3f}")

    # nb972 deploy for Pearson
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(
        np.float64)

    # First pass: seed=42
    from tabpfn import TabPFNRegressor

    def fit_predict_one(seed: int) -> np.ndarray:
        t_seed = time.time()
        print(f"\n[seed {seed}] fitting TabPFNRegressor ...")
        reg = TabPFNRegressor(
            device="cpu",
            random_state=seed,
            ignore_pretraining_limits=True,
        )
        reg.fit(X_tr, y_tr)
        preds = reg.predict(X_te).astype(np.float64)
        elapsed = time.time() - t_seed
        print(f"   seed {seed}: pred mean={preds.mean():.3f}  "
              f"std={preds.std():.3f}  elapsed={elapsed:.1f}s")
        return preds

    seed_preds = {}
    seed_preds[SEEDS[0]] = fit_predict_one(SEEDS[0])
    in_rae_s0 = in_rae(y_unb, seed_preds[SEEDS[0]][unb_idx])
    pr_s0, _ = pearsonr(seed_preds[SEEDS[0]], te_nb972)
    print(f"\n[seed {SEEDS[0]}] in_RAE(253) = {in_rae_s0:.4f}  "
          f"Pearson vs nb972 = {pr_s0:.4f}")

    bagged = seed_preds[SEEDS[0]]
    did_bag = False
    seed_inrae = {SEEDS[0]: in_rae_s0}
    seed_pearson = {SEEDS[0]: float(pr_s0)}

    if pr_s0 < 0.95 and (time.time() - t0) < 600:
        did_bag = True
        for s in SEEDS[1:]:
            if (time.time() - t0) > 780:
                print(f"   skipping seed {s} -- wall time budget exceeded")
                break
            seed_preds[s] = fit_predict_one(s)
            ir = in_rae(y_unb, seed_preds[s][unb_idx])
            pr, _ = pearsonr(seed_preds[s], te_nb972)
            seed_inrae[s] = ir
            seed_pearson[s] = float(pr)
            print(f"   seed {s}: in_RAE = {ir:.4f}  Pearson vs nb972 = "
                  f"{pr:.4f}")
        stack = np.stack([seed_preds[s] for s in seed_preds.keys()], axis=0)
        bagged = stack.mean(axis=0)

    in_rae_final = in_rae(y_unb, bagged[unb_idx])
    pr_final, _ = pearsonr(bagged, te_nb972)
    beats_nb1070 = in_rae_final < NB1070_REF

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   per-seed in_RAE: {seed_inrae}")
    print(f"   per-seed Pearson(nb972): {seed_pearson}")
    print(f"   final in_RAE(253) = {in_rae_final:.4f}  "
          f"(nb1070 ref = {NB1070_REF:.4f})")
    print(f"   final Pearson vs nb972 = {pr_final:.4f}")
    print(f"   bagged? = {did_bag}  beats_nb1070 = {beats_nb1070}")

    plain = SUBMISSIONS / f"{TAG}_tabpfn_pca50.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te["name"].values,
        "pEC50": bagged.astype(np.float32),
    }).to_csv(plain, index=False)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", bagged.astype(np.float32))
    print(f"[save] {plain}")
    print(f"[save] te_{TAG}.npy")

    summary = {
        "tag": TAG,
        "pca_dim": PCA_DIM,
        "pca_var_ratio_sum": var_ratio,
        "seeds_run": list(seed_preds.keys()),
        "seed_in_rae": seed_inrae,
        "seed_pearson_vs_nb972": seed_pearson,
        "did_bag": bool(did_bag),
        "in_rae_final": float(in_rae_final),
        "pearson_with_nb972": float(pr_final),
        "beats_nb1070": bool(beats_nb1070),
        "nb1070_ref": NB1070_REF,
        "wall_sec": round(time.time() - t0, 2),
        "plain_submission": str(plain),
    }
    out = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("in_rae_final", "pearson_with_nb972", "beats_nb1070",
              "did_bag", "wall_sec", "plain_submission"):
        print(f"  {k}: {res.get(k)}")
