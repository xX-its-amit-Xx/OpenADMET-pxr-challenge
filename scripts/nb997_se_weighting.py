"""nb997 — test inverse-variance (assay-SE) weighting: down-weight the noisy low-range pEC50
labels (SE up to 0.51) so the model stops fitting label noise. Grounded in the SE-by-range data.
(1) SE-by-range figure. (2) does SE-weighted training beat unweighted? 4139->253 deploy + scaffold-CV
degradation curve, multi-seed. Output -> C:/pxr_struct/dash/.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = "data/processed"; OUT = "C:/pxr_struct/dash"; os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 101, 202, 303, 404, 505, 606]


def main():
    tr = load_train().dropna(subset=["pec50", "pec50_se"]).reset_index(drop=True)
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y253 = np.load(f"{D}/_audit_unblind_y.npy")
    y = tr["pec50"].to_numpy(float); se = tr["pec50_se"].to_numpy(float)
    se = np.clip(se, 0.05, 2.0)
    smiles = tr["smiles"].tolist(); te_smiles = te["smiles"].to_numpy()[unb].tolist()

    # ---- (1) SE-by-range figure ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = [(0, 3), (3, 4), (4, 5), (5, 6), (6, 9)]
    xs = [f"{a}-{b}" for a, b in bins]
    med = [np.median(se[(y >= a) & (y < b)]) for a, b in bins]
    ns = [int(((y >= a) & (y < b)).sum()) for a, b in bins]
    bars = ax.bar(range(len(bins)), med, color=["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"])
    for i, (m, n) in enumerate(zip(med, ns)): ax.text(i, m + 0.01, f"SE={m:.2f}\nn={n}", ha="center", fontsize=9)
    ax.axhline(0.05, ls="--", c="grey", label="high-range floor 0.05")
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(xs); ax.set_xlabel("pEC50 range")
    ax.set_ylabel("median assay SE (log units)")
    ax.set_title("Assay noise EXPLODES at low pEC50 — labels under ~4 are ~10x noisier\n"
                 "=> down-weight them (inverse-variance) instead of fitting the noise")
    ax.legend(); plt.tight_layout(); plt.savefig(f"{OUT}/nb997_se_by_range.png", dpi=140); plt.close()

    Xtr = impute(combined(smiles)).astype(np.float32)
    Xte = impute(combined(te_smiles)).astype(np.float32)
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in smiles]

    # weighting schemes
    schemes = {"unweighted": np.ones_like(se), "inv_se": 1.0 / se, "inv_se2": 1.0 / se ** 2,
               "inv_se2_cap": np.clip(1.0 / se ** 2, 0, 50)}

    # ---- (2a) deploy 4139->253 ----
    print("=== deploy (train 4139, predict 253) RAE by weighting ===")
    dep = {}
    for nm, w in schemes.items():
        m = lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, n_jobs=4, verbose=-1)
        m.fit(Xtr, y, sample_weight=w)
        dep[nm] = round(float(rae(y253, m.predict(Xte))), 4)
        print(f"  {nm:14s} 253-RAE = {dep[nm]}")

    # ---- (2b) scaffold-CV degradation, multi-seed: inv_se2_cap vs unweighted ----
    print("\n=== scaffold-CV multi-seed: inv_se2_cap vs unweighted (does weighting generalize?) ===")
    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        oof_u = np.full(len(y), np.nan); oof_w = np.full(len(y), np.nan)
        for tri, vai in folds:
            oof_u[vai] = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(Xtr[tri], y[tri]).predict(Xtr[vai])
            oof_w[vai] = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(Xtr[tri], y[tri], sample_weight=schemes["inv_se2_cap"][tri]).predict(Xtr[vai])
        d = rae(y, oof_w) - rae(y, oof_u)
        rows.append({"seed": seed, "unw": round(rae(y, oof_u), 4), "wtd": round(rae(y, oof_w), 4), "delta": round(d, 5)})
        print(f"  seed {seed}: unweighted={rae(y,oof_u):.4f} se-weighted={rae(y,oof_w):.4f} delta={d:+.5f}")
    dd = np.array([r["delta"] for r in rows]); stable = dd.mean() < 0 and abs(dd.mean()) > dd.std()
    print(f"\nSE-weighting delta: mean={dd.mean():+.5f} std={dd.std():.5f} wins={int((dd<0).sum())}/{len(SEEDS)} stable={stable}")
    print(">>> SE-weighting HELPS (real)" if stable else ">>> SE-weighting not stably better on RAE (RAE penalizes low-range either way)")
    json.dump({"deploy_253": dep, "cv_rows": rows, "delta_mean": float(dd.mean()), "stable": bool(stable)},
              open(f"{OUT}/nb997_se_weighting.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb997_se_by_range.png + nb997_se_weighting.json")


if __name__ == "__main__":
    main()
