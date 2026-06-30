"""nb1093 — DIVERSITY-AWARE acquisition (resolves the FXR backfire; the deployable 'what we could have done').

nb1091: targeted-by-max-similarity-to-test gave +0.14 RAE on PPARg (800 cpds == 2266 random) but BACKFIRED on FXR
(naive nearest-to-test collapses train diversity -> worse at mid budget). The fix: a COVERAGE objective that spreads
selection across ALL test regions, not just the easy cluster. Compare at each budget n, same fixed blinded test:

  - RANDOM          : blind baseline (3-seed avg)
  - MAXSIM          : greedy nearest-to-test (nb1091's naive oracle)
  - COVERAGE-GREEDY : facility-location / max-min-coverage — each pick maximizes the gain in min test coverage
                      (sum of improvements in per-test max-sim-to-selected). diversity-aware, still test-informed.

If coverage-greedy beats BOTH random and maxsim on BOTH targets (incl. FXR), that is the robust acquisition strategy
PXR Phase-2 should use: measure compounds that COVER the test manifold, not just resemble its center.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

P = "data/processed"
PARQ = "data/external/papyrus_pxr_nr.parquet"
TARGETS = ["FXR", "PPARg"]
SIZES = [100, 200, 400, 800, 1600]
N_TEST = 250
SEED = 42


def murcko(s):
    m = Chem.MolFromSmiles(str(s))
    if not m: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception: return None


def lgbm(Xtr, ytr, Xte):
    m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbose=-1)
    m.fit(Xtr, ytr); return m.predict(Xte)


def coverage_greedy(sim_pool_te, budget):
    """Greedily pick pool rows maximizing gain in summed per-test max-coverage (facility location). Returns indices into pool."""
    nP, nT = sim_pool_te.shape
    cov = np.zeros(nT); chosen = []; avail = np.ones(nP, bool)
    # precompute is heavy (nP x nT each step); use vectorized marginal gain = sum(max(sim-cov,0))
    for _ in range(min(budget, nP)):
        gain = np.where(avail[:, None], np.maximum(sim_pool_te - cov[None, :], 0), -1).sum(1)
        j = int(np.argmax(gain)); chosen.append(j); avail[j] = False
        cov = np.maximum(cov, sim_pool_te[j])
    return np.array(chosen)


def run(name, df):
    df_t = df[df["target_name"] == name].drop_duplicates("inchikey").reset_index(drop=True)
    df_t["scaf"] = df_t["std_smiles"].map(murcko); df_t = df_t.dropna(subset=["scaf"]).reset_index(drop=True)
    folds = scaffold_kfold_indices(df_t["scaf"].tolist(), n_splits=max(2, round(len(df_t) / N_TEST)), seed=SEED)
    te = min((f[1] for f in folds), key=lambda ix: abs(len(ix) - N_TEST))[:N_TEST]
    te_set = set(te.tolist()); pool = np.array([i for i in range(len(df_t)) if i not in te_set])
    smi = df_t["std_smiles"].tolist(); y = df_t["pec50"].to_numpy().astype(float)
    Xc = impute(combined(smi)); yte = y[te]
    M = morgan_fp_batch(smi).astype(np.float32); s = M.sum(1)
    sim_pte = (M[pool] @ M[te].T); sim_pte = sim_pte / np.clip(s[pool, None] + s[te][None, :] - sim_pte, 1, None)
    order_max = pool[np.argsort(sim_pte.max(1))[::-1]]
    cov_order = pool[coverage_greedy(sim_pte, max(SIZES))]      # full coverage-greedy ranking once
    print(f"[{name}] pool={len(pool)} test={len(te)}", flush=True)

    sizes = [n for n in SIZES if n <= len(pool)]
    curve = []
    for n in sizes:
        rr = [rae(yte, lgbm(Xc[np.random.default_rng(SEED + sd).permutation(pool)[:n]],
                            y[np.random.default_rng(SEED + sd).permutation(pool)[:n]], Xc[te])) for sd in range(3)]
        r_rand = float(np.mean(rr))
        r_max = float(rae(yte, lgbm(Xc[order_max[:n]], y[order_max[:n]], Xc[te])))
        r_cov = float(rae(yte, lgbm(Xc[cov_order[:n]], y[cov_order[:n]], Xc[te])))
        curve.append(dict(n=int(n), random=r_rand, maxsim=r_max, coverage=r_cov))
        print(f"  n={n:5d}  random={r_rand:.4f}  maxsim={r_max:.4f}  coverage={r_cov:.4f}  "
              f"(cov vs rand {r_cov-r_rand:+.4f}, cov vs maxsim {r_cov-r_max:+.4f})", flush=True)
    return dict(target=name, pool=int(len(pool)), curve=curve)


def main():
    df = pd.read_parquet(PARQ)
    df = df[df["standard_type"].isin(["EC50", "AC50"])].reset_index(drop=True)
    out = [run(t, df) for t in TARGETS]
    json.dump(out, open(f"{P}/nb1093_acquisition.json", "w"), indent=2)
    print("\n=== SMART ACQUISITION (coverage-greedy vs random vs naive maxsim) ===")
    for o in out:
        cov_wins = np.mean([x["coverage"] < x["random"] for x in o["curve"]])
        best_n = min(o["curve"], key=lambda x: x["coverage"])
        full_rand = o["curve"][-1]["random"]
        match = next((x["n"] for x in o["curve"] if x["coverage"] <= full_rand), None)
        print(f"\n{o['target']}: coverage-greedy beats random at {cov_wins*100:.0f}% of budgets; "
              f"reaches random-at-{o['curve'][-1]['n']} ({full_rand:.3f}) "
              + (f"with n={match} ({match/o['curve'][-1]['n']*100:.0f}% data)" if match else "(not within tested budgets)"))
    print(f"\nwrote {P}/nb1093_acquisition.json")


if __name__ == "__main__":
    main()
