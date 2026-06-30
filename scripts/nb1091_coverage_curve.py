"""nb1091 — COVERAGE CURVE + TARGETED-vs-RANDOM acquisition (the actionable 'what could we have done').

nb1090 proved (on ground truth) that the lever is COVERAGE: FXR/PPARg gained -0.16/-0.23 RAE from 400->full train.
This quantifies the SHAPE and the OracleDelta:

  For FXR, PPARg (rich) on a FIXED scaffold-disjoint blinded test:
    - RANDOM curve : RAE vs train size n in {50,100,200,400,800,1600,full}, train cpds sampled at random (blind)
    - TARGETED curve: same sizes, but train cpds chosen GREEDILY as the most-similar-to-test available
                      (the 'if we had known where to measure' oracle; still scaffold-disjoint -> no leakage)
    - elbow: where does the random curve flatten? (is PXR's ~4139 past it or still climbing?)
    - OracleDelta(n) = RAE_random(n) - RAE_targeted(n): the value of KNOWING where to sample at budget n.

Decisive for PXR: if targeted at small n ~ random at large n, then SMART acquisition substitutes for raw volume
-> Phase-2 active learning (measure compounds near the test analogs) is the concrete -0.1..-0.2 RAE move.
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
SIZES = [50, 100, 200, 400, 800, 1600]
N_TEST = 250
SEED = 42
rng = np.random.default_rng(SEED)


def murcko(s):
    m = Chem.MolFromSmiles(str(s))
    if not m: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception: return None


def lgbm(Xtr, ytr, Xte):
    m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=4, verbose=-1)
    m.fit(Xtr, ytr); return m.predict(Xte)


def run(name, df):
    df_t = df[df["target_name"] == name].drop_duplicates("inchikey").reset_index(drop=True)
    df_t["scaf"] = df_t["std_smiles"].map(murcko); df_t = df_t.dropna(subset=["scaf"]).reset_index(drop=True)
    folds = scaffold_kfold_indices(df_t["scaf"].tolist(), n_splits=max(2, round(len(df_t) / N_TEST)), seed=SEED)
    te = min((f[1] for f in folds), key=lambda ix: abs(len(ix) - N_TEST))
    te = te[:N_TEST]
    te_set = set(te.tolist()); pool = np.array([i for i in range(len(df_t)) if i not in te_set])
    smi = df_t["std_smiles"].tolist(); y = df_t["pec50"].to_numpy().astype(float)
    Xc = impute(combined(smi)); yte = y[te]
    full_n = len(pool)
    print(f"[{name}] pool={full_n} test={len(te)}", flush=True)

    # similarity of each pool compound to the test set (max Tanimoto) for targeted acquisition
    M = morgan_fp_batch(smi).astype(np.float32); s = M.sum(1)
    sim_pool_te = (M[pool] @ M[te].T)
    sim_pool_te = sim_pool_te / np.clip(s[pool, None] + s[te][None, :] - sim_pool_te, 1, None)
    pool_maxsim = sim_pool_te.max(1)
    order_targeted = pool[np.argsort(pool_maxsim)[::-1]]       # most test-similar first

    sizes = [n for n in SIZES if n <= full_n] + [full_n]
    curve = []
    for n in sizes:
        # random (avg of 3 seeds to de-noise)
        rr = []
        for sd in range(3):
            r = np.random.default_rng(SEED + sd)
            sub = r.permutation(pool)[:n]
            rr.append(rae(yte, lgbm(Xc[sub], y[sub], Xc[te])))
        rae_rand = float(np.mean(rr))
        # targeted: top-n most test-similar
        sub_t = order_targeted[:n]
        rae_targ = float(rae(yte, lgbm(Xc[sub_t], y[sub_t], Xc[te])))
        curve.append(dict(n=int(n), rae_random=rae_rand, rae_targeted=rae_targ,
                          oracle_delta=rae_rand - rae_targ,
                          mean_trainsim=float(pool_maxsim[np.argsort(pool_maxsim)[::-1][:n]].mean())))
        print(f"  n={n:5d}  random={rae_rand:.4f}  targeted={rae_targ:.4f}  oracleD={rae_rand-rae_targ:+.4f}", flush=True)
    return dict(target=name, pool=full_n, n_test=int(len(te)), curve=curve)


def main():
    df = pd.read_parquet(PARQ)
    df = df[df["standard_type"].isin(["EC50", "AC50"])].reset_index(drop=True)
    out = [run(t, df) for t in TARGETS]
    json.dump(out, open(f"{P}/nb1091_coverage_curve.json", "w"), indent=2)

    print("\n=== COVERAGE CURVE INTERPRETATION ===")
    for o in out:
        c = o["curve"]; r = {x["n"]: x for x in c}
        small = c[0]["n"]; big = c[-1]["n"]
        print(f"\n{o['target']} (pool {o['pool']}):")
        print(f"  random RAE: n={small} {r[small]['rae_random']:.3f}  ->  n={big} {r[big]['rae_random']:.3f} "
              f"(coverage gain {r[small]['rae_random']-r[big]['rae_random']:+.3f})")
        # how much train does TARGETED need to match RANDOM-at-full?
        full_rand = r[big]["rae_random"]
        match = next((x["n"] for x in c if x["rae_targeted"] <= full_rand), None)
        if match:
            print(f"  TARGETED reaches random-at-full ({full_rand:.3f}) with only n={match} "
                  f"({match/big*100:.0f}% of the data) -> smart acquisition substitutes for volume")
        avg_oracle = np.mean([x["oracle_delta"] for x in c if x["n"] <= 800])
        print(f"  mean OracleDelta (n<=800) = {avg_oracle:+.4f} RAE = value of knowing where to measure")
    print(f"\nwrote {P}/nb1091_coverage_curve.json")


if __name__ == "__main__":
    main()
