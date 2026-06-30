"""nb1139 — test-adjacent GENERATION lever, honestly validated via a PSEUDO-TEST.

The benefit of generating on-manifold analogs near the TEST set lands in the TEST region, which our normal train-holdout
gate can't see. So we validate the METHOD on a held-out train sub-region: hold out a scaffold cluster as pseudo-test,
generate analogs near it (RDKit BRICS), label them by k-NN read-across from the TRAIN FOLD (independent of our model),
add as weighted extra training rows, and check if RAE on the pseudo-test IMPROVES vs baseline. If yes -> the lever works
and we apply it to the real 513; if null -> the lever is closed (honestly). 3 scaffold splits.
"""
import os, sys, itertools, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.chem import morgan_fp_batch, standardize
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem import AllChem
from lightgbm import LGBMRegressor
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def fp(smi):
    m = Chem.MolFromSmiles(str(smi))
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None


def gen_analogs(seed_smis, frag_pool, cap=600):
    """Fast 1-fragment-swap mutations of each seed compound (small fragment sets -> BRICSBuild yields instantly).
    Each analog = a seed with ONE BRICS fragment replaced by a pool fragment -> guaranteed near the seed."""
    import random
    rng = random.Random(0)
    pool = [f for f in frag_pool if Chem.MolFromSmiles(f)]
    if not pool: return []
    out = []
    for s in seed_smis:
        m = Chem.MolFromSmiles(str(s))
        if m is None: continue
        try: sfrags = list(BRICS.BRICSDecompose(m))
        except Exception: continue
        if len(sfrags) < 2: continue
        for _ in range(4):
            fr = list(sfrags); fr[rng.randrange(len(fr))] = rng.choice(pool)
            fmols = [x for x in (Chem.MolFromSmiles(f) for f in fr) if x is not None]
            try:
                for built in itertools.islice(BRICS.BRICSBuild(fmols), 2):
                    built.UpdatePropertyCache(strict=False); out.append(Chem.MolToSmiles(built)); break
            except Exception:
                pass
            if len(out) >= cap: return out
    return out


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    y = tr["pec50"].to_numpy(); smis = tr["smiles"].tolist()
    scaf = np.array([murcko(s) for s in smis])
    Xall = impute(combined(smis)).astype(np.float32)
    trfp = morgan_fp_batch(smis).astype(np.float32)   # (N,2048) for vectorized Tanimoto

    def tanimoto(A, B):  # A:(a,2048) B:(b,2048) -> (a,b)
        inter = A @ B.T
        return inter / (A.sum(1)[:, None] + B.sum(1)[None, :] - inter + 1e-9)

    results = []
    for seed in range(3):
        folds = scaffold_kfold_indices(scaf.tolist(), n_splits=round(len(y)/250), seed=200+seed)
        ho = min((f[1] for f in folds), key=lambda ix: abs(len(ix)-250))
        trn = np.array([i for i in range(len(y)) if i not in set(ho.tolist())])
        ho_smis = [smis[i] for i in ho]

        # baseline: train on trn, predict pseudo-test ho
        lo, hi = np.quantile(y[trn], 0.05), np.quantile(y[trn], 0.98)
        base = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(Xall[trn], y[trn])
        rae_base = rae(y[ho], np.clip(base.predict(Xall[ho]), lo, hi))

        # generate analogs near the pseudo-test scaffolds
        print(f"seed {seed}: generating...", flush=True)
        pool = []
        for i in trn[:150]:
            try: pool += list(BRICS.BRICSDecompose(Chem.MolFromSmiles(smis[i])))
            except Exception: pass
        gen = gen_analogs(ho_smis[:40], list(dict.fromkeys(pool))[:120], cap=400)  # mutate up to 40 pseudo-test compounds
        print(f"seed {seed}: generated {len(gen)} raw", flush=True)
        canon = set()
        for g in gen:
            m = Chem.MolFromSmiles(g) if g else None
            if m is not None and 6 <= m.GetNumHeavyAtoms() <= 60:
                canon.add(Chem.MolToSmiles(m))
        gen = list(canon)
        if not gen:
            results.append((rae_base, rae_base, 0)); continue
        gfp = morgan_fp_batch(gen).astype(np.float32)

        # ON-MANIFOLD filter (vectorized): keep analogs near the pseudo-test, confidently read-across from train fold
        sim_ho = tanimoto(gfp, trfp[ho]).max(1)                 # (Ngen,)
        S = tanimoto(gfp, trfp[trn])                            # (Ngen, Ntrn)
        keep, labels, wts = [], [], []
        for j in range(len(gen)):
            if sim_ho[j] < 0.4:
                continue
            s = S[j]; top = np.argsort(-s)[:5]
            if s[top].mean() < 0.3 or s.max() > 0.98:           # need confident read-across, skip near-dup of train
                continue
            w = s[top]; lab = float((w * y[trn][top]).sum() / w.sum())
            keep.append(gen[j]); labels.append(lab); wts.append(float(s[top].mean()))
        if not keep:
            results.append((rae_base, rae_base, 0)); continue

        Xgen = impute(combined(keep)).astype(np.float32)
        Xaug = np.vstack([Xall[trn], Xgen]); yaug = np.concatenate([y[trn], np.array(labels)])
        sw = np.concatenate([np.ones(len(trn)), np.array(wts) * 0.5])  # down-weight synthetic by read-across confidence
        aug = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, n_jobs=4, verbose=-1).fit(Xaug, yaug, sample_weight=sw)
        rae_aug = rae(y[ho], np.clip(aug.predict(Xall[ho]), lo, hi))
        results.append((rae_base, rae_aug, len(keep)))
        print(f"seed {seed}: pseudo-test base {rae_base:.4f} | +{len(keep)} synth {rae_aug:.4f} | delta {rae_aug-rae_base:+.4f}", flush=True)

    rb = np.mean([r[0] for r in results]); ra = np.mean([r[1] for r in results]); ng = int(np.mean([r[2] for r in results]))
    verdict = "HELPS" if ra < rb - 0.002 else ("noise" if abs(ra-rb) <= 0.002 else "HURTS")
    print(f"\nPSEUDO-TEST mean: base {rb:.4f} | +synth {ra:.4f} | delta {ra-rb:+.4f} ({ng} analogs/fold) -> {verdict}")
    json.dump({"base": round(rb,4), "aug": round(ra,4), "delta": round(ra-rb,4), "n_analogs": ng, "verdict": verdict},
              open(f"{P}/nb1139_test_adjacent_gen.json","w"), indent=2)


if __name__ == "__main__":
    main()
