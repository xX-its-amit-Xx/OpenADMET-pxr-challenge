"""nb1111 — BRICS fragment-overlap COVERAGE feature (research TIER-1; orthogonal to the chempropembed sink).

BELKA finding: in analog-expansion test sets the exploitable structure is SHARED FRAGMENTS, not deep representation.
A test compound can be whole-molecule Tanimoto-novel yet have every PIECE seen in train. Build, per compound, a
fragment-SUPPORT coverage signal (discrete set-overlap, NOT a learned embedding) and test it the honest way:
  - corr(frag-support, nb3200 |error|)  -> does low fragment support flag high error? (abstention gate)
  - corr(frag-support, nb3200 signed err)-> does low support flag OVER-prediction? (shrinkage direction)
  - residual model (y-anchor) from frag features, 30-seed cross-fit -> blend delta + corr-with-error (deploy gate)
Only earns deployment if it clears corr-with-error. Figures -> C:/pxr_work/figures.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from collections import Counter
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = "data/processed"; FIG = "C:/pxr_work/figures"; os.makedirs(FIG, exist_ok=True)


def brics_frags(smi):
    m = Chem.MolFromSmiles(str(smi))
    if not m: return []
    try: return list(BRICS.BRICSDecompose(m, minFragmentSize=2))
    except Exception: return []


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); resid = y - anchor; aerr = np.abs(resid)

    print("BRICS decomposing train+test...", flush=True)
    tr_frags = [brics_frags(s) for s in tr["smiles"]]
    te_frags = [brics_frags(s) for s in te["smiles"].to_numpy()[unb]]
    vocab = Counter(f for fl in tr_frags for f in set(fl))    # train fragment -> # train compounds containing it
    ntr = len(tr)

    def feats(fraglist):
        rows = []
        for fl in fraglist:
            u = set(fl)
            if not u:
                rows.append([0, 0, 0, 0, 0]); continue
            sup = [vocab.get(f, 0) for f in u]                # train support per fragment
            seen = [s > 0 for s in sup]
            rows.append([np.mean(seen),                       # frac fragments seen in train
                         min(sup),                            # rarest fragment's train support
                         np.mean(sup),                        # mean support
                         np.log1p(min(sup)),                  # log rarest
                         len(u)])                             # n fragments
        return np.array(rows, float)

    Fte = feats(te_frags)
    names = ["frac_seen", "min_support", "mean_support", "log_min_support", "n_frags"]
    print("\n=== corr of each coverage feature with nb3200 error (the gate test) ===")
    for j, nm in enumerate(names):
        c_abs = np.corrcoef(Fte[:, j], aerr)[0, 1]
        c_sgn = np.corrcoef(Fte[:, j], resid)[0, 1]
        print(f"  {nm:16s} corr|err| {c_abs:+.3f}   corr(signed resid) {c_sgn:+.3f}")

    # residual deploy gate: predict signed residual from coverage feats, 30-seed scaffold-CV
    scaf = [murcko(s) for s in te["smiles"].to_numpy()[unb]]
    Fz = (Fte - Fte.mean(0)) / (Fte.std(0) + 1e-9)
    deltas, corrs = [], []
    for seed in range(1200, 1230):
        oof = np.zeros(len(y))
        for trn, val in scaffold_kfold_indices(scaf, n_splits=5, seed=seed):
            m = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05, n_jobs=4, verbose=-1, random_state=seed)
            m.fit(Fz[trn], resid[trn]); oof[val] = m.predict(Fz[val])
        corrs.append(np.corrcoef(oof, resid)[0, 1] if oof.std() > 1e-9 else 0)
        b = rae(y, anchor)
        for w in np.linspace(0, 1.5, 31): b = min(b, rae(y, anchor + w * oof))
        deltas.append(b - rae(y, anchor))
    print(f"\n=== DEPLOY GATE (residual-on-nb3200 from coverage feats, 30 seeds) ===")
    print(f"  corr(resid_pred, error) {np.mean(corrs):+.3f} | blend_delta {np.mean(deltas):+.4f} | "
          f"frac_improved {np.mean(np.array(deltas)<-1e-6):.2f}")

    # figure: error vs fragment support
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(Fte[:, 1], aerr, s=18, alpha=0.6, c=(y <= np.quantile(y, .25)), cmap="coolwarm")
    ax[0].set_xlabel("rarest-fragment train support"); ax[0].set_ylabel("nb3200 |error|")
    ax[0].set_title("Error vs fragment coverage (red=low activity)")
    ax[1].scatter(Fte[:, 0], resid, s=18, alpha=0.6, color="#34495e")
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xlabel("frac fragments seen in train")
    ax[1].set_ylabel("nb3200 signed residual (y-pred)"); ax[1].set_title("Over/under-prediction vs fragment coverage")
    plt.tight_layout(); plt.savefig(f"{FIG}/nb1111_fragment_overlap.png", dpi=115); plt.close()
    json.dump({"corr_err": float(np.mean(corrs)), "blend_delta": float(np.mean(deltas))},
              open(f"{P}/nb1111_fragment.json", "w"), indent=2)
    print(f"\nwrote {FIG}/nb1111_fragment_overlap.png")
    print("GATE: deployable only if corr-with-error stably >0 and blend_delta < ~-0.003.")


if __name__ == "__main__":
    main()
