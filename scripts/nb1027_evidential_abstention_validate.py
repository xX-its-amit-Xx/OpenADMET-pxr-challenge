"""nb1027 — validate EVIDENTIAL abstention (from Kaggle nb1026 T4 run) on the honest 253.

Prereq diagnostic: does evidential EPISTEMIC uncertainty predict nb3200's error? (corr with |err| and with
SIGNED err — F2 is OVER-prediction of novel inactives, so we expect uncertainty to track positive residual).
If yes -> uncertainty-gated shrinkage toward the low-active prior should reduce RAE on the novel tail.

Honest protocol: epistemic for the 253 comes from the evidential model (never saw 253 labels). The gate params
(shrink weight, uncertainty threshold, prior quantile) are fit per-fold on train-folds of the 253 and applied to
the held-out fold -> scaffold 5-fold x 30 seeds, same honest cross-fit as every other validation here.

Run after pulling: KaggleApi().kernels_output('knowledgegraphlover/pxr-challenge-nb1026', 'submissions/kaggle_nb1026')
"""
import os, sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "data/processed"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def find_evid():
    for p in ["submissions/kaggle_nb1026/found_evid_513.parquet"]:
        if os.path.exists(p):
            return p
    h = glob.glob("submissions/**/found_evid_513.parquet", recursive=True)
    return h[0] if h else None


def shrink(anchor, u_pct, w, u0, prior):
    """shrink anchor toward `prior` for high-uncertainty rows: only pulls predictions DOWN toward the low prior."""
    g = w / (1 + np.exp(-(u_pct - u0) / 0.1))
    out = anchor - g * (anchor - prior)
    return np.minimum(anchor, out) if prior < np.median(anchor) else out  # never raise a prediction


def fit_gate(anchor, y, u_pct, ytr_for_prior):
    best = (0.0, 1.0, np.median(anchor)); best_r = rae(y, anchor)
    for w in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for u0 in [0.6, 0.7, 0.8, 0.9]:
            for pq in [0.05, 0.10, 0.20]:
                prior = np.quantile(ytr_for_prior, pq)
                r = rae(y, shrink(anchor, u_pct, w, u0, prior))
                if r < best_r:
                    best_r, best = r, (w, u0, prior)
    return best


def main():
    pp = find_evid()
    assert pp, "no nb1026 evidential output — run the Kaggle T4 notebook and pull found_evid_513.parquet first"
    ev = pd.read_parquet(pp).sort_values("test_pos")
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    epi = ev["epistemic"].to_numpy()[unb]
    anchor = np.load(f"{D}/nb3200_pred_oof.npy")
    tr = load_train().dropna(subset=["pec50"]); tr_y = tr["pec50"].to_numpy()

    # --- prereq diagnostics ---
    err = y - anchor                       # signed (positive = under-pred; negative = over-pred)
    print(f"loaded {pp}")
    print(f"corr(epistemic, |err|)   = {np.corrcoef(epi, np.abs(err))[0,1]:+.3f}  (need >0 for abstention to help)")
    print(f"corr(epistemic, -err)    = {np.corrcoef(epi, -err)[0,1]:+.3f}  (>0 => uncertainty tracks OVER-prediction = F2)")

    te = load_test(); smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    trfp = morgan_fp_batch(tr["smiles"].tolist()).astype(bool); tefp = morgan_fp_batch(smiles).astype(bool)
    top1 = np.array([np.max((tefp[i] & trfp).sum(1) / np.clip((tefp[i] | trfp).sum(1), 1, None)) for i in range(len(smiles))])
    novel = top1 < np.median(top1)
    print(f"corr(epistemic, novelty[1-top1]) = {np.corrcoef(epi, 1-top1)[0,1]:+.3f}  (epi should be high on novel)")

    # --- honest cross-fit gate ---
    deltas = []
    for s in range(1400, 1430):
        folds = scaffold_kfold_indices(scaf, 5, seed=s)
        pred = anchor.copy()
        # uncertainty percentile computed on full 253 (label-free, no leak)
        u_pct = pd.Series(epi).rank(pct=True).to_numpy()
        for tri, vai in folds:
            w, u0, prior = fit_gate(anchor[tri], y[tri], u_pct[tri], tr_y)
            pred[vai] = shrink(anchor[vai], u_pct[vai], w, u0, prior)
        deltas.append(rae(y, pred) - rae(y, anchor))
    deltas = np.array(deltas); st = deltas.mean() < 0 and abs(deltas.mean()) > deltas.std()
    print(f"\nabstention-shrinkage (30 seeds cross-fit): delta {deltas.mean():+.5f} +/- {deltas.std():.5f} "
          f"wins {int((deltas<0).sum())}/30 stable={st}")

    # novelty-stratified single-shot (fixed a-priori gate: w=0.6,u0=0.8,prior=q10) for interpretability
    prior = np.quantile(tr_y, 0.10); u_pct = pd.Series(epi).rank(pct=True).to_numpy()
    pred = shrink(anchor, u_pct, 0.6, 0.8, prior)
    for nm, mk in [("NEAR", ~novel), ("NOVEL", novel)]:
        print(f"  {nm:6s} n={mk.sum():3d}  anchor {rae(y[mk], anchor[mk]):.4f}  shrunk {rae(y[mk], pred[mk]):.4f}")

    json.dump({"corr_epi_abserr": float(np.corrcoef(epi, np.abs(err))[0,1]),
               "corr_epi_overpred": float(np.corrcoef(epi, -err)[0,1]),
               "xfit_delta": float(deltas.mean()), "xfit_std": float(deltas.std()), "stable": bool(st)},
              open(f"{D}/nb1027_abstention.json", "w"), indent=2)


if __name__ == "__main__":
    main()
