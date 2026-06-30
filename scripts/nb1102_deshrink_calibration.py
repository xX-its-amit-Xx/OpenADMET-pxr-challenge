"""nb1102 — honest cross-fit DE-SHRINKAGE / calibration on nb3200 (targets the shrinkage the diagnostic found).

nb1101: nb3200 shrinks the dynamic range — over-predicts weak (+0.30), under-predicts strong (-0.26). The fix for
symmetric shrinkage is DE-SHRINKAGE (stretch away from the center), NOT a low-end shift. But in-sample stretch
overfits (our documented trap) -> test STRICTLY cross-fit: scaffold 5-fold, fit the calibration map on train-folds,
apply to the held-out fold, aggregate OOF, 30 fresh seeds. Compare honest RAE vs nb3200 0.4416.

Methods (low-df = safe at n=253; isotonic = high-df control expected to overfit):
  linear_stretch   pred' = c + s*(pred-c), fit s (1 param) minimizing train-fold RAE
  affine_platt     pred' = a*pred + b      (2 param)
  asym_stretch     separate s_lo/s_hi below/above center (2 param)
  quantile_down    shift predicted-low down by a fitted amount (research rec, directional)
  isotonic_xfit    cross-fit isotonic (high df; control)
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.isotonic import IsotonicRegression

P = "data/processed"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def fit_apply(method, ptr, ytr, pte):
    c = np.median(ptr)
    if method == "linear_stretch":
        best_s, best = 1.0, 1e9
        for s in np.linspace(0.7, 2.0, 53):
            r = rae(ytr, c + s * (ptr - c))
            if r < best: best, best_s = r, s
        return c + best_s * (pte - c), best_s
    if method == "affine_platt":
        # least-RAE affine via grid on a, b
        best, ba, bb = 1e9, 1.0, 0.0
        for a in np.linspace(0.7, 2.0, 27):
            res = ytr - a * ptr; b = np.median(res)
            r = rae(ytr, a * ptr + b)
            if r < best: best, ba, bb = r, a, b
        return ba * pte + bb, ba
    if method == "asym_stretch":
        best, bl, bh = 1e9, 1.0, 1.0
        for sl in np.linspace(0.7, 2.0, 18):
            for sh in np.linspace(0.7, 2.0, 18):
                q = np.where(ptr < c, c + sl * (ptr - c), c + sh * (ptr - c))
                r = rae(ytr, q)
                if r < best: best, bl, bh = r, sl, sh
        out = np.where(pte < c, c + bl * (pte - c), c + bh * (pte - c)); return out, (bl, bh)
    if method == "quantile_down":
        thr = np.quantile(ptr, 0.35); best, bd = 1e9, 0.0
        for d in np.linspace(0, 1.0, 41):
            q = ptr.copy(); q[q < thr] -= d
            r = rae(ytr, q)
            if r < best: best, bd = r, d
        out = pte.copy(); out[out < thr] -= bd; return out, bd
    if method == "isotonic_xfit":
        iso = IsotonicRegression(out_of_bounds="clip").fit(ptr, ytr)
        return iso.predict(pte), 0.0


def main():
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy")
    te = load_test(); scaf = [murcko(s) for s in te["smiles"].to_numpy()[unb]]
    base = rae(y, anchor)
    print(f"nb3200 anchor RAE {base:.4f}\n")
    methods = ["linear_stretch", "affine_platt", "asym_stretch", "quantile_down", "isotonic_xfit"]
    print(f"{'method':16s} {'mean_RAE':>9s} {'delta':>9s} {'frac_improved':>14s} {'mean_param':>22s}")
    out = {}
    for meth in methods:
        raes, params = [], []
        for seed in range(1200, 1230):
            oof = np.zeros(len(y))
            for trn, val in scaffold_kfold_indices(scaf, n_splits=5, seed=seed):
                cal, pr = fit_apply(meth, anchor[trn], y[trn], anchor[val]); oof[val] = cal
                params.append(pr)
            raes.append(rae(y, oof))
        mr = float(np.mean(raes)); fi = float(np.mean(np.array(raes) < base - 1e-9))
        pmean = np.mean([p if np.isscalar(p) else p[0] for p in params])
        out[meth] = dict(mean_rae=mr, delta=mr - base, frac_improved=fi)
        print(f"{meth:16s} {mr:>9.4f} {mr-base:>+9.4f} {fi:>14.2f} {pmean:>22.3f}")
    json.dump({"anchor": base, **out}, open(f"{P}/nb1102_deshrink.json", "w"), indent=2)
    print("\nGATE: a real deploy lever needs stable mean_RAE < 0.4416 with high frac_improved across the 30 seeds.")


if __name__ == "__main__":
    main()
