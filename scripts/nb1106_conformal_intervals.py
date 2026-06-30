"""nb1106 — Mondrian (stratified) split-conformal intervals for nb3200 (honest uncertainty, esp. the weak tail).

The research verdict: the weak-tail error is partly IRREDUCIBLE noise -> the right move is honest INTERVALS / abstention,
not confident point predictions. This builds distribution-free conformal intervals, STRATIFIED by predicted pEC50
(Mondrian), so the low/noisy stratum correctly gets WIDER intervals. Honest cross-fit: for each compound, the
nonconformity quantile is computed from the OTHER compounds in its stratum (scaffold-CV, leakage-free).

Reports: per-stratum 80/90% interval half-widths, empirical coverage (should hit nominal), and shows the weak stratum
needs wider intervals. Figure -> data/processed/figures/nb1106_conformal.png. This is a TRUST/calibration deliverable
(does NOT change RAE) — exactly what an academic reviewer asks for on a noisy potency tail.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = "data/processed"; FIG = f"{P}/figures"


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def main():
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy")
    te = load_test(); scaf = np.array([murcko(s) for s in te["smiles"].to_numpy()[unb]])
    nc = np.abs(y - anchor)                                   # nonconformity = |error|
    # stratum by predicted pEC50 tercile
    qt = np.quantile(anchor, [0, 1/3, 2/3, 1.0]); strat = np.digitize(anchor, qt[1:-1])
    names = {0: "predicted-LOW", 1: "predicted-MID", 2: "predicted-HIGH"}

    print(f"nb3200 RAE {rae(y, anchor):.4f}\n")
    print(f"{'stratum':16s} {'n':>4s} {'mean|err|':>9s} {'q80_halfwidth':>14s} {'q90_halfwidth':>14s} "
          f"{'cov80':>6s} {'cov90':>6s}")
    res = {}; widths90 = {}
    for s in [0, 1, 2]:
        m = strat == s
        # cross-fit conformal: each compound's quantile from OTHER scaffolds in-stratum (5-fold, avg 20 seeds)
        cov80s, cov90s, w80s, w90s = [], [], [], []
        for seed in range(1200, 1220):
            covered80 = np.zeros(m.sum(), bool); covered90 = np.zeros(m.sum(), bool)
            w80 = np.zeros(m.sum()); w90 = np.zeros(m.sum())
            idx = np.where(m)[0]
            for trn, val in scaffold_kfold_indices(scaf[m].tolist(), n_splits=5, seed=seed):
                q80 = np.quantile(nc[idx[trn]], 0.80); q90 = np.quantile(nc[idx[trn]], 0.90)
                for j, vi in enumerate(val):
                    covered80[vi] = nc[idx[vi]] <= q80; covered90[vi] = nc[idx[vi]] <= q90
                    w80[vi] = q80; w90[vi] = q90
            cov80s.append(covered80.mean()); cov90s.append(covered90.mean())
            w80s.append(w80.mean()); w90s.append(w90.mean())
        res[s] = dict(n=int(m.sum()), mean_err=float(nc[m].mean()), q80=float(np.mean(w80s)),
                      q90=float(np.mean(w90s)), cov80=float(np.mean(cov80s)), cov90=float(np.mean(cov90s)))
        widths90[s] = np.mean(w90s)
        print(f"{names[s]:16s} {m.sum():>4d} {nc[m].mean():>9.3f} {np.mean(w80s):>14.3f} {np.mean(w90s):>14.3f} "
              f"{np.mean(cov80s):>6.2f} {np.mean(cov90s):>6.2f}")

    print(f"\n-> weak/low stratum half-width {widths90[0]:.2f} vs high {widths90[2]:.2f} "
          f"({widths90[0]/max(widths90[2],1e-9):.1f}x wider) = the model HONESTLY flags the weak tail as uncertain")
    # figure: predictions with 90% conformal intervals, sorted by prediction
    order = np.argsort(anchor); hw = np.array([widths90[strat[i]] for i in range(len(y))])
    fig, ax = plt.subplots(figsize=(11, 5))
    xx = np.arange(len(y))
    ax.fill_between(xx, (anchor - hw)[order], (anchor + hw)[order], alpha=0.2, color="#2471a3", label="90% conformal interval")
    ax.plot(xx, anchor[order], color="#2471a3", lw=1, label="nb3200 prediction")
    ax.scatter(xx, y[order], s=10, color="#c0392b", alpha=0.6, label="true pEC50")
    ax.set_xlabel("test compounds (sorted by prediction)"); ax.set_ylabel("pEC50")
    ax.set_title("nb3200 predictions with Mondrian conformal intervals (wider where uncertain)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/nb1106_conformal.png", dpi=115); plt.close()
    json.dump({names[s]: res[s] for s in res}, open(f"{P}/nb1106_conformal.json", "w"), indent=2)
    print(f"wrote {FIG}/nb1106_conformal.png")


if __name__ == "__main__":
    main()
