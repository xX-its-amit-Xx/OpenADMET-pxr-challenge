"""nb990 — error diagnostics on nb3200's honest 253-unblind predictions.
Plots: (1) predicted vs actual (colored by scaffold novelty) + noise band, (2) |error| vs novelty
(the OOD wall), (3) residual histogram, (4) the worst-15 outlier molecules (2D structures + why).
All figures -> C:/pxr_struct/diag/. Compares nb3200 vs the chemprop_aux anchor.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

D = "data/processed"; OUT = "C:/pxr_struct/diag"; os.makedirs(OUT, exist_ok=True)
NOISE = 0.22  # pEC50 measurement SE (assay noise floor)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb]
    pred = np.load(f"{D}/nb3200_pred_oof.npy")
    anchor = np.load(f"{D}/te_chemprop_aux.npy")[unb]
    print(f"nb3200 RAE={rae(y,pred):.4f}  anchor RAE={rae(y,anchor):.4f}  n={len(y)}")

    # scaffold novelty: max Tanimoto to train + scaffold-in-train
    fp_te = morgan_fp_batch(list(smiles)); fp_tr = morgan_fp_batch(tr["smiles"].tolist())
    A = fp_te.astype(np.float32); B = fp_tr.astype(np.float32)
    inter = A @ B.T; u = A.sum(1)[:, None] + B.sum(1)[None, :] - inter; u[u == 0] = 1
    max_sim = (inter / u).max(1)
    tr_scaf = set(MurckoScaffold.MurckoScaffoldSmiles(s) for s in tr["smiles"] if Chem.MolFromSmiles(s))
    te_scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in smiles]
    novel = np.array([s is None or s not in tr_scaf for s in te_scaf])
    err = pred - y; aerr = np.abs(err)

    # ---- Figure 1: predicted vs actual ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax = axes[0]
    sc = ax.scatter(y, pred, c=max_sim, cmap="viridis", s=30, edgecolors="k", linewidths=0.3)
    lim = [min(y.min(), pred.min()) - 0.3, max(y.max(), pred.max()) + 0.3]
    ax.plot(lim, lim, "k-", lw=1); ax.fill_between(lim, [v - NOISE for v in lim], [v + NOISE for v in lim], color="grey", alpha=0.2, label=f"±noise ({NOISE})")
    ax.set_xlabel("actual pEC50"); ax.set_ylabel("nb3200 predicted"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(f"Predicted vs actual (RAE {rae(y,pred):.3f})\ncolor=max-sim-to-train"); ax.legend()
    plt.colorbar(sc, ax=ax, label="max Tanimoto to train")

    # ---- Figure 2: |error| vs novelty ----
    ax = axes[1]
    ax.scatter(max_sim[novel], aerr[novel], c="#d62728", s=25, label=f"novel scaffold (n={novel.sum()})", alpha=0.6)
    ax.scatter(max_sim[~novel], aerr[~novel], c="#1f77b4", s=25, label=f"seen scaffold (n={(~novel).sum()})", alpha=0.6)
    ax.axhline(NOISE, ls="--", c="grey", label=f"noise floor {NOISE}")
    # binned mean
    for lo, hi in [(0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1.01)]:
        m = (max_sim >= lo) & (max_sim < hi)
        if m.sum(): ax.plot([(lo+hi)/2], [aerr[m].mean()], "ks", ms=9)
    ax.set_xlabel("max Tanimoto to train (novelty ->left)"); ax.set_ylabel("|prediction error|")
    ax.set_title("Error vs scaffold novelty (the OOD wall)\nblack squares = binned mean |err|"); ax.legend()

    # ---- Figure 3: residual histogram ----
    ax = axes[2]
    ax.hist(err[~novel], bins=30, alpha=0.6, color="#1f77b4", label="seen")
    ax.hist(err[novel], bins=30, alpha=0.6, color="#d62728", label="novel")
    ax.axvline(0, c="k"); ax.axvline(err.mean(), c="g", ls="--", label=f"mean bias {err.mean():+.2f}")
    ax.set_xlabel("prediction error (pred - actual)"); ax.set_ylabel("count")
    ax.set_title(f"Residuals: novel std {err[novel].std():.2f} vs seen std {err[~novel].std():.2f}"); ax.legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/nb990_error_overview.png", dpi=140); plt.close()

    # ---- worst-15 outliers ----
    order = np.argsort(-aerr)[:15]
    mols, legs, rows = [], [], []
    for i in order:
        m = Chem.MolFromSmiles(str(smiles[i]))
        direction = "OVER" if err[i] > 0 else "UNDER"
        legs.append(f"act {y[i]:.2f} / pred {pred[i]:.2f} (err {err[i]:+.2f}, {direction})\n"
                    f"sim {max_sim[i]:.2f} {'NOVEL' if novel[i] else 'seen'} MW{Descriptors.MolWt(m):.0f}")
        mols.append(m)
        rows.append({"smiles": str(smiles[i]), "actual": float(y[i]), "pred": float(pred[i]),
                     "error": float(err[i]), "max_sim_train": float(max_sim[i]), "novel_scaffold": bool(novel[i])})
    img = Draw.MolsToGridImage(mols, legends=legs, molsPerRow=5, subImgSize=(300, 260))
    img.save(f"{OUT}/nb990_worst15_outliers.png")

    # summary stats
    over = (err > NOISE).sum(); under = (err < -NOISE).sum()
    summary = {"nb3200_rae": round(float(rae(y, pred)), 4), "anchor_rae": round(float(rae(y, anchor)), 4),
               "mean_bias": round(float(err.mean()), 3), "mae": round(float(aerr.mean()), 3),
               "novel_n": int(novel.sum()), "novel_mae": round(float(aerr[novel].mean()), 3),
               "seen_mae": round(float(aerr[~novel].mean()), 3),
               "n_overpredicted_>noise": int(over), "n_underpredicted_<-noise": int(under),
               "worst15": rows}
    json.dump(summary, open(f"{OUT}/nb990_summary.json", "w"), indent=2)
    print(f"MAE {aerr.mean():.3f} (novel {aerr[novel].mean():.3f} vs seen {aerr[~novel].mean():.3f}); "
          f"mean bias {err.mean():+.3f}; over>noise {over}, under<-noise {under}")
    print(f"saved -> {OUT}/nb990_error_overview.png + nb990_worst15_outliers.png + summary.json")


if __name__ == "__main__":
    main()
