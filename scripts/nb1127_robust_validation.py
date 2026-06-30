"""nb1127 — ROBUST validation: clean analog-expansion holdouts from TRAIN (never tuned against) + error visualization.

User's concern (correct): the 0.4416 on the 253 was achieved by tuning the pipeline (clip/blend/HPO) AGAINST the now-
revealed 253 labels over ~270 cycles -> selection-bias optimism; couldn't reach it in Phase-1 when 253 were blinded.
This builds a SELECTION-BIAS-FREE estimate: carve 5 analog-expansion holdouts from the 4139 train (scaffold-disjoint,
mimicking the test's train-similarity profile), train the SAME pipeline on the rest, predict the holdout. These RAEs
are clean (never used for any tuning). Compare to the 253. Uses combined features ONLY (chempropembed would LEAK on a
train-holdout since it was trained on all 4139 -> would mask the bias). Visualize error structure + the clean-vs-253 gap.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

P = "data/processed"; FIG = "C:/pxr_work/figures"; os.makedirs(FIG, exist_ok=True)


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def fpf(sm): return (morgan_fp_batch(sm).astype(np.float32) > 0).astype(np.float32)


def clipped_lgbm(Xtr, ytr, Xte):
    p = np.mean([lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, subsample=0.8,
                colsample_bytree=0.8, n_jobs=4, verbose=-1, random_state=s).fit(Xtr, ytr).predict(Xte) for s in range(3)], 0)
    lo, hi = np.quantile(ytr, 0.05), np.quantile(ytr, 0.98)
    return np.clip(p, lo, hi)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy")
    ytr = tr["pec50"].to_numpy()
    print("featurizing (combined only)...", flush=True)
    Xtr = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    Xte = impute(combined(te["smiles"].to_numpy()[unb].tolist())).astype(np.float32)
    scaf = np.array([murcko(s) for s in tr["smiles"]])

    # test's train-similarity profile (to match holdouts to)
    Ftr = fpf(tr["smiles"].tolist()); Fte = fpf(te["smiles"].to_numpy()[unb].tolist())
    inter = Fte @ Ftr.T; s = Ftr.sum(1); test_top1 = (inter / np.clip(Fte.sum(1)[:, None] + s[None, :] - inter, 1, None)).max(1)
    print(f"test->train top1 sim median {np.median(test_top1):.3f}\n")

    # combined-only on the 253 (the tuned-validation set, for comparison)
    p253 = clipped_lgbm(Xtr, ytr, Xte)
    rae253 = rae(y, p253)
    print(f"combined-LGBM on the 253 (the tuned validation set): RAE {rae253:.4f}")
    print(f"  (nb3200's 0.4416 adds chempropembed + residual + the 253-TUNED clip/blend on top)\n")

    # 5 clean analog-expansion holdouts from the train
    print("=== CLEAN analog-expansion holdouts from TRAIN (never tuned against) ===")
    holds = []
    for seed in range(5):
        folds = scaffold_kfold_indices(scaf.tolist(), n_splits=round(len(tr) / 250), seed=100 + seed)
        ho = min((f[1] for f in folds), key=lambda ix: abs(len(ix) - 250))
        trn = np.array([i for i in range(len(tr)) if i not in set(ho.tolist())])
        ph = clipped_lgbm(Xtr[trn], ytr[trn], Xtr[ho])
        # holdout's train-sim for context
        Fh = fpf(tr["smiles"].iloc[ho].tolist()); it = Fh @ Ftr[trn].T
        hs = (it / np.clip(Fh.sum(1)[:, None] + Ftr[trn].sum(1)[None, :] - it, 1, None)).max(1)
        r = rae(ytr[ho], ph); holds.append(r)
        print(f"  holdout {seed}: n={len(ho)} RAE {r:.4f} | holdout->train sim median {np.median(hs):.2f}")
    hm, hs_ = np.mean(holds), np.std(holds)
    print(f"\n  CLEAN holdout RAE = {hm:.4f} +/- {hs_:.4f}  (selection-bias-free pipeline estimate)")
    print(f"  vs combined-on-253 {rae253:.4f}  ->  GAP {rae253-hm:+.4f} "
          f"({'253 is EASIER/optimistic' if rae253 < hm - 0.01 else '253 ~representative' if abs(rae253-hm)<0.02 else '253 HARDER'})")
    print(f"  => the same selection/representativeness gap applies on top of nb3200's 0.4416 -> honest blinded est ~{0.4416 + max(0, hm-rae253):.2f}-{0.4416 + (hm-rae253) + 0.05:.2f}")

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    err = anchor - y
    ax[0].scatter(y, anchor, c=test_top1, cmap="viridis", s=26); lims = [y.min()-.3, y.max()+.3]
    ax[0].plot(lims, lims, "k--", lw=1); ax[0].set_xlabel("true pEC50"); ax[0].set_ylabel("nb3200 pred")
    ax[0].set_title(f"nb3200 on 253 (RAE 0.4416)\ncolor=train sim"); fig.colorbar(ax[0].collections[0], ax=ax[0])
    ax[1].hist(np.abs(err), bins=25, color="#c0392b", alpha=0.7); ax[1].axvline(np.median(np.abs(err)), color="k", ls="--")
    ax[1].set_xlabel("|error|"); ax[1].set_ylabel("count"); ax[1].set_title(f"nb3200 |error| on 253\nmedian {np.median(np.abs(err)):.2f}")
    ax[2].bar(["combined\non 253\n(tuned)"], [rae253], color="#2471a3")
    ax[2].bar(["clean train\nholdouts\n(untuned)"], [hm], yerr=hs_, color="#c0392b", capsize=5)
    ax[2].axhline(0.4416, color="green", ls="--", label="nb3200 claim 0.4416")
    ax[2].set_ylabel("RAE"); ax[2].set_title("Tuned-253 vs clean-holdout\n(the selection-bias gap)"); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(f"{FIG}/nb1127_robust_validation.png", dpi=115); plt.close()
    json.dump({"rae_253_combined": float(rae253), "clean_holdout_rae": float(hm), "clean_holdout_std": float(hs_),
               "gap": float(rae253 - hm)}, open(f"{P}/nb1127_robust_validation.json", "w"), indent=2)
    print(f"\nwrote {FIG}/nb1127_robust_validation.png")


if __name__ == "__main__":
    main()
