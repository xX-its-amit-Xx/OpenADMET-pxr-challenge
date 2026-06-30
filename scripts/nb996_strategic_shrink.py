"""nb996 — Phase D: strategic uncertainty+novelty-gated shrink toward low pEC50.

The user's hypothesis: if we TRUST high predictions (confident actives) and only shrink UNCERTAIN
NOVEL compounds toward the inactive range, we fix F2 (novel-inactive over-prediction) WITHOUT losing
the active signal. Tested honestly: (1) reliability diagnostic — is the model reliable at the active
end? (2) gated shrink = pred - lambda * gate, gate = novelty x uncertainty x (mid-prediction zone),
cross-fit lambda on fold-train, multi-seed verify the RAE delta. Don't overfit; don't shrink confident
actives. Base = nb3200 OOF on the 253.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; OUT = "C:/pxr_struct/dash"
SEEDS = [42, 101, 202, 303, 404, 505, 606]


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb].tolist()
    base = np.load(f"{D}/nb3200_pred_oof.npy")
    anchor = np.load(f"{D}/te_chemprop_aux.npy")[unb]

    # novelty + uncertainty (ensemble std of LGBM on chemprop_aux residual)
    fp_te = morgan_fp_batch(smiles); fp_tr = morgan_fp_batch(tr["smiles"].tolist())
    A = fp_te.astype(np.float32); B = fp_tr.astype(np.float32)
    inter = A @ B.T; u = A.sum(1)[:, None] + B.sum(1)[None, :] - inter; u[u == 0] = 1
    sim = (inter / u).max(1)
    Xc = impute(combined(smiles)).astype(np.float32); resid = y - anchor
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in smiles]

    # ---- (1) reliability diagnostic: is the active end trustworthy? ----
    print("=== reliability by nb3200 PREDICTED bin (253) ===")
    print(f"{'pred bin':12s} {'n':>4s} {'mean_actual':>11s} {'frac_active>=5':>14s} {'mean|err|':>9s}")
    for lo, hi in [(0, 3.5), (3.5, 4.5), (4.5, 5.5), (5.5, 9)]:
        m = (base >= lo) & (base < hi); n = int(m.sum())
        if n: print(f"[{lo},{hi})".ljust(12) + f" {n:4d} {y[m].mean():11.2f} {(y[m]>=5).mean():14.2f} {np.abs(base[m]-y[m]).mean():9.2f}")

    # ---- (2) gated strategic shrink, cross-fit + multi-seed ----
    def shrink(pred, unc, lam, floor, tau_sim, tau_pred):
        g_nov = 1/(1+np.exp((sim - tau_sim)/0.1))            # high when novel
        g_unc = unc/ (unc.max()+1e-9)                        # high when uncertain
        g_zone = ((pred > 3.0) & (pred < tau_pred)).astype(float)  # only the over-prediction mid-zone
        gate = g_nov * g_unc * g_zone
        return pred - lam * gate * np.maximum(pred - floor, 0)

    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        # per-fold ensemble uncertainty + cross-fit lambda
        new = base.copy()
        for tri, vai in folds:
            preds = np.column_stack([lgb.LGBMRegressor(n_estimators=300, num_leaves=48, learning_rate=0.05,
                    random_state=s, n_jobs=4, verbose=-1).fit(Xc[tri], resid[tri]).predict(Xc) for s in range(5)])
            unc = preds.std(1)                               # ensemble disagreement (all 253)
            floor = float(np.quantile(y[tri], 0.10))
            best = (0.0, rae(y[tri], base[tri]))
            for lam in np.linspace(0, 1.5, 16):
                r = rae(y[tri], shrink(base, unc, lam, floor, np.median(sim[tri]), 4.6)[tri])
                if r < best[1]: best = (lam, r)
            new[vai] = shrink(base, unc, best[0], floor, np.median(sim[tri]), 4.6)[vai]
        d = rae(y, new) - rae(y, base)
        rows.append({"seed": seed, "base": round(rae(y, base), 4), "shrunk": round(rae(y, new), 4), "delta": round(d, 5)})
        print(f"  seed {seed}: base={rae(y,base):.4f} shrunk={rae(y,new):.4f} delta={d:+.5f}")

    dd = np.array([r["delta"] for r in rows]); stable = dd.mean() < 0 and abs(dd.mean()) > dd.std()
    print("\n" + "=" * 60)
    print(f"strategic shrink delta vs nb3200: mean={dd.mean():+.5f} std={dd.std():.5f} "
          f"wins={int((dd<0).sum())}/{len(SEEDS)} stable={stable}")
    print(">>> strategic shrink HELPS (real, won't lose active signal)" if stable
          else ">>> shrink does NOT reliably help -> can't distinguish novel-active from novel-inactive (the core wall, confirmed by Phase-C poses)")
    print("=" * 60)
    json.dump({"rows": rows, "delta_mean": float(dd.mean()), "delta_std": float(dd.std()), "stable": bool(stable)},
              open(f"{OUT}/nb996_strategic_shrink.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb996_strategic_shrink.json")


if __name__ == "__main__":
    main()
