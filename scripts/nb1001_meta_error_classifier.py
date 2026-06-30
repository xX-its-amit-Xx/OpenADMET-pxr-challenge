"""nb1001 — meta error-direction model (user idea): learn to predict whether a base prediction is
'too high' vs 'too low', then temper predictions < 4 (the F2 over-prediction zone).

Two variants, both cross-fit on the 253 (honest) + multi-seed:
 (A) regressor: predict residual (base - y) from [combined, base_pred, novelty, uncertainty] -> subtract.
 (B) classifier: predict P(too-high) -> shift pred<4 down proportionally.
The real question: is the error DIRECTION predictable from features (or do F2 inactives look like actives)?
Base = nb3200 OOF. Apply variants, verify RAE delta vs nb3200 0.4416.
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
    base = np.load(f"{D}/nb3200_pred_oof.npy"); anchor = np.load(f"{D}/te_chemprop_aux.npy")[unb]
    err = base - y
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in smiles]
    Xc = impute(combined(smiles)).astype(np.float32)

    # novelty
    fp_te = morgan_fp_batch(smiles); fp_tr = morgan_fp_batch(tr["smiles"].tolist())
    A = fp_te.astype(np.float32); B = fp_tr.astype(np.float32)
    inter = A @ B.T; uu = A.sum(1)[:, None] + B.sum(1)[None, :] - inter; uu[uu == 0] = 1
    nov = (inter / uu).max(1)

    # is the error direction even predictable? cross-fit AUC of a sign classifier
    print("=== can we predict error SIGN (too-high=1) from features? honest cross-fit AUC ===")
    from sklearn.metrics import roc_auc_score
    sign = (err > 0).astype(int)
    aucs = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        p = np.full(len(y), np.nan)
        Xmeta = np.column_stack([Xc, base, nov])
        for tri, vai in folds:
            c = lgb.LGBMClassifier(n_estimators=300, num_leaves=32, learning_rate=0.05, n_jobs=4, verbose=-1).fit(Xmeta[tri], sign[tri])
            p[vai] = c.predict_proba(Xmeta[vai])[:, 1]
        aucs.append(roc_auc_score(sign, p))
    print(f"  sign-classifier cross-fit AUC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}  (0.5 = no signal)")

    # variant A: regressor predicts residual -> subtract (full + tempered for pred<4)
    def run_correction(kind):
        rows = []
        for seed in SEEDS:
            folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
            corr = base.copy()
            Xmeta = np.column_stack([Xc, base, nov])
            for tri, vai in folds:
                if kind == "reg_full" or kind == "reg_temper_lt4":
                    m = lgb.LGBMRegressor(n_estimators=300, num_leaves=32, learning_rate=0.05, n_jobs=4, verbose=-1).fit(Xmeta[tri], err[tri])
                    pe = m.predict(Xmeta[vai])
                    c = base[vai] - pe
                    if kind == "reg_temper_lt4":   # only correct (down) where base<4 and predicted too-high
                        mask = (base[vai] < 4.0) & (pe > 0)
                        c = base[vai].copy(); c[mask] = base[vai][mask] - 0.5 * pe[mask]
                    corr[vai] = c
                elif kind == "clf_temper_lt4":
                    cl = lgb.LGBMClassifier(n_estimators=300, num_leaves=32, learning_rate=0.05, n_jobs=4, verbose=-1).fit(Xmeta[tri], (err[tri] > 0).astype(int))
                    ph = cl.predict_proba(Xmeta[vai])[:, 1]   # P(too-high)
                    c = base[vai].copy()
                    mask = base[vai] < 4.0
                    c[mask] = base[vai][mask] - 0.8 * (ph[mask] - 0.5).clip(0) * 2   # shift down up to ~0.8 if confident too-high
                    corr[vai] = c
            rows.append(rae(y, corr) - rae(y, base))
        d = np.array(rows); return d.mean(), d.std(), int((d < 0).sum())

    print(f"\nbase nb3200 RAE = {rae(y, base):.4f}")
    res = {}
    for kind in ["reg_full", "reg_temper_lt4", "clf_temper_lt4"]:
        mu, sd, w = run_correction(kind)
        stable = mu < 0 and abs(mu) > sd
        res[kind] = {"delta_mean": round(float(mu), 5), "delta_std": round(float(sd), 5), "wins": f"{w}/{len(SEEDS)}", "stable": bool(stable)}
        print(f"  {kind:18s} delta={mu:+.5f} +/- {sd:.5f}  wins={w}/{len(SEEDS)}  stable={stable}")
    print("\n" + ("=" * 58))
    any_help = any(v["stable"] for v in res.values())
    print(">>> META-CORRECTION HELPS (error direction IS partly predictable)" if any_help
          else ">>> error direction NOT predictable from features (F2 inactives look like actives) -> same wall, 5th confirmation")
    print("=" * 58)
    json.dump({"sign_auc": round(float(np.mean(aucs)), 3), "base_rae": round(float(rae(y, base)), 4), "variants": res},
              open(f"{OUT}/nb1001_meta_error.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb1001_meta_error.json")


if __name__ == "__main__":
    main()
