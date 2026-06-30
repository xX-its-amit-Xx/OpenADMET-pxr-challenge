"""nb978 — does MODE-CONDITIONING help? (the within-skewer SAR test)

Phase 4: test is 73% skewer-mode. Hypothesis: a skewer-SPECIALIST model (trained only on
skewer compounds) captures within-skewer SAR better than the global model, because the global
model averages across modes. Test: per-mode, global-LGBM OOF-RAE-on-mode vs specialist-LGBM
OOF-RAE-on-mode (both honest scaffold-CV). If specialist < global on skewer, conditioning helps.
A different hypothesis from 'add a feature' (which keeps getting absorbed). CPU, all on C:.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"
OUT = "C:/pxr_struct"
MODES = ["A_tripod", "B_blade", "C_skewer", "D_blob", "E_reach"]


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    y = tr["pec50"].to_numpy(float)
    mode = np.load(f"{D}/nb974_train_mode.npy", allow_pickle=True)
    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in tr["smiles"]]
    folds = scaffold_kfold_indices(scaf, n_splits=5, seed=42)
    X = impute(combined(tr["smiles"].tolist())).astype(np.float32)

    def fit_oof(train_mask):
        """Global if train_mask all True; else specialist trained only on train_mask rows."""
        oof = np.full(len(y), np.nan)
        for tri, vai in folds:
            tr_idx = tri[train_mask[tri]] if train_mask is not None else tri
            if len(tr_idx) < 30:
                continue
            m = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                                  n_jobs=4, verbose=-1).fit(X[tr_idx], y[tr_idx])
            oof[vai] = m.predict(X[vai])
        return oof

    glob_oof = fit_oof(None)
    print(f"GLOBAL model scaffold-CV RAE (all): {rae(y, glob_oof):.4f}\n")
    print(f"{'mode':10s} {'n':>5s} {'global-on-mode':>14s} {'specialist-on-mode':>18s} {'delta':>8s}")
    res = {}
    for mo in MODES:
        mm = mode == mo
        n = int(mm.sum())
        if n < 50:
            print(f"{mo:10s} {n:5d}  (too few)"); continue
        spec_oof = fit_oof(mm)
        gm = rae(y[mm], glob_oof[mm])
        sm = rae(y[mm], spec_oof[mm]) if np.isfinite(spec_oof[mm]).all() else float("nan")
        res[mo] = {"n": n, "global_on_mode": round(gm, 4), "specialist": round(sm, 4),
                   "delta": round(sm - gm, 4)}
        print(f"{mo:10s} {n:5d} {gm:14.4f} {sm:18.4f} {sm-gm:+8.4f}")

    sk = res.get("C_skewer", {})
    print("\n" + "=" * 60)
    if sk and sk["delta"] < -0.005:
        print(f">>> SKEWER specialist BEATS global by {-sk['delta']:.4f} -> mode-conditioning HELPS; "
              f"build a mode-routed deploy model")
    else:
        print(">>> mode-specialist does NOT beat global (global already captures within-mode SAR via "
              "tree splits) -> conditioning absorbed; within-skewer SAR is the irreducible residual")
    print("=" * 60)
    json.dump({"global_all_rae": round(float(rae(y, glob_oof)), 4), "per_mode": res},
              open(f"{OUT}/nb978_mode_conditional.json", "w"), indent=2)
    print(f"saved -> {OUT}/nb978_mode_conditional.json")


if __name__ == "__main__":
    main()
