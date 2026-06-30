"""nb952 — scaffold-extrapolation stress test (the cloud-spend decision gate).

The honest question for the foundation-model bet is NOT "does it beat 0.4416 on
the selection-biased 253" — it's "does a broad-pretrained representation degrade
more gracefully on novel chemistry than our LGBM stack?" That predicts LB transfer
to the truly-blind 260, and it is measurable LOCALLY on the 4,139 train right now.

Method:
  - scaffold 5-fold CV on the 4,139 (each Bemis-Murcko scaffold entirely in one fold)
  - per fold: train LGBM-combined, predict the held-out fold -> honest OOF
  - per OOF compound: max Tanimoto (ECFP4) to that fold's TRAINING compounds
    = how "seen" its chemistry is
  - bin by that similarity; report MAE (denominator-free, comparable across bins
    AND across models) + RAE per bin = the DEGRADATION CURVE

The headline = MAE in the deep-extrapolation bin (sim < 0.3). A foundation model
earns paid cloud only if its curve is FLATTER there (same OOF protocol, same bins).
This is the LGBM reference curve; the Kaggle MolFormer notebook plugs its OOF into
the identical bins via compare_curve().
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]   # similarity-to-train bins
N_JOBS = 4                                     # leave headroom on 16 cores


def murcko(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) if sc is not None else None
    except Exception:
        return None


def max_tanimoto_to_train(fp_val, fp_train):
    """Max ECFP4 Tanimoto of each val row to any train row. Dense-bit numpy path.
    Tanimoto = |a&b| / (|a|+|b|-|a&b|). inter = V @ T.T."""
    V = fp_val.astype(np.float32); T = fp_train.astype(np.float32)
    inter = V @ T.T                                   # (nv, nt) shared on-bits
    a = V.sum(1)[:, None]; b = T.sum(1)[None, :]
    union = a + b - inter
    union[union == 0] = 1.0
    sim = inter / union
    return sim.max(1)


def curve(y, pred, max_sim):
    """MAE + RAE + n per similarity bin."""
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (max_sim >= lo) & (max_sim < hi)
        n = int(m.sum())
        if n == 0:
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "mae": None, "rae": None})
            continue
        mae = float(np.mean(np.abs(y[m] - pred[m])))
        r = float(rae(y[m], pred[m])) if n > 1 else None
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": n, "mae": round(mae, 4),
                     "rae": round(r, 4) if r is not None else None})
    return rows


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    smiles = tr["smiles"].tolist()
    y = tr["pec50"].to_numpy(float)
    print(f"train rows with pec50: {len(y)}")

    scaffolds = [murcko(s) for s in smiles]
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=42)

    print("featurizing (combined 2265) ...", flush=True)
    X = impute(combined(smiles))
    print("morgan fps for similarity ...", flush=True)
    fp = morgan_fp_batch(smiles)                       # (N, 2048) dense uint8

    oof = np.full(len(y), np.nan)
    max_sim = np.full(len(y), np.nan)
    for k, (tr_idx, va_idx) in enumerate(folds):
        model = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                                  n_jobs=N_JOBS, verbose=-1)
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict(X[va_idx])
        max_sim[va_idx] = max_tanimoto_to_train(fp[va_idx], fp[tr_idx])
        print(f"  fold {k}: train={len(tr_idx)} val={len(va_idx)} "
              f"val-RAE={rae(y[va_idx], oof[va_idx]):.4f} "
              f"median-sim={np.median(max_sim[va_idx]):.3f}", flush=True)

    overall = rae(y, oof)
    print(f"\nLGBM-combined scaffold-CV OOF RAE (4139): {overall:.4f}")
    print(f"\n{'DEGRADATION CURVE — LGBM-combined (the reference to beat)':^60s}")
    print(f"{'sim-to-train bin':18s} {'n':>5s} {'MAE':>8s} {'RAE':>8s}   (novel <- ... -> seen)")
    print("-" * 60)
    rows = curve(y, oof, max_sim)
    for r in rows:
        mae = f"{r['mae']:.4f}" if r["mae"] is not None else "   --"
        rr = f"{r['rae']:.4f}" if r["rae"] is not None else "   --"
        print(f"{r['bin']:18s} {r['n']:5d} {mae:>8s} {rr:>8s}")

    deep = next((r for r in rows if r["bin"] == "[0.0,0.3)"), None)
    print()
    if deep and deep["n"]:
        print(f">>> DEEP-EXTRAPOLATION GATE: MAE @ sim<0.3 (n={deep['n']}) = {deep['mae']:.4f}")
        print(">>> MolFormer must beat THIS (flatter low-sim curve) to earn paid cloud.")
    else:
        print(">>> few/no sim<0.3 compounds in train; extrapolation is milder than the 253.")

    np.save(f"{D}/nb952_lgbm_oof_4139.npy", oof)
    np.save(f"{D}/nb952_max_sim_4139.npy", max_sim)
    np.save(f"{D}/nb952_y_4139.npy", y)
    json.dump({"overall_rae": round(float(overall), 4), "bins": BINS,
               "lgbm_curve": rows,
               "deep_extrap_mae": deep["mae"] if deep else None},
              open(f"{D}/nb952_stress_curve.json", "w"), indent=2)
    print(f"\nsaved -> {D}/nb952_stress_curve.json (+ oof/max_sim/y .npy for MolFormer compare)")


if __name__ == "__main__":
    main()
