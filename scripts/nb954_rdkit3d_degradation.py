"""nb954 — does 3D geometry add ORTHOGONAL signal over 2D Morgan, especially at
the novel-scaffold end? The cheap (CPU) go/no-go before committing GPU to Uni-Mol.

Uni-Mol's premise is that 3D binding geometry carries information 2D fingerprints
cannot. RDKit can generate the SAME ETKDG conformers Uni-Mol uses and compute
hand-crafted 3D descriptors (shape: PMI/NPR/asphericity; WHIM; AUTOCORR3D). If
even these FLATTEN the degradation curve at sim<0.3 (or 'combined+3D' beats
'combined' there), the 3D axis has signal and Uni-Mol (a learned 209M-conformer
model) is well-motivated. If 3D adds nothing, the 3D prior is weak too.

Same scaffold folds (seed=42), same max-sim bins as nb952/nb953 -> directly
comparable. Conformers generated in parallel + checkpointed (Windows-safe).
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

D = "data/processed"
BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.01]
N_3D = 10 + 80 + 114        # shape(10) + AUTOCORR3D(80) + WHIM(114) = 204
CKPT = f"{D}/nb954_desc3d_ckpt.npy"


def desc3d_one(smi):
    """ETKDG conformer + 3D descriptor vector (len N_3D); NaN vector on failure."""
    nan = [np.nan] * N_3D
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return nan
        m = Chem.AddHs(m)
        p = AllChem.ETKDGv3(); p.randomSeed = 42; p.maxIterations = 200
        if AllChem.EmbedMolecule(m, p) != 0:
            # retry with random coords
            p.useRandomCoords = True
            if AllChem.EmbedMolecule(m, p) != 0:
                return nan
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        except Exception:
            pass
        shape = [Descriptors3D.Asphericity(m), Descriptors3D.Eccentricity(m),
                 Descriptors3D.InertialShapeFactor(m), Descriptors3D.NPR1(m),
                 Descriptors3D.NPR2(m), Descriptors3D.PMI1(m), Descriptors3D.PMI2(m),
                 Descriptors3D.PMI3(m), Descriptors3D.RadiusOfGyration(m),
                 Descriptors3D.SpherocityIndex(m)]
        ac = list(rdMolDescriptors.CalcAUTOCORR3D(m))      # 80
        wh = list(rdMolDescriptors.CalcWHIM(m))            # 114
        v = shape + ac + wh
        if len(v) != N_3D:
            return nan
        return [float(x) for x in v]
    except Exception:
        return nan


def compute_desc3d(smiles):
    """Parallel conformer+descriptor with checkpoint resume."""
    n = len(smiles)
    if os.path.exists(CKPT):
        X = np.load(CKPT)
        if X.shape == (n, N_3D):
            done = ~np.isnan(X[:, 0])
            print(f"resume: {int(done.sum())}/{n} already computed", flush=True)
            if done.all():
                return X
        else:
            X = np.full((n, N_3D), np.nan, np.float32)
    else:
        X = np.full((n, N_3D), np.nan, np.float32)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    todo = [i for i in range(n) if np.isnan(X[i, 0])]
    print(f"computing 3D descriptors for {len(todo)} molecules (6 workers) ...", flush=True)
    t0 = time.time(); done_ct = 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(desc3d_one, smiles[i]): i for i in todo}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                X[i] = fut.result()
            except Exception:
                pass
            done_ct += 1
            if done_ct % 300 == 0:
                np.save(CKPT, X)
                print(f"  {done_ct}/{len(todo)} ({time.time()-t0:.0f}s) "
                      f"fail_so_far={int(np.isnan(X[:,0]).sum())}", flush=True)
    np.save(CKPT, X)
    return X


def curve(y, pred, max_sim, rae):
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (max_sim >= lo) & (max_sim < hi); nn = int(m.sum())
        if nn == 0:
            rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": 0, "mae": None, "rae": None}); continue
        mae = float(np.mean(np.abs(y[m] - pred[m])))
        r = float(rae(y[m], pred[m])) if nn > 1 else None
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": nn, "mae": round(mae, 4),
                     "rae": round(r, 4) if r is not None else None})
    return rows


def main():
    from src.pxr.data import load_train
    from src.pxr.eval import rae, scaffold_kfold_indices
    from src.pxr.featurize import combined, impute
    import lightgbm as lgb

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    smiles = tr["smiles"].tolist()
    y = tr["pec50"].to_numpy(float)
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None
                 for s in smiles]
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=42)
    max_sim = np.load(f"{D}/nb952_max_sim_4139.npy")

    X3 = compute_desc3d(smiles)
    n_fail = int(np.isnan(X3[:, 0]).sum())
    print(f"3D descriptors: {X3.shape}  embed_failures={n_fail} ({100*n_fail/len(y):.1f}%)")

    # median-impute + clip the 3D block
    from sklearn.impute import SimpleImputer
    X3i = SimpleImputer(strategy="median").fit_transform(X3)
    X3i = np.clip(X3i, -1e6, 1e6).astype(np.float32)
    Xc = impute(combined(smiles)).astype(np.float32)
    Xcomb3d = np.hstack([Xc, X3i])

    lgbm_curve = json.load(open(f"{D}/nb952_stress_curve.json"))["lgbm_curve"]
    lgbm_deep = next(r for r in lgbm_curve if r["bin"] == "[0.0,0.3)")["mae"]
    lgbm_overall = json.load(open(f"{D}/nb952_stress_curve.json"))["overall_rae"]

    def run(X, tag):
        oof = np.full(len(y), np.nan)
        for tr_idx, va_idx in folds:
            mdl = lgb.LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05,
                                    n_jobs=4, verbose=-1).fit(X[tr_idx], y[tr_idx])
            oof[va_idx] = mdl.predict(X[va_idx])
        ov = rae(y, oof); rows = curve(y, oof, max_sim, rae)
        deep = next(r for r in rows if r["bin"] == "[0.0,0.3)")
        np.save(f"{D}/nb954_{tag}_oof.npy", oof)
        print(f"\n{tag}: overall scaffold-CV RAE = {ov:.4f}")
        print(f"{'sim-to-train':18s} {'n':>5s} {'MAE':>8s} {'RAE':>8s}"); print("-" * 44)
        for r in rows:
            mae = f"{r['mae']:.4f}" if r["mae"] is not None else "   --"
            rr = f"{r['rae']:.4f}" if r["rae"] is not None else "   --"
            print(f"{r['bin']:18s} {r['n']:5d} {mae:>8s} {rr:>8s}")
        return {"overall_rae": round(float(ov), 4), "curve": rows, "deep_extrap_mae": deep["mae"]}

    res = {"3d_only": run(X3i, "3d_only"), "combined_plus_3d": run(Xcomb3d, "combined_plus_3d")}

    print("\n" + "=" * 64)
    print("DEEP-EXTRAPOLATION (sim<0.3) MAE — does 3D help where we're weakest?")
    print(f"  LGBM-combined (2D ref)   : {lgbm_deep:.4f}   overall {lgbm_overall:.4f}")
    print(f"  3D-descriptors only      : {res['3d_only']['deep_extrap_mae']:.4f}   "
          f"overall {res['3d_only']['overall_rae']:.4f}")
    d2 = res["combined_plus_3d"]["deep_extrap_mae"]
    verdict = "3D ADDS at novel end -> Uni-Mol JUSTIFIED" if d2 < lgbm_deep else "3D adds nothing at novel end"
    print(f"  combined + 3D            : {d2:.4f}   "
          f"overall {res['combined_plus_3d']['overall_rae']:.4f}   <- {verdict}")
    print("=" * 64)
    print("Caveat: RDKit 3D descriptors are hand-crafted; Uni-Mol is LEARNED on 209M")
    print("conformers and may extract signal these miss. A negative here RAISES the bar")
    print("for Uni-Mol but does not kill it; a positive is a strong green light + cheap win.")

    json.dump({"lgbm_deep_extrap_mae": lgbm_deep, "lgbm_overall_rae": lgbm_overall,
               "results": res, "n_embed_fail": n_fail},
              open(f"{D}/nb954_rdkit3d_degradation.json", "w"), indent=2)
    print(f"\nsaved -> {D}/nb954_rdkit3d_degradation.json")


if __name__ == "__main__":
    main()
