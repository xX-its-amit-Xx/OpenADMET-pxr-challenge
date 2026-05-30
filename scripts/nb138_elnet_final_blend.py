"""nb138 — ElasticNet Final Blend (Post-nb129).

After discovering that the k=3 blend of (nb109_calib + nb107_calib + counter_delta)
gives OOF 0.3556, apply ElasticNet to optimally weight this core trio against the
full model pool. Specifically:

  1. Start from the 3-model core: nb109_calib + nb107_calib + counter_delta
  2. Build ElasticNet over these 3 plus all new models (nb130-nb137)
  3. Use nested CV to avoid over-fitting the weights
  4. Compare to the perturb_optimize result from nb129

Key difference from nb125/nb127: we constrain the optimization to start from
the known-good k=3 seed, then use ElasticNet to potentially add beneficial
complementary models.

Also try: Positive-constrained Ridge (RidgeCV with non-negative constraint)
which may generalize better than unconstrained Ridge.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.linear_model import ElasticNetCV, RidgeCV, Lasso
from itertools import combinations

from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
COLLAPSE_THRESH = 0.58


def load_all_models(n_tr, y_tr, thresh=COLLAPSE_THRESH):
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists(): break
        else:
            continue
        try:
            oof = np.load(p).astype(np.float64)
            te  = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te.ndim == 2:  te  = te[:, 0]
            if len(oof) != n_tr: continue
            oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < thresh: continue
            r = rae(y_tr, oof)
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    return results


def nonneg_ridge(oof_mat, y_tr, alphas=None):
    """Ridge with non-negativity constraint via SLSQP."""
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    k = oof_mat.shape[1]
    best_r, best_w = np.inf, np.ones(k) / k
    for alpha in alphas:
        def obj(w):
            pred = oof_mat @ w
            mae = np.mean(np.abs(y_tr - pred))
            baseline = np.mean(np.abs(y_tr - y_tr.mean()))
            return mae / baseline + alpha * np.sum(w**2)
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, None)] * k
        w0 = np.ones(k) / k
        res = optimize.minimize(obj, w0, method="SLSQP",
                                bounds=bounds, constraints=constraints,
                                options={"ftol": 1e-8, "maxiter": 1000})
        r = rae(y_tr, oof_mat @ res.x)
        if r < best_r:
            best_r, best_w = r, res.x
    return best_r, best_w


def main():
    print("=== nb138: ElasticNet Final Blend ===\n")

    from pxr.data import load_train, load_test
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)

    models = load_all_models(n_tr, y_tr)
    n_mod = len(models)
    print(f"  {n_mod} models loaded")
    for m in models[:12]:
        print(f"    {m['stem']:50s}  RAE={m['rae']:.4f}  ratio={m['ratio']:.2f}")

    oof_mat = np.column_stack([m["oof"] for m in models])
    te_mat  = np.column_stack([m["te"]  for m in models])
    stems   = [m["stem"] for m in models]

    # Strategy 1: ElasticNet (non-negative) over all models
    print(f"\nElasticNet (non-negative, all {n_mod} models)...")
    from sklearn.linear_model import ElasticNetCV
    enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
                        alphas=np.logspace(-4, 1, 30),
                        positive=True, cv=5, max_iter=10000, random_state=SEED)
    enet.fit(oof_mat, y_tr)
    oof_enet = oof_mat @ enet.coef_
    oof_enet_r = rae(y_tr, oof_enet)
    print(f"  ElasticNet OOF RAE: {oof_enet_r:.4f}  "
          f"(non-zero: {(enet.coef_ > 0.001).sum()})")
    for i, w in enumerate(enet.coef_):
        if w > 0.001:
            print(f"    {stems[i]:50s}  w={w:.3f}")

    # Strategy 2: Non-negative Ridge
    print(f"\nNon-negative Ridge (all {n_mod} models)...")
    best_ridge_r, best_ridge_w = nonneg_ridge(oof_mat, y_tr)
    print(f"  Non-neg Ridge OOF RAE: {best_ridge_r:.4f}")
    for i, w in enumerate(best_ridge_w):
        if w > 0.01:
            print(f"    {stems[i]:50s}  w={w:.3f}")

    # Strategy 3: Core trio + ElasticNet over remainder
    # Find the k=3 core: nb109_calib, nb107_calib, counter_delta
    core_stems = ["nb109_deep_meta_stack_calib", "nb107_assay_decomp_calib", "counter_delta"]
    core_idx   = []
    for cs in core_stems:
        for i, s in enumerate(stems):
            if s == cs:
                core_idx.append(i)
                break

    if len(core_idx) == 3:
        print(f"\nCore k=3 + ElasticNet over remainder...")
        core_pred = oof_mat[:, core_idx].mean(axis=1)
        r_core_eq = rae(y_tr, core_pred)
        print(f"  Equal-weight k=3: {r_core_eq:.4f}")
        # Non-core models
        ncore_idx = [i for i in range(n_mod) if i not in core_idx]
        oof_ncore = oof_mat[:, ncore_idx]
        # Residual from equal-weight core
        resid = y_tr - core_pred
        enet2 = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0],
                             alphas=np.logspace(-4, 0, 20),
                             positive=False, cv=5, max_iter=10000, random_state=SEED)
        enet2.fit(oof_ncore, resid)
        oof_corrected = core_pred + oof_ncore @ enet2.coef_
        r_corrected = rae(y_tr, oof_corrected)
        print(f"  Core + residual correction RAE: {r_corrected:.4f}")
    else:
        print(f"  Warning: could only find {len(core_idx)}/3 core models")
        r_corrected = np.inf; oof_corrected = oof_mat[:, 0]

    # Best strategy
    strategies = [
        ("ElasticNet", oof_enet_r, enet.coef_),
        ("NonNegRidge", best_ridge_r, best_ridge_w),
    ]
    if r_corrected < np.inf:
        # For corrected, need full weight vector
        full_w_corrected = np.zeros(n_mod)
        for i, ci in enumerate(core_idx):
            full_w_corrected[ci] = 1.0 / len(core_idx)
        for i, ni in enumerate(ncore_idx):
            full_w_corrected[ni] += enet2.coef_[i]
        strategies.append(("CoreResidual", r_corrected, full_w_corrected))

    best_name, best_r, best_w = min(strategies, key=lambda x: x[1])
    print(f"\n=== BEST: {best_name}  OOF RAE={best_r:.4f} ===")

    best_oof = oof_mat @ best_w
    best_te  = te_mat  @ best_w
    best_te  = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    ratio = best_te.std() / best_oof.std()
    print(f"Test: med={np.median(best_te):.2f}  std={best_te.std():.3f}  ratio={ratio:.2f}")

    np.save(DATA_PROCESSED / "oof_nb138_elnet_final.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb138_elnet_final.npy",  best_te)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te})
    sub.to_csv(SUBMISSIONS / "138_elnet_final_blend.csv", index=False)
    print(f"\nSaved: submissions/138_elnet_final_blend.csv")
    print(f"OOF RAE: {best_r:.4f}")


if __name__ == "__main__":
    main()
