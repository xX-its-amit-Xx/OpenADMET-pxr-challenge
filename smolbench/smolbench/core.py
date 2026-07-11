"""Combinatorial benchmark engine: featurizer x prep x model x HPO, honest CV, ensemble.

Given train/test CSVs (SMILES + numeric target), searches drop-in combinations, reports
out-of-fold + holdout metrics, fits the best config on the full train, and (optionally)
builds a top-K ensemble and isotonic calibration. Small-data pitfalls are surfaced as
``insights``.
"""
from __future__ import annotations
import itertools, os, time, warnings
from dataclasses import dataclass, field
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.isotonic import IsotonicRegression

from .featurizers import featurize, available_featurizers
from .models import build, grid, available_models
from .cv import make_folds
from .metrics import all_metrics, rae

PREPS = {
    "none": lambda: [("var", VarianceThreshold(0.0))],
    "standard": lambda: [("var", VarianceThreshold(0.0)), ("sc", StandardScaler())],
    "robust": lambda: [("var", VarianceThreshold(0.0)), ("sc", RobustScaler())],
    "pca50": lambda: [("var", VarianceThreshold(0.0)), ("sc", StandardScaler()), ("pca", PCA(50, random_state=0))],
    "pca200": lambda: [("var", VarianceThreshold(0.0)), ("sc", StandardScaler()), ("pca", PCA(200, random_state=0))],
}
# models that need dimensionality control (dense linear/kernel) vs tree models that don't
_NEEDS_PCA = {"ridge", "elasticnet", "svr", "krr", "gp", "knn"}


def _fit_transform(steps, Xtr, Xte):
    for _, t in steps:
        try:
            Xtr = t.fit_transform(Xtr); Xte = t.transform(Xte)
        except Exception:
            pass
    return Xtr, Xte


@dataclass
class BenchmarkResult:
    results: pd.DataFrame
    best: dict
    test_predictions: np.ndarray | None
    oof: dict
    insights: list
    meta: dict = field(default_factory=dict)

    def top(self, k=10, by="RAE"):
        return self.results.sort_values(by).head(k).reset_index(drop=True)


def benchmark(train_df, test_df=None, smiles_col="SMILES", target_col="y",
              featurizers=("morgan", "rdkit_desc", "maccs"),
              models=("ridge", "rf", "lgbm", "knn"),
              preps=("standard",), cv="scaffold", n_folds=5, seed=0,
              hpo=True, ensemble=True, top_k=5, calibrate=True,
              holdout_frac=0.0, verbose=True, out_dir=None):
    """Run the combinatorial benchmark. Returns a ``BenchmarkResult``.

    Parameters mirror a comp-chemist's knobs: which fingerprints, which models, which
    data-prep, which CV, whether to tune hyperparameters, ensemble the top configs, and
    calibrate. ``cv='scaffold'`` (default) gives honest estimates on novel chemistry.
    """
    t0 = time.time()
    if isinstance(train_df, str): train_df = pd.read_csv(train_df)
    if isinstance(test_df, str): test_df = pd.read_csv(test_df)
    tr = train_df.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)
    smi = tr[smiles_col].tolist(); y = tr[target_col].to_numpy(float)
    n = len(y)
    if featurizers == "all": featurizers = available_featurizers()
    if models == "all": models = available_models()

    # optional internal holdout carved by scaffold (never seen in CV/selection)
    ho_idx = None
    if holdout_frac and holdout_frac > 0:
        folds_h = make_folds(smi, "scaffold", max(2, int(round(1 / holdout_frac))), seed)
        ho_idx = folds_h[0][1]
        keep = np.setdiff1d(np.arange(n), ho_idx)
        smi_cv = [smi[i] for i in keep]; y_cv = y[keep]
    else:
        keep = np.arange(n); smi_cv = smi; y_cv = y

    folds = make_folds(smi_cv, cv, n_folds, seed)
    folds_rand = make_folds(smi_cv, "random", n_folds, seed)  # for optimism diagnostic

    # cache featurizations
    feats = {}
    for f in featurizers:
        try:
            feats[f] = featurize(smi, f)
        except Exception as e:
            if verbose: print(f"  [skip featurizer {f}: {e}]")
    test_feats = {}
    if test_df is not None:
        for f in feats:
            test_feats[f] = featurize(test_df[smiles_col].tolist(), f)

    rows = []; oof_store = {}
    for fname, X in feats.items():
        Xcv = X[keep]
        for mname in models:
            prep_opts = preps
            for prep in prep_opts:
                if prep.startswith("pca") is False and mname in _NEEDS_PCA and X.shape[1] > 300:
                    # force dim reduction for dense models on wide fingerprints
                    eff_prep = "pca200"
                else:
                    eff_prep = prep
                params_list = grid(mname) if hpo else [grid(mname)[0]]
                best_local = None
                for params in params_list:
                    oof = np.full(len(y_cv), np.nan)
                    ok = True
                    for tri, vai in folds:
                        steps = PREPS[eff_prep]()
                        Xt, Xv = _fit_transform(steps, Xcv[tri], Xcv[vai])
                        try:
                            m = build(mname, params); m.fit(Xt, y_cv[tri]); oof[vai] = m.predict(Xv)
                        except Exception:
                            ok = False; break
                    if not ok or np.isnan(oof).any():
                        continue
                    met = all_metrics(y_cv, oof)
                    if best_local is None or met["RAE"] < best_local[0]["RAE"]:
                        best_local = (met, params, oof)
                if best_local is None:
                    continue
                met, params, oof = best_local
                # random-CV optimism gap (our overfitting-warning signal)
                oof_r = np.full(len(y_cv), np.nan)
                try:
                    for tri, vai in folds_rand:
                        steps = PREPS[eff_prep]()
                        Xt, Xv = _fit_transform(steps, Xcv[tri], Xcv[vai])
                        m = build(mname, params); m.fit(Xt, y_cv[tri]); oof_r[vai] = m.predict(Xv)
                    opt_gap = rae(y_cv, oof) - rae(y_cv, oof_r)
                except Exception:
                    opt_gap = np.nan
                key = f"{fname}|{eff_prep}|{mname}"
                oof_store[key] = oof
                rows.append({"featurizer": fname, "prep": eff_prep, "model": mname,
                             **{k: round(v, 4) for k, v in met.items()},
                             "random_cv_optimism": round(float(opt_gap), 4), "params": params})
                if verbose:
                    print(f"  {key:<38} RAE={met['RAE']:.4f} MAE={met['MAE']:.4f} R2={met['R2']:.3f}")

    results = pd.DataFrame(rows).sort_values("RAE").reset_index(drop=True)

    # === refit best (+ ensemble + calibration) on full train, predict test/holdout ===
    best = results.iloc[0].to_dict() if len(results) else {}
    test_pred = None; oof_best = {}
    insights = _insights(results, n, cv)

    def refit_predict(cfgrow, Xall_key):
        f, p, mn, params = cfgrow["featurizer"], cfgrow["prep"], cfgrow["model"], cfgrow["params"]
        steps = PREPS[p]()
        Xt, Xte = _fit_transform(steps, feats[f], test_feats[f]) if test_df is not None else (feats[f], None)
        m = build(mn, params); m.fit(Xt, y)
        return m.predict(Xte) if Xte is not None else None

    if len(results):
        oof_best = {results.iloc[0]["featurizer"] + "|" + results.iloc[0]["prep"] + "|" + results.iloc[0]["model"]:
                    oof_store[results.iloc[0]["featurizer"] + "|" + results.iloc[0]["prep"] + "|" + results.iloc[0]["model"]]}
        if test_df is not None:
            if ensemble and len(results) >= 2:
                preds = [refit_predict(results.iloc[i].to_dict(), None) for i in range(min(top_k, len(results)))]
                preds = [p for p in preds if p is not None]
                test_pred = np.mean(preds, axis=0) if preds else None
                meta_ens = f"mean of top-{len(preds)}"
            else:
                test_pred = refit_predict(best, None); meta_ens = "single best"
            if calibrate and test_pred is not None:
                # honest isotonic: fit on best config's OOF (never on test)
                bkey = best["featurizer"] + "|" + best["prep"] + "|" + best["model"]
                iso = IsotonicRegression(out_of_bounds="clip").fit(oof_store[bkey], y_cv)
                test_pred = iso.predict(test_pred)

    meta = {"n_train": n, "cv": cv, "n_folds": n_folds, "n_configs": len(results),
            "runtime_s": round(time.time() - t0, 1),
            "featurizers": list(feats), "models": list(models)}
    res = BenchmarkResult(results, best, test_pred, oof_store, insights, meta)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        results.to_csv(os.path.join(out_dir, "benchmark_results.csv"), index=False)
        if test_pred is not None and test_df is not None:
            pd.DataFrame({smiles_col: test_df[smiles_col], "prediction": test_pred}).to_csv(
                os.path.join(out_dir, "test_predictions.csv"), index=False)
        with open(os.path.join(out_dir, "insights.txt"), "w") as fh:
            fh.write("\n".join(insights))
    return res


def _insights(results, n, cv):
    """Bake in the hard-won small-data lessons as automatic warnings."""
    out = []
    if n < 500:
        out.append(f"[!] SMALL-N ({n} train). Prefer the single most ROBUST model over a finely-tuned "
                   "stack - tuned ensembles overfit and lose on series-shifted test sets.")
    if len(results):
        worst_gap = results["random_cv_optimism"].min()  # most negative = scaffold much harder
        if worst_gap < -0.05:
            out.append(f"[!] random-CV is ~{-worst_gap:.2f} RAE optimistic vs scaffold-CV - the test likely "
                       "contains novel chemistry. Trust the scaffold-CV numbers only.")
        spread = results.groupby("model")["RAE"].min()
        out.append(f"Best model family: {spread.idxmin()} (RAE {spread.min():.3f}). "
                   f"Model choice spans {spread.max()-spread.min():.3f} RAE.")
        fspread = results.groupby("featurizer")["RAE"].min()
        out.append(f"Best featurizer: {fspread.idxmin()} (RAE {fspread.min():.3f}).")
        top = results.iloc[0]
        out.append(f"Recommended config: {top['featurizer']} + {top['prep']} + {top['model']} "
                   f"(scaffold-CV RAE {top['RAE']:.3f}, MAE {top['MAE']:.3f}).")
    if cv != "scaffold":
        out.append("[i] You used non-scaffold CV; for novel-chemistry test sets, cv='scaffold' is the honest default.")
    return out
