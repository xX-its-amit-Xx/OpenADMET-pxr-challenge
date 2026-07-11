"""Stability / variance estimation — the 'deep-N' lesson.

A single CV split's RAE has real seed variance. Two configs within ~1 std of each other
are NOT distinguishable. ``stability_check`` re-runs a config's CV over many seeds so you
promote a "winner" only if it clears the noise band.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from .featurizers import featurize
from .models import build, grid
from .cv import make_folds
from .core import PREPS, _fit_transform
from .metrics import rae, mae


def stability_check(train_df, config, smiles_col="SMILES", target_col="y",
                    cv="scaffold", n_folds=5, n_seeds=15, verbose=True):
    """Re-run one config's CV over ``n_seeds`` and report mean +/- std RAE.

    ``config`` is a dict with keys featurizer, prep, model, params (a row of
    ``BenchmarkResult.results``). Returns {rae_mean, rae_std, rae_ci95, mae_mean, seeds}.
    """
    if isinstance(train_df, str):
        train_df = pd.read_csv(train_df)
    tr = train_df.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)
    smi = tr[smiles_col].tolist(); y = tr[target_col].to_numpy(float)
    X = featurize(smi, config["featurizer"])
    raes, maes = [], []
    for s in range(n_seeds):
        folds = make_folds(smi, cv, n_folds, seed=s)
        oof = np.full(len(y), np.nan); ok = True
        for tri, vai in folds:
            steps = PREPS[config["prep"]]()
            Xt, Xv = _fit_transform(steps, X[tri], X[vai])
            try:
                m = build(config["model"], config.get("params", {})); m.fit(Xt, y[tri]); oof[vai] = m.predict(Xv)
            except Exception:
                ok = False; break
        if ok and not np.isnan(oof).any():
            raes.append(rae(y, oof)); maes.append(mae(y, oof))
    raes = np.array(raes)
    out = {"rae_mean": float(raes.mean()), "rae_std": float(raes.std()),
           "rae_ci95": [float(np.percentile(raes, 2.5)), float(np.percentile(raes, 97.5))],
           "mae_mean": float(np.mean(maes)), "n_seeds": len(raes)}
    if verbose:
        print(f"  {config['featurizer']}+{config['prep']}+{config['model']}: "
              f"RAE {out['rae_mean']:.4f} +/- {out['rae_std']:.4f}  "
              f"(CI95 [{out['rae_ci95'][0]:.4f}, {out['rae_ci95'][1]:.4f}], n={out['n_seeds']})")
    return out


def compare_top(train_df, result, smiles_col="SMILES", target_col="y", k=3, n_seeds=15, **kw):
    """Stability-check the top-k configs; flag which are truly distinguishable from #1."""
    rows = []
    for i in range(min(k, len(result.results))):
        cfg = result.results.iloc[i].to_dict()
        st = stability_check(train_df, cfg, smiles_col, target_col, n_seeds=n_seeds, **kw)
        rows.append({**{c: cfg[c] for c in ("featurizer", "prep", "model")}, **st})
    df = pd.DataFrame(rows)
    if len(df) > 1:
        top = df.iloc[0]; band = top["rae_std"]
        df["distinguishable_from_best"] = df["rae_mean"] > top["rae_mean"] + band
        df.loc[0, "distinguishable_from_best"] = False
    return df
