"""nb220 -- MMP transformation features (rdMMPA single-cut).

The medicinal chemist's "what changes when I swap this group" insight.

For each compound:
1. Single-bond cut -> list of (core, fragment) pairs (rdMMPA)
2. For each cut: find training compounds with same CORE (different fragments)
3. The "context-conditional pec50" = mean training pec50 with this core
4. Aggregate across all cuts for the compound

Features per compound:
- n_cuts_with_match: number of cuts that found training matches
- mean_context_pec50: average context-conditional prediction
- std_context_pec50: agreement
- max_context_pec50, min_context_pec50: best/worst case
- max_context_n: largest context match group size

Train LGBM with combined + MMP features.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMMPA
import lightgbm as lgb

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.featurize import combined as feat_combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
SEED = 42
COLLAPSE_THRESH = 0.58
PREV_BEST = 0.296172

t0 = time.time()


def fragment_compound(smi, max_cuts=1):
    """Returns list of (core_smiles, fragment_smiles) using rdMMPA single-cut."""
    if not isinstance(smi, str) or not smi:
        return []
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return []
    try:
        results = rdMMPA.FragmentMol(mol, maxCuts=max_cuts, resultsAsMols=False)
    except Exception:
        return []
    pairs = []
    for tup in results:
        if len(tup) != 2:
            continue
        core, chains = tup
        # Skip if core or chains is None
        if not core or not chains:
            continue
        # core may have multiple parts joined by '.'; normalize
        pairs.append((core, chains))
    return pairs


def main():
    print("=== nb220: MMP transformation features ===\n", flush=True)

    tr_df = load_train()
    te_df = load_test()
    y_tr = tr_df["pec50"].values.astype(np.float64)
    n_tr = len(tr_df)

    scaffolds = tr_df["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # Fragment all training compounds
    print(f"Fragmenting train compounds (single cut)...", flush=True)
    tr_frags = []
    for i, smi in enumerate(tr_df["smiles"]):
        tr_frags.append(fragment_compound(smi, max_cuts=1))
    n_total_cuts = sum(len(f) for f in tr_frags)
    print(f"  {n_total_cuts} total cuts ({time.time()-t0:.0f}s)", flush=True)

    # Fragment test compounds
    print(f"Fragmenting test compounds...", flush=True)
    te_frags = [fragment_compound(s, max_cuts=1) for s in te_df["smiles"]]
    print(f"  {sum(len(f) for f in te_frags)} test cuts ({time.time()-t0:.0f}s)\n", flush=True)

    def build_core_lookup(exclude_idx=None):
        """Build core -> [(idx, chain, pec50)] lookup."""
        lookup = {}
        excl = set(exclude_idx) if exclude_idx is not None else set()
        for i, fragl in enumerate(tr_frags):
            if i in excl: continue
            for core, chain in fragl:
                lookup.setdefault(core, []).append((i, chain, y_tr[i]))
        return lookup

    def mmp_features_for(query_frags, lookup, self_idx=None):
        """Compute MMP features for a query compound's fragment list."""
        n_match_cuts = 0
        ctx_pecs_per_cut = []  # list of pec50 lists (one per cut with match)
        max_match_n = 0
        for core, chain in query_frags:
            matches = lookup.get(core, [])
            if self_idx is not None:
                matches = [m for m in matches if m[0] != self_idx]
            if not matches:
                continue
            n_match_cuts += 1
            pecs = [m[2] for m in matches]
            ctx_pecs_per_cut.append(pecs)
            max_match_n = max(max_match_n, len(matches))

        if not ctx_pecs_per_cut:
            return np.array([0, 0.0, 0.0, 0.0, 0.0, 0, 0])

        # Aggregate per cut: take mean per cut, then aggregate across cuts
        cut_means = np.array([np.mean(p) for p in ctx_pecs_per_cut])
        cut_stds  = np.array([np.std(p) if len(p) > 1 else 0.0 for p in ctx_pecs_per_cut])

        return np.array([
            n_match_cuts,
            cut_means.mean(),
            cut_means.std() if len(cut_means) > 1 else 0.0,
            cut_means.max(),
            cut_means.min(),
            max_match_n,
            cut_stds.mean(),
        ])

    # ----- MMP features for train (fold-aware: exclude same-fold from lookup) -----
    print("Building MMP features for train (fold-aware)...", flush=True)
    mmp_tr = np.zeros((n_tr, 7))
    for fi, (tr_idx, va_idx) in enumerate(splits):
        # For va_idx compounds, build lookup from tr_idx ONLY
        lookup = build_core_lookup(exclude_idx=set(va_idx))
        for j, q_idx in enumerate(va_idx):
            mmp_tr[q_idx] = mmp_features_for(tr_frags[q_idx], lookup, self_idx=q_idx)
        print(f"  fold {fi+1}/{N_FOLDS} done ({time.time()-t0:.0f}s)", flush=True)

    # ----- MMP features for test (use all training) -----
    print("\nBuilding MMP features for test...", flush=True)
    full_lookup = build_core_lookup()
    mmp_te = np.zeros((len(te_df), 7))
    for j, frags in enumerate(te_frags):
        mmp_te[j] = mmp_features_for(frags, full_lookup)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Print stats
    print(f"\nMMP coverage:", flush=True)
    print(f"  Train: {(mmp_tr[:,0] > 0).sum()}/{n_tr} compounds had matched cuts", flush=True)
    print(f"  Test:  {(mmp_te[:,0] > 0).sum()}/{len(te_df)} compounds had matched cuts", flush=True)
    print(f"  Mean matches per train compound: {mmp_tr[:,0].mean():.1f}", flush=True)
    print(f"  Mean matches per test compound:  {mmp_te[:,0].mean():.1f}", flush=True)

    # Combine with base features
    print("\nBuilding combined feature matrix...", flush=True)
    X_tr_base = impute(feat_combined(tr_df["smiles"].tolist())).astype(np.float32)
    X_te_base = impute(feat_combined(te_df["smiles"].tolist())).astype(np.float32)

    X_tr = np.hstack([X_tr_base, mmp_tr.astype(np.float32)])
    X_te = np.hstack([X_te_base, mmp_te.astype(np.float32)])
    print(f"  shape: {X_tr.shape} (added 7 MMP features)\n", flush=True)

    # Train LGBM
    print("Training LGBM (5-fold scaffold CV)...", flush=True)
    oof = np.full(n_tr, np.nan)
    te_pred = np.zeros(len(te_df))

    for fi, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(
            n_estimators=2000, num_leaves=64, learning_rate=0.03,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, objective="regression_l1",
            random_state=SEED, verbose=-1,
        )
        m.fit(
            X_tr[tr_idx], y_tr[tr_idx],
            eval_set=[(X_tr[va_idx], y_tr[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof[va_idx] = m.predict(X_tr[va_idx])
        te_pred += m.predict(X_te) / N_FOLDS
        print(f"  fold {fi+1}/{N_FOLDS}: best_iter={m.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    r = rae(y_tr, oof)
    ratio = te_pred.std() / oof.std()
    flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
    beat = " ***BEATS PREV BEST***" if (ratio >= COLLAPSE_THRESH and r < PREV_BEST) else ""
    print(f"\n=== nb220 OOF: RAE={r:.6f}  ratio={ratio:.4f}  [{flag}]{beat} ===", flush=True)
    print(f"Total time: {time.time()-t0:.0f}s", flush=True)

    out_stem = "nb220_mmp_features"
    np.save(DATA_PROCESSED / f"oof_{out_stem}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{out_stem}.npy", te_pred)
    sub = pd.DataFrame({
        "SMILES": te_df["smiles"].values,
        "Molecule Name": te_df["name"].values,
        "pEC50": te_pred,
    })
    sub.to_csv(SUBMISSIONS / f"{out_stem}.csv", index=False)
    print(f"Saved: {out_stem}", flush=True)


if __name__ == "__main__":
    main()
