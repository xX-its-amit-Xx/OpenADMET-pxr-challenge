"""nb110 — Scaffold Prior + Residual Modeling.

Key insight: test set is analog expansion of 61 seed compounds. Within each
Murcko scaffold family in training, the pEC50 distribution is much tighter than
the global distribution. We can use scaffold-level priors to dramatically reduce
prediction uncertainty for test compounds with matching scaffold families.

Strategy:
  1. Group training compounds by Murcko scaffold
  2. For scaffold families with >= 3 members: compute mean, std, count
  3. Predict pEC50 = scaffold_group_prior + structural_residual
  4. The residual model only needs to learn LOCAL variation within scaffold families
  5. For test compounds whose scaffold is in training: apply scaffold prior
  6. For test compounds with novel scaffolds: fall back to global model

Additional features:
  - Scaffold-mean neighbor lookup (Tanimoto nearest scaffold, if exact not in training)
  - Per-scaffold selectivity statistics (scaffolds have characteristic selectivity)
  - Emax statistics per scaffold family
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from collections import defaultdict
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
MIN_SCAFFOLD_SIZE = 3   # min compounds per scaffold to use scaffold prior
SIM_THRESH = 0.40       # Tanimoto threshold for neighbor scaffold lookup

LGBM_PARAMS = dict(
    n_estimators=1500, num_leaves=64, learning_rate=0.04,
    min_child_samples=8, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)


def tanimoto_matrix(fps_q, fps_db, batch_size=512):
    """Compute Tanimoto similarity between query and database fingerprints."""
    N, M = len(fps_q), len(fps_db)
    result = np.zeros((N, M), dtype=np.float32)
    for i in range(0, N, batch_size):
        q = fps_q[i:i+batch_size].astype(np.float32)
        # Tanimoto: |A & B| / |A | B|
        inter = q @ fps_db.T                     # (b, M)
        sum_q = q.sum(1, keepdims=True)          # (b, 1)
        sum_d = fps_db.sum(1, keepdims=True).T   # (1, M)
        union = sum_q + sum_d - inter
        result[i:i+batch_size] = np.where(union > 0, inter / union, 0)
    return result


def compute_scaffold_features(smiles_list, scaffold_map, scaffold_stats):
    """
    For each compound, compute scaffold-group statistics as features.
    scaffold_map: smiles -> scaffold string
    scaffold_stats: scaffold -> dict(mean, std, count, mean_emax, mean_sel)
    Returns: (N, 6) float array
    """
    feats = []
    for smi in smiles_list:
        sc = scaffold_map.get(smi, "")
        if sc in scaffold_stats:
            s = scaffold_stats[sc]
            feats.append([
                s["mean"],           # scaffold mean pEC50
                s["std"],            # scaffold std pEC50
                np.log1p(s["count"]),# log scaffold size
                s["mean_sel"],       # scaffold mean selectivity
                s["mean_emax"],      # scaffold mean Emax
                1.0,                 # scaffold known flag
            ])
        else:
            # Unknown scaffold: use global statistics
            feats.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return np.array(feats, dtype=np.float32)


def find_scaffold_neighbors(te_fps, tr_fps, tr_scaffolds, scaffold_stats, threshold=SIM_THRESH):
    """
    For test compounds with unknown scaffold, find nearest training scaffold.
    Returns: (N_te, 5) — nearest known scaffold statistics
    """
    known_scaffolds = list(scaffold_stats.keys())
    n_te = len(te_fps)
    feats = np.zeros((n_te, 5), dtype=np.float32)

    # Batch similarity computation
    sim = tanimoto_matrix(te_fps, tr_fps)
    for i in range(n_te):
        sims_i = sim[i]
        mask = sims_i >= threshold
        if mask.sum() == 0:
            continue
        # Find the most common scaffold among neighbors
        neighbor_scaffolds = defaultdict(float)
        for j in np.where(mask)[0]:
            sc = tr_scaffolds[j]
            if sc in scaffold_stats:
                neighbor_scaffolds[sc] += sims_i[j]
        if not neighbor_scaffolds:
            continue
        best_sc = max(neighbor_scaffolds, key=neighbor_scaffolds.get)
        s = scaffold_stats[best_sc]
        feats[i] = [s["mean"], s["std"], np.log1p(s["count"]), s["mean_sel"], s["mean_emax"]]
    return feats


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp)
    if label:
        print(f"  [{label:50s}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  r={pr:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, Pearson=pr)


def main():
    print("=== nb110: Scaffold Prior + Residual Modeling ===\n")

    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr); n_te = len(te)

    # ── Assay metadata ────────────────────────────────────────────────────────
    emax_col  = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw  = raw_train[emax_col].values.astype(np.float64)
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names = raw_train["Molecule Name"].values
    pec50_null = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_median = np.nanmedian(pec50_null)
    null_imputed = np.where(np.isnan(pec50_null), null_median, pec50_null)
    selectivity  = y_tr - null_imputed
    emax_safe    = np.where(np.isnan(emax_raw), np.nanmedian(emax_raw), emax_raw)

    # ── Compute Murcko scaffolds ──────────────────────────────────────────────
    print("Computing Murcko scaffolds...")
    tr_scaffolds = [bemis_murcko(s) for s in tr["smiles"].tolist()]
    te_scaffolds = [bemis_murcko(s) for s in te["smiles"].tolist()]
    tr_scaffold_set = set(tr_scaffolds)
    smiles_to_scaffold_tr = dict(zip(tr["smiles"].tolist(), tr_scaffolds))
    smiles_to_scaffold_te = dict(zip(te["smiles"].tolist(), te_scaffolds))

    # Group training compounds by scaffold
    scaffold_groups = defaultdict(list)
    for idx, sc in enumerate(tr_scaffolds):
        scaffold_groups[sc].append(idx)

    # Compute per-scaffold statistics (ALL training data for test features)
    scaffold_stats_full = {}
    for sc, idxs in scaffold_groups.items():
        if len(idxs) >= MIN_SCAFFOLD_SIZE:
            pec50_sc = y_tr[idxs]
            sel_sc   = selectivity[idxs]
            emax_sc  = emax_safe[idxs]
            scaffold_stats_full[sc] = dict(
                mean=pec50_sc.mean(),
                std=pec50_sc.std() if len(idxs) > 1 else 0.0,
                count=len(idxs),
                mean_sel=sel_sc.mean(),
                mean_emax=emax_sc.mean(),
            )

    # Coverage analysis
    te_in_train_scaffold = sum(1 for sc in te_scaffolds if sc in scaffold_stats_full)
    tr_in_scaffold = sum(1 for sc in tr_scaffolds if sc in scaffold_stats_full)
    print(f"Scaffold families (>= {MIN_SCAFFOLD_SIZE} members): {len(scaffold_stats_full)}")
    print(f"Train coverage: {tr_in_scaffold}/{n_tr} ({100*tr_in_scaffold/n_tr:.1f}%)")
    print(f"Test coverage: {te_in_train_scaffold}/{n_te} ({100*te_in_train_scaffold/n_te:.1f}%)")

    # ── Compute Morgan FPs for neighbor scaffold lookup ───────────────────────
    print("\nComputing Morgan FPs for neighbor lookup...")
    tr_fps = morgan_fp_batch(tr["smiles"].tolist()).astype(np.float32)
    te_fps = morgan_fp_batch(te["smiles"].tolist()).astype(np.float32)

    # Scaffold features for test (using full training scaffold stats)
    print("Computing test scaffold features...")
    # First: direct scaffold match
    te_sc_feats_direct = compute_scaffold_features(
        te["smiles"].tolist(), smiles_to_scaffold_te, scaffold_stats_full
    )
    # For test compounds without direct scaffold match, find nearest training scaffold
    te_no_scaffold = te_sc_feats_direct[:, -1] == 0
    print(f"  Test with direct scaffold match: {(~te_no_scaffold).sum()}")
    print(f"  Test needing neighbor lookup: {te_no_scaffold.sum()}")

    if te_no_scaffold.sum() > 0:
        te_neighbor_feats = find_scaffold_neighbors(
            te_fps[te_no_scaffold], tr_fps, tr_scaffolds, scaffold_stats_full
        )
        te_sc_feats_direct[te_no_scaffold, :5] = te_neighbor_feats
        # Mark as "neighbor-based" (0.5 instead of 1.0 for known flag)
        te_sc_feats_direct[te_no_scaffold, 5] = 0.5

    # ── Structural features ───────────────────────────────────────────────────
    print("\nComputing structural features...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # Load best available OOF meta-features
    meta_oof = {}; meta_te = {}
    for stem in ["nb107_assay_decomp", "nb101_delta_base", "nb99_sc_bio_fp",
                 "grand_v6b", "lgbm_tuned", "catboost"]:
        oof_f = DATA_PROCESSED / f"oof_{stem}.npy"
        te_f  = DATA_PROCESSED / f"te_{stem}.npy"
        if oof_f.exists() and te_f.exists():
            oof_a = np.load(oof_f); te_a = np.load(te_f)
            if oof_a.ndim == 2: oof_a = oof_a[:, 0]
            if te_a.ndim == 2: te_a = te_a[:, 0]
            if len(oof_a) == n_tr:
                meta_oof[stem] = np.where(np.isfinite(oof_a), oof_a, np.nanmean(oof_a))
                meta_te[stem] = np.where(np.isfinite(te_a), te_a, np.nanmean(te_a))
                print(f"  Loaded: {stem} (OOF RAE={rae(y_tr, meta_oof[stem]):.4f})")

    M_tr_meta = np.column_stack(list(meta_oof.values())) if meta_oof else np.zeros((n_tr, 0))
    M_te_meta = np.column_stack(list(meta_te.values())) if meta_te else np.zeros((n_te, 0))

    # ── Scaffold CV with scaffold prior features ──────────────────────────────
    print("\n=== Scaffold CV ===")
    splits = scaffold_kfold_indices(tr_scaffolds, N_FOLDS, SEED)
    oof_sp = np.full(n_tr, np.nan)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        # Build FOLD-LEVEL scaffold stats (using only fold-train data)
        fold_scaffold_groups = defaultdict(list)
        for i in tr_idx:
            fold_scaffold_groups[tr_scaffolds[i]].append(i)

        fold_scaffold_stats = {}
        for sc, idxs in fold_scaffold_groups.items():
            if len(idxs) >= MIN_SCAFFOLD_SIZE:
                pec50_sc = y_tr[idxs]
                fold_scaffold_stats[sc] = dict(
                    mean=pec50_sc.mean(),
                    std=pec50_sc.std() if len(idxs) > 1 else 0.0,
                    count=len(idxs),
                    mean_sel=selectivity[idxs].mean(),
                    mean_emax=emax_safe[idxs].mean(),
                )

        # Scaffold features for fold's val compounds
        va_smiles = [tr["smiles"].tolist()[i] for i in va_idx]
        va_sc_feats = compute_scaffold_features(
            va_smiles,
            dict(zip(tr["smiles"].tolist(), tr_scaffolds)),
            fold_scaffold_stats
        )
        # For val compounds not in fold-train scaffold, use fold-level neighbor lookup
        va_no_sc = va_sc_feats[:, -1] == 0
        if va_no_sc.sum() > 0:
            va_fps = tr_fps[va_idx][va_no_sc]
            fold_tr_fps = tr_fps[tr_idx]
            fold_tr_scaffolds = [tr_scaffolds[i] for i in tr_idx]
            va_nbr = find_scaffold_neighbors(va_fps, fold_tr_fps, fold_tr_scaffolds,
                                              fold_scaffold_stats, threshold=SIM_THRESH)
            va_sc_feats[va_no_sc, :5] = va_nbr
            va_sc_feats[va_no_sc, 5] = 0.5

        # Scaffold features for fold's train compounds
        tr_smiles_fold = [tr["smiles"].tolist()[i] for i in tr_idx]
        tr_sc_feats = compute_scaffold_features(
            tr_smiles_fold,
            dict(zip(tr["smiles"].tolist(), tr_scaffolds)),
            fold_scaffold_stats
        )

        # Final feature matrices
        X_tr_f = np.hstack([X_tr[tr_idx], tr_sc_feats, M_tr_meta[tr_idx]])
        X_va_f = np.hstack([X_tr[va_idx], va_sc_feats, M_tr_meta[va_idx]])

        m = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_tr_f, label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_va_f, label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)]
        )
        oof_sp[va_idx] = m.predict(X_va_f)
        fold_rae = rae(y_tr[va_idx], oof_sp[va_idx])
        sc_coverage = (va_sc_feats[:, -1] > 0).mean()
        print(f"  fold {fold+1}  RAE={fold_rae:.4f}  scaffold_cov={100*sc_coverage:.0f}%",
              flush=True)

    full_metrics(y_tr, oof_sp, "nb110_scaffold_prior_residual")

    # ── Test predictions ──────────────────────────────────────────────────────
    X_te_full = np.hstack([X_te, te_sc_feats_direct, M_te_meta])
    X_tr_full = np.hstack([
        X_tr,
        compute_scaffold_features(tr["smiles"].tolist(), smiles_to_scaffold_tr, scaffold_stats_full),
        M_tr_meta
    ])
    m_final = lgb.train(
        dict(LGBM_PARAMS, n_estimators=1000),
        lgb.Dataset(X_tr_full, label=y_tr),
        callbacks=[lgb.log_evaluation(-1)]
    )
    te_preds = m_final.predict(X_te_full)
    te_preds = np.clip(te_preds, y_tr.min() - 0.5, y_tr.max() + 0.5)

    print(f"\nTest: min={te_preds.min():.2f}  med={np.median(te_preds):.2f}  "
          f"max={te_preds.max():.2f}  std={te_preds.std():.3f}")

    # ── Blend with nb109 if available ─────────────────────────────────────────
    nb109_oof = DATA_PROCESSED / "oof_nb109_deep_meta_stack.npy"
    nb109_te  = DATA_PROCESSED / "te_nb109_deep_meta_stack.npy"
    if nb109_oof.exists():
        o109 = np.load(nb109_oof); t109 = np.load(nb109_te)
        print("\nSweeping blend with nb109:")
        best_r, best_a = rae(y_tr, oof_sp), 0.0
        for alpha in np.arange(0.0, 1.01, 0.1):
            blend = (1 - alpha) * oof_sp + alpha * o109
            r = rae(y_tr, blend)
            print(f"  alpha(109)={alpha:.1f}  RAE={r:.4f}")
            if r < best_r:
                best_r, best_a = r, alpha
        te_blend = (1 - best_a) * te_preds + best_a * t109
        te_blend = np.clip(te_blend, y_tr.min() - 0.5, y_tr.max() + 0.5)
        sub_blend = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_blend})
        sub_blend.to_csv(SUBMISSIONS / "110_scaffold_109_blend.csv", index=False)
        print(f"  Best blend alpha(109)={best_a:.1f}  RAE={best_r:.4f}")
        print(f"  Blend test: med={np.median(te_blend):.2f}  std={te_blend.std():.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(DATA_PROCESSED / "oof_nb110_scaffold_prior.npy", oof_sp)
    np.save(DATA_PROCESSED / "te_nb110_scaffold_prior.npy", te_preds)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds})
    sub.to_csv(SUBMISSIONS / "110_scaffold_prior_residual.csv", index=False)
    print(f"\nSaved: submissions/110_scaffold_prior_residual.csv")
    print(f"OOF RAE = {rae(y_tr, oof_sp):.4f}")


if __name__ == "__main__":
    main()
