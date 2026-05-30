"""nb219 -- Use the in-house single-concentration PXR screen (21k records,
10,870 unique compounds, 8,477 NEW vs CRC training set).

This is gold data we've underutilized:
  - Same target (PXR), same lab, same assay (just single point)
  - t-statistic vs pEC50: Spearman rho = 0.588 (better than any external)
  - log2_fc vs pEC50: rho = 0.522
  - Provides 8,477 new compounds for PXR signal beyond the 4,139 CRC compounds

Strategy:
  A. Build per-compound aggregated single-conc summary (median over replicates)
  B. Learn calibration map from single-conc features -> pEC50 using 2,393 overlap
  C. Apply map to the 8,477 single-conc-only compounds -> synthetic pEC50 labels
  D. Train augmented LGBM on combined CRC + synthetic, with sample weights
     reflecting (label uncertainty for synthetic vs. measured for CRC)
  E. Also build a multi-task variant: shared features, learn to predict
     both pEC50 (CRC head) and t_statistic (single-conc head). Use the
     CRC head for evaluation.

Expected to genuinely help: the 8,477 new compounds expand chemical space
coverage in the same activity domain.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import HuberRegressor

from pxr.data import load_train, load_test, load_single_conc, load_counter
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

COLLAPSE_THRESH = 0.58

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)


def aggregate_single_conc(sc):
    """Per-compound aggregation of the single-concentration screen."""
    agg = sc.groupby("smiles", as_index=False).agg(
        log2_fc=("log2_fc_estimate", "median"),
        log2_fc_max=("log2_fc_estimate", "max"),
        log2_fc_se=("log2_fc_stderr", "median"),
        t_stat=("t_statistic", "median"),
        t_stat_max=("t_statistic", "max"),
        neg_log_fdr=("neg_log10_fdr", "max"),
        cohens_d=("cohens_d", "median"),
        n_obs=("log2_fc_estimate", "count"),
    )
    print(f"  Aggregated single-conc: {len(agg):,} unique compounds")
    return agg


def fit_calibration_map(joint_df):
    """Learn map from single-conc features -> pEC50 using overlapping compounds."""
    feats = ["log2_fc", "log2_fc_max", "t_stat", "t_stat_max", "neg_log_fdr", "cohens_d"]
    X = joint_df[feats].copy()
    # Cap infinite values (e.g., -log10(fdr=0) = inf)
    for c in feats:
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        cap = X[c].quantile(0.99)
        X[c] = X[c].clip(upper=cap)
    X = X.fillna(X.median()).values
    y = joint_df["pec50"].values

    # Huber regression handles outliers in single-conc noise
    reg = HuberRegressor(max_iter=500)
    reg.fit(X, y)
    pred = reg.predict(X)
    r_pearson = pearsonr(y, pred).statistic
    r_spearman = spearmanr(y, pred).correlation
    mae = np.mean(np.abs(y - pred))
    print(f"  Calibration on {len(joint_df):,} overlap compounds:")
    print(f"    Pearson r = {r_pearson:.3f}  Spearman ρ = {r_spearman:.3f}  MAE = {mae:.3f}")
    print(f"    Feature weights: {dict(zip(feats, np.round(reg.coef_, 3)))}")
    return reg, feats, mae


def apply_calibration(sc_only, reg, feats):
    X = sc_only[feats].copy()
    for c in feats:
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        cap = X[c].quantile(0.99)
        X[c] = X[c].clip(upper=cap)
    X = X.fillna(X.median()).values
    return reg.predict(X)


def featurize(smiles_list):
    X = combined(smiles_list)
    return impute(X)


def main():
    print("=== nb219: Single-conc data augmentation ===\n")

    tr = load_train()
    te_df = load_test()
    sc = load_single_conc()

    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()
    pxr_train_set = set(smiles_tr)

    oof_base = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_base  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    base_rae = rae(y_tr, oof_base)
    print(f"Base nb197: OOF RAE={base_rae:.4f}  ratio={te_base.std()/oof_base.std():.3f}\n")

    # ── A. Aggregate single-conc ──────────────────────────────────────────────
    print("[A] Aggregating single-conc...")
    sc_agg = aggregate_single_conc(sc)

    # ── B. Learn calibration map on overlap ───────────────────────────────────
    print("\n[B] Learning calibration map (single_conc features -> pEC50):")
    joint = tr[["smiles", "pec50"]].merge(sc_agg, on="smiles", how="inner")
    print(f"  Overlap CRC+single-conc: {len(joint):,}")
    reg, calib_feats, calib_mae = fit_calibration_map(joint)

    # ── C. Synthetic labels for single-conc-only compounds ────────────────────
    sc_only = sc_agg[~sc_agg["smiles"].isin(pxr_train_set)].copy()
    print(f"\n[C] Single-conc-only compounds (not in CRC): {len(sc_only):,}")
    sc_only["pseudo_pec50"] = apply_calibration(sc_only, reg, calib_feats)
    print(f"  Synthetic pEC50 distribution: mean={sc_only['pseudo_pec50'].mean():.2f}  "
          f"std={sc_only['pseudo_pec50'].std():.2f}  "
          f"range=[{sc_only['pseudo_pec50'].min():.2f}, {sc_only['pseudo_pec50'].max():.2f}]")

    # Sanity: ensure SMILES parse cleanly before featurizing
    from rdkit import Chem
    sc_only["mol_ok"] = sc_only["smiles"].map(lambda s: Chem.MolFromSmiles(str(s)) is not None)
    sc_only = sc_only[sc_only["mol_ok"]].reset_index(drop=True)
    print(f"  After SMILES sanity: {len(sc_only):,}")

    # ── D. Build augmented training set with sample weights ───────────────────
    print("\n[D] Featurizing augmented training set...")
    X_tr_base = featurize(smiles_tr)
    X_te = featurize(smiles_te)
    X_aux = featurize(sc_only["smiles"].tolist())
    y_aux = sc_only["pseudo_pec50"].values.astype(np.float64)

    # Sample weights: CRC = 1.0; single-conc = scaled by calibration trust
    # Use cohens_d magnitude (signal strength) and n_obs (replication)
    w_crc = np.ones(len(y_tr))
    sc_trust = sc_only[["cohens_d", "neg_log_fdr", "n_obs"]].copy()
    # Replace inf with NaN then fill all NaN with 0
    sc_trust = sc_trust.replace([np.inf, -np.inf], np.nan).fillna(0)
    sc_trust["cohens_d_abs"] = sc_trust["cohens_d"].abs()
    # Robust max: 99th pct to avoid extreme single-point spikes
    cd_max = max(sc_trust["cohens_d_abs"].quantile(0.99), 1e-6)
    nf_max = max(sc_trust["neg_log_fdr"].quantile(0.99), 1e-6)
    no_max = max(np.log1p(sc_trust["n_obs"]).max(), 1e-6)
    sc_trust["score"] = (sc_trust["cohens_d_abs"].clip(upper=cd_max) / cd_max +
                         sc_trust["neg_log_fdr"].clip(upper=nf_max) / nf_max +
                         np.log1p(sc_trust["n_obs"]) / no_max) / 3
    sc_trust["score"] = sc_trust["score"].clip(0.05, 1.0)  # floor of 0.05
    base_aux_weight = 0.30  # synthetic labels < CRC labels
    w_aux = base_aux_weight * sc_trust["score"].values
    w_aux = np.where(np.isfinite(w_aux), w_aux, 0.05)  # final safety net
    print(f"  Augmentation weight stats: mean={w_aux.mean():.3f}  range=[{w_aux.min():.3f}, {w_aux.max():.3f}]")
    print(f"  Combined train size: {len(y_tr):,} CRC + {len(y_aux):,} aug = {len(y_tr)+len(y_aux):,}")

    # ── E. Scaffold CV with augmentation ──────────────────────────────────────
    print("\n[E] Scaffold 5-fold CV with single-conc augmentation:")
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    results = {}
    for name, use_aug in [("base_only", False), ("aug_30pct", True), ("aug_50pct", True)]:
        if name == "aug_50pct":
            w_aux_use = 0.50 / 0.30 * w_aux  # scale to 50% base
        else:
            w_aux_use = w_aux

        oof = np.zeros(len(y_tr))
        te_preds = []
        for tr_idx, va_idx in folds:
            X_combined = np.vstack([X_tr_base[tr_idx], X_aux]) if use_aug else X_tr_base[tr_idx]
            y_combined = np.concatenate([y_tr[tr_idx], y_aux]) if use_aug else y_tr[tr_idx]
            w_combined = np.concatenate([w_crc[tr_idx], w_aux_use]) if use_aug else w_crc[tr_idx]

            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(X_combined, y_combined, sample_weight=w_combined,
                  eval_set=[(X_tr_base[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_tr_base[va_idx])
            te_preds.append(m.predict(X_te))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {name:12s}: OOF RAE={r:.4f}  te_std={te_pred.std():.4f}  "
              f"ratio={ratio:.3f}  [{flag}]")
        results[name] = (oof, te_pred, r, ratio)

    # ── F. Blend with nb197 ──────────────────────────────────────────────────
    best_aug = min([(k, v) for k, v in results.items() if k != "base_only"],
                   key=lambda x: x[1][2])
    aug_name, (oof_aug, te_aug, r_aug, ratio_aug) = best_aug
    print(f"\n[F] Best augmented model: {aug_name} (OOF={r_aug:.4f})")
    print("    Blending with nb197...")
    best_blend, best_r_bl = None, 999
    for w in np.arange(0.05, 0.75, 0.05):
        oof_bl = (1-w)*oof_base + w*oof_aug
        te_bl  = (1-w)*te_base  + w*te_aug
        r_bl   = rae(y_tr, oof_bl)
        ratio_bl = te_bl.std() / oof_bl.std()
        flag = "PASS" if ratio_bl >= COLLAPSE_THRESH else "FAIL"
        print(f"  w={w:.2f}: OOF={r_bl:.4f}  ratio={ratio_bl:.3f}  [{flag}]")
        if ratio_bl >= COLLAPSE_THRESH and r_bl < best_r_bl:
            best_r_bl = r_bl
            best_blend = (w, oof_bl, te_bl, ratio_bl)

    # ── G. Save ───────────────────────────────────────────────────────────────
    saved = []
    # Always save the augmented LGBM OOF/test arrays — they're a new ensemble candidate
    # even if they don't beat nb197 alone.
    np.save(DATA_PROCESSED / f"oof_nb219_{aug_name}.npy", oof_aug)
    np.save(DATA_PROCESSED / f"te_nb219_{aug_name}.npy", te_aug)
    print(f"  [always-save] oof_nb219_{aug_name}.npy + te_nb219_{aug_name}.npy"
          f" (OOF={r_aug:.4f}, te_std={te_aug.std():.4f})")
    if ratio_aug >= COLLAPSE_THRESH and r_aug < base_rae:
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_aug})
        sub.to_csv(SUBMISSIONS / f"219_{aug_name}.csv", index=False)
        saved.append(f"219_{aug_name} OOF={r_aug:.4f}")

    if best_blend and best_r_bl < base_rae:
        w_b, oof_b, te_b, ratio_b = best_blend
        name = f"219_blend_w{int(w_b*100):02d}"
        np.save(DATA_PROCESSED / f"oof_{name}.npy", oof_b)
        np.save(DATA_PROCESSED / f"te_{name}.npy", te_b)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_b})
        sub.to_csv(SUBMISSIONS / f"{name}.csv", index=False)
        saved.append(f"{name} OOF={best_r_bl:.4f}  ratio={ratio_b:.3f}")

    print(f"\n=== Saved: {saved or ['none']}")


if __name__ == "__main__":
    main()
