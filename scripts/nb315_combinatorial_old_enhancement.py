"""nb315 -- Combinatorial old-approach enhancement.

Pick three historically strong stacks and retrain each with EVERY new
feature source we now have access to:

    v1 = nb107  (mechanistic assay decomposition)
    v2 = nb145  (XGBoost meta-stack)
    v3 = nb117  (KNN residual correction)

Added feature sources:
  (a) ADMET-fp from nb314 (10 d) if available, else RDKit-only (217 d)
  (b) Boltz iPTM-proxy features from boltz_dargason_features_test.parquet
      (test side only -> use mean-imputation for train rows)
      + the boltz pseudo-pec50 (te_boltz_pseudo_pec50.npy)
  (c) Per-compound pharmacophore class one-hot from nb305 SMARTS logic

Each variant is retrained with scaffold 5-fold CV, capped at 4000 feature
dimensions.  Outputs:
    oof_nb315_v1_nb107_enh.npy, te_nb315_v1_nb107_enh.npy
    oof_nb315_v2_nb145_enh.npy, te_nb315_v2_nb145_enh.npy
    oof_nb315_v3_nb117_enh.npy, te_nb315_v3_nb117_enh.npy
"""
from __future__ import annotations

import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.featurize import combined, rdkit_desc, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# Pharmacophore SMARTS (mirrors nb305)
from rdkit import Chem
CLASS_SMARTS = {
    "C1_ansamycin": ["[OX2H1]C1=C(O)C(=O)c2c(c1=O)C(=O)[#6]2"],
    "C2_steroid":   ["C1CC2CCC3CCCC4CCCC1C2C34", "C1CC2CCC3CCCCC3C2CC1"],
    "C3_statin":    ["C(=O)[OH]", "[CX4][CX4][CX4][CX4][CX4][CX4][CX4][CX4]"],
    "C4_taxane":    ["C1CC2CCC3(C)C1CC3C(=O)O2"],
    "C5_biaryl_azole": ["c1ccc(-c2ncc[nH]2)cc1", "c1ccc(-c2nnc[nH]2)cc1",
                        "c1ccc(-c2nccs2)cc1"],
    "C6_sulfonamide":  ["S(=O)(=O)N", "c1ccc(S(=O)(=O))cc1"],
    "C7_other":     ["[*]"],
}
CLASS_NAMES = list(CLASS_SMARTS.keys())

SEED = 42
N_FOLDS = 5
MAX_FEAT_DIM = 4000


def pharm_onehot(smiles_list):
    out = np.zeros((len(smiles_list), 7), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None: out[i, 6] = 1.0; continue
        for ci, c_name in enumerate(CLASS_NAMES):
            for patt in CLASS_SMARTS[c_name]:
                try:
                    p = Chem.MolFromSmarts(patt)
                    if p is not None and m.HasSubstructMatch(p):
                        out[i, ci] = 1.0; break
                except Exception:
                    pass
        if out[i].sum() == 0: out[i, 6] = 1.0
    return out


def load_admet_fp(smi_tr, smi_te):
    """Return (admet_tr, admet_te). Prefer nb314 saved-style features.

    If nb314 produced per-task feature columns we'd load them, but it
    saves only the LGBM-aggregated oof/te.  We fall back to using the
    OOF/te of nb314 as a single proxy feature, padded with RDKit
    descriptors so every variant still gets *some* ADMET-style signal.
    """
    oof_p = DATA_PROCESSED / "oof_nb314_tdc.npy"
    te_p  = DATA_PROCESSED / "te_nb314_tdc.npy"
    if oof_p.exists() and te_p.exists():
        ot = np.load(oof_p).reshape(-1, 1).astype(np.float32)
        tt = np.load(te_p).reshape(-1, 1).astype(np.float32)
        print(f"  ADMET-fp via nb314 OOF/test: {ot.shape}")
        # add RDKit descriptors for ADMET breadth
        Rt = impute(rdkit_desc(smi_tr)).astype(np.float32)
        Rs = impute(rdkit_desc(smi_te)).astype(np.float32)
        return np.hstack([ot, Rt]), np.hstack([tt, Rs])
    print("  nb314 not found -> using RDKit descriptors only as ADMET proxy")
    Rt = impute(rdkit_desc(smi_tr)).astype(np.float32)
    Rs = impute(rdkit_desc(smi_te)).astype(np.float32)
    return Rt, Rs


def load_boltz_features(smi_te_names, n_tr):
    """Return (boltz_tr, boltz_te) feature matrices.

    Boltz features are available only for test compounds (184/513 in the
    dargason parquet).  For training rows we backfill with column means
    so the feature is still usable for training without leaking test info.
    """
    p = DATA_PROCESSED / "boltz_dargason_features_test.parquet"
    if not p.exists():
        print("  Boltz parquet missing -> skip boltz features.")
        return None, None
    df = pd.read_parquet(p)
    feat_cols = [c for c in df.columns if c != "name"]
    feat_cols = [c for c in feat_cols
                 if any(k in c for k in ("iptm", "ptm", "plddt", "confidence"))]
    feat_cols = feat_cols[:30]  # keep dim tight
    print(f"  Boltz features: {len(feat_cols)} cols on {len(df)} test rows")

    name_to_row = {r["name"]: r[feat_cols].values.astype(np.float32)
                   for _, r in df.iterrows()}
    means = df[feat_cols].mean(axis=0).values.astype(np.float32)

    te_mat = np.zeros((len(smi_te_names), len(feat_cols)), dtype=np.float32)
    for i, nm in enumerate(smi_te_names):
        te_mat[i] = name_to_row.get(nm, means)

    # iptm pseudo-pec50 column (test-only)
    pseudo_p = DATA_PROCESSED / "te_boltz_pseudo_pec50.npy"
    if pseudo_p.exists():
        pseudo = np.load(pseudo_p).astype(np.float32)
        if pseudo.shape[0] == te_mat.shape[0]:
            te_mat = np.hstack([te_mat, pseudo.reshape(-1, 1)])
            feat_cols = feat_cols + ["pseudo_pec50"]

    tr_mat = np.tile(np.append(means, np.array([0.0], dtype=np.float32))
                     if "pseudo_pec50" in feat_cols else means,
                     (n_tr, 1)).astype(np.float32)
    return tr_mat, te_mat


def cap_dim(X_tr, X_te, max_dim):
    if X_tr.shape[1] <= max_dim: return X_tr, X_te
    # drop the Morgan tail to fit dim budget: morgan is the first 2048 cols
    keep = max_dim
    return X_tr[:, :keep], X_te[:, :keep]


# ───────────────────────── Variant 1: nb107-like (assay decomposition) ──────────
def variant_v1_nb107(X_tr, X_te, y, splits, raw_train, raw_counter, mol_names):
    """Train assay-decomposition LGBM on enhanced features."""
    LGBM_AUX = dict(n_estimators=500, num_leaves=32, learning_rate=0.05,
                    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1,
                    n_jobs=4)
    LGBM_MAIN = dict(n_estimators=1500, num_leaves=64, learning_rate=0.03,
                     min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                     reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1,
                     n_jobs=4, objective='mae')

    emax_raw = raw_train["Emax.vs.pos.ctrl_estimate (dimensionless)"].values.astype(np.float64)
    emax_log = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    pec50_null = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_med = np.nanmedian(pec50_null)
    null_imp = np.where(np.isnan(pec50_null), null_med, pec50_null)
    selectivity = y - null_imp
    has_null = (~np.isnan(pec50_null)).astype(np.float32)

    # AUX OOF
    oof_em = np.zeros(len(y)); oof_nl = np.zeros(len(y)); oof_sl = np.zeros(len(y))
    te_em = []; te_nl = []; te_sl = []
    for tri, vai in splits:
        m1 = lgb.LGBMRegressor(**LGBM_AUX); m1.fit(X_tr[tri], emax_log[tri])
        oof_em[vai] = 10.0 ** m1.predict(X_tr[vai]); te_em.append(10.0 ** m1.predict(X_te))
        m2 = lgb.LGBMRegressor(**LGBM_AUX); m2.fit(X_tr[tri], null_imp[tri])
        oof_nl[vai] = m2.predict(X_tr[vai]); te_nl.append(m2.predict(X_te))
        m3 = lgb.LGBMRegressor(**LGBM_AUX); m3.fit(X_tr[tri], selectivity[tri])
        oof_sl[vai] = m3.predict(X_tr[vai]); te_sl.append(m3.predict(X_te))
    te_em = np.mean(te_em, axis=0); te_nl = np.mean(te_nl, axis=0); te_sl = np.mean(te_sl, axis=0)

    aug_tr = np.column_stack([oof_em, oof_nl, oof_sl, has_null,
                              np.log1p(np.clip(oof_em, 0, None))]).astype(np.float32)
    aug_te = np.column_stack([te_em, te_nl, te_sl,
                              np.zeros(len(X_te), dtype=np.float32),
                              np.log1p(np.clip(te_em, 0, None))]).astype(np.float32)

    X_tr_aug = np.hstack([X_tr, aug_tr]).astype(np.float32)
    X_te_aug = np.hstack([X_te, aug_te]).astype(np.float32)
    X_tr_aug, X_te_aug = cap_dim(X_tr_aug, X_te_aug, MAX_FEAT_DIM)

    oof = np.zeros(len(y)); te_preds = []
    for fold, (tri, vai) in enumerate(splits):
        m = lgb.LGBMRegressor(**LGBM_MAIN)
        m.fit(X_tr_aug[tri], y[tri], eval_set=[(X_tr_aug[vai], y[vai])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        oof[vai] = m.predict(X_tr_aug[vai])
        te_preds.append(m.predict(X_te_aug))
    te_final = np.mean(te_preds, axis=0)
    return oof, te_final


# ───────────────────────── Variant 2: nb145-like (XGB meta-stack) ───────────────
def variant_v2_nb145(X_tr, X_te, y, splits):
    """XGBoost meta-stack on enhanced features (depth-3 shallow stack)."""
    xparams = dict(n_estimators=500, max_depth=3, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                   reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
                   device="cpu", verbosity=0, n_jobs=4, random_state=SEED)

    # Stack 4 meta-models we already have available
    PRIOR = [
        ("nb145_xgb_stack",   "oof_nb145_xgb_stack.npy",   "te_nb145_xgb_stack.npy"),
        ("nb107_assay",       "oof_nb107_assay_decomp.npy","te_nb107_assay_decomp.npy"),
        ("nb117_knn",         "oof_nb117_knn_residual.npy","te_nb117_knn_residual.npy"),
        ("nb239_full_slsqp",  "oof_nb239_full_slsqp.npy",  "te_nb239_full_slsqp.npy"),
    ]
    meta_oof_cols, meta_te_cols = [], []
    for _, op, tp in PRIOR:
        op_p = DATA_PROCESSED / op; tp_p = DATA_PROCESSED / tp
        if not op_p.exists() or not tp_p.exists(): continue
        o = np.load(op_p).astype(np.float32); t = np.load(tp_p).astype(np.float32)
        if o.shape[0] != X_tr.shape[0] or t.shape[0] != X_te.shape[0]: continue
        o = np.where(np.isfinite(o), o, np.nanmean(o))
        t = np.where(np.isfinite(t), t, np.nanmean(t))
        meta_oof_cols.append(o); meta_te_cols.append(t)
    if meta_oof_cols:
        meta_oof = np.column_stack(meta_oof_cols).astype(np.float32)
        meta_te  = np.column_stack(meta_te_cols).astype(np.float32)
    else:
        meta_oof = np.zeros((X_tr.shape[0], 0), dtype=np.float32)
        meta_te  = np.zeros((X_te.shape[0], 0), dtype=np.float32)

    X_tr_stack = np.hstack([meta_oof, X_tr]).astype(np.float32)
    X_te_stack = np.hstack([meta_te,  X_te]).astype(np.float32)
    X_tr_stack, X_te_stack = cap_dim(X_tr_stack, X_te_stack, MAX_FEAT_DIM)

    oof = np.zeros(len(y)); te_preds = []
    for fold, (tri, vai) in enumerate(splits):
        m = xgb.XGBRegressor(**xparams)
        m.fit(X_tr_stack[tri], y[tri], verbose=False)
        oof[vai] = m.predict(X_tr_stack[vai])
        te_preds.append(m.predict(X_te_stack))
    return oof, np.mean(te_preds, axis=0)


# ───────────────────────── Variant 3: nb117-like (KNN residual correction) ──────
def variant_v3_nb117(X_tr, X_te, y, splits, smi_tr, smi_te):
    """LGBM base + KNN residual correction with enhanced features used in the LGBM."""
    LGBM = dict(n_estimators=1200, num_leaves=64, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1,
                n_jobs=4, objective='mae')

    Xt, Xs = cap_dim(X_tr.astype(np.float32), X_te.astype(np.float32), MAX_FEAT_DIM)

    # 1) base LGBM OOF on enhanced features
    base_oof = np.zeros(len(y)); base_te = []
    for tri, vai in splits:
        m = lgb.LGBMRegressor(**LGBM)
        m.fit(Xt[tri], y[tri], eval_set=[(Xt[vai], y[vai])],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)])
        base_oof[vai] = m.predict(Xt[vai]); base_te.append(m.predict(Xs))
    base_te = np.mean(base_te, axis=0)

    # 2) Tanimoto-weighted KNN residual correction
    fps_tr = morgan_fp_batch(smi_tr).astype(np.float32)
    fps_te = morgan_fp_batch(smi_te).astype(np.float32)
    res = y - base_oof

    def tanimoto_mat(Q, D, batch=128):
        n_q, n_d = len(Q), len(D)
        out = np.zeros((n_q, n_d), dtype=np.float32)
        d_norm = (D * D).sum(1)
        for s in range(0, n_q, batch):
            e = min(s + batch, n_q)
            q = Q[s:e]; q_norm = (q * q).sum(1, keepdims=True)
            dot = q @ D.T; union = q_norm + d_norm[None, :] - dot
            out[s:e] = np.where(union > 0, dot / union, 0.0)
        return out

    def knn_pred(sim_mat, residuals, k=5, thr=0.25):
        n = len(sim_mat); out = np.zeros(n)
        for i in range(n):
            r = sim_mat[i]; mask = r >= thr
            if mask.sum() == 0: continue
            idx = np.argsort(-r[mask])[:k]
            tops = r[mask][idx]; topr = residuals[mask][idx]
            out[i] = (tops * topr).sum() / max(tops.sum(), 1e-9)
        return out

    # OOF correction
    oof_corr = np.zeros(len(y))
    for tri, vai in splits:
        sm = tanimoto_mat(fps_tr[vai], fps_tr[tri])
        oof_corr[vai] = knn_pred(sm, res[tri], k=5)

    # sweep alpha
    best_alpha, best_rae = 0.0, rae(y, base_oof)
    for a in np.arange(0.0, 1.01, 0.05):
        r = rae(y, base_oof + a * oof_corr)
        if r < best_rae: best_rae, best_alpha = r, a

    sim_te = tanimoto_mat(fps_te, fps_tr)
    te_corr = np.clip(knn_pred(sim_te, res, k=5), -0.5, 0.5)
    oof_final = base_oof + best_alpha * oof_corr
    te_final  = base_te + best_alpha * te_corr
    print(f"  v3 KNN-residual: best_alpha={best_alpha:.2f}  base_RAE={rae(y, base_oof):.4f}"
          f"  -> {rae(y, oof_final):.4f}")
    return oof_final, te_final


def report(name, oof, te, y):
    r = rae(y, oof); sp, _ = stats.spearmanr(y, oof)
    ratio = te.std() / oof.std() if oof.std() > 0 else 0.0
    print(f"  {name:30s}  RAE={r:.4f}  Spearman={sp:.4f}  te_std={te.std():.3f}  "
          f"ratio={ratio:.2f}")
    return r, sp, ratio


def slsqp_weight(oof_target, base_oof, te_target, base_te, y, n_seeds=80):
    """Return SLSQP weight for `oof_target` when blended with nb239 base."""
    M = np.column_stack([base_oof, oof_target])
    def loss(w): return rae(y, M @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * 2
    best = None
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        w0 = rng.dirichlet(np.ones(2))
        try:
            res = minimize(loss, w0, method="SLSQP", bounds=bounds,
                           constraints=cons, options={"ftol": 1e-9})
            if best is None or res.fun < best.fun: best = res
        except Exception:
            continue
    return best


def main():
    print("=== nb315: Combinatorial old-approach enhancement ===\n")

    tr = load_train(); te = load_test()
    y = tr["pec50"].values.astype(np.float64)
    n_tr = len(y); n_te = len(te)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    smi_tr = tr["smiles"].tolist()
    smi_te = te["smiles"].tolist()
    name_te = te["name"].tolist()
    mol_names_tr = tr["name"].values

    # Build the three new feature blocks
    print("--- Building enhancement feature blocks ---")
    X_tr_base = impute(combined(smi_tr)).astype(np.float32)
    X_te_base = impute(combined(smi_te)).astype(np.float32)
    print(f"  base combined: train {X_tr_base.shape}  test {X_te_base.shape}")

    admet_tr, admet_te = load_admet_fp(smi_tr, smi_te)
    print(f"  admet block:   train {admet_tr.shape}  test {admet_te.shape}")

    boltz_tr, boltz_te = load_boltz_features(name_te, n_tr)
    if boltz_tr is None:
        boltz_tr = np.zeros((n_tr, 0), dtype=np.float32)
        boltz_te = np.zeros((n_te, 0), dtype=np.float32)
    print(f"  boltz block:   train {boltz_tr.shape}  test {boltz_te.shape}")

    pharm_tr = pharm_onehot(smi_tr)
    pharm_te = pharm_onehot(smi_te)
    print(f"  pharm block:   train {pharm_tr.shape}  test {pharm_te.shape}")

    # Combine; cap-dim per variant
    X_tr_enh = np.hstack([X_tr_base, admet_tr, boltz_tr, pharm_tr]).astype(np.float32)
    X_te_enh = np.hstack([X_te_base, admet_te, boltz_te, pharm_te]).astype(np.float32)
    print(f"  enhanced feature dim: {X_tr_enh.shape[1]}  (cap={MAX_FEAT_DIM})\n")

    raw_train = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")

    summary = {}

    # ---- Variant 1: nb107 enhanced ----
    print("--- Variant 1: nb107 assay-decomposition (enhanced features) ---")
    oof1, te1 = variant_v1_nb107(X_tr_enh, X_te_enh, y, splits,
                                  raw_train, raw_counter, mol_names_tr)
    te1 = np.clip(te1, y.min() - 0.5, y.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb315_v1_nb107_enh.npy", oof1)
    np.save(DATA_PROCESSED / "te_nb315_v1_nb107_enh.npy",  te1)
    r1, sp1, ra1 = report("v1_nb107_enh", oof1, te1, y)
    summary["v1_nb107_enh"] = (r1, sp1, ra1, oof1, te1)

    # ---- Variant 2: nb145 enhanced ----
    print("\n--- Variant 2: nb145 XGB meta-stack (enhanced features) ---")
    oof2, te2 = variant_v2_nb145(X_tr_enh, X_te_enh, y, splits)
    te2 = np.clip(te2, y.min() - 0.5, y.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb315_v2_nb145_enh.npy", oof2)
    np.save(DATA_PROCESSED / "te_nb315_v2_nb145_enh.npy",  te2)
    r2, sp2, ra2 = report("v2_nb145_enh", oof2, te2, y)
    summary["v2_nb145_enh"] = (r2, sp2, ra2, oof2, te2)

    # ---- Variant 3: nb117 enhanced ----
    print("\n--- Variant 3: nb117 KNN residual correction (enhanced features) ---")
    oof3, te3 = variant_v3_nb117(X_tr_enh, X_te_enh, y, splits, smi_tr, smi_te)
    te3 = np.clip(te3, y.min() - 0.5, y.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb315_v3_nb117_enh.npy", oof3)
    np.save(DATA_PROCESSED / "te_nb315_v3_nb117_enh.npy",  te3)
    r3, sp3, ra3 = report("v3_nb117_enh", oof3, te3, y)
    summary["v3_nb117_enh"] = (r3, sp3, ra3, oof3, te3)

    # ---- SLSQP weights vs nb239 base ----
    nb239_oof_p = DATA_PROCESSED / "oof_nb239_full_slsqp.npy"
    nb239_te_p  = DATA_PROCESSED / "te_nb239_full_slsqp.npy"
    if nb239_oof_p.exists() and nb239_te_p.exists():
        base_oof = np.load(nb239_oof_p); base_te = np.load(nb239_te_p)
        print(f"\n--- 2-way SLSQP vs nb239 (base RAE={rae(y, base_oof):.4f}) ---")
        for nm, (r, sp, ra, o, t) in summary.items():
            best = slsqp_weight(o, base_oof, t, base_te, y)
            if best is None: continue
            print(f"  {nm:25s}  blend_RAE={best.fun:.4f}  w(variant)={best.x[1]:.4f}")

    # Save submissions for each variant
    for nm, (_, _, _, o, t) in summary.items():
        sub = pd.DataFrame({"Molecule Name": te["name"].values,
                            "SMILES":         te["smiles"].values,
                            "pEC50":          t})
        out = SUBMISSIONS / f"315_{nm}.csv"
        sub.to_csv(out, index=False)
        print(f"  saved {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
