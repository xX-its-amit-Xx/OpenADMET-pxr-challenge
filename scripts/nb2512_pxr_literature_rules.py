"""nb2512 -- PXR literature SAR rules as binary expert-prior features.

NEW PARADIGM (cycle 200+):
    Cycle-134 paradigm exhaustion + cycle-169 ceiling cluster (nb2171 0.4682)
    show that every learned-feature axis (LASSO/Boruta/MI/PermImp/PySR/spectral)
    is paradigm-matched with SHAP-on-LGBM and cannot break the ceiling. This
    script bypasses learned features and encodes known PXR pharmacophores
    directly as binary SMARTS substructure features. These are "expert prior"
    features: their presence is a direct chemistry signal from medchem
    literature (rifampicin, hyperforin, T0901317/CITCO, paclitaxel-like
    steroids, resveratrol/stilbene, biphenyl/clotrimazole, chlorpyrifos
    nitrophenyl, lithocholic-acid steroid A-ring, naphthyl hydrophobe,
    bridged bicyclic norbornane-like promiscuity flag).

    The expert prior is ORTHOGONAL to learned-feature rankings because the
    SMARTS are hand-coded by chemistry (not optimized to LGBM splits) and
    encode multi-atom patterns that single atom-pair / Morgan bits do not
    cleanly represent. We feed them alongside the existing X_117 pyramid
    matrix (5-way + ChEMBL kNN), build a K=25 RFE residual on
    chemprop_aux, and slot the K=25 anchor into the standard nb2171 pyramid
    (chemprop_aux + nb1191 + nb503 + nb562) under scaffold-CV + SLSQP +
    rank-stretch.

PROTOCOL:
    1. Define 10 PXR literature SMARTS (hardcoded below).
    2. For TRAIN(4139), UNB-as-test-subset(253), TEST(513) compute binary
       substructure-present features via RDKit.
    3. Concatenate with cached X_117 (per-row) -> X_127 (n, 127).
    4. Greedy backward RFE on chemprop_aux residual (X_127 -> K=25):
       at each iteration drop the column whose removal most reduces the
       residual MSE under 5-fold KFold cross-fit.
    5. Build a K=25 residual anchor: chemprop_aux + LGBM(MSE) on the 25
       surviving features, mean-bag over 5 seeds {0,1,7,42,137}, KFold
       5-fold cross-fit per seed. Save nb2512_mean_bag_oof_K25 (253) and
       te_nb2512_K25 (513).
    6. Slot into 4-anchor pyramid {nb2512_K25, chemprop_aux, nb1191, nb503,
       nb562}. SLSQP convex blend (w>=0, sum=1) per scaffold-fold +
       rank-stretch (grid 1.000..1.150). 5-fold scaffold-CV on 253 across
       kf_seeds {1001..1005}.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4601 -> MARGINAL_BEAT
    else              -> FAIL

OUTPUTS:
    scripts/nb2512_pxr_literature_rules.py
    data/processed/nb2512_summary.json
    data/processed/nb2512_pred_oof.npy       (253,) float32
    data/processed/te_nb2512.npy             (513,) float32
    data/processed/nb2512_mean_bag_oof_K25.npy  (253,) float32
    data/processed/te_nb2512_K25.npy         (513,) float32
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger
from scipy.optimize import minimize
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2512"

# ---------------------------------------------------------------------------
# 10 PXR LITERATURE SAR SMARTS (binary substructure present / absent)
# ---------------------------------------------------------------------------
# Sources: PXR (NR1I2) medchem literature -- Watkins 2001 (rifampicin co-crystal),
# Xue 2007 (hyperforin), Synold 2001 (paclitaxel/RU486), Honkakoski 2003 (CITCO),
# Synold/Cherian 2009 (T0901317), Squires 2004 (lithocholic acid), pocket
# tolerance of bridged bicyclics (norbornane promiscuity 2002-2010 reviews).
PXR_RULES = [
    # 1. Rifampicin core fragment (ansa-bridged naphthalenone hint)
    ("rifampicin_core",    "O=C1C=Cc2cccc(O)c21"),
    # 2. Hyperforin acylphloroglucinol
    ("hyperforin_acyl",    "OC1=C(O)C(=C(O)C1=O)CC(C)C"),
    # 3. Steroidal A-ring (taxane/paclitaxel-like 6-6-6-5 partial)
    ("steroidal_aring",    "C1CCC2CCC3(C(C1)CC2)CCC4"),
    # 4. trans-stilbene (resveratrol-like)
    ("trans_stilbene",     "C=C/c1ccc(O)cc1"),
    # 5. Diphenyl methane (avermectin-like)
    ("diphenyl_methane",   "c1ccc(Cc2ccccc2)cc1"),
    # 6. Biphenyl (clotrimazole-like)
    ("biphenyl",           "c1ccc(-c2ccccc2)cc1"),
    # 7. Nitrophenyl (chlorpyrifos-like, PXR activator)
    ("nitrophenyl",        "c1ccc([N+](=O)[O-])cc1"),
    # 8. tert-butyl group (CITCO-like, T0901317)
    ("tert_butyl",         "CC(C)(C)"),
    # 9. Hydrophobic naphthyl (large hydrophobic PXR ligand class)
    ("naphthyl",           "c1ccc2ccccc2c1"),
    # 10. Bridged bicyclic / norbornane-like (PXR promiscuity flag)
    ("bridged_bicyclic",   "C1CC2CCCC1CC2"),
]

# ---------------------------------------------------------------------------
# Stage 1 config: K=25 RFE on X_127 residual to chemprop_aux
# ---------------------------------------------------------------------------
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"

RFE_TARGET_K = 25
RFE_FOLDS = 5
RFE_SEED = 42

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

# ---------------------------------------------------------------------------
# Stage 2 config: pyramid SLSQP + rank-stretch
# ---------------------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601
NB2171_REF_OOF = 0.4682

CHEMPROP_AUX_REF = 0.6216


# ============================================================================
# Helpers
# ============================================================================

def _lgbm_params(seed):
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=int(seed),
        n_jobs=2,
        verbosity=-1,
    )


def smarts_features(smiles_list, smarts_specs):
    """Compute binary (substructure present / absent) features.

    Returns (N, len(smarts_specs)) uint8 array and the list of feature names.
    A compound that fails to parse contributes all-zeros.
    """
    patts = []
    names = []
    for nm, smarts in smarts_specs:
        p = Chem.MolFromSmarts(smarts)
        if p is None:
            print(f"   [warn] could not compile SMARTS {nm!r}: {smarts}")
            patts.append(None)
        else:
            patts.append(p)
        names.append(nm)
    X = np.zeros((len(smiles_list), len(patts)), dtype=np.uint8)
    for i, s in enumerate(smiles_list):
        if not isinstance(s, str) or not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        for j, p in enumerate(patts):
            if p is None:
                continue
            if mol.HasSubstructMatch(p):
                X[i, j] = 1
    return X, names


def residual_oof_cv(X, residual, seed):
    """Single-seed KFold residual cross-fit; returns oof_resid (n,) float64."""
    n = len(residual)
    kf = KFold(n_splits=RFE_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def rfe_backward_K(X_unb, residual, target_K, seed=RFE_SEED):
    """Greedy backward elimination: drop column whose removal most reduces
    residual MSE (== improves cross-fit MSE the most) until target_K remain.

    Returns surviving column indices (sorted) and full per-iteration trace.
    """
    p = X_unb.shape[1]
    surviving = list(range(p))
    trace = []
    # Baseline (all features) score
    base_oof = residual_oof_cv(X_unb[:, surviving], residual, seed)
    base_mse = float(np.mean((residual - base_oof) ** 2))
    trace.append({"K": len(surviving), "mse": base_mse, "dropped": None})
    print(f"   RFE start K={len(surviving)}  mse={base_mse:.4f}")
    while len(surviving) > target_K:
        best_drop_idx = None
        best_mse = float("inf")
        # Try dropping each remaining column; pick the drop that minimizes mse
        for j in range(len(surviving)):
            trial = surviving[:j] + surviving[j + 1:]
            oof_j = residual_oof_cv(X_unb[:, trial], residual, seed)
            mse_j = float(np.mean((residual - oof_j) ** 2))
            if mse_j < best_mse:
                best_mse = mse_j
                best_drop_idx = j
        dropped_col = surviving[best_drop_idx]
        surviving = surviving[:best_drop_idx] + surviving[best_drop_idx + 1:]
        trace.append({
            "K": len(surviving),
            "mse": best_mse,
            "dropped": int(dropped_col),
        })
        if len(surviving) % 10 == 0 or len(surviving) <= target_K + 5:
            print(
                f"   RFE K={len(surviving):3d}  mse={best_mse:.4f}  "
                f"dropped col={dropped_col}"
            )
    surviving_sorted = sorted(surviving)
    return surviving_sorted, trace


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def _residual_cross_fit_one_seed(X, residual, seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def slsqp_simplex(P, y):
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r, best_s = r, float(s)
    return best_s, best_r


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w, fold_s = [], []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    return float(rae(y_unb, oof_blend)), oof_blend, fold_w, fold_s


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PXR literature SAR SMARTS rules (10 binary features)")
    print("=" * 78)

    # ---- Load test + truth ----
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = (
        te_df["smiles"].astype(str).tolist()
        if "smiles" in te_df.columns
        else te_df["SMILES"].astype(str).tolist()
    )
    te_names = (
        te_df["name"].values
        if "name" in te_df.columns
        else te_df["Molecule Name"].values
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    tr_df = load_train()
    n_train = len(tr_df)
    tr_smiles = (
        tr_df["smiles"].astype(str).tolist()
        if "smiles" in tr_df.columns
        else tr_df["SMILES"].astype(str).tolist()
    )
    print(f"[load] n_train={n_train}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_uniq_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_uniq_scaf}")

    # ---- Anchor ----
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    chemprop_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    assert te_anchor_513.shape == (n_test,)
    assert chemprop_oof.shape == (n_unb,)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    rae_anchor_oof = float(rae(y_unb, chemprop_oof))
    print(f"[anchor] te_chemprop[unb] RAE     = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")
    print(f"[anchor] nb1133 chemprop_aux OOF  = {rae_anchor_oof:.4f}")
    residual = y_unb - chemprop_oof

    # ---- Compute 10 SMARTS features on TRAIN / UNB / TEST ----
    print("\n" + "-" * 78)
    print(f"COMPUTE 10 PXR LITERATURE SMARTS  (TRAIN={n_train}, UNB={n_unb}, TEST={n_test})")
    print("-" * 78)
    Xs_tr, smarts_names = smarts_features(tr_smiles, PXR_RULES)
    Xs_te, _ = smarts_features(te_smiles, PXR_RULES)
    Xs_unb = Xs_te[unb_idx]
    print(f"[smarts] X_train={Xs_tr.shape}  X_unb={Xs_unb.shape}  X_te={Xs_te.shape}")
    print(f"[smarts] hit rates on TRAIN:")
    for nm, frac_tr, frac_te in zip(
        smarts_names,
        Xs_tr.mean(axis=0).tolist(),
        Xs_te.mean(axis=0).tolist(),
    ):
        print(f"   {nm:22s} train={frac_tr*100:5.2f}%  test={frac_te*100:5.2f}%")

    # ---- Load X_117 (unb, te) and concat to X_127 ----
    X_117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X_117_te = np.load(X117_TE_PATH).astype(np.float32)
    assert X_117_unb.shape == (n_unb, 117), f"X_117_unb {X_117_unb.shape}"
    assert X_117_te.shape == (n_test, 117), f"X_117_te {X_117_te.shape}"
    X_127_unb = np.concatenate([X_117_unb, Xs_unb.astype(np.float32)], axis=1)
    X_127_te = np.concatenate([X_117_te, Xs_te.astype(np.float32)], axis=1)
    print(f"[concat] X_127_unb={X_127_unb.shape}  X_127_te={X_127_te.shape}")

    # ---- K=25 backward RFE on chemprop_aux residual ----
    print("\n" + "-" * 78)
    print(f"K=25 BACKWARD RFE on X_127 residual (seed={RFE_SEED})")
    print("-" * 78)
    surviving_K25, rfe_trace = rfe_backward_K(
        X_127_unb, residual, target_K=RFE_TARGET_K, seed=RFE_SEED,
    )
    assert len(surviving_K25) == RFE_TARGET_K
    # Map global indices to feature provenance
    smarts_indices = list(range(117, 117 + len(PXR_RULES)))  # indices 117..126
    smarts_surviving = [j for j in surviving_K25 if j in smarts_indices]
    smarts_contributing_names = [smarts_names[j - 117] for j in smarts_surviving]
    x117_surviving = [j for j in surviving_K25 if j < 117]
    print(f"[rfe] K=25 surviving:")
    print(f"   X_117 cols kept ({len(x117_surviving)}): {x117_surviving}")
    print(f"   SMARTS cols kept ({len(smarts_surviving)}): {smarts_surviving}")
    print(f"   SMARTS contributing: {smarts_contributing_names}")

    X_unb_K25 = X_127_unb[:, surviving_K25].astype(np.float32)
    X_te_K25 = X_127_te[:, surviving_K25].astype(np.float32)

    # ---- Build K=25 anchor: chemprop_aux + LGBM residual mean-bag ----
    print("\n" + "-" * 78)
    print(f"K=25 RESIDUAL LGBM  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_rae = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof = _residual_cross_fit_one_seed(X_unb_K25, residual, s)
        per_seed_corrected[i] = chemprop_oof + resid_oof
        per_seed_rae.append(float(rae(y_unb, chemprop_oof + resid_oof)))
        te_resid_s = _train_full_then_predict_te(X_unb_K25, residual, X_te_K25, s)
        per_seed_te_resid[i] = te_resid_s
        print(f"   seed={s:3d}: rae_corr={per_seed_rae[-1]:.4f}  wall={time.time()-ts:.1f}s")
    mean_bag_oof_K25 = per_seed_corrected.mean(axis=0)
    mean_bag_te_resid_K25 = per_seed_te_resid.mean(axis=0)
    te_K25_513 = te_anchor_513 + mean_bag_te_resid_K25
    rae_K25_mean_bag = float(rae(y_unb, mean_bag_oof_K25))
    rae_K25_per_seed_mean = float(np.mean(per_seed_rae))
    print(f"\n[K25] per-seed mean RAE = {rae_K25_per_seed_mean:.4f}")
    print(f"[K25] mean-bag RAE      = {rae_K25_mean_bag:.4f}")
    print(f"[K25] anchor OOF RAE    = {rae_anchor_oof:.4f}  (delta {rae_K25_mean_bag - rae_anchor_oof:+.4f})")

    oof_K25_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof_K25.npy"
    te_K25_path = DATA_PROCESSED / f"te_{TAG}_K25.npy"
    np.save(oof_K25_path, mean_bag_oof_K25.astype(np.float32))
    np.save(te_K25_path, te_K25_513.astype(np.float32))
    print(f"[save] {oof_K25_path}")
    print(f"[save] {te_K25_path}")

    # ---- Stage 2: 4-anchor pyramid ----
    print("\n" + "=" * 78)
    print("STAGE 2: 4-ANCHOR PYRAMID {nb2512_K25, chemprop_aux, nb1191, nb503, nb562}")
    print("=" * 78)

    # nb1191 te cached on disk (PRE-clean nb1191 deploy vector)
    te_nb1191 = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
    te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    # Note: the task says "K=25 RFE on 127 ... fit pyramid". Use the same
    # 5-anchor pyramid layout as nb2171/nb2240 but with K=25 in slot 0.
    # Use cached OOFs reconstructed identically to nb2240's pyramid module.
    # Cached OOF readers:
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)

    # nb1191 OOF reconstruction (PRE-clean stack; see nb2171.reconstruct_nb1191_oof)
    NB1150_SLSQP4_OOFS = [
        "nb1133_chemprop_aux_pred_oof.npy",
        "nb503_pred_oof.npy",
        "nb1133_nb1014_pred_oof.npy",
        "nb2103_mean_bag_oof_K28.npy",
    ]
    NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]
    NB1191_DEPLOY_WEIGHTS = {
        "chemprop_aux": 0.0,
        "nb1150":       0.641721304028517,
        "nb1158_K32":   0.23970131778546713,
        "nb2112_K28":   0.11857737818601592,
    }
    NB1191_DEPLOY_S = 1.031

    cols_1150 = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing sub-anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols_1150.append(v)
    nb1150_oof = np.column_stack(cols_1150) @ np.asarray(
        NB1150_SLSQP4_WEIGHTS, dtype=np.float64
    )
    nb1158_oof = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb2112_oof = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    blend_1191 = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu_1191 = float(blend_1191.mean())
    nb1191_oof = mu_1191 + NB1191_DEPLOY_S * (blend_1191 - mu_1191)

    anchors_list = [
        ("nb2512_K25",   mean_bag_oof_K25.astype(np.float64), te_K25_513.astype(np.float64)),
        ("chemprop_aux", chemprop_oof,                         te_anchor_513),
        ("nb1191",       nb1191_oof,                           te_nb1191),
        ("nb503",        nb503_oof,                            te_nb503),
        ("nb562",        nb562_oof,                            te_nb562),
    ]
    indiv_rae = {}
    oof_cols, te_cols = [], []
    print("\n[anchors]")
    for disp, oof, te_arr in anchors_list:
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_test,), f"{disp} te {te_arr.shape}"
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:14s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K_anch = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K_anch}")

    # ---- Scaffold 5-fold CV across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fw, fs = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_s": [float(x) for x in fs],
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(
            f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  mean_s={np.mean(fs):.3f}  "
            f"w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}"
        )
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(
        f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
        f"(+/- {pooled_rae_std_seeds:.4f})"
    )
    print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")

    # ---- Deploy ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit weights on 253; mean(fold_s) across all 5 seeds)")
    print("-" * 78)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean([s for r in per_seed for s in r["fold_s"]]))
    in_rae_final = float(rae(y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    w_str = ", ".join(f"{disp}={w:.4f}" for (disp, _, _), w in zip(anchors_list, w_deploy))
    print(f"   deploy weights      = {w_str}")
    print(f"   deploy mu / s       = {mu_deploy:.4f} / {s_deploy:.4f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}")
    print(f"   te[unb_idx] RAE     = {te_unb_rae:.4f}")
    print(f"   te(513) mean/std    = {deploy_te.mean():.3f}/{deploy_te.std():.3f}")

    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae
    print(
        f"\n[LB-band] {LB_W_OOF:.2f}*OOF + {LB_W_TE:.2f}*te_unb = {lb_band_est:.4f}"
    )

    # ---- Gate ----
    if pooled_rae_mean_seeds < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean_seeds < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    delta_vs_nb2171 = pooled_rae_mean_seeds - NB2171_REF_OOF
    print("\n" + "-" * 78)
    print(f"GATE  promote<{GATE_PROMOTE:.4f}  marginal<{GATE_MARGINAL:.4f}  "
          f"nb2171_ref={NB2171_REF_OOF:.4f}")
    print("-" * 78)
    print(f"   pooled_rae      = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta vs nb2171 = {delta_vs_nb2171:+.4f}")
    print(f"   verdict         = {verdict}")

    # ---- Save pred_oof + te artefacts ----
    pred_oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(pred_oof_path, mean_oof.astype(np.float32))
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {pred_oof_path}  shape={mean_oof.shape}")
    print(f"[save] {te_npy_path}  shape={deploy_te.shape}")

    summary = {
        "tag": TAG,
        "method": "pxr_literature_smarts_rules_K25_RFE_then_5anchor_pyramid",
        "smarts_rules": [
            {"name": nm, "smarts": sm} for nm, sm in PXR_RULES
        ],
        "smarts_hit_rate_train": {
            nm: float(Xs_tr[:, j].mean())
            for j, nm in enumerate(smarts_names)
        },
        "smarts_hit_rate_test": {
            nm: float(Xs_te[:, j].mean())
            for j, nm in enumerate(smarts_names)
        },
        "smarts_hit_rate_unb": {
            nm: float(Xs_unb[:, j].mean())
            for j, nm in enumerate(smarts_names)
        },
        "smarts_contributing": smarts_contributing_names,
        "smarts_contributing_indices_global": [int(j) for j in smarts_surviving],
        "x117_surviving_indices": [int(j) for j in x117_surviving],
        "rfe_target_K": RFE_TARGET_K,
        "rfe_seed": RFE_SEED,
        "rfe_folds": RFE_FOLDS,
        "rfe_trace_tail": rfe_trace[-15:],
        "K25_surviving_indices_in_127": [int(j) for j in surviving_K25],
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "rae_K25_per_seed_mean": rae_K25_per_seed_mean,
        "rae_K25_mean_bag": rae_K25_mean_bag,
        "delta_K25_vs_anchor_oof": rae_K25_mean_bag - rae_anchor_oof,
        "anchor_chemprop_oof_rae": rae_anchor_oof,
        "anchor_chemprop_te_unb_rae": rae_anchor,
        "anchor_oof_rae_unb": indiv_rae,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "n_train": int(n_train),
        "n_unique_scaffolds": n_uniq_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "mean_rae": pooled_rae_mean_seeds,
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(anchors_list, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "compare_nb2171_oof": NB2171_REF_OOF,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_promote_target": GATE_PROMOTE,
        "gate_marginal_target": GATE_MARGINAL,
        "verdict": verdict,
        "promote": bool(verdict == "PROMOTE"),
        "marginal_beat": bool(verdict == "MARGINAL_BEAT"),
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "outputs": {
            "pred_oof_npy": str(pred_oof_path),
            "te_npy": str(te_npy_path),
            "mean_bag_oof_K25_npy": str(oof_K25_path),
            "te_K25_npy": str(te_K25_path),
        },
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K=25 mean-bag RAE       = {rae_K25_mean_bag:.4f}")
    print(f"   pooled RAE (5 seeds)    = {pooled_rae_mean_seeds:.4f}")
    print(f"   delta vs nb2171         = {delta_vs_nb2171:+.4f}")
    print(f"   verdict                 = {verdict}")
    print(f"   SMARTS contributing     = {smarts_contributing_names}")
    print(f"   LB band                 = {lb_band_est:.4f}")
    print(f"   wall                    = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "rae_K25_mean_bag",
        "delta_K25_vs_anchor_oof",
        "delta_vs_nb2171",
        "verdict",
        "promote",
        "marginal_beat",
        "smarts_contributing",
        "deploy_weights",
        "deploy_s",
        "lb_band_estimate",
    ):
        print(f"  {k}: {res.get(k)}")
