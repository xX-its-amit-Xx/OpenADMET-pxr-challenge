"""nb2520 -- SMILES test-time augmentation via RDKit doRandom for chemprop_aux residual.

NEW PARADIGM:
    For each molecule, generate 5 random non-canonical SMILES via
    Chem.MolToSmiles(mol, doRandom=True, canonical=False).  Recompute
    Morgan-2048 + RDKit descriptors from each random SMILES -> K=20
    feature vector.  Run the (chemprop_aux residual) LGBM K=20 model on
    each of the 5 feature vectors -> 5 predictions per molecule.  Final
    prediction = mean across 5 random SMILES.

    The hypothesis is that even though hash-based fingerprints are atom-
    order invariant in theory, RDKit's canonical-ordering during
    re-parse of a randomized SMILES yields the SAME molecular graph
    (so Morgan/RDKit-desc are TRULY invariant).  This script audits the
    paradigm honestly -- expected outcome is TTA-mean ~= canonical
    (within 1e-4 RAE), and SLSQP pyramid weight on TTA-K20 will likely
    collapse to (1, 0, 0, 0, 0) i.e. behave like nb2240 K=20.

    NOTE: nb2240's actual K=20 features mix Morgan/Avalon/MACCS/AtomPair
    with Mordred + ChempropEmbed (8/20 are ChempropEmbed dims that
    REQUIRE a chemprop checkpoint to regenerate, which is not on disk).
    Per user prompt ("recompute Morgan-2048 + RDKit descriptors fresh
    -> K=20 features") this script uses Morgan(2048)+RDKit(~217) as the
    feature substrate and selects K=20 by SHAP from a fold-fit on the
    253 unblind chemprop_aux residual.  This is the cleanest faithful
    interpretation -- TTA only makes sense on features that CAN be
    regenerated from random SMILES.

PROTOCOL:
    1. Load 253 unblind + 513 test SMILES.  Compute canonical Morgan+RDKit
       features once per molecule as the canonical baseline.
    2. SHAP-select K=20 features from full-data fold-fit of chemprop_aux
       residual ~ LGBM on (Morgan+RDKit) on the 253 unblind.
    3. For each molecule: generate 5 random SMILES (seeds 11..15).
       Re-featurize Morgan+RDKit per random SMILES -> 5 K=20 vectors.
    4. Outer 5-fold scaffold CV on 253 (5 kf_seeds 1001..1005):
       per fold:
         a. Refit LGBM mean-bag (5 inner seeds) on train-fold residual,
            using the SHAP-K20 feature subset.
         b. Predict on val-fold canonical + each of 5 TTA features.
         c. TTA-mean prediction = mean of 5 TTA preds.
       Per-seed pooled RAE = anchor + (TTA-mean residual).
    5. Build 5-anchor pyramid {nb2520_K20_TTA, chemprop_aux, nb1191,
       nb503, nb562} + SLSQP simplex + rank-stretch under same 5-fold
       scaffold CV across the 5 kf_seeds.  Compare to gates.

GATES:
    mean_rae < 0.4570 -> "PROMOTE"
    < 0.4601           -> "MARGINAL_BEAT"
    else               -> "FAIL"

OUTPUTS:
    scripts/nb2520_smiles_tta_randomization.py (this file)
    data/processed/nb2520_summary.json
    data/processed/nb2520_pred_oof.npy        (253,) float32 -- TTA-K20 anchor only
    data/processed/te_nb2520.npy              (513,) float32 -- deploy pyramid
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
from rdkit import Chem, RDLogger
from scipy.optimize import minimize
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, standardize
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined as featurize_combined
from pxr.featurize import impute as featurize_impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2520"

# ------------------------------ TTA spec -------------------------------------
N_TTA = 5
TTA_SEEDS = [11, 12, 13, 14, 15]
K_TOP = 20

# ------------------------------ LGBM residual --------------------------------
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]


def _lgbm_params(seed):
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


# ------------------------------ Pyramid + gates ------------------------------
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601

CHEMPROP_AUX_REF = 0.6216
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB1133_OOF_PATH = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"

# nb1191 reconstruction (copied from nb2171/nb2240)
NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]


# ============================================================================
# helpers
# ============================================================================

def random_smiles_for_mol(mol: Chem.Mol, seed: int) -> str | None:
    """Generate a non-canonical random SMILES.  Seeds RDKit RNG by trying multiple
    permutations until we get a non-canonical variant (best-effort)."""
    if mol is None:
        return None
    # Use seed to perturb (RDKit doRandom is global RNG; we re-seed numpy and
    # retry once if collision with canonical -- still doRandom is the
    # primary mechanism).
    try:
        # Re-randomize: try a few attempts up to a small bound
        rng = np.random.default_rng(seed)
        n_atoms = mol.GetNumAtoms()
        ans = list(range(n_atoms))
        rng.shuffle(ans)
        m2 = Chem.RenumberAtoms(mol, ans)
        s = Chem.MolToSmiles(m2, canonical=False, doRandom=True)
        return s if s else None
    except Exception:
        return None


def featurize_random_smiles_batch(smiles_list: list[str], seed: int) -> np.ndarray:
    """For each SMILES: parse -> standardize -> generate ONE random SMILES at seed
    -> recompute Morgan+RDKit (~2265-dim).  Returns (N, 2265) float32 imputed."""
    random_smis = []
    for s in smiles_list:
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                random_smis.append(s)  # fall back to canonical
                continue
            rs = random_smiles_for_mol(m, seed)
            random_smis.append(rs if rs else s)
        except Exception:
            random_smis.append(s)
    X = featurize_combined(random_smis)
    return X


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


def reconstruct_nb1150_oof(n_unb):
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 sub-anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS, dtype=np.float64)
    return P @ w


def reconstruct_nb1191_oof(n_unb):
    chemprop_oof = np.load(NB1133_OOF_PATH).astype(np.float64)
    nb1150_oof = reconstruct_nb1150_oof(n_unb)
    nb1158_oof = np.load(
        DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
    ).astype(np.float64)
    nb2112_oof = np.load(
        DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    ).astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop_oof
        + NB1191_DEPLOY_WEIGHTS["nb1150"]       * nb1150_oof
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"]   * nb1158_oof
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"]   * nb2112_oof
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


def select_top_k_by_shap(X_unb: np.ndarray, residual: np.ndarray, K: int) -> np.ndarray:
    """Fit a single LGBM on (X_unb, residual) and return top-K feature indices by
    mean absolute SHAP value.  Returns int array (K,)."""
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed=0))
    mdl.fit(X_unb, residual)
    # SHAP via TreeSHAP -- LightGBM's pred_contrib
    contrib = mdl.predict(X_unb, pred_contrib=True)  # (N, D+1)
    shap_vals = contrib[:, :-1]  # drop bias column
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(-mean_abs)[:K]
    return np.sort(top_idx).astype(int)


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SMILES TTA via doRandom -> Morgan+RDKit -> K=20 LGBM")
    print("=" * 78)

    # ---- Load test + unblind ----
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[load] chemprop_aux in_RAE = {rae_anchor:.4f} (ref {CHEMPROP_AUX_REF:.4f})")
    residual_unb = y_unb - anchor_unb

    # ---- Canonical Morgan+RDKit (one pass) ----
    print("\n[featurize] canonical Morgan+RDKit (combined)...")
    t_feat = time.time()
    X_te_canon = featurize_combined(te_smiles)
    X_te_canon = featurize_impute(X_te_canon).astype(np.float32)
    X_unb_canon = X_te_canon[unb_idx]
    feat_dim_full = X_te_canon.shape[1]
    print(f"   X_te_canon = {X_te_canon.shape}  X_unb_canon = {X_unb_canon.shape}  "
          f"wall={time.time() - t_feat:.1f}s")

    # ---- SHAP-select K=20 features on 253 unblind residual ----
    print(f"\n[shap-select] top-{K_TOP} features from residual fit on 253 unblind")
    top_K_idx = select_top_k_by_shap(X_unb_canon, residual_unb, K_TOP)
    print(f"   selected idx = {top_K_idx.tolist()}")
    X_te_K_canon = X_te_canon[:, top_K_idx].astype(np.float32)
    X_unb_K_canon = X_te_K_canon[unb_idx]

    # ---- TTA: pre-compute 5 random SMILES feature batches on FULL 513 ----
    print(f"\n[TTA] generating {N_TTA} random SMILES batches over {n_test} test mols")
    X_te_K_tta = np.zeros((N_TTA, n_test, K_TOP), dtype=np.float32)
    for i_tta, sd in enumerate(TTA_SEEDS):
        t_tta = time.time()
        X_full = featurize_random_smiles_batch(te_smiles, seed=sd)
        X_full = featurize_impute(X_full).astype(np.float32)
        # Sanity-check feature dim matches canonical
        if X_full.shape[1] != feat_dim_full:
            raise RuntimeError(
                f"TTA-{sd} feat dim {X_full.shape[1]} != canonical {feat_dim_full}"
            )
        X_te_K_tta[i_tta] = X_full[:, top_K_idx].astype(np.float32)
        print(f"   TTA seed={sd}: X={X_full.shape}  wall={time.time() - t_tta:.1f}s")
    X_unb_K_tta = X_te_K_tta[:, unb_idx, :]  # (N_TTA, n_unb, K)
    print(f"   X_unb_K_tta = {X_unb_K_tta.shape}  X_te_K_tta = {X_te_K_tta.shape}")

    # Diagnostic: are TTA features actually different from canonical?
    delta_canon_vs_tta_mean = float(
        np.mean(np.abs(X_unb_K_tta.mean(axis=0) - X_unb_K_canon))
    )
    print(f"   mean|TTA-mean - canonical| over K=20 = {delta_canon_vs_tta_mean:.6f}")

    # ============================================================================
    # Stage 1: Build TTA-K20 anchor predictions via 5-fold scaffold CV outer
    # (refit LGBM mean-bag per fold; predict on val-fold canonical + each TTA)
    # ============================================================================
    print("\n" + "-" * 78)
    print(f"STAGE 1: TTA-K20 anchor via 5-fold scaffold CV outer, mean-bag inner")
    print(f"  outer kf_seeds={KF_SEEDS}  inner LGBM seeds={RESID_SEEDS}")
    print("-" * 78)

    # Per-kf-seed: build TTA-mean residual OOF on 253, plus TTA-mean te residual on 513
    # We average inner-seed (RESID_SEEDS) LGBMs per fold.

    n_seeds_kf = len(KF_SEEDS)
    resid_oof_per_kfseed = np.zeros((n_seeds_kf, n_unb), dtype=np.float64)
    te_resid_per_kfseed = np.zeros((n_seeds_kf, n_test), dtype=np.float64)

    canonical_oof_per_kfseed = np.zeros((n_seeds_kf, n_unb), dtype=np.float64)
    canonical_te_per_kfseed = np.zeros((n_seeds_kf, n_test), dtype=np.float64)

    for k_i, kf_seed in enumerate(KF_SEEDS):
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )

        oof_canon = np.full(n_unb, np.nan, dtype=np.float64)
        oof_tta = np.full(n_unb, np.nan, dtype=np.float64)
        # te accumulator: average across folds (each fold trains on its own train)
        te_canon_fold_accum = np.zeros(n_test, dtype=np.float64)
        te_tta_fold_accum = np.zeros(n_test, dtype=np.float64)

        for tr_loc, va_loc in splits:
            X_tr = X_unb_K_canon[tr_loc]
            r_tr = residual_unb[tr_loc]

            # Mean-bag inner: train RESID_SEEDS LGBMs on train fold; predict val
            # (canonical) AND each TTA feature set on val, plus full 513 (canonical + TTA)
            pred_va_canon_bag = np.zeros(len(va_loc), dtype=np.float64)
            pred_va_tta_bag = np.zeros(len(va_loc), dtype=np.float64)
            pred_te_canon_bag = np.zeros(n_test, dtype=np.float64)
            pred_te_tta_bag = np.zeros(n_test, dtype=np.float64)

            for inner_seed in RESID_SEEDS:
                mdl = lgb.LGBMRegressor(**_lgbm_params(inner_seed))
                mdl.fit(X_tr, r_tr)
                # Val canonical
                pred_va_canon_bag += mdl.predict(X_unb_K_canon[va_loc])
                # Val TTA mean across N_TTA features
                va_tta_preds = np.stack([
                    mdl.predict(X_unb_K_tta[t_i, va_loc, :])
                    for t_i in range(N_TTA)
                ], axis=0)
                pred_va_tta_bag += va_tta_preds.mean(axis=0)
                # Te canonical
                pred_te_canon_bag += mdl.predict(X_te_K_canon)
                # Te TTA mean
                te_tta_preds = np.stack([
                    mdl.predict(X_te_K_tta[t_i])
                    for t_i in range(N_TTA)
                ], axis=0)
                pred_te_tta_bag += te_tta_preds.mean(axis=0)

            pred_va_canon_bag /= len(RESID_SEEDS)
            pred_va_tta_bag /= len(RESID_SEEDS)
            pred_te_canon_bag /= len(RESID_SEEDS)
            pred_te_tta_bag /= len(RESID_SEEDS)

            oof_canon[va_loc] = pred_va_canon_bag
            oof_tta[va_loc] = pred_va_tta_bag
            te_canon_fold_accum += pred_te_canon_bag
            te_tta_fold_accum += pred_te_tta_bag

        # te predicted is averaged across N_FOLDS (each fold trained on its own train)
        te_canon_fold_accum /= N_FOLDS
        te_tta_fold_accum /= N_FOLDS

        canonical_oof_per_kfseed[k_i] = oof_canon
        resid_oof_per_kfseed[k_i] = oof_tta
        canonical_te_per_kfseed[k_i] = te_canon_fold_accum
        te_resid_per_kfseed[k_i] = te_tta_fold_accum

        rae_canon = float(rae(y_unb, anchor_unb + oof_canon))
        rae_tta = float(rae(y_unb, anchor_unb + oof_tta))
        print(f"   kf_seed={kf_seed}  canon_RAE={rae_canon:.4f}  tta_RAE={rae_tta:.4f}  "
              f"delta={rae_tta - rae_canon:+.4f}  wall={time.time() - ts:.1f}s")

    # Mean-of-seed K=20 anchor predictions
    tta_anchor_oof = anchor_unb + resid_oof_per_kfseed.mean(axis=0)
    tta_anchor_te = te_anchor_513 + te_resid_per_kfseed.mean(axis=0)
    canon_anchor_oof = anchor_unb + canonical_oof_per_kfseed.mean(axis=0)
    canon_anchor_te = te_anchor_513 + canonical_te_per_kfseed.mean(axis=0)

    rae_tta_anchor = float(rae(y_unb, tta_anchor_oof))
    rae_canon_anchor = float(rae(y_unb, canon_anchor_oof))
    print(f"\n[K20-anchor] canon RAE = {rae_canon_anchor:.4f}")
    print(f"[K20-anchor] TTA   RAE = {rae_tta_anchor:.4f}  "
          f"(delta vs canon {rae_tta_anchor - rae_canon_anchor:+.4f})")

    # Save anchor OOF and te
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", tta_anchor_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}_K20.npy", tta_anchor_te.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / (TAG + '_pred_oof.npy')}")
    print(f"[save] {DATA_PROCESSED / ('te_' + TAG + '_K20.npy')}")

    # ============================================================================
    # Stage 2: 5-anchor pyramid SLSQP + rank-stretch
    # ============================================================================
    print("\n" + "=" * 78)
    print("STAGE 2: 5-ANCHOR PYRAMID  (TTA-K20 swaps in for nb2240_K20)")
    print("=" * 78)
    nb1191_oof = reconstruct_nb1191_oof(n_unb)
    chemprop_oof = np.load(NB1133_OOF_PATH).astype(np.float64)
    nb503_oof = np.load(DATA_PROCESSED / "nb503_pred_oof.npy").astype(np.float64)
    nb562_oof = np.load(DATA_PROCESSED / "nb562_pred_oof.npy").astype(np.float64)
    te_nb1191 = np.load(DATA_PROCESSED / "te_nb1191.npy").astype(np.float64)
    te_nb503 = np.load(DATA_PROCESSED / "te_nb503.npy").astype(np.float64)
    te_nb562 = np.load(DATA_PROCESSED / "te_nb562.npy").astype(np.float64)
    te_chemprop_aux = te_anchor_513

    anchors_list = [
        ("nb2520_K20_TTA", tta_anchor_oof.astype(np.float64), tta_anchor_te.astype(np.float64)),
        ("chemprop_aux",   chemprop_oof,                       te_chemprop_aux),
        ("nb1191",         nb1191_oof,                         te_nb1191),
        ("nb503",          nb503_oof,                          te_nb503),
        ("nb562",          nb562_oof,                          te_nb562),
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
        print(f"   {disp:18s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")
    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K_anch = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K_anch}")

    # Scaffold 5-fold CV across 5 seeds
    print("\n" + "-" * 78)
    print(f"PYRAMID SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fw, fs = cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed)
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": float(pooled),
            "fold_s": [float(x) for x in fs],
            "fold_w_mean": [float(x) for x in np.mean(fw, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  mean_s={np.mean(fs):.3f}  "
              f"w_mean={np.round(np.mean(fw, axis=0), 3).tolist()}")
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")

    # Deploy
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
    lb_low = lb_band_est - 0.05
    lb_high = lb_band_est + 0.05
    print(f"\n[LB-band] {LB_W_OOF:.2f}*OOF + {LB_W_TE:.2f}*te_unb = {lb_band_est:.4f}  "
          f"[{lb_low:.4f}, {lb_high:.4f}]")

    # ---- GATE EVAL ----
    print("\n" + "-" * 78)
    print(f"GATE EVALUATION (PROMOTE<{GATE_PROMOTE:.4f}; MARGINAL_BEAT<{GATE_MARGINAL:.4f})")
    print("-" * 78)
    if pooled_rae_mean_seeds < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif pooled_rae_mean_seeds < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"   pyramid mean RAE = {pooled_rae_mean_seeds:.4f}  ->  {verdict}")

    # ---- Save outputs ----
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_smiles_tta.csv"
    if verdict in ("PROMOTE", "MARGINAL_BEAT"):
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (verdict {verdict})")
    else:
        print(f"[skip] verdict FAIL -- no submission CSV written")

    summary = {
        "tag": TAG,
        "method": "smiles_tta_doRandom_5x_morgan_rdkit_K20_lgbm_residual",
        "n_tta": N_TTA,
        "tta_seeds": TTA_SEEDS,
        "K_top": K_TOP,
        "top_K_idx_in_combined": [int(j) for j in top_K_idx],
        "feat_dim_full": int(feat_dim_full),
        "rae_anchor_chemprop_aux": rae_anchor,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "rae_K20_canon": rae_canon_anchor,
        "rae_K20_TTA": rae_tta_anchor,
        "delta_TTA_vs_canon_K20_anchor": rae_tta_anchor - rae_canon_anchor,
        "mean_abs_feat_delta_TTA_vs_canon": delta_canon_vs_tta_mean,
        "resid_folds": RESID_FOLDS,
        "resid_seeds": RESID_SEEDS,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "anchors": [a[0] for a in anchors_list],
        "anchor_oof_rae_unb": indiv_rae,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(anchors_list, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_low": lb_low,
        "lb_band_high": lb_high,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if verdict in ("PROMOTE", "MARGINAL_BEAT") else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   K=20 anchor RAE (TTA)        = {rae_tta_anchor:.4f}")
    print(f"   K=20 anchor RAE (canon)      = {rae_canon_anchor:.4f}")
    print(f"   pyramid mean RAE (5 seeds)   = {pooled_rae_mean_seeds:.4f}")
    print(f"   verdict                      = {verdict}")
    print(f"   LB band                      = {lb_band_est:.4f}")
    print(f"   wall                         = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "rae_K20_canon",
        "rae_K20_TTA",
        "delta_TTA_vs_canon_K20_anchor",
        "mean_abs_feat_delta_TTA_vs_canon",
        "pooled_rae_mean_seeds",
        "verdict",
        "deploy_weights",
        "deploy_s",
        "lb_band_estimate",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
