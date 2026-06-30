"""nb1630 -- Deploy nb1622 blend (0.55*nb1561 + 0.45*nb1612) to 513-row CSV.

PROTOCOL
    1. Anchor = chemprop_aux on 513 (PRE-unblind).
    2. Build 6 deploy residuals on 513 (one per family):
         - top-25 AtomPair (SHAP-pruned on 253 unblind), 5-seed LGBM-Huber bag
           fit on ALL 253 unblind; predict residual on 513; mean across seeds.
         - top-20 MACCS, top-20 Mordred, top-20 ChempropEmbed, top-30 Avalon,
           top-50 ChemBERTa: same recipe.
         Each per-family feature matrix = top-K family columns + pred_chembl_513
         + sim_chembl_513 (matches nb1612 / nb1623 recipe).
    3. te_nb1612_deploy = chemprop_aux_513 + sum_k(w_k * resid_k_513), where
       w_k = nb1612 SLSQP mean-over-folds weights:
         AtomPair = 0.0969, MACCS = 0.0000, Mordred = 0.2719,
         ChempropEmbed = 0.3791, Avalon = 0.0752, ChemBERTa = 0.1768.
    4. te_nb1630 = 0.55 * te_nb1570_mean + 0.45 * te_nb1612_deploy.
    5. Save te_nb1612_deploy.npy, te_nb1630.npy, submissions CSV.

REPORT: te mean/std/min/max, in_RAE on 253 unblind, CSV path.

HONEST LB ANCHORS
    cross-fit blend (nb1622): 0.5136   grid w=0.55: 0.5130
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
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1630"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Mean SLSQP weights from nb1612 (5-fold cross-fit, mean over folds).
# Family order: AtomPair, MACCS, Mordred, ChempropEmbed, Avalon, ChemBERTa.
NB1612_SLSQP_W = {
    "AtomPair":      0.09687802305012036,
    "MACCS":         9.673810836372207e-17,
    "Mordred":       0.27194502557348726,
    "ChempropEmbed": 0.37912058950494737,
    "Avalon":        0.07524461637088213,
    "ChemBERTa":     0.17681174550056286,
}

# nb1612 family K-tuned indices.
TOP_K = {
    "AtomPair":      25,
    "MACCS":         20,
    "Mordred":       20,
    "ChempropEmbed": 20,
    "Avalon":        30,
    "ChemBERTa":     50,
}
FAMILIES = ["AtomPair", "MACCS", "Mordred", "ChempropEmbed", "Avalon", "ChemBERTa"]

RESID_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
CHEMBERTA_TE_PATH = DATA_PROCESSED / "chemberta_test_emb.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

TE_NB1570_MEAN_PATH = DATA_PROCESSED / "te_nb1570_mean.npy"

W_NB1561 = 0.55
W_NB1612 = 0.45
HONEST_LB_CROSSFIT = 0.5136
HONEST_LB_GRID = 0.5130

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X = np.load(mte_p).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"Mordred shape mismatch: {X.shape}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_family_te(family: str, n_test: int) -> np.ndarray:
    if family == "AtomPair":
        return np.load(ATOMPAIR_TE_PATH).astype(np.float32)
    if family == "MACCS":
        return np.load(MACCS_TE_PATH).astype(np.float32)
    if family == "Mordred":
        return _load_mordred_test(n_test)
    if family == "ChempropEmbed":
        X = np.load(CHEMPROP_EMBED_TE_PATH).astype(np.float32)
        return np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    if family == "Avalon":
        return np.load(AVALON_TE_PATH).astype(np.float32)
    if family == "ChemBERTa":
        X = np.load(CHEMBERTA_TE_PATH).astype(np.float32)
        return np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    raise ValueError(f"unknown family: {family}")


def _compute_shap_importance(X: np.ndarray, residual: np.ndarray, seed: int = 0):
    mdl = LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X, residual)
    try:
        import shap
        explainer = shap.TreeExplainer(mdl)
        sv = explainer.shap_values(X)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 0]
        imp = np.abs(sv).mean(axis=0)
        return imp.astype(np.float64), "shap_tree_explainer"
    except Exception:
        imp = mdl.booster_.feature_importance(importance_type="gain")
        return imp.astype(np.float64), "lgbm_gain_fallback"


def _build_family_deploy_residual(
    family: str,
    X_fam_te: np.ndarray,
    pred_chembl_513: np.ndarray,
    sim_chembl_513: np.ndarray,
    residual_unb: np.ndarray,
    unb_idx: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, dict]:
    """Compute SHAP top-K on 253, train 5-seed LGBM-Huber bag fit on ALL 253,
    predict residual on 513, return mean across seeds.

    Returns
    -------
    mean_resid_513 : (513,) float64
    info : dict (top_idx, shap_src, per_seed_train_corr)
    """
    n_fam = int(X_fam_te.shape[1])

    # 253 slice
    X_fam_unb = X_fam_te[unb_idx].astype(np.float32)
    pred_chembl_unb = pred_chembl_513[unb_idx].astype(np.float32)
    sim_chembl_unb = sim_chembl_513[unb_idx].astype(np.float32)

    X_full_unb = np.concatenate(
        [X_fam_unb,
         pred_chembl_unb.reshape(-1, 1),
         sim_chembl_unb.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)

    imp_full, imp_src = _compute_shap_importance(X_full_unb, residual_unb, seed=0)
    fam_imp = imp_full[:n_fam]
    top_k_eff = min(top_k, n_fam)
    top_order = np.argsort(-fam_imp)
    top_idx = top_order[:top_k_eff].astype(int)

    # PRUNED matrices on 253 (train) and 513 (deploy)
    X_pruned_unb = np.concatenate(
        [X_fam_unb[:, top_idx],
         pred_chembl_unb.reshape(-1, 1),
         sim_chembl_unb.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)
    X_pruned_513 = np.concatenate(
        [X_fam_te[:, top_idx].astype(np.float32),
         pred_chembl_513.reshape(-1, 1),
         sim_chembl_513.reshape(-1, 1)],
        axis=1,
    ).astype(np.float32)

    n_test = X_pruned_513.shape[0]
    per_seed_513 = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_train_rae = []
    for i, s in enumerate(RESID_SEEDS):
        mdl = LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_pruned_unb, residual_unb)
        per_seed_513[i] = mdl.predict(X_pruned_513).astype(np.float64)
        # In-sample train residual prediction for diagnostic
        in_sample_pred = mdl.predict(X_pruned_unb)
        # RAE of corrected (anchor + resid) is computed outside; here just MAE
        mae_train = float(np.mean(np.abs(residual_unb - in_sample_pred)))
        per_seed_train_rae.append(mae_train)

    mean_resid_513 = per_seed_513.mean(axis=0)
    info = {
        "family": family,
        "n_fam_bits": n_fam,
        "top_k_eff": int(top_k_eff),
        "shap_source": imp_src,
        "feat_dim_pruned": int(X_pruned_unb.shape[1]),
        "per_seed_train_mae": per_seed_train_rae,
        "resid_513_mean": float(mean_resid_513.mean()),
        "resid_513_std": float(mean_resid_513.std()),
    }
    return mean_resid_513, info


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1622 blend (0.55*nb1561 + 0.45*nb1612) to 513 CSV")
    print(f"          honest LB cross-fit = {HONEST_LB_CROSSFIT}  "
          f"grid w=0.55 = {HONEST_LB_GRID}")
    print(f"          nb1612 SLSQP weights = "
          f"{ {k: round(v, 4) for k, v in NB1612_SLSQP_W.items()} }")
    print("=" * 78)

    # ---- Test metadata ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    else:
        cand = [c for c in te.columns if "name" in c.lower()]
        if not cand:
            raise KeyError(f"No Molecule Name column ({te.columns.tolist()})")
        mol_names = te[cand[0]].astype(str).tolist()

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Anchor on 513 (chemprop_aux) ----
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor shape mismatch {te_anchor_513.shape}")
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual_unb = y_unb - anchor_unb
    print(f"[anchor] chemprop_aux in_RAE(unb) = {rae_anchor:.4f}")
    print(f"[resid] mean={residual_unb.mean():+.4f}  "
          f"std={residual_unb.std():.4f}")

    # ---- nb1561 deploy (precomputed by nb1570) ----
    if not TE_NB1570_MEAN_PATH.exists():
        raise FileNotFoundError(f"missing {TE_NB1570_MEAN_PATH}")
    te_nb1561 = np.load(TE_NB1570_MEAN_PATH).astype(np.float64)
    if te_nb1561.shape[0] != n_test:
        raise ValueError(f"te_nb1570_mean shape mismatch {te_nb1561.shape}")
    rae_nb1561 = float(rae(y_unb, te_nb1561[unb_idx]))
    print(f"[nb1561] te_nb1570_mean in_RAE(unb) = {rae_nb1561:.4f}")

    # ---- ChEMBL kNN features on 513 (cached) ----
    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)
    if pred_chembl_513.shape[0] != n_test or sim_chembl_513.shape[0] != n_test:
        raise ValueError("chembl cache shape mismatch")
    print(f"[chembl] pred_chembl_513 mean={pred_chembl_513.mean():.3f}  "
          f"std={pred_chembl_513.std():.3f}")
    print(f"[chembl] sim_chembl_513  mean={sim_chembl_513.mean():.3f}  "
          f"std={sim_chembl_513.std():.3f}")

    # ---- Build 6 deploy residuals (513,) per family ----
    print("\n" + "=" * 78)
    print("6 PER-FAMILY DEPLOY RESIDUALS (513,) -- 5-seed LGBM-Huber bag on ALL 253")
    print("=" * 78)
    family_resid_513 = {}
    family_info = []
    for family in FAMILIES:
        t_fam = time.time()
        X_fam_te = _load_family_te(family, n_test)
        if X_fam_te.shape[0] != n_test:
            raise ValueError(f"{family} shape {X_fam_te.shape}")
        resid_513, info = _build_family_deploy_residual(
            family=family,
            X_fam_te=X_fam_te,
            pred_chembl_513=pred_chembl_513,
            sim_chembl_513=sim_chembl_513,
            residual_unb=residual_unb,
            unb_idx=unb_idx,
            top_k=TOP_K[family],
        )
        family_resid_513[family] = resid_513
        info["wall_sec"] = round(time.time() - t_fam, 2)
        family_info.append(info)

        # Diagnostic: per-family in_RAE if it were applied alone
        corr_unb = anchor_unb + resid_513[unb_idx]
        in_rae_fam = float(rae(y_unb, corr_unb))
        info["in_rae_unb_alone"] = in_rae_fam
        print(f"   {family:<14s} (K={info['top_k_eff']:>3})  "
              f"resid_513 mean={info['resid_513_mean']:+.4f}  "
              f"std={info['resid_513_std']:.4f}  "
              f"in_RAE_alone={in_rae_fam:.4f}  "
              f"wall={info['wall_sec']}s")

    # ---- te_nb1612_deploy = chemprop_aux + sum(w_k * resid_k) ----
    w_sum = sum(NB1612_SLSQP_W[f] for f in FAMILIES)
    print(f"\n[weights] sum of nb1612 SLSQP weights = {w_sum:.6f}")
    combined_resid_513 = np.zeros(n_test, dtype=np.float64)
    for family in FAMILIES:
        combined_resid_513 += NB1612_SLSQP_W[family] * family_resid_513[family]
    te_nb1612_deploy = te_anchor_513 + combined_resid_513

    rae_nb1612_deploy = float(rae(y_unb, te_nb1612_deploy[unb_idx]))
    print(f"[nb1612_deploy] in_RAE(unb) = {rae_nb1612_deploy:.4f}")
    print(f"[nb1612_deploy] mean={te_nb1612_deploy.mean():.4f}  "
          f"std={te_nb1612_deploy.std():.4f}  "
          f"min={te_nb1612_deploy.min():.4f}  "
          f"max={te_nb1612_deploy.max():.4f}")

    # ---- te_nb1630 = 0.55 * te_nb1561 + 0.45 * te_nb1612_deploy ----
    te_nb1630 = W_NB1561 * te_nb1561 + W_NB1612 * te_nb1612_deploy
    rae_nb1630 = float(rae(y_unb, te_nb1630[unb_idx]))

    print("\n" + "=" * 78)
    print("FINAL BLEND  te_nb1630 = 0.55*te_nb1570_mean + 0.45*te_nb1612_deploy")
    print("=" * 78)
    print(f"   te_nb1630   mean={te_nb1630.mean():.4f}  "
          f"std={te_nb1630.std():.4f}  "
          f"min={te_nb1630.min():.4f}  "
          f"max={te_nb1630.max():.4f}")
    print(f"   in_RAE(unb) = {rae_nb1630:.4f}")
    print(f"   honest LB anchor cross-fit = {HONEST_LB_CROSSFIT}  "
          f"grid w=0.55 = {HONEST_LB_GRID}")
    print(f"   d_vs_nb1561        = {rae_nb1630 - rae_nb1561:+.4f}")
    print(f"   d_vs_nb1612_deploy = {rae_nb1630 - rae_nb1612_deploy:+.4f}")

    # ---- Save NPYs ----
    te_nb1612_path = DATA_PROCESSED / "te_nb1612_deploy.npy"
    te_nb1630_path = DATA_PROCESSED / "te_nb1630.npy"
    np.save(te_nb1612_path, te_nb1612_deploy.astype(np.float32))
    np.save(te_nb1630_path, te_nb1630.astype(np.float32))
    print(f"\n[save] {te_nb1612_path}")
    print(f"[save] {te_nb1630_path}")

    # ---- Save CSV ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1622.csv"
    df = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1630.astype(np.float64),
    })
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}  rows={len(df)}  cols={list(df.columns)}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "rae_anchor": rae_anchor,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "blend_w_nb1561": W_NB1561,
        "blend_w_nb1612": W_NB1612,
        "honest_lb_crossfit": HONEST_LB_CROSSFIT,
        "honest_lb_grid_w_055": HONEST_LB_GRID,
        "families_order": FAMILIES,
        "top_k_config": TOP_K,
        "nb1612_slsqp_weights_used": NB1612_SLSQP_W,
        "nb1612_slsqp_weights_sum": float(w_sum),
        "resid_seeds": RESID_SEEDS,
        "family_info": family_info,
        "rae_nb1561_te1570_mean": rae_nb1561,
        "rae_nb1612_deploy": rae_nb1612_deploy,
        "rae_nb1630": rae_nb1630,
        "te_nb1612_deploy_stats": {
            "mean": float(te_nb1612_deploy.mean()),
            "std": float(te_nb1612_deploy.std()),
            "min": float(te_nb1612_deploy.min()),
            "max": float(te_nb1612_deploy.max()),
        },
        "te_nb1630_stats": {
            "mean": float(te_nb1630.mean()),
            "std": float(te_nb1630.std()),
            "min": float(te_nb1630.min()),
            "max": float(te_nb1630.max()),
        },
        "te_nb1612_deploy_path": str(te_nb1612_path),
        "te_nb1630_path": str(te_nb1630_path),
        "csv_path": str(csv_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_test", "n_unb",
        "rae_anchor",
        "rae_nb1561_te1570_mean",
        "rae_nb1612_deploy",
        "rae_nb1630",
        "te_nb1612_deploy_stats",
        "te_nb1630_stats",
        "honest_lb_crossfit",
        "honest_lb_grid_w_055",
        "te_nb1612_deploy_path",
        "te_nb1630_path",
        "csv_path",
    ):
        print(f"  {k}: {res.get(k)}")
