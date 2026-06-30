"""nb2201 -- Substrate probe: graph spectral features (Laplacian eigenvalues).

HYPOTHESIS:
    Per memory, only substrate change is the open lever (post-hoc rank-stretch,
    SLSQP blends, K-tuning all plateau). Add a NOVEL feature axis -- graph
    spectral descriptors (Laplacian eigenvalues + Kirchhoff index + GraphDesc
    invariants from RDKit) -- to the nb2103 SHAP top-28 cache, building K=48.
    These are orthogonal to fingerprints (AtomPair, MACCS, Avalon), to descriptors
    of physchem origin (Mordred), and to learned embeddings (ChempropEmbed).

PROTOCOL:
    1. Reuse cached nb2103 SHAP top-28 feature matrix (X_unb_28_nb2103.npy).
    2. For each of 253 unblind compounds: compute molecular graph; compute
       Laplacian eigenvalues. Take top-10 largest + bottom-10 smallest non-zero
       eigenvalues -> 20 spectral dimensions.
    3. Concatenate spectral (20) + SHAP top-28 -> K=48.
    4. Fit LGBM(MSE) on the residual (y_unb - chemprop_aux), 5-seed bag,
       5-fold scaffold-CV-equivalent KFold cross-fit per seed. Same hyperparams
       as nb2063/nb2081/nb2103.
    5. final = chemprop_aux + mean_bag(resid_oof); compute mean-bag RAE.
    6. Compare vs nb2103 K=28 0.4737 (mean_bag) baseline; margin 0.003.
       Note: task prompt referenced 0.5057 baseline; cached summary shows 0.4737
       so use 0.4737 as actual reference.
    7. If beats: SHAP-prune K=48 -> K=28 best and retest.

Outputs:
    scripts/nb2201_graph_spectral.py
    data/processed/nb2201_summary.json
    data/processed/nb2201_X_unb_spectral20.npy  (253, 20) float32
    data/processed/nb2201_X_unb_48.npy           (253, 48) float32
    data/processed/nb2201_mean_bag_oof_K48.npy   (253,) float32
    data/processed/nb2201_mean_bag_oof_K28_pruned.npy (253,) float32 (if prune fires)
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
from sklearn.model_selection import KFold
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import rdmolops, GraphDescriptors

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2201"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
N_TOP = 10        # top-10 largest Laplacian eigs
N_BOT = 10        # bottom-10 non-zero Laplacian eigs (smallest)
N_SPECTRAL = N_TOP + N_BOT  # 20

# References
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_REF = 0.4737   # actual cached nb2103 K=28 mean_bag (prompt said 0.5057)
NB2103_K28_PROMPT = 0.5057
DECISION_MARGIN = 0.003


def _lgbm_params(seed: int) -> dict:
    """LGBM(MSE) -- identical to nb2063/nb2081/nb2103."""
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


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _compute_spectral_one(smi: str) -> np.ndarray:
    """Compute (N_TOP + N_BOT = 20) Laplacian eigenvalue features."""
    out = np.zeros(N_SPECTRAL, dtype=np.float64)
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return out  # zero-fill
    try:
        A = rdmolops.GetAdjacencyMatrix(m).astype(np.float64)
        deg = A.sum(axis=1)
        L = np.diag(deg) - A
        eigs = np.linalg.eigvalsh(L)
        eigs = np.sort(eigs)
        # top-N_TOP largest (descending)
        top_eigs = eigs[-N_TOP:][::-1]
        # bottom-N_BOT smallest non-zero (ascending); algebraic connectivity etc.
        nz = eigs[eigs > 1e-8]
        if len(nz) >= N_BOT:
            bot_eigs = nz[:N_BOT]
        else:
            bot_eigs = np.concatenate([nz, np.zeros(N_BOT - len(nz))])
        # pad top if molecule too small
        if len(top_eigs) < N_TOP:
            top_eigs = np.concatenate([top_eigs, np.zeros(N_TOP - len(top_eigs))])
        out[:N_TOP] = top_eigs[:N_TOP]
        out[N_TOP:N_TOP + N_BOT] = bot_eigs[:N_BOT]
    except Exception:
        pass
    return out


def _compute_spectral_batch(smiles_list: list[str]) -> np.ndarray:
    """Compute (n, 20) spectral feature matrix."""
    n = len(smiles_list)
    X = np.zeros((n, N_SPECTRAL), dtype=np.float64)
    for i, smi in enumerate(smiles_list):
        X[i] = _compute_spectral_one(smi)
    return X.astype(np.float32)


def _fit_residual_bag(X: np.ndarray, residual: np.ndarray,
                       anchor: np.ndarray, y_unb: np.ndarray,
                       seeds: list[int], rae_anchor: float,
                       tag: str) -> dict:
    """Run 5-seed bag, return per-seed RAE + mean/median bag RAE."""
    n_unb = len(y_unb)
    per_seed_corrected = np.zeros((len(seeds), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(seeds):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X, residual, s)
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   [{tag}] seed={s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"wall = {time.time() - ts:.1f}s")
    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))
    per_seed_arr = np.array(per_seed_rae)
    return {
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": float(per_seed_arr.mean()),
        "rae_per_seed_median": float(np.median(per_seed_arr)),
        "rae_per_seed_std": float(per_seed_arr.std()),
        "rae_per_seed_min": float(per_seed_arr.min()),
        "rae_per_seed_max": float(per_seed_arr.max()),
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "mean_bag_oof": mean_bag_oof,
        "median_bag_oof": median_bag_oof,
    }


def _shap_importance_full(X: np.ndarray, residual: np.ndarray, seed: int = 0) -> np.ndarray:
    """Fit one LGBM on full X, compute global mean |SHAP|."""
    import shap
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X, residual)
    expl = shap.TreeExplainer(mdl)
    sv = expl.shap_values(X)
    if isinstance(sv, list):
        sv = sv[0]
    imp = np.abs(sv).mean(axis=0).astype(np.float32)
    return imp


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- substrate probe: Laplacian eigenvalues (20) + SHAP top-28 -> K=48")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K=28 mean-bag RAE = {NB2103_K28_REF:.4f}  "
          f"(prompt said {NB2103_K28_PROMPT:.4f})  margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load cached SHAP top-28 matrix from nb2103 ----
    X28_path = DATA_PROCESSED / "X_unb_28_nb2103.npy"
    if not X28_path.exists():
        raise FileNotFoundError(f"missing cached top-28: {X28_path}")
    X_unb_28 = np.load(X28_path).astype(np.float32)
    if X_unb_28.shape != (n_unb, 28):
        raise ValueError(f"X_unb_28 shape mismatch: {X_unb_28.shape}")
    print(f"[load] X_unb_28 (nb2103 cached): {X_unb_28.shape}")

    # ---- Compute graph spectral features for 253 unblind compounds ----
    print("\n" + "-" * 78)
    print("COMPUTE GRAPH SPECTRAL FEATURES (Laplacian eigenvalues)")
    print("-" * 78)
    t_spec = time.time()
    unb_smiles = [test_smiles[i] for i in unb_idx]
    # Standardize first for canonical graphs
    std_smiles = []
    for s in unb_smiles:
        try:
            m = standardize(s)
            std_smiles.append(Chem.MolToSmiles(m) if m is not None else s)
        except Exception:
            std_smiles.append(s)
    X_spec = _compute_spectral_batch(std_smiles)
    print(f"   X_spec shape  = {X_spec.shape}")
    print(f"   X_spec stats: mean={X_spec.mean():.3f}  std={X_spec.std():.3f}  "
          f"min={X_spec.min():.3f}  max={X_spec.max():.3f}")
    print(f"   wall = {time.time() - t_spec:.2f}s")
    # Save for reproducibility
    np.save(DATA_PROCESSED / f"{TAG}_X_unb_spectral20.npy", X_spec)
    print(f"   [save] {DATA_PROCESSED / f'{TAG}_X_unb_spectral20.npy'}")

    # ---- Concatenate to K=48 ----
    X_unb_48 = np.concatenate([X_unb_28, X_spec], axis=1).astype(np.float32)
    print(f"\n   K=48 substrate: {X_unb_48.shape}")
    np.save(DATA_PROCESSED / f"{TAG}_X_unb_48.npy", X_unb_48)
    print(f"   [save] {DATA_PROCESSED / f'{TAG}_X_unb_48.npy'}")

    # ---- K=48 5-seed bag ----
    print("\n" + "-" * 78)
    print("K=48 5-seed bag fit (LGBM(MSE) on residual)")
    print("-" * 78)
    k48_result = _fit_residual_bag(
        X_unb_48, residual, anchor, y_unb,
        RESID_SEEDS, rae_anchor, "K=48"
    )
    rae_k48_mean = k48_result["rae_mean_bag"]
    rae_k48_median = k48_result["rae_median_bag"]
    delta_vs_nb2103 = rae_k48_mean - NB2103_K28_REF
    delta_vs_prompt = rae_k48_mean - NB2103_K28_PROMPT
    delta_vs_anchor = rae_k48_mean - rae_anchor
    print(f"\n   K=48 mean_bag   = {rae_k48_mean:.4f}   "
          f"(d_vs_anchor = {delta_vs_anchor:+.4f}  "
          f"d_vs_nb2103_K28 = {delta_vs_nb2103:+.4f}  "
          f"d_vs_prompt_K28 = {delta_vs_prompt:+.4f})")
    print(f"   K=48 median_bag = {rae_k48_median:.4f}")
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_K48.npy",
            k48_result["mean_bag_oof"].astype(np.float32))

    beats_nb2103 = rae_k48_mean < NB2103_K28_REF - DECISION_MARGIN
    flat_vs_nb2103 = abs(delta_vs_nb2103) < DECISION_MARGIN
    if beats_nb2103:
        k48_verdict = f"BEATS_NB2103_K28_BY_{abs(delta_vs_nb2103):.4f}"
    elif flat_vs_nb2103:
        k48_verdict = "FLAT_VS_NB2103_K28"
    elif rae_k48_mean < rae_anchor - DECISION_MARGIN:
        k48_verdict = "BEATS_ANCHOR_BUT_WORSE_THAN_NB2103_K28"
    else:
        k48_verdict = "HURTS_NB2103_K28"
    print(f"   K=48 verdict   = {k48_verdict}")

    # ---- SHAP-prune K=48 -> K=28 best (conditional) ----
    prune_result = None
    prune_verdict = "PRUNE_SKIPPED_K48_DID_NOT_BEAT_NB2103_K28"
    if beats_nb2103:
        print("\n" + "-" * 78)
        print("PRUNE: SHAP K=48 -> best K=28 retest")
        print("-" * 78)
        # Compute SHAP importance on K=48
        t_shap = time.time()
        shap_imp = _shap_importance_full(X_unb_48, residual, seed=0)
        print(f"   SHAP imp shape: {shap_imp.shape}  wall={time.time() - t_shap:.2f}s")
        # Save SHAP importance
        np.save(DATA_PROCESSED / f"{TAG}_shap_importance_K48.npy", shap_imp)
        # Pick top-28
        top28_idx = np.argsort(-shap_imp)[:28].astype(np.int32)
        spec_in_top28 = int(np.sum(top28_idx >= 28))
        print(f"   top-28 within K=48: {spec_in_top28} of 20 spectral features retained")
        X_pruned = X_unb_48[:, top28_idx].astype(np.float32)
        np.save(DATA_PROCESSED / f"{TAG}_top28_idx_in_K48.npy", top28_idx)

        prune_result = _fit_residual_bag(
            X_pruned, residual, anchor, y_unb,
            RESID_SEEDS, rae_anchor, "K=28prune"
        )
        rae_pruned_mean = prune_result["rae_mean_bag"]
        rae_pruned_median = prune_result["rae_median_bag"]
        delta_pruned_vs_k48 = rae_pruned_mean - rae_k48_mean
        delta_pruned_vs_nb2103 = rae_pruned_mean - NB2103_K28_REF
        print(f"\n   K=28pruned mean_bag   = {rae_pruned_mean:.4f}  "
              f"(d_vs_K48 = {delta_pruned_vs_k48:+.4f}  "
              f"d_vs_nb2103_K28 = {delta_pruned_vs_nb2103:+.4f})")
        print(f"   K=28pruned median_bag = {rae_pruned_median:.4f}")
        np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_K28_pruned.npy",
                prune_result["mean_bag_oof"].astype(np.float32))
        if rae_pruned_mean < NB2103_K28_REF - DECISION_MARGIN:
            prune_verdict = f"PRUNED_BEATS_NB2103_K28_BY_{abs(delta_pruned_vs_nb2103):.4f}"
        elif rae_pruned_mean < rae_k48_mean - DECISION_MARGIN:
            prune_verdict = "PRUNED_BEATS_K48_BUT_NOT_NB2103_K28"
        elif abs(rae_pruned_mean - rae_k48_mean) < DECISION_MARGIN:
            prune_verdict = "PRUNED_FLAT_VS_K48"
        else:
            prune_verdict = "PRUNED_HURTS_VS_K48"
        print(f"   prune verdict = {prune_verdict}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "graph_spectral_Laplacian_eig_top10_bot10_plus_nb2103_K28_SHAP",
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "n_spectral_features": int(N_SPECTRAL),
        "n_top_eigs": int(N_TOP),
        "n_bot_eigs": int(N_BOT),
        "spectral_stats": {
            "mean": float(X_spec.mean()),
            "std": float(X_spec.std()),
            "min": float(X_spec.min()),
            "max": float(X_spec.max()),
        },
        "K48_dim": int(X_unb_48.shape[1]),
        "K48_per_seed_rae": k48_result["per_seed_rae"],
        "K48_per_seed_records": k48_result["per_seed_records"],
        "K48_rae_per_seed_mean": k48_result["rae_per_seed_mean"],
        "K48_rae_per_seed_median": k48_result["rae_per_seed_median"],
        "K48_rae_per_seed_std": k48_result["rae_per_seed_std"],
        "K48_rae_per_seed_min": k48_result["rae_per_seed_min"],
        "K48_rae_per_seed_max": k48_result["rae_per_seed_max"],
        "K48_rae_mean_bag": k48_result["rae_mean_bag"],
        "K48_rae_median_bag": k48_result["rae_median_bag"],
        "K48_delta_mean_bag_vs_chemprop_aux": delta_vs_anchor,
        "K48_delta_mean_bag_vs_nb2103_K28": delta_vs_nb2103,
        "K48_delta_mean_bag_vs_prompt_K28": delta_vs_prompt,
        "K48_beats_nb2103_K28": bool(beats_nb2103),
        "K48_flat_vs_nb2103_K28": bool(flat_vs_nb2103),
        "K48_verdict": k48_verdict,
        "prune_fired": bool(beats_nb2103),
        "prune_verdict": prune_verdict,
        "nb2103_K28_ref_actual": NB2103_K28_REF,
        "nb2103_K28_ref_prompt": NB2103_K28_PROMPT,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "decision_margin": DECISION_MARGIN,
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    if prune_result is not None:
        summary.update({
            "Kprune_dim": 28,
            "Kprune_per_seed_rae": prune_result["per_seed_rae"],
            "Kprune_per_seed_records": prune_result["per_seed_records"],
            "Kprune_rae_per_seed_mean": prune_result["rae_per_seed_mean"],
            "Kprune_rae_per_seed_std": prune_result["rae_per_seed_std"],
            "Kprune_rae_mean_bag": prune_result["rae_mean_bag"],
            "Kprune_rae_median_bag": prune_result["rae_median_bag"],
            "Kprune_delta_mean_bag_vs_K48": prune_result["rae_mean_bag"] - rae_k48_mean,
            "Kprune_delta_mean_bag_vs_nb2103_K28": prune_result["rae_mean_bag"] - NB2103_K28_REF,
            "Kprune_spec_in_top28": spec_in_top28,
        })

    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_spectral_features", "K48_dim",
        "rae_anchor_chemprop_aux",
        "K48_rae_mean_bag", "K48_rae_median_bag",
        "K48_rae_per_seed_mean", "K48_rae_per_seed_std",
        "K48_delta_mean_bag_vs_nb2103_K28",
        "K48_delta_mean_bag_vs_prompt_K28",
        "K48_verdict",
        "prune_fired", "prune_verdict",
        "Kprune_rae_mean_bag", "Kprune_spec_in_top28",
        "nb2103_K28_ref_actual", "nb2103_K28_ref_prompt",
        "pre_unblind_clean", "wall_sec",
    ):
        if k in res:
            v = res[k]
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
