"""nb2460 -- TabPFN transformer for chemprop_aux residual on 117-col substrate.

CONTEXT:
    TabPFN is a transformer pre-trained on synthetic tabular datasets that can
    do zero-shot tabular regression up to 1000 samples and 100 features. Tests
    whether a foundation-model paradigm orthogonal to the LGBM/MLP/Chemprop
    family that dominates the existing pyramid can extract residual signal vs
    the nb2240 K=20 anchor (PRIMARY-1, scaffold-CV RAE 0.4682).

PROTOCOL:
    1. Substrate:
         anchor    = te_nb2240_K20.npy (513,)  -> deploy anchor
         oof_anchor= nb2240_mean_bag_oof_K20.npy (253,)
         truth     = _audit_unblind_y.npy (253,)
         features  = data/processed/pyramid/X_117_unb.npy (253, 117)
                     data/processed/pyramid/X_117_te.npy  (513, 117)
                 ^ if missing, rebuild via nb2240 helpers
         residual  = truth - oof_anchor
    2. Truncate X_117 to top-100 features by variance on UNION of unb + te.
    3. TabPFNRegressor(device='cpu', n_estimators=8) on residual under
       5-fold scaffold-CV on the 253 unblind. Fit on fold-train residuals,
       predict fold-val residuals. Add back to oof_anchor.
    4. Compute per-fold + pooled RAE vs truth.
    5. Deploy refit on full 253 residual; predict 513 -> te_nb2460.npy.
    6. Gate: mean_rae < 0.4570 -> PROMOTE; else FAIL.

Outputs:
    scripts/nb2460_tabpfn_residual.py
    data/processed/nb2460_summary.json
    data/processed/nb2460_pred_oof.npy   (253,) standalone corrected OOF
    data/processed/te_nb2460.npy         (513,) deploy
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
# Prevent TabPFN from opening a browser / blocking on stdin for license auth.
# If TABPFN_TOKEN is not pre-set, the model download will raise TabPFNLicenseError
# which we catch and report as INSTALL_FAILED in the summary.
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2460"

# ---- paths ----
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
TRUTH_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"

PYRAMID_DIR = DATA_PROCESSED / "pyramid"
X117_UNB_PATH = PYRAMID_DIR / "X_117_unb.npy"
X117_TE_PATH = PYRAMID_DIR / "X_117_te.npy"

# ---- protocol ----
N_FOLDS = 5
KF_SEED = 1001
MAX_FEATURES = 100   # TabPFN hard limit
MAX_SAMPLES = 1000   # TabPFN hard limit (253 << 1000)
N_ESTIMATORS = 8     # TabPFN ensemble size (was N_ensemble_configurations)
GATE_RAE = 0.4570


def _save_summary(payload: dict):
    out = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[save] {out}")


def _try_install_tabpfn():
    """Return TabPFNRegressor or None on failure."""
    try:
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor
    except Exception:
        pass
    print("[install] attempting `pip install tabpfn`")
    import subprocess
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "tabpfn"],
        capture_output=True, text=True, timeout=600,
    )
    print(proc.stdout[-2000:])
    print(proc.stderr[-2000:])
    if proc.returncode != 0:
        return None
    try:
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor
    except Exception as e:
        print(f"[install] post-install import failed: {e}")
        return None


def _rebuild_X117():
    """Rebuild the 117-col 5-way feature matrix on (253, 513) using nb2240 helpers."""
    print("[fallback] X_117 cache missing, rebuilding via nb2240 helpers")
    import importlib.util
    spec_path = Path(__file__).resolve().parent / "nb2240_nb2171_k20.py"
    spec = importlib.util.spec_from_file_location("_nb2240_mod", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from pxr.chem import standardize, morgan_fp_batch
    from rdkit import Chem

    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(UNB_IDX_PATH)

    # summary loads
    with open(mod.NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(mod.NB1392_SUMMARY) as f:
        sum_1392 = json.load(f)
    with open(mod.NB1484_SUMMARY) as f:
        sum_1484 = json.load(f)
    with open(mod.NB1523_SUMMARY) as f:
        sum_1523 = json.load(f)
    with open(mod.NB1524_SUMMARY) as f:
        sum_1524 = json.load(f)
    with open(mod.NB1541_SUMMARY) as f:
        sum_1541 = json.load(f)

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = mod._extract_best_K_record(sum_1523, "per_K_records", best_K_key="best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = mod._extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    X_ap_te = mod._load_npy_test(mod.ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = mod._load_npy_test(mod.MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = mod._load_mordred_test(n_test_expected=n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = mod._load_npy_test(mod.CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = mod._load_npy_test(mod.AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

    pool = mod._load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = mod._safe_inchikey(m)
        if ik is not None:
            test_inchikeys.add(ik)
    pool = pool[~pool["inchikey"].isin(test_inchikeys)].reset_index(drop=True)
    fp_pool = morgan_fp_batch(pool["std_smiles"].tolist())
    keep_pool = fp_pool.sum(axis=1) > 0
    if not keep_pool.all():
        pool = pool[keep_pool].reset_index(drop=True)
        fp_pool = fp_pool[keep_pool]
    pool_labels = pool["pec50"].to_numpy(dtype=np.float32)
    pool_median = float(np.median(pool_labels))
    std_test_smiles = [Chem.MolToSmiles(m) if m is not None else "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = mod._tanimoto_topk(fp_test, fp_pool, k=mod.KNN_K)
    pred_chembl_pec50, mean_sim = mod._knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te = np.concatenate(
        [
            X_ap_te, X_maccs_te, X_mord_te, X_emb_te, X_av_te,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te.shape[1] == 117, f"feat_dim {X_te.shape[1]} != 117"
    X_unb = X_te[unb_idx].astype(np.float32)

    # cache for next time
    PYRAMID_DIR.mkdir(parents=True, exist_ok=True)
    np.save(X117_UNB_PATH, X_unb)
    np.save(X117_TE_PATH, X_te)
    print(f"[cache] saved {X117_UNB_PATH}  +  {X117_TE_PATH}")
    return X_unb, X_te


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- TabPFN transformer on nb2240_K20 residual")
    print("=" * 78)

    # ---- install / import tabpfn ----
    TabPFNRegressor = _try_install_tabpfn()
    if TabPFNRegressor is None:
        print("[fatal] tabpfn install failed; writing FAIL summary")
        _save_summary({
            "tag": TAG,
            "verdict": "INSTALL_FAILED",
            "gate_rae": GATE_RAE,
            "wall_sec": round(time.time() - t0, 2),
        })
        return {"verdict": "INSTALL_FAILED"}

    # ---- load substrate ----
    anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)
    y_unb = np.load(TRUTH_PATH).astype(np.float64)
    unb_idx = np.load(UNB_IDX_PATH)
    n_unb = len(y_unb)
    n_test = len(anchor_te)
    assert anchor_oof.shape == (n_unb,), f"anchor_oof shape {anchor_oof.shape}"
    print(f"[load] anchor_te {anchor_te.shape}  anchor_oof {anchor_oof.shape}  y_unb {y_unb.shape}")

    rae_anchor_oof = float(rae(y_unb, anchor_oof))
    rae_anchor_te_unb = float(rae(y_unb, anchor_te[unb_idx]))
    print(f"[anchor] nb2240_K20 oof RAE         = {rae_anchor_oof:.4f}")
    print(f"[anchor] nb2240_K20 te[unb_idx] RAE = {rae_anchor_te_unb:.4f}")

    residual = y_unb - anchor_oof
    print(f"[residual] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- load or rebuild X_117 ----
    if X117_UNB_PATH.exists() and X117_TE_PATH.exists():
        X_unb_117 = np.load(X117_UNB_PATH).astype(np.float32)
        X_te_117 = np.load(X117_TE_PATH).astype(np.float32)
        print(f"[load] cached X_117 unb={X_unb_117.shape}  te={X_te_117.shape}")
    else:
        try:
            X_unb_117, X_te_117 = _rebuild_X117()
        except Exception as e:
            print(f"[fatal] X_117 rebuild failed: {e}")
            _save_summary({
                "tag": TAG,
                "verdict": "FEATURE_BUILD_FAILED",
                "error": str(e),
                "gate_rae": GATE_RAE,
                "wall_sec": round(time.time() - t0, 2),
            })
            return {"verdict": "FEATURE_BUILD_FAILED", "error": str(e)}

    assert X_unb_117.shape == (n_unb, 117), f"unb {X_unb_117.shape}"
    assert X_te_117.shape == (n_test, 117), f"te {X_te_117.shape}"

    # ---- truncate to top-MAX_FEATURES by variance on union (unb+te) ----
    X_union = np.concatenate([X_unb_117, X_te_117], axis=0)
    var_full = X_union.var(axis=0)
    k = min(MAX_FEATURES, X_unb_117.shape[1])
    top_var_idx = np.argsort(-var_full)[:k]
    X_unb = X_unb_117[:, top_var_idx].astype(np.float32)
    X_te = X_te_117[:, top_var_idx].astype(np.float32)
    print(f"[feat] truncated to top-{k} by variance: X_unb {X_unb.shape}  X_te {X_te.shape}")
    print(f"[feat] variance range: max={var_full[top_var_idx[0]]:.3f} min_kept={var_full[top_var_idx[-1]]:.3f}")

    # ---- scaffolds ----
    te = load_test()
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- preflight: instantiate + try a tiny fit to surface model-download / license gate ----
    try:
        _probe = TabPFNRegressor(
            device="cpu",
            n_estimators=N_ESTIMATORS,
            random_state=KF_SEED,
            ignore_pretraining_limits=True,
        )
        _probe.fit(X_unb[:4], residual[:4].astype(np.float32))
    except Exception as e:
        msg = str(e)
        # license / browser-auth / network failure -> treat as INSTALL_FAILED
        print(f"[fatal] TabPFN preflight failed (likely model-weights license gate): {e!r}")
        _save_summary({
            "tag": TAG,
            "verdict": "INSTALL_FAILED",
            "reason": "tabpfn_license_or_weights_download_blocked",
            "error": msg[:500],
            "gate_rae": GATE_RAE,
            "tabpfn_version": _safe_tabpfn_version(),
            "wall_sec": round(time.time() - t0, 2),
        })
        return {"verdict": "INSTALL_FAILED", "error": msg[:500]}

    # ---- 5-fold scaffold CV: fit TabPFN on fold-train residual, predict fold-val ----
    print("\n" + "-" * 78)
    print(f"5-FOLD SCAFFOLD CV  TabPFNRegressor(device=cpu, n_estimators={N_ESTIMATORS})")
    print("-" * 78)
    splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                    shuffle=True, seed=KF_SEED)
    oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold = []
    for f_idx, (tr_loc, va_loc) in enumerate(splits):
        ts = time.time()
        model = TabPFNRegressor(
            device="cpu",
            n_estimators=N_ESTIMATORS,
            random_state=KF_SEED + f_idx,
            ignore_pretraining_limits=True,
        )
        model.fit(X_unb[tr_loc], residual[tr_loc].astype(np.float32))
        pred_va = model.predict(X_unb[va_loc])
        oof_resid[va_loc] = pred_va
        corrected_va = anchor_oof[va_loc] + pred_va
        fold_rae = float(rae(y_unb[va_loc], corrected_va))
        per_fold.append({
            "fold": f_idx,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "rae": fold_rae,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   fold={f_idx}  n_tr={len(tr_loc):3d} n_va={len(va_loc):3d}  "
              f"fold_RAE={fold_rae:.4f}  wall={time.time()-ts:.1f}s")

    corrected_oof = anchor_oof + oof_resid
    rae_corrected = float(rae(y_unb, corrected_oof))
    fold_raes = np.array([r["rae"] for r in per_fold])
    mean_rae = float(fold_raes.mean())
    std_rae = float(fold_raes.std())
    print(f"\n[cv] pooled corrected RAE = {rae_corrected:.4f}")
    print(f"[cv] mean fold-RAE         = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"[cv] delta vs anchor       = {rae_corrected - rae_anchor_oof:+.4f}")

    # ---- gate ----
    gate_pass = mean_rae < GATE_RAE
    verdict = "PROMOTE" if gate_pass else "FAIL"
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   gate_threshold       = {GATE_RAE:.4f}")
    print(f"   mean fold-RAE        = {mean_rae:.4f}")
    print(f"   pooled corrected RAE = {rae_corrected:.4f}")
    print(f"   verdict              = {verdict}")

    # ---- deploy: refit TabPFN on full 253 residual; predict 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit TabPFN on all 253; predict 513)")
    print("-" * 78)
    model_deploy = TabPFNRegressor(
        device="cpu",
        n_estimators=N_ESTIMATORS,
        random_state=KF_SEED,
        ignore_pretraining_limits=True,
    )
    model_deploy.fit(X_unb, residual.astype(np.float32))
    te_resid = model_deploy.predict(X_te).astype(np.float64)
    te_deploy = (anchor_te + te_resid).astype(np.float32)
    te_deploy = np.clip(te_deploy, 3.0, 9.0).astype(np.float32)
    te_unb_in = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"   te_deploy mean={te_deploy.mean():.3f}  std={te_deploy.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")

    # ---- save ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, corrected_oof.astype(np.float32))
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "tabpfn_transformer_on_nb2240_K20_residual",
        "anchor": "nb2240_K20",
        "rae_anchor_oof": rae_anchor_oof,
        "rae_anchor_te_unb": rae_anchor_te_unb,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "tabpfn_n_estimators": N_ESTIMATORS,
        "tabpfn_device": "cpu",
        "tabpfn_version": _safe_tabpfn_version(),
        "feat_dim_full": 117,
        "feat_dim_used": int(k),
        "feat_truncation": "top_by_variance_on_union",
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "per_fold": per_fold,
        "rae_corrected_pooled": rae_corrected,
        "mean_fold_rae": mean_rae,
        "std_fold_rae": std_rae,
        "delta_vs_anchor": rae_corrected - rae_anchor_oof,
        "gate_rae": GATE_RAE,
        "gate_pass": bool(gate_pass),
        "verdict": verdict,
        "te_unb_in_sample_rae": te_unb_in,
        "deploy_te_mean": float(te_deploy.mean()),
        "deploy_te_std": float(te_deploy.std()),
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    _save_summary(summary)

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor (nb2240_K20) RAE        = {rae_anchor_oof:.4f}")
    print(f"   TabPFN corrected pooled RAE    = {rae_corrected:.4f}")
    print(f"   mean fold-RAE                  = {mean_rae:.4f}")
    print(f"   delta vs anchor                = {rae_corrected - rae_anchor_oof:+.4f}")
    print(f"   gate ({GATE_RAE:.4f})              = {verdict}")
    print(f"   wall                           = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


def _safe_tabpfn_version():
    try:
        import tabpfn
        return str(getattr(tabpfn, "__version__", "unknown"))
    except Exception:
        return "unknown"


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "verdict",
        "mean_fold_rae",
        "rae_corrected_pooled",
        "delta_vs_anchor",
        "gate_pass",
        "te_unb_in_sample_rae",
    ):
        print(f"  {k}: {res.get(k)}")
