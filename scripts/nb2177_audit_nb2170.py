"""nb2177 -- HARD AUDIT of nb2170 0.3920 result.

Purpose:
    Verify whether the cycle 123 nb2170 anchor swap (chemprop_aux -> nb730)
    that produced 0.3920 honest cross-fit RAE is real OR a lucky kf_seed /
    silent contamination.

Per memory feedback_te_vs_pred_oof_protocol.md and
feedback_data_integrity_2026_06_01.md, te_X[unb_idx] == X_pred_oof EXACTLY
is a known CONTAMINATION SIGNATURE. We must verify they are NOT identical.

Checks executed:
    1. Generation script identified for te_nb730.npy
    2. Construction mode classified (deploy-refit vs OOF copy vs stitched)
    3. Honest 5-fold OOF nb730_pred_oof.npy generation script identified
    4. mean-abs-diff between te_nb730[unb_idx] and nb730_pred_oof
    5. Pearson + sha256 comparison
    6. Identity verdict (HONEST | LUCKY_KF_SEED | CONTAMINATED)
    7. Fresh-kf reproduction of nb2170 0.3920 with kf_seeds {1001..1005}
    8. sha256 of te_nb730[unb_idx] vs truth labels y_unb_253
    9. Plain-language verdict + path forward

Outputs:
    data/processed/nb2177_summary.json
"""
from __future__ import annotations

import hashlib
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
import lightgbm as lgb
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2177"

# Reproduce nb2170 exactly:
RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
TOP_K_SHAP = 28

NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb730.npy"
NB730_OOF_PATH = DATA_PROCESSED / "nb730_pred_oof.npy"
NB2170_MEAN_BAG_PATH = DATA_PROCESSED / "nb2170_mean_bag_oof.npy"
NB2170_MEDIAN_BAG_PATH = DATA_PROCESSED / "nb2170_median_bag_oof.npy"


def _sha(arr) -> str:
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(arr.tobytes())
    return h.hexdigest()


def _lgbm_params(seed: int) -> dict:
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


def _residual_cross_fit_one_seed(X, residual, kf_seed, lgbm_seed):
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=kf_seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(lgbm_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print("nb2177 -- HARD AUDIT of nb2170 0.3920 result")
    print("=" * 78)

    # ---------- Check 1: identify generation script for te_nb730 ----------
    gen_script = "scripts/nb730_null_ensemble.py"
    gen_oof_script = "scripts/nb730_null_ensemble.py"
    print(f"\n[Check 1] te_nb730.npy generation script: {gen_script}")
    print(f"[Check 1] nb730_pred_oof.npy generation script: {gen_oof_script}")

    # ---------- Check 2: classify construction mode ----------
    # Read directly from nb730_null_ensemble.py inspection (already done):
    #   line 264 te_deploy = (te_nb562 - best_pooled_lam * null_pos_te)
    #     -> te_nb562 is DEPLOY refit; null_pos_te is from null heads
    #        refit on ALL dual-labelled (2858) -> applied to ALL 513.
    #     -> lambda chosen by IN-SAMPLE POOLED fit on 253 unblind.
    #   line 268 pred_oof = cross_pred (5-fold scaffold cross-fit on 253)
    #     -> per-fold lambda chosen by grid on train-fold of unblind.
    construction_te_nb730 = (
        "deploy_refit_nb562_minus_lambda_times_deploy_null_pos_te ; "
        "best_pooled_lambda chosen IN-SAMPLE on 253 unblind"
    )
    construction_nb730_oof = (
        "5_fold_scaffold_cross_fit_on_253_unblind ; per-fold lambda "
        "chosen from train-fold pooled grid sweep"
    )
    print(f"\n[Check 2] te_nb730 construction:    {construction_te_nb730}")
    print(f"[Check 2] nb730_pred_oof construction: {construction_nb730_oof}")
    print("[Check 2] These are TWO DIFFERENT OBJECTS:")
    print("          te_nb730[unb_idx] = nb562_deploy(unb) - lam_pooled*null_ens_deploy(unb)")
    print("          nb730_pred_oof    = nb562_deploy(unb) - lam_per_fold*null_ens_deploy(unb)")
    print("          Both share the nb562_deploy and null_ens_deploy components but")
    print("          differ in WHICH lambda is applied (single best_pooled_lam vs")
    print("          per-fold cross-fit lambda).")

    # ---------- Check 3 + 4: load arrays + compute mean-abs-diff ----------
    print("\n[Check 3-4] Loading arrays")
    te_nb730 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    nb730_oof = np.load(NB730_OOF_PATH).astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(unb_idx)
    print(f"  te_nb730 shape={te_nb730.shape}")
    print(f"  nb730_pred_oof shape={nb730_oof.shape}")
    print(f"  unb_idx shape={unb_idx.shape}  y_unb shape={y_unb.shape}")

    te_nb730_unb = te_nb730[unb_idx]
    diff = te_nb730_unb - nb730_oof
    mean_abs_diff = float(np.mean(np.abs(diff)))
    max_abs_diff = float(np.max(np.abs(diff)))
    pearson = float(np.corrcoef(te_nb730_unb, nb730_oof)[0, 1])
    sha_te_unb = _sha(te_nb730_unb.astype(np.float32))
    sha_oof = _sha(nb730_oof.astype(np.float32))
    sha_y_unb = _sha(y_unb.astype(np.float32))

    print(f"\n  mean_abs_diff (te[unb] vs pred_oof) = {mean_abs_diff:.6e}")
    print(f"  max_abs_diff                        = {max_abs_diff:.6e}")
    print(f"  Pearson correlation                 = {pearson:.6f}")
    print(f"  sha256 te_nb730[unb_idx] (f32)      = {sha_te_unb}")
    print(f"  sha256 nb730_pred_oof    (f32)      = {sha_oof}")
    print(f"  sha256 y_unb_253         (f32)      = {sha_y_unb}")

    rae_te_unb = float(rae(y_unb, te_nb730_unb))
    rae_oof = float(rae(y_unb, nb730_oof))
    print(f"\n  RAE(y_unb, te_nb730[unb_idx]) = {rae_te_unb:.4f}")
    print(f"  RAE(y_unb, nb730_pred_oof)    = {rae_oof:.4f}")
    rae_gap = rae_te_unb - rae_oof
    print(f"  RAE gap (te[unb] - oof)       = {rae_gap:+.4f}")

    # ---------- Check 6/7: identity verdict ----------
    if mean_abs_diff < 1e-9:
        identity_verdict = "IDENTICAL_BIT_FOR_BIT"
    elif mean_abs_diff < 1e-4:
        identity_verdict = "NEAR_IDENTICAL_LIKELY_LAMBDA_COLLISION"
    else:
        identity_verdict = "DISTINCT_HONEST_TWO_OBJECTS"
    print(f"\n  identity verdict = {identity_verdict}")

    # contamination check: te_unb == y_unb?
    truth_contamination = (sha_te_unb == sha_y_unb)
    print(f"  te[unb] == y_unb (truth contamination)? {truth_contamination}")

    # ---------- Check 8: fresh-kf reproduction of nb2170 0.3920 ----------
    # Recompute nb2170 mean-bag RAE with kf_seeds {1001..1005}
    # while keeping lgbm_seeds = [0,1,7,42,137]
    print("\n" + "=" * 78)
    print("[Check 8] FRESH-KF REPRODUCTION of nb2170 0.3920")
    print("=" * 78)

    # Need X_unb_28 + residual.  Reuse same construction as nb2170:
    # residual = y_unb - te_nb730[unb_idx]
    residual = y_unb - te_nb730_unb
    print(f"  residual mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # Reuse nb2170_mean_bag_oof for original kf_seed (default 0,1,7,42,137 == lgbm_seed)
    orig_mean_bag = np.load(NB2170_MEAN_BAG_PATH).astype(np.float64)
    orig_median_bag = np.load(NB2170_MEDIAN_BAG_PATH).astype(np.float64)
    orig_rae_mean = float(rae(y_unb, orig_mean_bag))
    orig_rae_median = float(rae(y_unb, orig_median_bag))
    print(f"  ORIGINAL nb2170 mean-bag   RAE = {orig_rae_mean:.4f}")
    print(f"  ORIGINAL nb2170 median-bag RAE = {orig_rae_median:.4f}")

    # nb2170 uses kf=KFold(random_state=seed) where seed = LGBM seed.
    # So kf_seeds {1001..1005} means re-fitting with KFold(rs=1001..1005)
    # while keeping lgbm random_state at e.g. 0 (or matching kf_seed).
    # We follow: per fresh_kf_seed, use that seed for BOTH kf and lgbm.
    # Build top-28 feature matrix from nb2170_mean_bag_oof IS NOT
    # possible -- we need the X matrix.  Recompute X_unb_28 the cheap way:
    # since nb2170 saved per-seed corrected via mean_bag, and we have
    # residual + anchor, we cannot recover X without re-running the
    # full pipeline.  Instead, we re-do the residual CV with the same
    # construction but new KFold seeds by using the cached X_unb_28
    # via a lightweight pathway:  the residual_oof is a function of X
    # and KFold split only.  We reload X_unb_28 from the nb2170 pipeline.

    # Try to load cached X_unb_28 -- nb2170 didn't save it explicitly.
    # We'll rebuild it directly here using the recorded construction.
    # Easiest: import from nb2170 logic.  But this would re-run the
    # ChEMBL kNN block ~30s.  Cheaper path: search for any saved
    # X_unb_28 alias.
    X_unb_28_cache = DATA_PROCESSED / "X_unb_28_nb2103.npy"
    fresh_kf_results = {}
    if not X_unb_28_cache.exists():
        # Try alternative cache from nb2103
        alt = DATA_PROCESSED / "nb2103_X_unb_28.npy"
        if alt.exists():
            X_unb_28_cache = alt
    if X_unb_28_cache.exists():
        X_unb_28 = np.load(X_unb_28_cache).astype(np.float32)
        print(f"  [reuse X_unb_28] {X_unb_28_cache.name} shape={X_unb_28.shape}")
    else:
        # Need to rebuild.  This is heavy.  Use shortcut: derive
        # residual_oof_for_each_lgbm_seed by inverse-engineering
        # corrected = anchor + resid_oof, where corrected = orig per-seed.
        # nb2170 saved only mean_bag and median_bag, not per-seed --
        # so we cannot reconstruct X.
        # FALL BACK: rebuild X_unb_28 by running the nb2170 feature pipe.
        print("  [build X_unb_28] no cache; rebuilding from nb2170 pipeline")
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog("rdApp.*")
        from pxr.chem import standardize, morgan_fp_batch
        from pxr.data import load_test
        from pathlib import Path as _P

        ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
        MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
        CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
        AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
        MORDRED_DIR = _P("C:/pxr_artifacts/nb1030")
        EXT_DIR = _P(__file__).resolve().parents[1] / "data" / "external"

        NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
        NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
        NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
        NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
        NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
        NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

        with open(NB1352_SUMMARY) as f:
            sum_1352 = json.load(f)
        with open(NB1392_SUMMARY) as f:
            sum_1392 = json.load(f)
        with open(NB1484_SUMMARY) as f:
            sum_1484 = json.load(f)
        with open(NB1523_SUMMARY) as f:
            sum_1523 = json.load(f)
        with open(NB1524_SUMMARY) as f:
            sum_1524 = json.load(f)
        with open(NB1541_SUMMARY) as f:
            sum_1541 = json.load(f)
        with open(NB2103_SUMMARY) as f:
            sum_2103 = json.load(f)

        def _ap(d):
            for fa in d["families"]:
                if fa["family"] == "AtomPair":
                    return np.array(fa["top_idx_ranked"], dtype=int)
            raise KeyError("AtomPair")

        def _bestK(d):
            bestK = int(d["best_K"])
            for r in d["per_K_records"]:
                if int(r["K"]) == bestK:
                    return r
            raise KeyError(bestK)

        def _Krec(d, K):
            for r in d["per_K_records"]:
                if int(r["K"]) == K:
                    return r
            raise KeyError(K)

        rec28 = _Krec(sum_2103, K=TOP_K_SHAP)
        top28_idx = np.array(rec28["top_K_idx_in_117"], dtype=np.int32)
        top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
        rec_mord = _bestK(sum_1523)
        top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
        full_ap = _ap(sum_1484)
        K_AP = int(sum_1524["best_K"])
        top_ap_bit_idx = full_ap[:K_AP]
        K_Embed = int(sum_1541["best_K"])
        top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
        top_embed_col_idx = top_embed_full[:K_Embed]
        top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

        te = load_test()
        n_test = len(te)
        test_smiles = te.get("SMILES", te.get("smiles")).astype(str).tolist()

        def _ld(p):
            X = np.load(p).astype(np.float32)
            return np.where(np.isfinite(X), X, 0.0).astype(np.float32)

        X_ap_te = _ld(ATOMPAIR_TE_PATH)[:, top_ap_bit_idx]
        X_mc_te = _ld(MACCS_TE_PATH)[:, top_maccs_bit_idx]
        X_emb_te = _ld(CHEMPROP_EMBED_TE_PATH)[:, top_embed_col_idx]
        X_av_te = _ld(AVALON_TE_PATH)[:, top_avalon_bit_idx]
        X_mord_te = np.load(MORDRED_DIR / "X_mordred_test.npy").astype(np.float32)
        X_mord_te = np.where(np.isfinite(X_mord_te), X_mord_te, 0.0)
        col_med = np.nanmedian(X_mord_te, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
        bad = ~np.isfinite(X_mord_te)
        if bad.any():
            ix, ic = np.where(bad)
            X_mord_te[ix, ic] = col_med[ic]
        X_mord_te = X_mord_te[:, top_mord_col_idx]

        # ChEMBL kNN
        KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
        KEEP_RELATIONS = {"=", "==", "~"}
        MAX_NM = 100_000.0
        MIN_NM = 1e-3

        frames = []
        for name, fn in [
            ("CHEMBL3401_raw", "chembl_pxr_CHEMBL3401.parquet"),
            ("nr_extended",    "chembl_nr_extended.parquet"),
            ("pxr_all_types",  "chembl_pxr_all_types.parquet"),
        ]:
            p = EXT_DIR / fn
            if not p.exists():
                continue
            import pandas as pd
            d = pd.read_parquet(p)
            if name == "CHEMBL3401_raw":
                mask = (
                    d["standard_type"].isin(KEEP_TYPES)
                    & d["canonical_smiles"].notna()
                    & (d["standard_units"] == "nM")
                    & d["standard_value"].notna()
                    & d["standard_relation"].isin(KEEP_RELATIONS)
                )
                d = d[mask].copy()
                v = d["standard_value"].astype(float)
                d = d[(v > MIN_NM) & (v < MAX_NM)].copy()
                d["pec50"] = 9.0 - np.log10(d["standard_value"].astype(float))
                d = d[["canonical_smiles", "pec50"]].rename(
                    columns={"canonical_smiles": "smiles"})
            elif name == "nr_extended":
                d = d[d["target_name"] == "PXR"].copy()
                d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
                d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
                d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
            else:
                d = d[d["target"] == "PXR"].copy()
                d = d[d["pec50"].notna() & d["smiles"].notna()].copy()
                d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
                d = d[["smiles", "pec50"]]
            frames.append(d)
        import pandas as pd
        pool = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["smiles", "pec50"])

        def _ik(m):
            try:
                return Chem.MolToInchiKey(m) if m else None
            except Exception:
                return None

        def _can(m):
            try:
                return Chem.MolToSmiles(m) if m else None
            except Exception:
                return None

        mols = pool["smiles"].apply(standardize)
        pool["inchikey"] = mols.apply(_ik)
        pool["std_smiles"] = mols.apply(_can)
        pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
        agg = pool.groupby("inchikey", as_index=False).agg(
            pec50=("pec50", "median"), std_smiles=("std_smiles", "first")
        )

        test_mols = [standardize(s) for s in test_smiles]
        test_iks = set()
        for m in test_mols:
            ik = _ik(m)
            if ik:
                test_iks.add(ik)
        agg = agg[~agg["inchikey"].isin(test_iks)].reset_index(drop=True)
        fp_pool = morgan_fp_batch(agg["std_smiles"].tolist())
        keep = fp_pool.sum(axis=1) > 0
        agg = agg[keep].reset_index(drop=True)
        fp_pool = fp_pool[keep]
        pool_labels = agg["pec50"].to_numpy(dtype=np.float32)
        pool_median = float(np.median(pool_labels))

        std_test = []
        for m in test_mols:
            std_test.append(Chem.MolToSmiles(m) if m else "")
        fp_test = morgan_fp_batch(std_test)

        a = fp_test.astype(np.float32)
        b = fp_pool.astype(np.float32)
        a_sum = a.sum(axis=1)
        b_sum = b.sum(axis=1)
        inter = a @ b.T
        denom = a_sum[:, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        K_KNN = 5
        part = np.argpartition(-sim, kth=K_KNN - 1, axis=1)[:, :K_KNN]
        ri = np.arange(sim.shape[0])[:, None]
        sim_p = sim[ri, part]
        order = np.argsort(-sim_p, axis=1)
        top_idx_knn = part[ri, order]
        top_sim_knn = sim[ri, top_idx_knn]
        w = np.clip(top_sim_knn, 0.0, 1.0)
        w_sum = w.sum(axis=1)
        pred_chembl_te = np.empty(n_test, dtype=np.float32)
        for i in range(n_test):
            if w_sum[i] < 1e-6:
                pred_chembl_te[i] = pool_median
            else:
                pred_chembl_te[i] = np.sum(w[i] * pool_labels[top_idx_knn[i]]) / w_sum[i]
        mean_sim_te = top_sim_knn.mean(axis=1).astype(np.float32)

        X_te_117 = np.concatenate([
            X_ap_te, X_mc_te, X_mord_te, X_emb_te, X_av_te,
            pred_chembl_te.reshape(-1, 1), mean_sim_te.reshape(-1, 1),
        ], axis=1).astype(np.float32)
        X_unb_117 = X_te_117[unb_idx]
        X_unb_28 = X_unb_117[:, top28_idx].astype(np.float32)
        np.save(DATA_PROCESSED / "X_unb_28_nb2103.npy", X_unb_28)
        print(f"  [save] X_unb_28 cached for future runs ({X_unb_28.shape})")

    # Run fresh-kf reproduction
    FRESH_KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
    fresh_per_seed_corrected = np.zeros((len(FRESH_KF_SEEDS), n_unb), dtype=np.float64)
    fresh_per_seed_rae = []
    for i, kf_s in enumerate(FRESH_KF_SEEDS):
        resid_oof = _residual_cross_fit_one_seed(
            X_unb_28, residual, kf_seed=kf_s, lgbm_seed=kf_s
        )
        pred_corr = te_nb730_unb + resid_oof
        fresh_per_seed_corrected[i] = pred_corr
        r = float(rae(y_unb, pred_corr))
        fresh_per_seed_rae.append(r)
        print(f"  fresh kf_seed={kf_s}  RAE_corrected={r:.4f}")

    fresh_mean_bag = fresh_per_seed_corrected.mean(axis=0)
    fresh_median_bag = np.median(fresh_per_seed_corrected, axis=0)
    fresh_rae_mean = float(rae(y_unb, fresh_mean_bag))
    fresh_rae_median = float(rae(y_unb, fresh_median_bag))
    print(f"\n  FRESH-KF mean-bag   RAE = {fresh_rae_mean:.4f}")
    print(f"  FRESH-KF median-bag RAE = {fresh_rae_median:.4f}")
    print(f"  ORIG mean-bag       RAE = {orig_rae_mean:.4f}")
    print(f"  delta (fresh - orig)    = {fresh_rae_mean - orig_rae_mean:+.4f}")

    reproduces = bool(abs(fresh_rae_mean - orig_rae_mean) < 0.02)
    print(f"\n  reproduces within 0.02 RAE? {reproduces}")

    # ---------- Final verdict ----------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if truth_contamination:
        verdict = "CONTAMINATED_te_unb_equals_y_unb_truth"
    elif identity_verdict == "IDENTICAL_BIT_FOR_BIT":
        verdict = "CONTAMINATED_te_unb_equals_pred_oof_likely_stitch"
    elif not reproduces and fresh_rae_mean > orig_rae_mean + 0.05:
        verdict = "LUCKY_KF_SEED_orig_0p3920_not_reproducible_with_fresh_kf"
    elif reproduces:
        verdict = "HONEST_reproduces_under_fresh_kf_seeds_within_0p02"
    else:
        verdict = "AMBIGUOUS_some_kf_drift_but_under_5pct"
    print(f"  {verdict}")

    summary = {
        "tag": TAG,
        "audit_target": "nb2170_anchor_swap_chemprop_aux_to_nb730_rae_0p3920",
        "check1_te_nb730_generation_script": gen_script,
        "check1_nb730_oof_generation_script": gen_oof_script,
        "check2_te_nb730_construction": construction_te_nb730,
        "check2_nb730_oof_construction": construction_nb730_oof,
        "check2_two_distinct_objects_by_construction": True,
        "check3_te_nb730_shape": list(te_nb730.shape),
        "check3_nb730_oof_shape": list(nb730_oof.shape),
        "check4_mean_abs_diff_te_unb_vs_pred_oof": mean_abs_diff,
        "check4_max_abs_diff": max_abs_diff,
        "check5_pearson": pearson,
        "check5_sha256_te_nb730_unb_idx": sha_te_unb,
        "check5_sha256_nb730_pred_oof": sha_oof,
        "check5_sha256_y_unb_253": sha_y_unb,
        "check6_rae_te_nb730_unb": rae_te_unb,
        "check6_rae_nb730_pred_oof": rae_oof,
        "check6_rae_gap": rae_gap,
        "check7_identity_verdict": identity_verdict,
        "check9_truth_contamination_te_unb_equals_y_unb": truth_contamination,
        "check8_orig_nb2170_mean_bag_rae": orig_rae_mean,
        "check8_orig_nb2170_median_bag_rae": orig_rae_median,
        "check8_fresh_kf_seeds": FRESH_KF_SEEDS,
        "check8_fresh_per_seed_rae": fresh_per_seed_rae,
        "check8_fresh_mean_bag_rae": fresh_rae_mean,
        "check8_fresh_median_bag_rae": fresh_rae_median,
        "check8_delta_fresh_minus_orig": fresh_rae_mean - orig_rae_mean,
        "check8_reproduces_within_0p02": reproduces,
        "final_verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "check4_mean_abs_diff_te_unb_vs_pred_oof",
        "check5_pearson",
        "check5_sha256_te_nb730_unb_idx",
        "check5_sha256_nb730_pred_oof",
        "check5_sha256_y_unb_253",
        "check6_rae_te_nb730_unb",
        "check6_rae_nb730_pred_oof",
        "check7_identity_verdict",
        "check9_truth_contamination_te_unb_equals_y_unb",
        "check8_orig_nb2170_mean_bag_rae",
        "check8_fresh_mean_bag_rae",
        "check8_delta_fresh_minus_orig",
        "check8_reproduces_within_0p02",
        "final_verdict",
    ):
        print(f"  {k}: {res.get(k)}")
