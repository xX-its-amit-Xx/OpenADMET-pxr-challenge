"""nb2713 -- HyperOpt Parzen-window TPE hyperparameter search for nb2240 K=20.

CONTEXT (cycle 134+ paradigm exhaustion -> hyperparam axis, sister to nb2582):
    nb2582 used Optuna's TPE (multivariate Parzen) for the same nb2240 K=20
    LGBM tuning task.  nb2713 swaps the sampler implementation to HyperOpt's
    classical Parzen-window TPE.  Same protocol (5 kf_seeds, scaffold ...
    wait -- the inner CV here is KFold(5) as in nb2582, NOT scaffold-CV --
    so trial scores remain comparable to the nb2240 K=20 mean-bag 0.4601
    baseline.  HyperOpt's TPE uses a slightly different exploration profile
    (per-dim Parzen by default, gamma=0.25 default split) than Optuna's
    multivariate variant -- this is the only paradigm delta.

PROTOCOL (mirrors nb2582):
    1. Rebuild the K=20 feature matrix exactly as nb2240 (RFE survivors
       from nb2231 -> 117-col 5-way + ChEMBL kNN -> slice to K=20).
    2. Compute chemprop_aux residual on 253 unblind.
    3. HyperOpt fmin with tpe.suggest, 60 max_evals.  Search space:
         max_depth          quniform-int  [3, 8]
         num_leaves         quniform-int  [7, 63]
         learning_rate      loguniform    [0.01, 0.10]
         reg_lambda         loguniform    [0.5, 10.0]
         min_child_samples  quniform-int  [3, 20]
       Objective: mean RAE across 5 kf_seeds {0, 1, 7, 42, 137}, each seed
       runs KFold(5, shuffle=True, random_state=seed) on 253 -> OOF residual
       -> rae(y_unb, anchor + oof_residual).  Trial loss = mean of 5
       per-seed RAEs.
    4. Best trial -> deploy refit: train LGBM on all 253 with best params
       (one model per seed in {0, 1, 7, 42, 137}), predict residual on 513,
       mean-bag across 5 seeds.  Save mean-bag OOF + te.
    5. Gate: best mean_rae < 0.4570 -> PROMOTE; < 0.4598 -> MARGINAL_BEAT;
       else FAIL.

Outputs:
    data/processed/nb2713_summary.json
    data/processed/nb2713_mean_bag_oof.npy   (253,) float32
    data/processed/te_nb2713.npy             (513,) float32
    data/processed/nb2713_best_params.json
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
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.stochastic import sample as hp_sample  # noqa: F401  (validate import)
from rdkit import Chem
from rdkit import RDLogger
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2713"

# --- HyperOpt config ---
N_TRIALS = 60
SAMPLER_SEED = 20260608  # reproducible HyperOpt random state (rstate)
KF_SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5

# --- Reference points ---
NB2240_K20_MEAN_BAG_RAE = 0.4601   # current K=20 baseline (fixed-HP)
PROMOTE_THRESHOLD = 0.4570
MARGINAL_THRESHOLD = 0.4598

# --- Anchor / data paths ---
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

EXT_DIR = Path(__file__).resolve().parents[1] / "data" / "external"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"

KEEP_TYPES = {"EC50", "IC50", "Ki", "Kd", "AC50", "Potency"}
KEEP_RELATIONS = {"=", "==", "~"}
MAX_NM = 100_000.0
MIN_NM = 1e-3
KNN_K = 5
SIM_FLOOR = 1e-6


# ============================================================================
# Feature build helpers (re-used from nb2240 / nb2582)
# ============================================================================

def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _safe_can_smiles(mol):
    try:
        return Chem.MolToSmiles(mol) if mol is not None else None
    except Exception:
        return None


def _load_chembl_pool() -> pd.DataFrame:
    frames = []
    p1 = EXT_DIR / "chembl_pxr_CHEMBL3401.parquet"
    if p1.exists():
        d = pd.read_parquet(p1)
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
        d["pec50_raw"] = 9.0 - np.log10(d["standard_value"].astype(float))
        d = d[["canonical_smiles", "pec50_raw"]].rename(
            columns={"canonical_smiles": "smiles", "pec50_raw": "pec50"}
        )
        d["src"] = "CHEMBL3401_raw"
        frames.append(d)
    p2 = EXT_DIR / "chembl_nr_extended.parquet"
    if p2.exists():
        d = pd.read_parquet(p2)
        d = d[d["target_name"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["std_smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
        d["src"] = "nr_extended"
        frames.append(d)
    p3 = EXT_DIR / "chembl_pxr_all_types.parquet"
    if p3.exists():
        d = pd.read_parquet(p3)
        d = d[d["target"] == "PXR"].copy()
        d = d[d["pec50"].notna() & d["smiles"].notna()].copy()
        d["pec50"] = d["pec50"].astype(float)
        d = d[(d["pec50"] >= 3.0) & (d["pec50"] <= 11.0)].copy()
        d = d[["smiles", "pec50"]]
        d["src"] = "pxr_all_types"
        frames.append(d)
    if not frames:
        raise FileNotFoundError("No local ChEMBL PXR parquets found")
    pool = pd.concat(frames, ignore_index=True)
    mols = pool["smiles"].apply(standardize)
    pool["inchikey"] = mols.apply(_safe_inchikey)
    pool["std_smiles"] = mols.apply(_safe_can_smiles)
    pool = pool[pool["inchikey"].notna() & pool["std_smiles"].notna()].copy()
    agg = (
        pool.groupby("inchikey", as_index=False)
        .agg(pec50=("pec50", "median"),
             std_smiles=("std_smiles", "first"),
             src_first=("src", "first"),
             n_meas=("pec50", "count"))
    )
    agg = agg.rename(columns={"src_first": "src"})
    return agg


def _tanimoto_topk(fp_q, fp_pool, k):
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    n_pool = b.shape[0]
    top_idx = np.zeros((n_q, k), dtype=np.int32)
    top_sim = np.zeros((n_q, k), dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        if k >= n_pool:
            idx_part = np.argsort(-sim, axis=1)[:, :k]
        else:
            part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            row_idx = np.arange(e - s)[:, None]
            sim_part = sim[row_idx, part]
            order = np.argsort(-sim_part, axis=1)
            idx_part = part[row_idx, order]
        row_idx = np.arange(e - s)[:, None]
        top_idx[s:e] = idx_part
        top_sim[s:e] = sim[row_idx, idx_part]
    return top_idx, top_sim


def _knn_predict(top_idx, top_sim, pool_labels, fallback):
    w = np.clip(top_sim, 0.0, 1.0)
    w_sum = w.sum(axis=1)
    n_q = top_idx.shape[0]
    pred = np.empty(n_q, dtype=np.float32)
    for i in range(n_q):
        if w_sum[i] < SIM_FLOOR:
            pred[i] = fallback
        else:
            pred[i] = np.sum(w[i] * pool_labels[top_idx[i]]) / w_sum[i]
    mean_sim = top_sim.mean(axis=1).astype(np.float32)
    return pred, mean_sim


def _load_mordred_test(n_test_expected):
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing -- run nb1030 first ({mte_p})")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape mismatch: {X_te_m.shape}")
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_npy_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx_from_nb1484(sum_1484):
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict, records_key, best_K_key="best_K"):
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def build_K20_matrices():
    """Reconstruct (X_unb_K20, X_te_K20) exactly as nb2240."""
    te = load_test()
    n_test = len(te)
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns \
        else te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")

    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    surviving_K20 = list(nb2231["snapshots"]["20"]["surviving_idx_in_117"])
    surviving_K20_names = list(nb2231["snapshots"]["20"]["surviving_names"])
    assert len(surviving_K20) == 20

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

    top_maccs_bit_idx = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    rec_mord = _extract_best_K_record(sum_1523, "per_K_records", best_K_key="best_K")
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]
    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]
    top_avalon_bit_idx = np.array(sum_1392["top_avalon_bit_indices_ranked"], dtype=int)

    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)[:, top_avalon_bit_idx].astype(np.float32)

    # ChEMBL kNN
    pool = _load_chembl_pool()
    test_mols = [standardize(s) for s in te_smiles]
    test_inchikeys = set()
    for m in test_mols:
        ik = _safe_inchikey(m)
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
    std_test_smiles = []
    for m in test_mols:
        if m is None:
            std_test_smiles.append("")
        else:
            std_test_smiles.append(Chem.MolToSmiles(m))
    fp_test = morgan_fp_batch(std_test_smiles)
    top_idx_knn, top_sim_knn = _tanimoto_topk(fp_test, fp_pool, k=KNN_K)
    pred_chembl_pec50, mean_sim = _knn_predict(
        top_idx_knn, top_sim_knn, pool_labels, fallback=pool_median
    )

    X_te_full = np.concatenate(
        [
            X_ap_te,
            X_maccs_te,
            X_mord_te,
            X_emb_te,
            X_av_te,
            pred_chembl_pec50.reshape(-1, 1).astype(np.float32),
            mean_sim.reshape(-1, 1).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert X_te_full.shape[1] == 117, f"feat_dim {X_te_full.shape[1]} != 117"

    X_te_K20 = X_te_full[:, surviving_K20].astype(np.float32)
    X_unb_K20 = X_te_K20[unb_idx]
    return X_unb_K20, X_te_K20, unb_idx, n_test, surviving_K20, surviving_K20_names


# ============================================================================
# HyperOpt search space + objective + final fit
# ============================================================================

# HyperOpt search space (Parzen-window TPE).  Integers come back as floats
# from hp.quniform -- we cast in lgbm_params_from_dict().
SEARCH_SPACE = {
    "max_depth":         hp.quniform("max_depth", 3, 8, 1),
    "num_leaves":        hp.quniform("num_leaves", 7, 63, 1),
    "learning_rate":     hp.loguniform("learning_rate", np.log(0.01), np.log(0.10)),
    "reg_lambda":        hp.loguniform("reg_lambda", np.log(0.5),  np.log(10.0)),
    "min_child_samples": hp.quniform("min_child_samples", 3, 20, 1),
}


def lgbm_params_from_dict(params, seed):
    return dict(
        objective="regression",
        max_depth=int(round(float(params["max_depth"]))),
        num_leaves=int(round(float(params["num_leaves"]))),
        n_estimators=300,
        learning_rate=float(params["learning_rate"]),
        min_child_samples=int(round(float(params["min_child_samples"]))),
        reg_lambda=float(params["reg_lambda"]),
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def cross_fit_one_seed(X, residual, params):
    n = len(residual)
    seed = params["random_state"]
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def make_objective(X_unb, anchor_unb, residual, y_unb, t0_ref, state):
    """state: dict tracking best_so_far + trial_log (mutated as fmin calls)."""
    def objective(raw_params):
        per_seed_rae = []
        for seed in KF_SEEDS:
            params = lgbm_params_from_dict(raw_params, seed)
            oof_resid = cross_fit_one_seed(X_unb, residual, params)
            r = float(rae(y_unb, anchor_unb + oof_resid))
            per_seed_rae.append(r)
        loss = float(np.mean(per_seed_rae))

        # log + cumulative best
        n = state["n_evals"]
        prev_best = state["best_so_far"]
        new_best = min(prev_best, loss) if prev_best is not None else loss
        state["best_so_far"] = new_best

        # canonicalize for logging
        canon = {
            "max_depth": int(round(float(raw_params["max_depth"]))),
            "num_leaves": int(round(float(raw_params["num_leaves"]))),
            "learning_rate": float(raw_params["learning_rate"]),
            "reg_lambda": float(raw_params["reg_lambda"]),
            "min_child_samples": int(round(float(raw_params["min_child_samples"]))),
        }
        elapsed = time.time() - t0_ref
        print(f"   trial {n:3d}: loss={loss:.4f}  params={canon}  "
              f"best={new_best:.4f}  wall={elapsed:.1f}s")
        state["trial_log"].append({
            "number": int(n),
            "value": loss,
            "params": canon,
            "per_seed_rae": per_seed_rae,
        })
        state["n_evals"] = n + 1
        return {"loss": loss, "status": STATUS_OK, "params_canon": canon}
    return objective


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- HyperOpt Parzen-TPE HP search for nb2240 K=20 LGBM")
    print("=" * 78)

    # ---- Load truth + anchor ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual = y_unb - anchor_unb
    print(f"[load] n_unb={n_unb}  chemprop_aux in_RAE={rae_anchor:.4f}")

    # ---- Rebuild K=20 feature matrix ----
    print("[feat] building K=20 matrix (RFE survivors from nb2231)")
    X_unb_K20, X_te_K20, unb_idx_check, n_test, surviving_K20, surviving_names = build_K20_matrices()
    assert np.array_equal(unb_idx, unb_idx_check)
    print(f"[feat] X_unb_K20={X_unb_K20.shape}  X_te_K20={X_te_K20.shape}")

    # ---- HyperOpt TPE search ----
    print("\n" + "-" * 78)
    print(f"HYPEROPT TPE  max_evals={N_TRIALS}  kf_seeds={KF_SEEDS}  folds={N_FOLDS}")
    print(f"baseline nb2240 K=20 mean-bag RAE = {NB2240_K20_MEAN_BAG_RAE:.4f}")
    print("-" * 78)

    trials = Trials()
    state = {"best_so_far": None, "n_evals": 0, "trial_log": []}
    objective = make_objective(X_unb_K20, anchor_unb, residual, y_unb, t0, state)

    rstate = np.random.default_rng(SAMPLER_SEED)
    best_raw = fmin(
        fn=objective,
        space=SEARCH_SPACE,
        algo=tpe.suggest,
        max_evals=N_TRIALS,
        trials=trials,
        rstate=rstate,
        show_progressbar=False,
        verbose=False,
    )

    # Reconstruct best params + best loss from trials
    best_idx = int(np.argmin([t["result"]["loss"] for t in trials.trials]))
    best_value = float(trials.trials[best_idx]["result"]["loss"])
    best_params_canon = trials.trials[best_idx]["result"]["params_canon"]
    print(f"\n[opt] best trial #{best_idx}  loss={best_value:.4f}")
    print(f"[opt] best params: {best_params_canon}")

    # ---- Deploy refit with best params: mean-bag across 5 seeds ----
    print("\n" + "-" * 78)
    print(f"DEPLOY  best-params mean-bag refit on 253 -> predict 513  seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae_final = []
    for i, seed in enumerate(KF_SEEDS):
        params = lgbm_params_from_dict(best_params_canon, seed)
        oof_resid = cross_fit_one_seed(X_unb_K20, residual, params)
        per_seed_corrected[i] = anchor_unb + oof_resid
        per_seed_rae_final.append(float(rae(y_unb, anchor_unb + oof_resid)))
        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(X_unb_K20, residual)
        per_seed_te_resid[i] = mdl.predict(X_te_K20).astype(np.float64)
        print(f"   seed={seed:3d}: rae_corr={per_seed_rae_final[-1]:.4f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_513 = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    mean_bag_rae = float(rae(y_unb, mean_bag_oof))
    per_seed_mean = float(np.mean(per_seed_rae_final))

    print(f"\n[deploy] per-seed mean RAE = {per_seed_mean:.4f}")
    print(f"[deploy] mean-bag RAE      = {mean_bag_rae:.4f}")
    print(f"[deploy] delta vs nb2240   = {mean_bag_rae - NB2240_K20_MEAN_BAG_RAE:+.4f}")

    # ---- Gate ----
    if best_value < PROMOTE_THRESHOLD:
        verdict = "PROMOTE"
    elif best_value < MARGINAL_THRESHOLD:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    print(f"   best_trial_value     = {best_value:.4f}")
    print(f"   promote threshold    = {PROMOTE_THRESHOLD:.4f}")
    print(f"   marginal threshold   = {MARGINAL_THRESHOLD:.4f}")
    print(f"   verdict              = {verdict}")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    bp_path = DATA_PROCESSED / f"{TAG}_best_params.json"
    np.save(oof_path, mean_bag_oof.astype(np.float32))
    np.save(te_path, te_513)
    with open(bp_path, "w") as f:
        json.dump(best_params_canon, f, indent=2)
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")
    print(f"[save] {bp_path}")

    summary = {
        "tag": TAG,
        "method": "hyperopt_parzen_tpe_hp_search_nb2240_K20",
        "sampler": "hyperopt.tpe.suggest",
        "n_trials": N_TRIALS,
        "sampler_seed": SAMPLER_SEED,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_unb": n_unb,
        "n_te": n_test,
        "rae_anchor_chemprop_aux": rae_anchor,
        "search_space": {
            "max_depth": [3, 8, "quniform-int"],
            "num_leaves": [7, 63, "quniform-int"],
            "learning_rate": [0.01, 0.10, "loguniform"],
            "reg_lambda": [0.5, 10.0, "loguniform"],
            "min_child_samples": [3, 20, "quniform-int"],
        },
        "baseline_nb2240_K20_mean_bag_rae": NB2240_K20_MEAN_BAG_RAE,
        "promote_threshold": PROMOTE_THRESHOLD,
        "marginal_threshold": MARGINAL_THRESHOLD,
        "best_trial_number": int(best_idx),
        "best_trial_value": best_value,
        "best_params": best_params_canon,
        "deploy_per_seed_rae": per_seed_rae_final,
        "deploy_per_seed_mean_rae": per_seed_mean,
        "deploy_mean_bag_rae": mean_bag_rae,
        "delta_vs_nb2240_K20": mean_bag_rae - NB2240_K20_MEAN_BAG_RAE,
        "verdict": verdict,
        "k20_surviving_idx_in_117": [int(j) for j in surviving_K20],
        "k20_surviving_names": surviving_names,
        "oof_path": str(oof_path),
        "te_path": str(te_path),
        "best_params_path": str(bp_path),
        "trial_log": state["trial_log"],
        "te_mean": float(te_513.mean()),
        "te_std": float(te_513.std()),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   sampler               = hyperopt.tpe.suggest")
    print(f"   n_trials              = {N_TRIALS}")
    print(f"   best_trial_value      = {best_value:.4f}")
    print(f"   best_params           = {best_params_canon}")
    print(f"   deploy mean-bag RAE   = {mean_bag_rae:.4f}")
    print(f"   delta vs nb2240 K=20  = {mean_bag_rae - NB2240_K20_MEAN_BAG_RAE:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_trial_number",
        "best_trial_value",
        "best_params",
        "deploy_per_seed_mean_rae",
        "deploy_mean_bag_rae",
        "delta_vs_nb2240_K20",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
