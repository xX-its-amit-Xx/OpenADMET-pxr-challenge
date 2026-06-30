"""nb1782 -- DEPLOY nb1771 LGBM(small-leaves) 5-way K-tuned 117-col -> 513.

HANDOFF FROM nb1771 (LGBM huber, max_depth=4, num_leaves=15, n_est=300, lr=0.03,
min_child_samples=5, reg_lambda=2 on 5-way K-tuned 117-col matrix):
    For each seed s in {0, 1, 7, 42, 137}:
        LGBM(huber, depth=4, num_leaves=15, n_est=300, lr=0.03, min_child=5,
             reg_lambda=2, random_state=s)
        Fit on ALL 253 unblind residuals (y_unb - anchor_unb).
        Predict residual on 513.
    Stack 5 deploys -> (5, 513).
    Row-level MEAN + MEDIAN bag across 5 seeds -> (513,) each.
    te_nb1782_mean   = te_chemprop_aux + mean_bag_residual_513
    te_nb1782_median = te_chemprop_aux + median_bag_residual_513

Features: 117-col 5-way K-tuned (identical to nb1554/nb1561/nb1570/nb1771)
    top-25 AtomPair       (nb1524 best_K)
    top-20 MACCS          (nb1352 fixed)
    top-20 Mordred        (nb1523 best_K)
    top-20 ChempropEmbed  (nb1541 best_K)
    top-30 Avalon         (nb1392 SHAP top-30)
    pred_chembl_pec50 + mean_sim   (ChEMBL kNN-5, cached on 513)

Honest LB anchors: 0.5100 mean / 0.5044 median (PRE-unblind, predicted LB 0.513/0.507).

Outputs:
    data/processed/te_nb1782_mean.npy             (513,) float32
    data/processed/te_nb1782_median.npy           (513,) float32
    data/processed/nb1782_summary.json
    submissions/nb1782_deploy_nb1771_mean.csv     SMILES, Molecule Name, pEC50
    submissions/nb1782_deploy_nb1771_median.csv   SMILES, Molecule Name, pEC50
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

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1782"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# Match nb1771 seeds exactly
DEPLOY_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1392_SUMMARY = DATA_PROCESSED / "nb1392_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1523_SUMMARY = DATA_PROCESSED / "nb1523_summary.json"
NB1524_SUMMARY = DATA_PROCESSED / "nb1524_summary.json"
NB1541_SUMMARY = DATA_PROCESSED / "nb1541_summary.json"

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

HONEST_LB_ANCHOR_MEAN = 0.5100
HONEST_LB_ANCHOR_MEDIAN = 0.5044


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
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


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_npy_test(path: Path, n_test_expected: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"missing cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"shape mismatch {path}: {X.shape}")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_atompair_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "AtomPair":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("AtomPair entry not found in nb1484_summary.json")


def _extract_best_K_record(sum_dict: dict, records_key: str,
                           best_K_key: str = "best_K") -> dict:
    best_K = int(sum_dict[best_K_key])
    for r in sum_dict[records_key]:
        if int(r["K"]) == best_K:
            return r
    raise KeyError(f"best_K {best_K} not found in {records_key}")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1771 LGBM(small-leaves) 5-way K-tuned -> 513")
    print(f"         seeds = {DEPLOY_SEEDS}")
    print(f"         honest LB MEAN   anchor = {HONEST_LB_ANCHOR_MEAN}")
    print(f"         honest LB MEDIAN anchor = {HONEST_LB_ANCHOR_MEDIAN}")
    print("=" * 78)

    # ---- Load test metadata ----
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
            raise KeyError(
                f"No Molecule Name column in test ({te.columns.tolist()})"
            )
        mol_names = te[cand[0]].astype(str).tolist()
    print(f"[load] n_test={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- Anchor (PRE-unblind chemprop_aux) ----
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"Anchor missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"anchor shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[anchor] te_{ANCHOR}  in_RAE(unb_idx)={rae_anchor:.4f}  (ref 0.6216)")
    residual_unb = y_unb - anchor_unb
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- Family top-K indices (same as nb1771) ----
    for p in (NB1352_SUMMARY, NB1392_SUMMARY, NB1484_SUMMARY,
              NB1523_SUMMARY, NB1524_SUMMARY, NB1541_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
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

    top_maccs_bit_idx = np.array(
        sum_1352["top_maccs_bit_indices_ranked"], dtype=int
    )

    rec_mord = _extract_best_K_record(sum_1523, "per_K_records",
                                       best_K_key="best_K")
    K_Mord_best = int(rec_mord["K"])
    top_mord_col_idx = np.array(rec_mord["top_col_idx"], dtype=int)
    assert K_Mord_best == int(sum_1523["best_K"])

    full_ap_ranked = _extract_atompair_top_idx_from_nb1484(sum_1484)
    K_AP_best = int(sum_1524["best_K"])
    top_ap_bit_idx = full_ap_ranked[:K_AP_best]

    K_Embed_best = int(sum_1541["best_K"])
    top_embed_full = np.array(sum_1541["top_dim_order_top100"], dtype=int)
    top_embed_col_idx = top_embed_full[:K_Embed_best]

    top_avalon_bit_idx = np.array(
        sum_1392["top_avalon_bit_indices_ranked"], dtype=int
    )
    K_Avalon_used = int(len(top_avalon_bit_idx))

    n_top_maccs = int(len(top_maccs_bit_idx))
    n_top_mord = int(len(top_mord_col_idx))
    n_top_ap = int(len(top_ap_bit_idx))
    n_top_embed = int(len(top_embed_col_idx))
    n_top_avalon = int(K_Avalon_used)
    print(f"\n[reuse] top-{n_top_ap}     AtomPair bits  (nb1524 K={K_AP_best})")
    print(f"[reuse] top-{n_top_maccs}     MACCS bits     (nb1352)")
    print(f"[reuse] top-{n_top_mord}     Mordred cols   (nb1523 K={K_Mord_best})")
    print(f"[reuse] top-{n_top_embed}     ChempropEmbed  (nb1541 K={K_Embed_best})")
    print(f"[reuse] top-{n_top_avalon}     Avalon bits    (nb1392 SHAP K=30)")

    # ---- Load 513 raw feature caches ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    print(f"\n[feat] X_ap_te_top      = {X_ap_te_top.shape}")

    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    print(f"[feat] X_maccs_te_top   = {X_maccs_te_top.shape}")

    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    print(f"[feat] X_mord_te_top    = {X_mord_te_top.shape}")

    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    print(f"[feat] X_emb_te_top     = {X_emb_te_top.shape}")

    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)
    print(f"[feat] X_av_te_top      = {X_av_te_top.shape}")

    # ---- ChEMBL kNN features on 513 (cached) ----
    if not PRED_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(f"missing {PRED_CHEMBL_513_PATH}")
    if not SIM_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(f"missing {SIM_CHEMBL_513_PATH}")
    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)
    if pred_chembl_513.shape[0] != n_test or sim_chembl_513.shape[0] != n_test:
        raise ValueError(
            f"chembl cache shape mismatch "
            f"({pred_chembl_513.shape}, {sim_chembl_513.shape})"
        )
    print(f"[chembl] pred_chembl_513 mean={pred_chembl_513.mean():.3f}  "
          f"std={pred_chembl_513.std():.3f}")
    print(f"[chembl] sim_chembl_513  mean={sim_chembl_513.mean():.3f}  "
          f"std={sim_chembl_513.std():.3f}")

    # ---- Build COMBINED 5-way K-tuned 117-col 513 matrix ----
    X_te = np.concatenate(
        [
            X_ap_te_top,
            X_maccs_te_top,
            X_mord_te_top,
            X_emb_te_top,
            X_av_te_top,
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_te.shape[1]
    expected_dim = (n_top_ap + n_top_maccs + n_top_mord
                    + n_top_embed + n_top_avalon + 2)
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    X_unb = X_te[unb_idx].astype(np.float32)
    print(f"\n   COMBINED 5-WAY K-TUNED 513 matrix: {X_te.shape}  "
          f"(top-{n_top_ap} AP + top-{n_top_maccs} MACCS + "
          f"top-{n_top_mord} Mord + top-{n_top_embed} Embed + "
          f"top-{n_top_avalon} Aval + 2)")
    print(f"   X_unb sliced from X_te by unb_idx: {X_unb.shape}")

    # ---- Per-seed LGBM small-leaf deploy ----
    print("\n" + "=" * 78)
    print(f"PER-SEED DEPLOY x {len(DEPLOY_SEEDS)} seeds  "
          f"LGBM(huber, depth=4, num_leaves=15, n_est=300, lr=0.03, "
          f"min_child=5, reg_lambda=2)")
    print("=" * 78)

    per_seed_resid_513 = np.zeros((len(DEPLOY_SEEDS), n_test), dtype=np.float64)
    per_seed_records = []

    for si, s in enumerate(DEPLOY_SEEDS):
        ts = time.time()
        mdl = lgb.LGBMRegressor(**_lgbm_params(int(s)))
        mdl.fit(X_unb, residual_unb)
        resid_in = mdl.predict(X_unb)
        corr_in = anchor_unb + resid_in
        in_rae_s = float(rae(y_unb, corr_in))
        resid_513 = mdl.predict(X_te).astype(np.float64)
        per_seed_resid_513[si] = resid_513
        per_seed_records.append({
            "seed": int(s),
            "in_sample_rae_253": in_rae_s,
            "resid_513_mean": float(resid_513.mean()),
            "resid_513_std": float(resid_513.std()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"     seed {s:3d}: in_RAE_253={in_rae_s:.4f}  "
              f"resid_513 mean={resid_513.mean():+.4f}  "
              f"std={resid_513.std():.4f}  "
              f"wall={time.time() - ts:.1f}s")

    # ---- Row-level bag aggregations across 5 seeds ----
    mean_bag_resid_513 = per_seed_resid_513.mean(axis=0)
    median_bag_resid_513 = np.median(per_seed_resid_513, axis=0)

    te_nb1782_mean = te_anchor_513 + mean_bag_resid_513
    te_nb1782_median = te_anchor_513 + median_bag_resid_513

    in_rae_mean = float(rae(y_unb, te_nb1782_mean[unb_idx]))
    in_rae_median = float(rae(y_unb, te_nb1782_median[unb_idx]))

    print("\n" + "=" * 78)
    print("BAG DEPLOY SUMMARY")
    print("=" * 78)
    print(f"   te_nb1782_mean    mean={te_nb1782_mean.mean():.4f}  "
          f"std={te_nb1782_mean.std():.4f}  "
          f"min={te_nb1782_mean.min():.4f}  "
          f"max={te_nb1782_mean.max():.4f}")
    print(f"   te_nb1782_median  mean={te_nb1782_median.mean():.4f}  "
          f"std={te_nb1782_median.std():.4f}  "
          f"min={te_nb1782_median.min():.4f}  "
          f"max={te_nb1782_median.max():.4f}")
    print(f"   in_RAE(unb) MEAN   = {in_rae_mean:.4f}  "
          f"(honest LB anchor {HONEST_LB_ANCHOR_MEAN})")
    print(f"   in_RAE(unb) MEDIAN = {in_rae_median:.4f}  "
          f"(honest LB anchor {HONEST_LB_ANCHOR_MEDIAN})")
    print(f"   d_vs_anchor MEAN   = {in_rae_mean - rae_anchor:+.4f}")
    print(f"   d_vs_anchor MEDIAN = {in_rae_median - rae_anchor:+.4f}")

    # ---- Save NPYs ----
    te_mean_path = DATA_PROCESSED / f"te_{TAG}_mean.npy"
    te_median_path = DATA_PROCESSED / f"te_{TAG}_median.npy"
    np.save(te_mean_path, te_nb1782_mean.astype(np.float32))
    np.save(te_median_path, te_nb1782_median.astype(np.float32))
    print(f"\n[save] {te_mean_path}")
    print(f"[save] {te_median_path}")

    # ---- Save CSVs ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    csv_mean_path = SUBMISSIONS / f"{TAG}_deploy_nb1771_mean.csv"
    csv_median_path = SUBMISSIONS / f"{TAG}_deploy_nb1771_median.csv"

    df_mean = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1782_mean.astype(np.float64),
    })
    df_median = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1782_median.astype(np.float64),
    })
    df_mean.to_csv(csv_mean_path, index=False)
    df_median.to_csv(csv_median_path, index=False)
    print(f"[save] {csv_mean_path}  rows={len(df_mean)}  "
          f"cols={list(df_mean.columns)}")
    print(f"[save] {csv_median_path}  rows={len(df_median)}  "
          f"cols={list(df_median.columns)}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "parent_method": "nb1771",
        "parent_variants": ["mean_bag", "median_bag"],
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "anchor_kind": "PRE_unblind_te_513",
        "rae_anchor": rae_anchor,
        "model_family": "LightGBM",
        "lgbm_objective": "huber",
        "lgbm_alpha": 1.0,
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "deploy_seeds": DEPLOY_SEEDS,
        "K_AP_best": K_AP_best,
        "K_Mord_best": K_Mord_best,
        "K_Embed_best": K_Embed_best,
        "K_Avalon_used": K_Avalon_used,
        "K_MACCS_fixed": n_top_maccs,
        "feat_dim": int(feat_dim),
        "feat_breakdown": {
            "atompair": n_top_ap,
            "maccs": n_top_maccs,
            "mordred": n_top_mord,
            "chemprop_embed": n_top_embed,
            "avalon": n_top_avalon,
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "per_seed_records": per_seed_records,
        "te_nb1782_mean_stats": {
            "mean": float(te_nb1782_mean.mean()),
            "std": float(te_nb1782_mean.std()),
            "min": float(te_nb1782_mean.min()),
            "max": float(te_nb1782_mean.max()),
        },
        "te_nb1782_median_stats": {
            "mean": float(te_nb1782_median.mean()),
            "std": float(te_nb1782_median.std()),
            "min": float(te_nb1782_median.min()),
            "max": float(te_nb1782_median.max()),
        },
        "in_rae_unb_mean": in_rae_mean,
        "in_rae_unb_median": in_rae_median,
        "delta_mean_vs_anchor": in_rae_mean - rae_anchor,
        "delta_median_vs_anchor": in_rae_median - rae_anchor,
        "honest_lb_anchor_mean_nb1771": HONEST_LB_ANCHOR_MEAN,
        "honest_lb_anchor_median_nb1771": HONEST_LB_ANCHOR_MEDIAN,
        "te_mean_npy_path": str(te_mean_path),
        "te_median_npy_path": str(te_median_path),
        "csv_mean_path": str(csv_mean_path),
        "csv_median_path": str(csv_median_path),
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
        "feat_dim", "feat_breakdown",
        "deploy_seeds",
        "te_nb1782_mean_stats", "te_nb1782_median_stats",
        "in_rae_unb_mean", "in_rae_unb_median",
        "delta_mean_vs_anchor", "delta_median_vs_anchor",
        "honest_lb_anchor_mean_nb1771",
        "honest_lb_anchor_median_nb1771",
        "csv_mean_path", "csv_median_path",
        "te_mean_npy_path", "te_median_npy_path",
    ):
        print(f"  {k}: {res.get(k)}")
