"""nb1583 -- DEPLOY nb1571 blend (0.55 * nb1561_BoB + 0.45 * nb1560_BoB) -> 513.

Protocol:
1. te_nb1561_bob   == te_nb1570_mean (already built; CatBoost MAE BoB deploy).
2. te_nb1560_bob   built here: 5 outer x 5 inner LGBM-Huber on COMBINED 117-col
   5-way K-tuned matrix (top-25 AP + top-20 MACCS + top-20 Mord + top-20 chemprop
   embed + top-30 Avalon + pred_chembl + sim).
       - inner seed = o*1000 + offset, offsets [0,1,7,42,137]
       - LGBM(huber alpha=1.0, d3, leaves=7, n_est=80, lr=0.05) on (X_unb, residual_unb)
       - predict residual on 513 -> resid_513 (513,)
       - per_outer = mean of 5 inner deploys -> (5, 513) stack
       - te_nb1560_bob = 513-row BoB mean across 5 outer (then + anchor)
3. te_nb1583 = 0.55 * te_nb1570_mean + 0.45 * te_nb1560_bob.
4. CSV at submissions/nb1583_deploy_nb1571.csv.

Anchor:      chemprop_aux  (PRE-unblind, te[unb_idx] in_RAE 0.6216)
Honest LB:   nb1571 cross-fit 0.5130, predicted LB ~0.516

Outputs:
    data/processed/te_nb1560_bob.npy        (513,) float32
    data/processed/te_nb1583.npy            (513,) float32
    data/processed/nb1583_summary.json
    submissions/nb1583_deploy_nb1571.csv    SMILES, Molecule Name, pEC50
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
from lightgbm import LGBMRegressor
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1583"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# nb1571 weights (grid-best, also matches in-sample SLSQP ~ 0.558)
W_NB1561 = 0.55   # weight on nb1570_mean (CatBoost BoB deploy = te_nb1561_bob)
W_NB1560 = 0.45   # weight on te_nb1560_bob (LGBM BoB deploy)

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_OFFSETS = [0, 1, 7, 42, 137]

# 117-col 5-way K-tuned matrix indices (same as nb1570)
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

HONEST_LB_ANCHOR_NB1571 = 0.5130


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
    print(f"{TAG} -- DEPLOY nb1571 blend (0.55 * nb1561_BoB + 0.45 * nb1560_BoB)")
    print(f"         outer seeds   = {OUTER_SEEDS}")
    print(f"         inner offsets = {INNER_OFFSETS}  (inner_seed = o*1000 + offset)")
    print(f"         honest LB nb1571 cross-fit anchor = {HONEST_LB_ANCHOR_NB1571}")
    print(f"         W_nb1561={W_NB1561}  W_nb1560={W_NB1560}")
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
            raise KeyError(f"No Molecule Name col in test ({te.columns.tolist()})")
        mol_names = te[cand[0]].astype(str).tolist()
    print(f"[load] n_test={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- Anchor ----
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor shape mismatch: {te_anchor_513.shape}")
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[anchor] chemprop_aux in_RAE(unb_idx)={rae_anchor:.4f}  (ref 0.6216)")
    residual_unb = y_unb - anchor_unb
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- Family top-K indices (identical to nb1570) ----
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
    print(f"[reuse] top-{n_top_ap} AP  top-{n_top_maccs} MACCS  top-{n_top_mord} Mord  "
          f"top-{n_top_embed} Embed  top-{n_top_avalon} Avalon")

    # ---- Load 513 raw feature caches ----
    X_ap_te = _load_npy_test(ATOMPAIR_TE_PATH, n_test)
    X_ap_te_top = X_ap_te[:, top_ap_bit_idx].astype(np.float32)
    X_maccs_te = _load_npy_test(MACCS_TE_PATH, n_test)
    X_maccs_te_top = X_maccs_te[:, top_maccs_bit_idx].astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_mord_te_top = X_mord_te[:, top_mord_col_idx].astype(np.float32)
    X_emb_te = _load_npy_test(CHEMPROP_EMBED_TE_PATH, n_test)
    X_emb_te_top = X_emb_te[:, top_embed_col_idx].astype(np.float32)
    X_av_te = _load_npy_test(AVALON_TE_PATH, n_test)
    X_av_te_top = X_av_te[:, top_avalon_bit_idx].astype(np.float32)

    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)
    if pred_chembl_513.shape[0] != n_test or sim_chembl_513.shape[0] != n_test:
        raise ValueError("chembl cache shape mismatch")
    print(f"[chembl] pred_chembl mean={pred_chembl_513.mean():.3f} "
          f"std={pred_chembl_513.std():.3f}  "
          f"sim mean={sim_chembl_513.mean():.3f}")

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
    print(f"[feat] COMBINED 5-WAY K-TUNED 513 matrix: {X_te.shape}  "
          f"({n_top_ap}+{n_top_maccs}+{n_top_mord}+"
          f"{n_top_embed}+{n_top_avalon}+2)")

    # ---- Outer-bag LGBM Huber deploy ----
    print("\n" + "=" * 78)
    print("OUTER-BAG DEPLOY x 5 inner-seed LGBM(huber a=1.0, d3, l7, n80, lr0.05)")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)

    outer_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_records = []

    for oi, o in enumerate(OUTER_SEEDS):
        t_outer = time.time()
        inner_seeds = [int(o) * 1000 + int(s) for s in INNER_OFFSETS]
        print(f"\n  --- outer seed {o}  inner_seeds = {inner_seeds} ---")

        inner_resid_513 = np.zeros((len(inner_seeds), n_test), dtype=np.float64)
        per_inner_records = []
        for si, s_inner in enumerate(inner_seeds):
            ts = time.time()
            mdl = LGBMRegressor(**_lgbm_params(int(s_inner)))
            mdl.fit(X_unb, residual_unb)
            resid_in = mdl.predict(X_unb)
            corr_in = anchor_unb + resid_in
            in_rae_s = float(rae(y_unb, corr_in))
            resid_513 = mdl.predict(X_te).astype(np.float64)
            inner_resid_513[si] = resid_513
            per_inner_records.append({
                "inner_seed": int(s_inner),
                "in_sample_rae_253": in_rae_s,
                "resid_513_mean": float(resid_513.mean()),
                "resid_513_std": float(resid_513.std()),
                "wall_sec": round(time.time() - ts, 2),
            })
            print(f"    [outer {o:3d}] inner {s_inner:6d}: "
                  f"in_RAE_253={in_rae_s:.4f}  "
                  f"resid_513 mean={resid_513.mean():+.4f}  "
                  f"std={resid_513.std():.4f}  "
                  f"wall={time.time() - ts:.1f}s")

        per_outer_resid_513 = inner_resid_513.mean(axis=0)
        outer_resid_513[oi] = per_outer_resid_513

        per_outer_corr_unb = anchor_unb + per_outer_resid_513[unb_idx]
        in_rae_outer = float(rae(y_unb, per_outer_corr_unb))

        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "per_inner_records": per_inner_records,
            "per_outer_resid_513_mean": float(per_outer_resid_513.mean()),
            "per_outer_resid_513_std": float(per_outer_resid_513.std()),
            "per_outer_in_rae_unb": in_rae_outer,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"    [outer {o:3d}] per_outer_resid_513 mean="
              f"{per_outer_resid_513.mean():+.4f}  "
              f"std={per_outer_resid_513.std():.4f}  "
              f"in_RAE(unb)={in_rae_outer:.4f}  "
              f"wall={time.time() - t_outer:.1f}s")

    # ---- BoB row-level MEAN over 5 outer ----
    bob_mean_resid_513 = outer_resid_513.mean(axis=0)
    te_nb1560_bob = (te_anchor_513 + bob_mean_resid_513).astype(np.float64)
    in_rae_nb1560_bob = float(rae(y_unb, te_nb1560_bob[unb_idx]))
    print(f"\n[nb1560_bob] mean={te_nb1560_bob.mean():.4f}  "
          f"std={te_nb1560_bob.std():.4f}  "
          f"min={te_nb1560_bob.min():.4f}  max={te_nb1560_bob.max():.4f}")
    print(f"[nb1560_bob] in_RAE(unb_idx) = {in_rae_nb1560_bob:.4f}")

    # ---- Load nb1561_bob (= te_nb1570_mean, already deployed) ----
    te_nb1561_bob_path = DATA_PROCESSED / "te_nb1570_mean.npy"
    te_nb1561_bob = np.load(te_nb1561_bob_path).astype(np.float64)
    if te_nb1561_bob.shape[0] != n_test:
        raise ValueError(f"te_nb1570_mean shape mismatch: {te_nb1561_bob.shape}")
    in_rae_nb1561_bob = float(rae(y_unb, te_nb1561_bob[unb_idx]))
    print(f"\n[nb1561_bob = te_nb1570_mean] mean={te_nb1561_bob.mean():.4f}  "
          f"std={te_nb1561_bob.std():.4f}")
    print(f"[nb1561_bob] in_RAE(unb_idx) = {in_rae_nb1561_bob:.4f}")

    # ---- Blend ----
    te_nb1583 = (W_NB1561 * te_nb1561_bob + W_NB1560 * te_nb1560_bob).astype(np.float64)
    in_rae_nb1583 = float(rae(y_unb, te_nb1583[unb_idx]))

    print("\n" + "=" * 78)
    print("BLEND SUMMARY  te_nb1583 = 0.55 * te_nb1561_bob + 0.45 * te_nb1560_bob")
    print("=" * 78)
    print(f"  te_nb1583  mean={te_nb1583.mean():.4f}  "
          f"std={te_nb1583.std():.4f}  "
          f"min={te_nb1583.min():.4f}  "
          f"max={te_nb1583.max():.4f}")
    print(f"  in_RAE(unb) nb1583   = {in_rae_nb1583:.4f}")
    print(f"  in_RAE(unb) nb1561_bob (ref) = {in_rae_nb1561_bob:.4f}")
    print(f"  in_RAE(unb) nb1560_bob       = {in_rae_nb1560_bob:.4f}")
    print(f"  honest LB anchor (nb1571 xfit) = {HONEST_LB_ANCHOR_NB1571}  "
          f"(predicted LB ~0.516)")

    # ---- Save NPYs ----
    te_nb1560_path = DATA_PROCESSED / "te_nb1560_bob.npy"
    te_nb1583_path = DATA_PROCESSED / "te_nb1583.npy"
    np.save(te_nb1560_path, te_nb1560_bob.astype(np.float32))
    np.save(te_nb1583_path, te_nb1583.astype(np.float32))
    print(f"\n[save] {te_nb1560_path}")
    print(f"[save] {te_nb1583_path}")

    # ---- CSV ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1571.csv"
    df_out = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1583.astype(np.float64),
    })
    df_out.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}  rows={len(df_out)}  cols={list(df_out.columns)}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "parent_method": "nb1571",
        "blend_recipe": "0.55 * te_nb1561_bob + 0.45 * te_nb1560_bob",
        "w_nb1561": W_NB1561,
        "w_nb1560": W_NB1560,
        "te_nb1561_bob_source": str(te_nb1561_bob_path),
        "anchor": "chemprop_aux",
        "anchor_path": str(ANCHOR_TE_PATH),
        "rae_anchor": rae_anchor,
        "outer_seeds": OUTER_SEEDS,
        "inner_offsets": INNER_OFFSETS,
        "inner_seed_formula": "o * 1000 + offset",
        "lgbm_params": _lgbm_params(0),
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
        "per_outer_records": per_outer_records,
        "te_nb1560_bob_stats": {
            "mean": float(te_nb1560_bob.mean()),
            "std": float(te_nb1560_bob.std()),
            "min": float(te_nb1560_bob.min()),
            "max": float(te_nb1560_bob.max()),
        },
        "te_nb1561_bob_stats": {
            "mean": float(te_nb1561_bob.mean()),
            "std": float(te_nb1561_bob.std()),
            "min": float(te_nb1561_bob.min()),
            "max": float(te_nb1561_bob.max()),
        },
        "te_nb1583_stats": {
            "mean": float(te_nb1583.mean()),
            "std": float(te_nb1583.std()),
            "min": float(te_nb1583.min()),
            "max": float(te_nb1583.max()),
        },
        "in_rae_unb_nb1560_bob": in_rae_nb1560_bob,
        "in_rae_unb_nb1561_bob": in_rae_nb1561_bob,
        "in_rae_unb_nb1583": in_rae_nb1583,
        "honest_lb_anchor_nb1571_crossfit": HONEST_LB_ANCHOR_NB1571,
        "te_nb1560_bob_path": str(te_nb1560_path),
        "te_nb1583_path": str(te_nb1583_path),
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
        "feat_dim", "feat_breakdown",
        "te_nb1560_bob_stats", "te_nb1561_bob_stats", "te_nb1583_stats",
        "in_rae_unb_nb1560_bob", "in_rae_unb_nb1561_bob", "in_rae_unb_nb1583",
        "honest_lb_anchor_nb1571_crossfit",
        "csv_path", "te_nb1583_path", "te_nb1560_bob_path",
    ):
        print(f"  {k}: {res.get(k)}")
