"""nb1533 -- Deploy nb1520 (grid-best blend of nb1512 BoB + nb1500 BoB) to 513 CSV.

nb1520 found in-sample best at w=0.45 (RAE 0.5191) for
    p = w * nb1512_bob_mean + (1-w) * nb1500_bob_mean.
Honest 5-fold cross-fit RAE = 0.5214 (slightly in-sample-optimistic; submit
as alternative candidate alongside nb1510 deploys).

PROTOCOL
    1. te_nb1510_mean.npy already exists (nb1500 BoB MEAN deploy).
    2. Build te_nb1512_bob_mean (CatBoost 4-way 112-col residual deploy):
         - features = AtomPair-top30 + MACCS-top20 + Mordred-top30 +
           ChempropEmbed-top30 + pred_chembl_pec50 + mean_sim (= 112 cols).
           Top-K indices pinned from nb1352 / nb1364 / nb1373 / nb1484.
         - For each outer seed o in {0, 1, 7, 42, 137}, inner_seeds(o) =
           [o*1000+s for s in {0,1,7,42,137}]. Each of the 25 (outer, inner)
           fits is one full CatBoost(MAE, d=4, n=200, lr=0.05, l2=5) trained
           on residual = y_unb - chemprop_aux[unb_idx] using all 253 rows,
           then predicts residual on 513.
         - Per outer, inner-bag = mean over 5 inner-seed 513-row residuals
           (matches nb1512's nb1501_o pooling rule).
         - BoB MEAN = row-mean over the 5 outer 513-row residuals.
         - te_nb1512_bob_mean = te_chemprop_aux + BoB MEAN resid_513.
    3. te_nb1533 = 0.45 * te_nb1512_bob_mean + 0.55 * te_nb1510_mean
    4. submissions/nb1533_deploy_nb1520.csv  (SMILES, Molecule Name, pEC50)
       data/processed/te_nb1533.npy            (513,) float32
       data/processed/te_nb1512_bob_mean.npy   (513,) float32  (byproduct)

Caveat: in-sample 0.5191, cross-fit 0.5214 (-0.0023 sub-margin vs nb1500
BoB mean 0.5236, within 0.003 decision margin).
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
from catboost import CatBoostRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1533"
DEPLOY_OF = "nb1520"

W_NB1512 = 0.45
W_NB1500 = 0.55

ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB1510_MEAN_TE = DATA_PROCESSED / "te_nb1510_mean.npy"

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
MORDRED_TE_PATH = Path("C:/pxr_artifacts/nb1030/X_mordred_test.npy")
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"

NB1352_SUMMARY = DATA_PROCESSED / "nb1352_summary.json"
NB1364_SUMMARY = DATA_PROCESSED / "nb1364_summary.json"
NB1373_SUMMARY = DATA_PROCESSED / "nb1373_summary.json"
NB1484_SUMMARY = DATA_PROCESSED / "nb1484_summary.json"
NB1520_SUMMARY = DATA_PROCESSED / "nb1520_summary.json"

OUTER_SEEDS = [0, 1, 7, 42, 137]
INNER_BASE_SEEDS = [0, 1, 7, 42, 137]

TE_NB1512_OUT = DATA_PROCESSED / "te_nb1512_bob_mean.npy"
TE_NB1533_OUT = DATA_PROCESSED / f"te_{TAG}.npy"

SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"
SUB_OUT = SUB_DIR / f"{TAG}_deploy_nb1520.csv"

NB1520_INSAMPLE_REF = 0.5191
NB1520_CROSSFIT_REF = 0.5214


def _cat_params(seed: int) -> dict:
    return dict(
        loss_function="MAE",
        depth=4,
        iterations=200,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
    )


def _load_mordred_te(n_test_expected: int) -> np.ndarray:
    if not MORDRED_TE_PATH.exists():
        raise FileNotFoundError(f"Mordred cache missing: {MORDRED_TE_PATH}")
    X = np.load(MORDRED_TE_PATH).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"Mordred test shape mismatch: {X.shape}")
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _extract_embed_top_idx_from_nb1484(sum_1484: dict) -> np.ndarray:
    for f in sum_1484["families"]:
        if f["family"] == "ChempropEmbed":
            return np.array(f["top_idx_ranked"], dtype=int)
    raise KeyError("ChempropEmbed entry not found in nb1484_summary.json")


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Deploy nb1520 (in-sample best grid w=0.45 of nb1512 + nb1500 BoB)")
    print(f"       w_nb1512 = {W_NB1512}   w_nb1500 = {W_NB1500}")
    print(f"       in-sample anchor  = {NB1520_INSAMPLE_REF:.4f}")
    print(f"       cross-fit anchor  = {NB1520_CROSSFIT_REF:.4f}  "
          f"(sub-margin vs 0.5236 nb1500)")
    print(f"       outer seeds       = {OUTER_SEEDS}")
    print(f"       inner base seeds  = {INNER_BASE_SEEDS}")
    print(f"       inner_seeds(o)    = [o*1000+s for s in base]")
    print("=" * 78)

    # ---- Load test + truth + anchor + nb1510 mean ----
    te_df = load_test()
    n_test = len(te_df)
    smiles_col = "smiles" if "smiles" in te_df.columns else "SMILES"
    name_col = None
    for cand in ("molecule_name", "Molecule Name", "name", "compound_id"):
        if cand in te_df.columns:
            name_col = cand
            break
    if name_col is None:
        raise KeyError(f"No Molecule-Name column found; cols = {list(te_df.columns)}")
    print(f"[load] n_test={n_test}  smiles_col={smiles_col!r}  name_col={name_col!r}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape != (n_test,):
        raise ValueError(f"chemprop_aux te shape mismatch: {te_anchor_513.shape}")
    print(f"[load] te_chemprop_aux: mean={te_anchor_513.mean():.4f}  "
          f"std={te_anchor_513.std():.4f}")

    if not NB1510_MEAN_TE.exists():
        raise FileNotFoundError(f"nb1510 mean te missing: {NB1510_MEAN_TE}")
    te_nb1510_mean = np.load(NB1510_MEAN_TE).astype(np.float64)
    if te_nb1510_mean.shape != (n_test,):
        raise ValueError(f"te_nb1510_mean shape mismatch: {te_nb1510_mean.shape}")
    print(f"[load] te_nb1510_mean:  mean={te_nb1510_mean.mean():.4f}  "
          f"std={te_nb1510_mean.std():.4f}  "
          f"min={te_nb1510_mean.min():.4f}  max={te_nb1510_mean.max():.4f}")

    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(UNB_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual_unb = y_unb - anchor_unb
    print(f"[load] n_unb={n_unb}  in_RAE(anchor) = {rae_anchor:.4f}")
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- Load pinned SHAP top-idx (matches nb1512) ----
    for p in (NB1352_SUMMARY, NB1364_SUMMARY, NB1373_SUMMARY, NB1484_SUMMARY):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}")
    with open(NB1352_SUMMARY) as f:
        sum_1352 = json.load(f)
    with open(NB1364_SUMMARY) as f:
        sum_1364 = json.load(f)
    with open(NB1373_SUMMARY) as f:
        sum_1373 = json.load(f)
    with open(NB1484_SUMMARY) as f:
        sum_1484 = json.load(f)
    top_maccs = np.array(sum_1352["top_maccs_bit_indices_ranked"], dtype=int)
    top_mord = np.array(sum_1364["top_mordred_col_indices_ranked"], dtype=int)
    top_ap = np.array(sum_1373["top_atompair_bit_indices_ranked"], dtype=int)
    top_emb = _extract_embed_top_idx_from_nb1484(sum_1484)
    n_ap, n_maccs, n_mord, n_emb = (
        len(top_ap), len(top_maccs), len(top_mord), len(top_emb)
    )
    print(f"[pin] AP={n_ap}  MACCS={n_maccs}  Mord={n_mord}  Emb={n_emb}")

    # ---- Build 112-col feature matrix on 513 (and slice to 253) ----
    X_ap_te = np.load(ATOMPAIR_TE_PATH)
    X_maccs_te = np.load(MACCS_TE_PATH)
    X_mord_te = _load_mordred_te(n_test)
    X_emb_te = np.load(CHEMPROP_EMBED_TE_PATH).astype(np.float32)
    X_emb_te = np.where(np.isfinite(X_emb_te), X_emb_te, 0.0).astype(np.float32)

    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)

    X_513 = np.concatenate(
        [
            X_ap_te[:, top_ap].astype(np.float32),
            X_maccs_te[:, top_maccs].astype(np.float32),
            X_mord_te[:, top_mord].astype(np.float32),
            X_emb_te[:, top_emb].astype(np.float32),
            pred_chembl_513.reshape(-1, 1),
            sim_chembl_513.reshape(-1, 1),
        ],
        axis=1,
    ).astype(np.float32)
    feat_dim = X_513.shape[1]
    expected_dim = n_ap + n_maccs + n_mord + n_emb + 2
    if feat_dim != expected_dim:
        raise ValueError(f"feat_dim {feat_dim} != expected {expected_dim}")
    X_unb = X_513[unb_idx]
    print(f"[feat] X_513 = {X_513.shape}   X_unb = {X_unb.shape}")

    # ---- Outer x Inner CatBoost deploy fits ----
    print("\n" + "=" * 78)
    print("OUTER x INNER CATBOOST DEPLOY FIT on 253 unb residual + PREDICT 513")
    print("=" * 78)
    n_outer = len(OUTER_SEEDS)
    n_inner = len(INNER_BASE_SEEDS)
    outer_resid_513 = np.zeros((n_outer, n_test), dtype=np.float64)
    per_outer_records = []
    per_outer_inner_seeds: list[list[int]] = []

    for oi, o in enumerate(OUTER_SEEDS):
        inner_seeds = [int(o * 1000 + s) for s in INNER_BASE_SEEDS]
        per_outer_inner_seeds.append(inner_seeds)
        t_outer = time.time()
        print(f"\n   --- outer seed {o}  inner seeds = {inner_seeds} ---")
        inner_resid_513 = np.zeros((n_inner, n_test), dtype=np.float64)
        per_seed_in_rae: list[float] = []
        for ii, isd in enumerate(inner_seeds):
            ts = time.time()
            mdl = CatBoostRegressor(**_cat_params(isd))
            mdl.fit(X_unb, residual_unb)
            r_513 = mdl.predict(X_513)
            inner_resid_513[ii] = r_513
            # in-sample diagnostic
            te_seed = te_anchor_513 + r_513
            in_rae = float(rae(y_unb, te_seed[unb_idx]))
            per_seed_in_rae.append(in_rae)
            print(f"     outer {o:3d}  inner {isd:6d}  resid_513 std={r_513.std():.4f}  "
                  f"in_RAE_unb(in-sample)={in_rae:.4f}  wall={time.time()-ts:.1f}s")
        # inner-bag mean = nb1501_o aggregation rule
        blend_o_513 = inner_resid_513.mean(axis=0)
        outer_resid_513[oi] = blend_o_513
        te_blend_o = te_anchor_513 + blend_o_513
        in_rae_blend = float(rae(y_unb, te_blend_o[unb_idx]))
        per_outer_records.append({
            "outer_seed": int(o),
            "inner_seeds": inner_seeds,
            "per_seed_in_RAE_unb": per_seed_in_rae,
            "blend_resid_513_mean": float(blend_o_513.mean()),
            "blend_resid_513_std": float(blend_o_513.std()),
            "in_RAE_unb_blend_in_sample": in_rae_blend,
            "wall_sec": round(time.time() - t_outer, 2),
        })
        print(f"   outer {o:3d}  inner-bag mean  resid_513 mean={blend_o_513.mean():+.4f}  "
              f"std={blend_o_513.std():.4f}  in_RAE_unb(in-sample)={in_rae_blend:.4f}  "
              f"wall={time.time()-t_outer:.1f}s")

    # ---- BoB MEAN over outer axis ----
    bob_mean_resid_513 = outer_resid_513.mean(axis=0)
    te_nb1512_bob_mean = te_anchor_513 + bob_mean_resid_513
    in_rae_nb1512_bob_mean = float(rae(y_unb, te_nb1512_bob_mean[unb_idx]))
    print("\n" + "=" * 78)
    print("BoB MEAN over 5 per-outer 513-row residuals")
    print("=" * 78)
    print(f"[nb1512_bob_mean] te mean={te_nb1512_bob_mean.mean():.4f}  "
          f"std={te_nb1512_bob_mean.std():.4f}  "
          f"min={te_nb1512_bob_mean.min():.4f}  max={te_nb1512_bob_mean.max():.4f}")
    print(f"[nb1512_bob_mean] in_RAE_unb(in-sample) = {in_rae_nb1512_bob_mean:.4f}")

    # Save nb1512 deploy byproduct
    np.save(TE_NB1512_OUT, te_nb1512_bob_mean.astype(np.float32))
    print(f"[save] {TE_NB1512_OUT}")

    # ---- Final blend: 0.45 * nb1512 + 0.55 * nb1510_mean ----
    te_nb1533 = W_NB1512 * te_nb1512_bob_mean + W_NB1500 * te_nb1510_mean
    in_rae_nb1533 = float(rae(y_unb, te_nb1533[unb_idx]))
    te_stats = {
        "mean": float(te_nb1533.mean()),
        "std": float(te_nb1533.std()),
        "min": float(te_nb1533.min()),
        "max": float(te_nb1533.max()),
    }
    print("\n" + "=" * 78)
    print(f"FINAL BLEND  te_nb1533 = {W_NB1512}*te_nb1512 + {W_NB1500}*te_nb1510_mean")
    print("=" * 78)
    print(f"[te_nb1533] mean={te_stats['mean']:.4f}  std={te_stats['std']:.4f}  "
          f"min={te_stats['min']:.4f}  max={te_stats['max']:.4f}")
    print(f"[te_nb1533] in_RAE_unb(in-sample) = {in_rae_nb1533:.4f}")
    print(f"            nb1520 in-sample ref  = {NB1520_INSAMPLE_REF:.4f}")
    print(f"            nb1520 cross-fit ref  = {NB1520_CROSSFIT_REF:.4f}")

    np.save(TE_NB1533_OUT, te_nb1533.astype(np.float32))
    print(f"[save] {TE_NB1533_OUT}")

    # ---- Submission CSV ----
    SUB_DIR.mkdir(parents=True, exist_ok=True)
    smiles_arr = te_df[smiles_col].astype(str).to_numpy()
    names_arr = te_df[name_col].astype(str).to_numpy()
    out_df = pd.DataFrame({
        "SMILES": smiles_arr,
        "Molecule Name": names_arr,
        "pEC50": te_nb1533.astype(np.float64),
    })
    if len(out_df) != n_test:
        raise ValueError(f"CSV row mismatch: {len(out_df)} vs {n_test}")
    out_df.to_csv(SUB_OUT, index=False)
    print(f"[save] {SUB_OUT}  rows={len(out_df)}  cols={list(out_df.columns)}")

    # ---- Pearson sanity vs nb1520 OOF ----
    pearson_nb1533_vs_nb1520 = None
    nb1520_oof_p = DATA_PROCESSED / "nb1520_best_oof.npy"
    if nb1520_oof_p.exists():
        v = np.load(nb1520_oof_p).astype(np.float64)
        if v.shape == (n_unb,):
            try:
                pearson_nb1533_vs_nb1520 = float(np.corrcoef(
                    te_nb1533[unb_idx], v
                )[0, 1])
                print(f"[sanity] Pearson(te_nb1533[unb], nb1520_best_oof) = "
                      f"{pearson_nb1533_vs_nb1520:.4f}")
            except Exception:
                pass

    summary = {
        "tag": TAG,
        "deploy_of": DEPLOY_OF,
        "w_nb1512": W_NB1512,
        "w_nb1500": W_NB1500,
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "nb1510_mean_path": str(NB1510_MEAN_TE),
        "outer_seeds": OUTER_SEEDS,
        "inner_base_seeds": INNER_BASE_SEEDS,
        "per_outer_inner_seeds": per_outer_inner_seeds,
        "feat_dim": int(feat_dim),
        "feat_breakdown": {
            "atompair": int(n_ap),
            "maccs": int(n_maccs),
            "mordred": int(n_mord),
            "chemprop_embed": int(n_emb),
            "pred_chembl_pec50": 1,
            "mean_sim": 1,
            "total": int(feat_dim),
        },
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "rae_anchor_chemprop_aux_in_RAE_unb": rae_anchor,
        "residual_mean": float(residual_unb.mean()),
        "residual_std": float(residual_unb.std()),
        "per_outer_records": per_outer_records,
        "in_RAE_unb_nb1512_bob_mean_in_sample": in_rae_nb1512_bob_mean,
        "te_nb1512_bob_mean_path": str(TE_NB1512_OUT),
        "te_nb1533_stats": te_stats,
        "in_RAE_unb_nb1533_in_sample": in_rae_nb1533,
        "nb1520_in_sample_ref": NB1520_INSAMPLE_REF,
        "nb1520_cross_fit_ref": NB1520_CROSSFIT_REF,
        "caveat": "in-sample 0.5191; honest 5-fold cross-fit 0.5214; "
                  "sub-margin (-0.0022) vs nb1500 BoB mean 0.5236.",
        "pearson_te_nb1533_unb_vs_nb1520_oof": pearson_nb1533_vs_nb1520,
        "te_nb1533_path": str(TE_NB1533_OUT),
        "csv_path": str(SUB_OUT),
        "wall_sec": round(time.time() - t0, 2),
    }
    sum_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {sum_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_test", "n_unb",
        "rae_anchor_chemprop_aux_in_RAE_unb",
        "in_RAE_unb_nb1512_bob_mean_in_sample",
        "te_nb1533_stats",
        "in_RAE_unb_nb1533_in_sample",
        "nb1520_in_sample_ref",
        "nb1520_cross_fit_ref",
        "pearson_te_nb1533_unb_vs_nb1520_oof",
        "te_nb1512_bob_mean_path",
        "te_nb1533_path",
        "csv_path",
    ):
        print(f"  {k}: {res.get(k)}")
