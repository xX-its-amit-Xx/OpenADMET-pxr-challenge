"""nb1300 -- ChemBERTa-77M-MTR embeddings as DIRECT residual features.

Hypothesis:
    Prior ChemBERTa attempts (nb1601/nb1602 in feedback_2026_06_01_session_summary)
    used the embeddings inside SLSQP meta-stacks and collapsed to 0-weight.
    Try the embeddings as a DIRECT residual-LGBM feature axis (concatenated
    with MACCS-167) instead of a stack contributor.  If the foundation model
    carries scaffold-diverse signal beyond MACCS dictionary + curated PXR
    bioactivity counts, a shallow Huber bag on (MACCS-167 + ChemBERTa-emb)
    should beat nb1183 (MACCS-only, 0.5513) and ideally nb1242 (0.5431).

Cached embeddings:
    data/processed/tr_chemberta.npy   (4139, 384) float32  -- ChemBERTa-77M-MTR [CLS]
    data/processed/te_chemberta.npy   ( 513, 384) float32  -- ChemBERTa-77M-MTR [CLS]
(Built by nb213_chemberta_embed.py via HF DeepChem/ChemBERTa-77M-MTR; CPU.)

Protocol:
    1. Load anchor nb1070_pred_oof on 253 unblind rows.
    2. residual = y_unb - nb1070_pred_oof
    3. Slice te_chemberta to 253 unblind rows  -> X_emb (253, 384)
       Slice te_maccs to 253 unblind rows      -> X_maccs (253, 167)
    4. OPTION A (preferred at d=384): full embedding concat -> (253, 551)
       OPTION B (PCA-100): if d > 200 we also try a top-100 PCA reduction
       (fit on FULL pool: train + test) -> (253, 267)
    5. 5-seed bag shallow LGBM Huber (depth=3, leaves=7, n_est=80, lr=0.05,
       min_child=20, alpha=1.0).  5-fold cross-fit per seed.  Same capacity
       as nb1183 / nb1242.
    6. Pick the better variant (raw vs PCA-100) by pooled mean-bag RAE.
    7. Verdict at 0.003 margin vs nb1183 (0.5513) and vs nb1242 (0.5431).

Orthogonality probe:
    Pearson(mean_bag_oof, nb1242_mean_bag_oof) on 253 unblind rows -- low
    correlation supports that the foundation-embedding axis is orthogonal
    to the ChEMBL bioactivity kNN axis.

NO deploy refit -- 253-only honest cross-fit diagnostic.

Outputs:
    data/processed/nb1300_summary.json
    data/processed/nb1300_mean_bag_oof.npy       (253,)    float32
    data/processed/nb1300_per_seed_corrected_oof.npy (5, 253) float32
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
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1300"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMBERTA_TR_PATH = DATA_PROCESSED / "tr_chemberta.npy"
CHEMBERTA_TE_PATH = DATA_PROCESSED / "te_chemberta.npy"

PCA_DIM = 100

NB1070_REF = 0.5771
NB1183_REF = 0.5513    # MACCS residual bag
NB1242_REF = 0.5431    # ChEMBL-kNN residual bag (current best feature axis)
DECISION_MARGIN = 0.003


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


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _bag_over_seeds(
    X_unb: np.ndarray,
    residual: np.ndarray,
    anchor_oof: np.ndarray,
    y_unb: np.ndarray,
    seeds: list[int],
    label: str,
) -> dict:
    n_unb = len(y_unb)
    per_seed_corrected = np.zeros((len(seeds), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    print(f"\n[{label}] feat dim = {X_unb.shape[1]}  n_unb = {n_unb}")
    for i, s in enumerate(seeds):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor_oof + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": rae_s - float(rae(y_unb, anchor_oof)),
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"|resid|.std = {resid_oof_s.std():.3f}")
    mean_bag = per_seed_corrected.mean(axis=0)
    median_bag = np.median(per_seed_corrected, axis=0)
    rae_mean = float(rae(y_unb, mean_bag))
    rae_median = float(rae(y_unb, median_bag))
    arr = np.array(per_seed_rae)
    print(f"   pooled mean_bag  RAE = {rae_mean:.4f}")
    print(f"   pooled median_bag RAE = {rae_median:.4f}")
    return {
        "label": label,
        "feature_dim": int(X_unb.shape[1]),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "per_seed_mean": float(arr.mean()),
        "per_seed_median": float(np.median(arr)),
        "per_seed_std": float(arr.std()),
        "per_seed_min": float(arr.min()),
        "per_seed_max": float(arr.max()),
        "rae_mean_bag": rae_mean,
        "rae_median_bag": rae_median,
        "per_seed_corrected": per_seed_corrected,
        "mean_bag_oof": mean_bag,
        "median_bag_oof": median_bag,
    }


def _orthogonality_pearson(a: np.ndarray, ref_path: Path) -> float | None:
    if not ref_path.exists():
        return None
    ref = np.load(ref_path).astype(np.float64)
    if ref.shape[0] != a.shape[0]:
        return None
    if a.std() == 0 or ref.std() == 0:
        return float("nan")
    return float(np.corrcoef(a.astype(np.float64), ref)[0, 1])


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ChemBERTa-77M-MTR embeddings as DIRECT residual features")
    print(f"        anchor = {ANCHOR}; seeds = {RESID_SEEDS}; folds = {RESID_FOLDS}")
    print("=" * 78)

    # ---- Inputs ----
    needed = {
        "te_chemberta.npy": CHEMBERTA_TE_PATH,
        "tr_chemberta.npy": CHEMBERTA_TR_PATH,
        "te_maccs.npy":     MACCS_TE_PATH,
        "_audit_unblind_idx.npy": DATA_PROCESSED / "_audit_unblind_idx.npy",
        "_audit_unblind_y.npy":   DATA_PROCESSED / "_audit_unblind_y.npy",
        f"{ANCHOR}_pred_oof.npy": DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy",
    }
    missing = [k for k, p in needed.items() if not p.exists()]
    if missing:
        summary = {
            "tag": TAG,
            "status": "MISSING_INPUTS",
            "missing": missing,
            "note": (
                "ChemBERTa embedding cache or anchor not found.  Skipping "
                "gracefully -- no transformers re-embed attempted at this stage."
            ),
            "wall_sec": round(time.time() - t0, 2),
        }
        with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("MISSING:", missing)
        return summary

    te = load_test()
    n_test = len(te)
    unb_idx = np.load(needed["_audit_unblind_idx.npy"])
    y_unb = np.load(needed["_audit_unblind_y.npy"]).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_oof = np.load(needed[f"{ANCHOR}_pred_oof.npy"]).astype(np.float64)
    rae_anchor = float(rae(y_unb, anchor_oof))
    residual = y_unb - anchor_oof
    print(f"[anchor] {ANCHOR} pooled RAE = {rae_anchor:.4f}  "
          f"(ref ~{NB1070_REF:.4f})")
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Features ----
    tr_emb = np.load(CHEMBERTA_TR_PATH).astype(np.float32)
    te_emb = np.load(CHEMBERTA_TE_PATH).astype(np.float32)
    if te_emb.shape[0] != n_test:
        raise ValueError(
            f"te_chemberta shape mismatch: {te_emb.shape} vs n_test={n_test}"
        )
    raw_dim = int(te_emb.shape[1])
    print(f"[feat] tr_emb {tr_emb.shape}  te_emb {te_emb.shape}  "
          f"raw_dim = {raw_dim}")
    print(f"       embedding source: DeepChem/ChemBERTa-77M-MTR (cached, CPU-built)")
    print(f"       cache paths: {CHEMBERTA_TR_PATH.name}, {CHEMBERTA_TE_PATH.name}")

    # PCA reduction (fit on pooled train + test embeddings -- unsupervised, leak-safe)
    pooled = np.concatenate([tr_emb, te_emb], axis=0).astype(np.float32)
    eff_pca = min(PCA_DIM, pooled.shape[1] - 1, pooled.shape[0] - 1)
    pca = PCA(n_components=eff_pca, random_state=0)
    pooled_red = pca.fit_transform(pooled).astype(np.float32)
    te_emb_red = pooled_red[len(tr_emb):]
    pca_var = float(pca.explained_variance_ratio_.sum())
    print(f"[pca]  pooled fit -> top-{eff_pca}  cum-var = {pca_var:.4f}")
    print(f"       te_emb_red {te_emb_red.shape}")

    X_maccs_te = np.load(MACCS_TE_PATH)
    if X_maccs_te.shape[0] != n_test:
        raise ValueError(
            f"MACCS te cache shape mismatch: {X_maccs_te.shape} vs n_test={n_test}"
        )
    X_maccs_unb = X_maccs_te[unb_idx].astype(np.float32)
    X_emb_unb_raw = te_emb[unb_idx].astype(np.float32)
    X_emb_unb_red = te_emb_red[unb_idx].astype(np.float32)

    # ---- Variant A: MACCS-167 + raw 384-d ChemBERTa ----
    X_A = np.concatenate([X_maccs_unb, X_emb_unb_raw], axis=1).astype(np.float32)
    # ---- Variant B: MACCS-167 + PCA-100 ChemBERTa ----
    X_B = np.concatenate([X_maccs_unb, X_emb_unb_red], axis=1).astype(np.float32)

    print("\n" + "-" * 78)
    print("RESIDUAL BAG -- VARIANT A: MACCS-167 + raw ChemBERTa-384")
    print("-" * 78)
    bag_A = _bag_over_seeds(
        X_A, residual, anchor_oof, y_unb, RESID_SEEDS, "A_raw_384"
    )

    print("\n" + "-" * 78)
    print(f"RESIDUAL BAG -- VARIANT B: MACCS-167 + PCA-{eff_pca} ChemBERTa")
    print("-" * 78)
    bag_B = _bag_over_seeds(
        X_B, residual, anchor_oof, y_unb, RESID_SEEDS, f"B_pca_{eff_pca}"
    )

    # ---- Pick winner by mean_bag RAE ----
    if bag_A["rae_mean_bag"] <= bag_B["rae_mean_bag"]:
        winner = bag_A
        winner_tag = "A_raw_384"
    else:
        winner = bag_B
        winner_tag = f"B_pca_{eff_pca}"
    print("\n" + "-" * 78)
    print(f"WINNER VARIANT = {winner_tag}  "
          f"(rae_mean_bag = {winner['rae_mean_bag']:.4f})")
    print("-" * 78)

    rae_mean_bag = winner["rae_mean_bag"]
    rae_median_bag = winner["rae_median_bag"]
    mean_bag_oof = winner["mean_bag_oof"]
    median_bag_oof = winner["median_bag_oof"]
    per_seed_corrected = winner["per_seed_corrected"]

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_REF - DECISION_MARGIN
    beats_nb1242 = rae_mean_bag < NB1242_REF - DECISION_MARGIN

    if beats_nb1242:
        verdict = "CHEMBERTA_RESIDUAL_BEATS_NB1242_NEW_PRIMARY_CANDIDATE"
    elif beats_nb1183:
        verdict = "CHEMBERTA_RESIDUAL_BEATS_NB1183_BUT_NOT_NB1242"
    elif beats_nb1070:
        verdict = "CHEMBERTA_RESIDUAL_HELPS_NB1070_BUT_NOT_NB1183"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "CHEMBERTA_RESIDUAL_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "CHEMBERTA_RESIDUAL_HURTS_NB1070"
    print(f"verdict = {verdict}")

    # ---- Orthogonality vs nb1242 (and nb1183) ----
    print("\n" + "-" * 78)
    print("ORTHOGONALITY PROBE")
    print("-" * 78)
    pearson_vs_nb1242 = _orthogonality_pearson(
        mean_bag_oof, DATA_PROCESSED / "nb1242_mean_bag_oof.npy"
    )
    pearson_vs_nb1183 = _orthogonality_pearson(
        mean_bag_oof, DATA_PROCESSED / "nb1183_mean_bag_oof.npy"
    )
    print(f"   pearson(nb1300_mean_bag, nb1242_mean_bag) = {pearson_vs_nb1242}")
    print(f"   pearson(nb1300_mean_bag, nb1183_mean_bag) = {pearson_vs_nb1183}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "embedding_model": "DeepChem/ChemBERTa-77M-MTR",
        "embedding_cache_train": str(CHEMBERTA_TR_PATH),
        "embedding_cache_test": str(CHEMBERTA_TE_PATH),
        "embedding_source": "pre-cached (built by nb213_chemberta_embed.py, CPU)",
        "raw_embedding_dim": raw_dim,
        "pca_dim": eff_pca,
        "pca_explained_var_ratio": pca_var,
        "maccs_dim": int(X_maccs_unb.shape[1]),
        "variant_A_feature_dim": int(X_A.shape[1]),
        "variant_B_feature_dim": int(X_B.shape[1]),
        "winner_variant": winner_tag,
        "winner_feature_dim": int(winner["feature_dim"]),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_huber_alpha": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "variant_A": {
            "feature_dim": bag_A["feature_dim"],
            "per_seed_rae": bag_A["per_seed_rae"],
            "rae_mean_bag": bag_A["rae_mean_bag"],
            "rae_median_bag": bag_A["rae_median_bag"],
            "per_seed_mean": bag_A["per_seed_mean"],
            "per_seed_median": bag_A["per_seed_median"],
            "per_seed_std": bag_A["per_seed_std"],
            "per_seed_min": bag_A["per_seed_min"],
            "per_seed_max": bag_A["per_seed_max"],
            "per_seed_records": bag_A["per_seed_records"],
        },
        "variant_B": {
            "feature_dim": bag_B["feature_dim"],
            "per_seed_rae": bag_B["per_seed_rae"],
            "rae_mean_bag": bag_B["rae_mean_bag"],
            "rae_median_bag": bag_B["rae_median_bag"],
            "per_seed_mean": bag_B["per_seed_mean"],
            "per_seed_median": bag_B["per_seed_median"],
            "per_seed_std": bag_B["per_seed_std"],
            "per_seed_min": bag_B["per_seed_min"],
            "per_seed_max": bag_B["per_seed_max"],
            "per_seed_records": bag_B["per_seed_records"],
        },
        "per_seed_rae": winner["per_seed_rae"],
        "rae_per_seed_mean": winner["per_seed_mean"],
        "rae_per_seed_median": winner["per_seed_median"],
        "rae_per_seed_std": winner["per_seed_std"],
        "rae_per_seed_min": winner["per_seed_min"],
        "rae_per_seed_max": winner["per_seed_max"],
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_REF,
        "delta_mean_bag_vs_nb1242": rae_mean_bag - NB1242_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "beats_nb1242": bool(beats_nb1242),
        "verdict": verdict,
        "pearson_vs_nb1242_mean_bag": pearson_vs_nb1242,
        "pearson_vs_nb1183_mean_bag": pearson_vs_nb1183,
        "nb1070_ref_pooled": NB1070_REF,
        "nb1183_ref": NB1183_REF,
        "nb1242_ref": NB1242_REF,
        "decision_margin": DECISION_MARGIN,
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
        "embedding_model", "raw_embedding_dim", "pca_dim",
        "winner_variant", "winner_feature_dim",
        "rae_anchor_nb1070", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_nb1070",
        "delta_mean_bag_vs_nb1183",
        "delta_mean_bag_vs_nb1242",
        "beats_nb1070", "beats_nb1183", "beats_nb1242",
        "pearson_vs_nb1242_mean_bag",
        "pearson_vs_nb1183_mean_bag",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
