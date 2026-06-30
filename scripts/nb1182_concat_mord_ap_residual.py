"""nb1182 -- Concat-feature single residual model on [Mordred, AtomPair]
(1533 + 2048 = 3581) over nb1070 anchor.

Hypothesis:
    Parallel to nb1171's concat experiment, but swaps Morgan+RDKit -> AtomPair
    as the second axis. AtomPair fingerprints have stronger bit-level
    orthogonality to Mordred-derived descriptor signal than Morgan does,
    so cross-feature splits (Mordred-Burden x AtomPair-bit) in a single
    shallow LGBM may capture residual structure that a separately-trained
    Mordred + AtomPair 2-way blend cannot.

    Reference numbers (pooled RAE on 253 unblind):
      nb1070 anchor          ~ 0.5790
      nb1153 (Mordred-only)  = 0.5640
      nb1180 (placeholder)   = 0.5600   (used if nb1180 hasn't landed yet)

    Verdict @ 0.003 margin vs nb1153 AND vs nb1180 placeholder.

Protocol per seed s in {0, 1, 7, 42, 137}:
  1. Anchor = nb1070 pred_oof (constant across seeds).
  2. residual = y_unb - nb1070_oof
  3. KFold(n=5, shuffle=True, random_state=s) on 253 unblind.
  4. Shallow LGBM Huber (max_depth=3, num_leaves=7, n_est=80, lr=0.05,
     min_child_samples=20, alpha=1.0) on concat features
     [Mordred (1533), AtomPair (2048)] = (253, 3581); cross-fit
     residual OOF.
  5. pred_corrected_s = nb1070_oof + residual_oof_s; pooled RAE.

Then mean-bag pooled cross-fit RAE = RAE(y_unb, mean over seeds of pred_corr_s).
Median-bag same with np.median.

Outputs:
  data/processed/nb1182_per_seed_corrected_oof.npy  (5, 253) float32
  data/processed/nb1182_mean_bag_oof.npy            (253,) float32
  data/processed/nb1182_median_bag_oof.npy          (253,) float32
  data/processed/nb1182_summary.json
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
from lightgbm import LGBMRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1182"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")
ATOMPAIR_CACHE = DATA_PROCESSED / "te_atompair.npy"   # (513, 2048) uint8
ATOMPAIR_CACHE_2048 = DATA_PROCESSED / "te_atompair2048.npy"  # alt name

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5790
NB1153_MEAN_BAG_REF = 0.5640   # Mordred-only residual on nb1070
NB1180_MEAN_BAG_REF = 0.5600   # placeholder per task spec
MARGIN = 0.003


def _lgbm_params(seed: int) -> dict:
    """Shallow LGBM Huber -- identical capacity to nb1130 / nb1153 / nb1171."""
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


def _median_impute(X: np.ndarray) -> np.ndarray:
    X = np.where(np.isfinite(X), X, np.nan).astype(np.float32)
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X[idx_r, idx_c] = col_med[idx_c]
    return X


def _load_mordred_unblind(n_test_expected: int, unb_idx: np.ndarray) -> np.ndarray:
    """Load cached Mordred (513 x 1533) and slice to unblind rows."""
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(
            f"Mordred cache missing -- run nb1030 first ({mte_p})"
        )
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs "
            f"n_test={n_test_expected}"
        )
    X_te_m = _median_impute(X_te_m)
    return X_te_m[unb_idx]


def _build_atompair_from_smiles(smiles_list, n_bits: int = 2048) -> np.ndarray:
    """Fallback builder: hashed AtomPair fingerprint via rdMolDescriptors."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    X = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bv = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(
            mol, nBits=n_bits
        )
        # extract bit positions
        for bit in bv.GetOnBits():
            X[i, bit] = 1
    return X


def _load_atompair_unblind(
    n_test_expected: int, unb_idx: np.ndarray, te_df
) -> np.ndarray:
    """Load cached AtomPair (513 x 2048) and slice to unblind rows.

    Prefers te_atompair2048.npy then te_atompair.npy; falls back to
    on-the-fly construction from test SMILES if neither is present.
    """
    if ATOMPAIR_CACHE_2048.exists():
        path = ATOMPAIR_CACHE_2048
    elif ATOMPAIR_CACHE.exists():
        path = ATOMPAIR_CACHE
    else:
        path = None

    if path is not None:
        X_te_a = np.load(path).astype(np.float32)
        if X_te_a.shape[0] != n_test_expected:
            raise ValueError(
                f"AtomPair test shape mismatch: {X_te_a.shape} vs "
                f"n_test={n_test_expected}  (path={path})"
            )
        print(f"[feat]   AtomPair source = cached: {path}")
    else:
        print("[feat]   AtomPair cache missing -- constructing from test SMILES")
        smi = (te_df["smiles"] if "smiles" in te_df.columns
               else te_df["SMILES"]).tolist()
        X_te_a = _build_atompair_from_smiles(smi, n_bits=2048).astype(np.float32)
        if X_te_a.shape[0] != n_test_expected:
            raise ValueError(
                f"Built AtomPair shape mismatch: {X_te_a.shape} vs "
                f"n_test={n_test_expected}"
            )
        # write the canonical cache so downstream notebooks can re-use it
        np.save(ATOMPAIR_CACHE_2048, X_te_a.astype(np.uint8))
        print(f"[feat]   wrote {ATOMPAIR_CACHE_2048}")

    # AtomPair bits are 0/1; median-impute is harmless / no-op if no NaNs.
    X_te_a = _median_impute(X_te_a)
    return X_te_a[unb_idx]


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- CONCAT-feature residual LGBM on nb1070 anchor")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - nb1070_pred_oof")
    print(f"          features = Mordred (1533) || AtomPair (2048) = 3581")
    print(f"          LGBM: max_depth=3, num_leaves=7, n_est=80, lr=0.05, "
          f"min_child_samples=20, obj=huber(alpha=1.0)")
    print("=" * 78)

    te = load_test()
    n_test = len(te)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    anchor_path = DATA_PROCESSED / f"{ANCHOR}_pred_oof.npy"
    if not anchor_path.exists():
        raise FileNotFoundError(
            f"{anchor_path} not found; required anchor OOF "
            f"(run nb1070 first)."
        )
    anchor_oof = np.load(anchor_path).astype(np.float64)
    if anchor_oof.shape[0] != n_unb:
        raise ValueError(
            f"{anchor_path} shape mismatch: "
            f"{anchor_oof.shape} vs n_unb={n_unb}"
        )
    rae_anchor = float(rae(y_unb, anchor_oof))
    print(f"[load] {ANCHOR}_pred_oof.npy shape={anchor_oof.shape}  "
          f"pooled RAE = {rae_anchor:.4f}  (ref ~{NB1070_REF_POOLED:.4f})")

    residual = y_unb - anchor_oof
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    print(f"[feat] loading Mordred cache + slicing to unblind ...")
    X_mord = _load_mordred_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat]   Mordred  unblind shape = {X_mord.shape}")
    print(f"[feat] loading AtomPair cache + slicing to unblind ...")
    X_ap = _load_atompair_unblind(
        n_test_expected=n_test, unb_idx=unb_idx, te_df=te
    )
    print(f"[feat]   AtomPair unblind shape = {X_ap.shape}")

    X_unb = np.concatenate([X_mord, X_ap], axis=1).astype(np.float32)
    print(f"[feat] CONCAT X_unb shape = {X_unb.shape}  "
          f"({X_mord.shape[1]} + {X_ap.shape[1]} = {X_unb.shape[1]})")

    # Final safety re-impute across the concat (covers any cross-cache
    # residual NaNs).
    if not np.all(np.isfinite(X_unb)):
        X_unb = _median_impute(X_unb)
        print("[feat] CONCAT had non-finite cells -> median-imputed")

    print("\n" + "-" * 78)
    print("PER-SEED RESIDUAL CROSS-FIT (depth=3 shallow, CONCAT 3581)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        resid_oof_s = _residual_cross_fit_one_seed(X_unb, residual, s)
        pred_corr_s = anchor_oof + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_nb1070": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_nb1070 = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}  "
          f"(d_vs_nb1070 = {rae_median_bag - rae_anchor:+.4f})")
    print(f"   nb1153 mean_bag ref    = {NB1153_MEAN_BAG_REF:.4f}  "
          f"(Mordred-only residual)")
    print(f"   nb1180 mean_bag ref    = {NB1180_MEAN_BAG_REF:.4f}  "
          f"(placeholder)")

    beats_nb1070 = rae_mean_bag < rae_anchor - MARGIN
    beats_nb1153 = rae_mean_bag < NB1153_MEAN_BAG_REF - MARGIN
    beats_nb1180 = rae_mean_bag < NB1180_MEAN_BAG_REF - MARGIN

    if beats_nb1153 and beats_nb1180:
        verdict = "CONCAT_MORD_AP_BEATS_BOTH_NB1153_AND_NB1180"
    elif beats_nb1153 and not beats_nb1180:
        verdict = "CONCAT_MORD_AP_BEATS_NB1153_BUT_NOT_NB1180_PLACEHOLDER"
    elif beats_nb1180 and not beats_nb1153:
        verdict = "CONCAT_MORD_AP_BEATS_NB1180_PLACEHOLDER_BUT_NOT_NB1153"
    elif beats_nb1070:
        verdict = "CONCAT_MORD_AP_HELPS_NB1070_BUT_NOT_NB1153_OR_NB1180"
    elif abs(rae_mean_bag - NB1153_MEAN_BAG_REF) < MARGIN:
        verdict = "CONCAT_MORD_AP_TIES_NB1153_NO_NEW_CROSS_FEATURE_SIGNAL"
    else:
        verdict = "CONCAT_MORD_AP_HURTS_NB1153"
    print(f"   verdict                = {verdict}")

    np.save(DATA_PROCESSED / f"{TAG}_per_seed_corrected_oof.npy",
            per_seed_corrected.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof.npy",
            mean_bag_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_median_bag_oof.npy",
            median_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_per_seed_corrected_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof.npy'}")
    print(f"[save] {DATA_PROCESSED / f'{TAG}_median_bag_oof.npy'}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "feature_source": "concat_mordred_1533_plus_atompair_2048",
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "feature_dim_mordred": int(X_mord.shape[1]),
        "feature_dim_atompair": int(X_ap.shape[1]),
        "lgbm_max_depth": 3,
        "lgbm_num_leaves": 7,
        "lgbm_n_estimators": 80,
        "lgbm_learning_rate": 0.05,
        "lgbm_min_child_samples": 20,
        "lgbm_alpha_huber": 1.0,
        "rae_anchor_nb1070": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_nb1070": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb1153": rae_mean_bag - NB1153_MEAN_BAG_REF,
        "delta_mean_bag_vs_nb1180": rae_mean_bag - NB1180_MEAN_BAG_REF,
        "margin": MARGIN,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1153": bool(beats_nb1153),
        "beats_nb1180_placeholder": bool(beats_nb1180),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1153_mean_bag_ref": NB1153_MEAN_BAG_REF,
        "nb1180_mean_bag_ref_placeholder": NB1180_MEAN_BAG_REF,
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
    for k in ("rae_anchor_nb1070", "per_seed_rae",
              "rae_per_seed_mean", "rae_per_seed_median",
              "rae_per_seed_std",
              "rae_mean_bag", "rae_median_bag",
              "delta_mean_bag_vs_nb1070",
              "delta_mean_bag_vs_nb1153",
              "delta_mean_bag_vs_nb1180",
              "beats_nb1070", "beats_nb1153", "beats_nb1180_placeholder",
              "verdict"):
        print(f"  {k}: {res.get(k)}")
