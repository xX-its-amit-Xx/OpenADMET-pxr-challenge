"""nb1214 -- RandomForestRegressor residual bag on nb1070 anchor, MACCS-166 features.

Hypothesis:
    RF averages independent (bootstrapped + feature-subsampled) trees; on the
    253-row x 166-bit MACCS substrate this may have lower variance than the
    boosted LGBM (nb1183) and ExtraTrees (nb1212) variants.  Probing whether
    the curated-dictionary MACCS signal benefits from a third orthogonal model
    class.  RF uses bagging + per-split feature subsampling -- a different
    variance-reduction mechanism than LGBM's sequential boosting or ExtraTrees'
    fully-random splits.

Protocol per seed s in {0, 1, 7, 42, 137}:
  1. Anchor = nb1070 pred_oof (constant across seeds).
  2. residual = y_unb - nb1070_oof
  3. KFold(n=5, shuffle=True, random_state=s) on 253 unblind rows.
  4. RandomForestRegressor(n_estimators=300, max_depth=8,
     min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=s)
     on MACCS-166 (drop bit 0 padding) sliced to unblind.
  5. pred_corrected_s = nb1070_oof + residual_oof_s; pooled RAE.

Mean-bag pooled cross-fit RAE = RAE(y_unb, mean over seeds of pred_corr_s).
Verdict at 0.003 margin vs nb1183 (0.5513).

Orthogonality + 3-way blend probes:
  - Pearson correlation between mean-bag corrected OOF and nb1183 / nb1212.
  - Naive 3-way mean of (nb1183, nb1212, nb1214) if all three exist;
    otherwise pairwise mean of available pair.

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1214_per_seed_corrected_oof.npy  (5, 253) float32
  data/processed/nb1214_mean_bag_oof.npy            (253,)   float32
  data/processed/nb1214_summary.json
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
from sklearn.ensemble import RandomForestRegressor

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1214"
ANCHOR = "nb1070"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]

MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"   # cached (513, 167) uint8

# Reference numbers (pooled RAE on 253 unblind).
NB1070_REF_POOLED = 0.5771
NB1183_MEAN_BAG_REF = 0.5513   # LGBM-Huber residual bag on nb1070 / MACCS
NB1212_MEAN_BAG_REF = None     # ExtraTrees variant -- read live if present.
DECISION_MARGIN = 0.003


def _rf_params(seed: int) -> dict:
    return dict(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        max_features=0.5,
        n_jobs=-1,
        random_state=seed,
    )


def _residual_cross_fit_one_seed(
    X: np.ndarray, residual: np.ndarray, seed: int
) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = RandomForestRegressor(**_rf_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _load_maccs166_unblind(n_test_expected: int, unb_idx: np.ndarray) -> np.ndarray:
    """Load cached MACCS test matrix, slice to 166 bits (drop bit 0 padding)."""
    if not MACCS_TE_PATH.exists():
        raise FileNotFoundError(f"MACCS test cache missing: {MACCS_TE_PATH}")
    X_te = np.load(MACCS_TE_PATH)
    if X_te.shape[0] != n_test_expected:
        raise ValueError(
            f"MACCS test cache shape mismatch: {X_te.shape} "
            f"vs n_test={n_test_expected}"
        )
    if X_te.shape[1] == 167:
        # Drop bit 0 (padding) -> MACCS-166
        X_te = X_te[:, 1:]
    elif X_te.shape[1] != 166:
        raise ValueError(
            f"MACCS test cache unexpected width: {X_te.shape[1]} "
            f"(expected 166 or 167)"
        )
    X_unb = X_te[unb_idx].astype(np.float32)
    return X_unb


def _orthogonality_probe(mean_bag_oof: np.ndarray) -> dict:
    out = {}
    for ref_tag in ("nb1183", "nb1212"):
        p = DATA_PROCESSED / f"{ref_tag}_mean_bag_oof.npy"
        if not p.exists():
            out[f"pearson_vs_{ref_tag}_mean_bag"] = None
            out[f"{ref_tag}_probe_error"] = f"missing {p}"
            continue
        try:
            ref = np.load(p).astype(np.float64)
            if ref.shape[0] != mean_bag_oof.shape[0]:
                out[f"pearson_vs_{ref_tag}_mean_bag"] = None
                out[f"{ref_tag}_probe_error"] = (
                    f"shape mismatch: ref={ref.shape} vs "
                    f"self={mean_bag_oof.shape}"
                )
                continue
            a = mean_bag_oof.astype(np.float64)
            if a.std() > 0 and ref.std() > 0:
                r = float(np.corrcoef(a, ref)[0, 1])
            else:
                r = float("nan")
            out[f"pearson_vs_{ref_tag}_mean_bag"] = r
        except Exception as e:
            out[f"pearson_vs_{ref_tag}_mean_bag"] = None
            out[f"{ref_tag}_probe_error"] = repr(e)
    return out


def _blend_probe(
    mean_bag_oof: np.ndarray,
    y_unb: np.ndarray,
) -> dict:
    """3-way (or pairwise) naive-mean blend on the corrected OOFs."""
    out = {
        "nb1183_present": False,
        "nb1212_present": False,
        "nb1214_present": True,  # self
        "pair_nb1183_nb1214_mean_rae": None,
        "pair_nb1212_nb1214_mean_rae": None,
        "pair_nb1183_nb1212_mean_rae": None,
        "three_way_mean_rae": None,
    }
    ref = {}
    for ref_tag in ("nb1183", "nb1212"):
        p = DATA_PROCESSED / f"{ref_tag}_mean_bag_oof.npy"
        if p.exists():
            r = np.load(p).astype(np.float64)
            if r.shape[0] == mean_bag_oof.shape[0]:
                ref[ref_tag] = r
                out[f"{ref_tag}_present"] = True

    if "nb1183" in ref:
        m = 0.5 * (ref["nb1183"] + mean_bag_oof)
        out["pair_nb1183_nb1214_mean_rae"] = float(rae(y_unb, m))
    if "nb1212" in ref:
        m = 0.5 * (ref["nb1212"] + mean_bag_oof)
        out["pair_nb1212_nb1214_mean_rae"] = float(rae(y_unb, m))
    if "nb1183" in ref and "nb1212" in ref:
        m = 0.5 * (ref["nb1183"] + ref["nb1212"])
        out["pair_nb1183_nb1212_mean_rae"] = float(rae(y_unb, m))
        m3 = (ref["nb1183"] + ref["nb1212"] + mean_bag_oof) / 3.0
        out["three_way_mean_rae"] = float(rae(y_unb, m3))
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- RandomForestRegressor residual bag on top of nb1070, "
          f"MACCS-166 features, {len(RESID_SEEDS)} KFold seeds")
    print(f"          seeds = {RESID_SEEDS}")
    print(f"          residual target = y_unb - nb1070_pred_oof")
    print(f"          features = MACCS-166 (drop bit 0)  ({MACCS_TE_PATH})")
    print(f"          RF: n_estimators=300, max_depth=8, "
          f"min_samples_leaf=10, max_features=0.5")
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

    print(f"[feat] loading cached MACCS test matrix, slicing to MACCS-166 "
          f"and {n_unb} unblind rows ...")
    X_unb = _load_maccs166_unblind(n_test_expected=n_test, unb_idx=unb_idx)
    print(f"[feat] X_unb shape = {X_unb.shape}  (MACCS-166 keys)")
    print(f"[feat] bit density (unb) = {X_unb.mean():.4f}  "
          f"const cols = {int((X_unb.var(axis=0) == 0).sum())}/{X_unb.shape[1]}")

    # ---- Per-seed residual cross-fit ----
    print("\n" + "-" * 78)
    print(f"PER-SEED RESIDUAL CROSS-FIT (RF n_est=300 depth=8, "
          f"MACCS-{X_unb.shape[1]})")
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
    print(f"   nb1183 mean_bag ref    = {NB1183_MEAN_BAG_REF:.4f}  "
          f"(LGBM-Huber residual on nb1070 / MACCS)")

    beats_nb1070 = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb1183 = rae_mean_bag < NB1183_MEAN_BAG_REF - DECISION_MARGIN

    if beats_nb1183:
        verdict = "RF_MACCS_BEATS_NB1183_THIRD_MODEL_CLASS_WINS"
    elif beats_nb1070:
        verdict = "RF_MACCS_HELPS_NB1070_BUT_NOT_NB1183"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "RF_MACCS_FLAT_NO_NEW_SIGNAL"
    else:
        verdict = "RF_MACCS_HURTS_NB1070"
    print(f"   beats_nb1183 (margin 0.003) = {beats_nb1183}")
    print(f"   verdict                = {verdict}")

    # ---- Orthogonality probe ----
    print("\n" + "-" * 78)
    print("ORTHOGONALITY PROBE (corrected mean-bag OOF vs prior MACCS variants)")
    print("-" * 78)
    ortho = _orthogonality_probe(mean_bag_oof)
    for k, v in ortho.items():
        if isinstance(v, float):
            print(f"   {k} = {v:+.4f}")
        else:
            print(f"   {k} = {v}")

    # ---- Blend probe ----
    print("\n" + "-" * 78)
    print("3-WAY / PAIRWISE BLEND PROBE")
    print("-" * 78)
    blend = _blend_probe(mean_bag_oof, y_unb)
    for k, v in blend.items():
        print(f"   {k} = {v}")

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
        "model_class": "RandomForestRegressor",
        "feature_source": "maccs_cached_166",
        "maccs_cache_test": str(MACCS_TE_PATH),
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "feature_dim": int(X_unb.shape[1]),
        "rf_n_estimators": 300,
        "rf_max_depth": 8,
        "rf_min_samples_leaf": 10,
        "rf_max_features": 0.5,
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
        "delta_mean_bag_vs_nb1183": rae_mean_bag - NB1183_MEAN_BAG_REF,
        "beats_nb1070": bool(beats_nb1070),
        "beats_nb1183": bool(beats_nb1183),
        "verdict": verdict,
        "nb1070_ref_pooled": NB1070_REF_POOLED,
        "nb1183_mean_bag_ref": NB1183_MEAN_BAG_REF,
        "decision_margin": DECISION_MARGIN,
        "orthogonality_probe": ortho,
        "blend_probe": blend,
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
              "delta_mean_bag_vs_nb1183",
              "beats_nb1070", "beats_nb1183",
              "verdict", "orthogonality_probe",
              "blend_probe"):
        print(f"  {k}: {res.get(k)}")
