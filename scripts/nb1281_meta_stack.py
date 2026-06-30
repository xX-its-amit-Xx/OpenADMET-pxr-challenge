"""nb1281 -- Meta-stack: train a small LGBM (and Ridge, SLSQP simplex, NNLS)
as level-2 stacker on multiple residual-learner OOFs.

Components (9 OOFs, each 253-dim):
  - nb1242_mean_bag_oof.npy   (ChEMBL+MACCS residual,    0.5431)
  - nb1252_bob_mean_oof.npy   (ChEMBL BoB,               0.5446)
  - nb1190_bob_mean_oof.npy   (triple-FP BoB,            0.5499)
  - nb1200_bob_mean_oof.npy   (MACCS BoB,                0.5495)
  - nb1211_mean_oof.npy       (BoB-blend,                0.5451)
  - nb1183_mean_bag_oof.npy   (MACCS standalone,         0.5513)
  - nb1130_mean_bag_oof.npy   (Morgan+RDKit,             0.5673)
  - nb1153_mean_bag_oof.npy   (Mordred,                  0.5640)
  - nb1172_mean_bag_oof.npy   (AtomPair,                 0.5659)

Verdict at 0.003 margin vs nb1251 (0.5394 best_fixed_w blend).
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
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1281"
N_FOLDS = 5
SEED = 42
NB1251_REF = 0.5394
MARGIN = 0.003

COMPONENT_FILES = [
    ("nb1242", "nb1242_mean_bag_oof.npy", 0.5431),
    ("nb1252", "nb1252_bob_mean_oof.npy", 0.5446),
    ("nb1190", "nb1190_bob_mean_oof.npy", 0.5499),
    ("nb1200", "nb1200_bob_mean_oof.npy", 0.5495),
    ("nb1211", "nb1211_mean_oof.npy",     0.5451),
    ("nb1183", "nb1183_mean_bag_oof.npy", 0.5513),
    ("nb1130", "nb1130_mean_bag_oof.npy", 0.5673),
    ("nb1153", "nb1153_mean_bag_oof.npy", 0.5640),
    ("nb1172", "nb1172_mean_bag_oof.npy", 0.5659),
]

LGBM_META = dict(
    objective="huber", alpha=1.0,
    n_estimators=50, num_leaves=3, max_depth=2, learning_rate=0.03,
    min_child_samples=30,
    subsample=0.8, colsample_bytree=1.0,
    reg_alpha=0.0, reg_lambda=0.0,
    random_state=SEED, verbose=-1, n_jobs=4,
)


def _slsqp_simplex(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w):
        diff = y_tr - P_tr @ w
        return float(np.mean(diff * diff))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(_loss, w0, method="SLSQP",
                   bounds=bnds, constraints=cons,
                   options={"ftol": 1e-10, "maxiter": 1000})
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def _cross_fit_lgbm(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for f, (tr, va) in enumerate(kf.split(np.arange(n))):
        model = lgb.LGBMRegressor(**LGBM_META)
        model.fit(P[tr], y[tr])
        oof[va] = model.predict(P[va])
    return oof


def _cross_fit_ridge(P: np.ndarray, y: np.ndarray,
                     alphas=(0.01, 0.1, 1.0, 10.0, 100.0)) -> tuple[np.ndarray, float]:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # inner alpha CV per fold
    best_alphas = []
    for f, (tr, va) in enumerate(kf.split(np.arange(n))):
        best_a = None
        best_r = float("inf")
        inner_kf = KFold(n_splits=3, shuffle=True, random_state=SEED + f)
        for a in alphas:
            err = 0.0
            for tr2, va2 in inner_kf.split(tr):
                ix_tr = tr[tr2]; ix_va = tr[va2]
                m = Ridge(alpha=a).fit(P[ix_tr], y[ix_tr])
                err += float(np.mean((y[ix_va] - m.predict(P[ix_va])) ** 2))
            if err < best_r:
                best_r = err
                best_a = a
        best_alphas.append(best_a)
        model = Ridge(alpha=best_a).fit(P[tr], y[tr])
        oof[va] = model.predict(P[va])
    # pooled best alpha for reporting (mode)
    from collections import Counter
    pooled = Counter(best_alphas).most_common(1)[0][0]
    return oof, float(pooled)


def _cross_fit_slsqp(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for f, (tr, va) in enumerate(kf.split(np.arange(n))):
        w = _slsqp_simplex(P[tr], y[tr])
        oof[va] = P[va] @ w
        fold_records.append({"fold": int(f), "weights": [float(x) for x in w]})
    return oof, fold_records


def _cross_fit_nnls(P: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for f, (tr, va) in enumerate(kf.split(np.arange(n))):
        model = LinearRegression(positive=True, fit_intercept=False)
        model.fit(P[tr], y[tr])
        oof[va] = model.predict(P[va])
        fold_records.append({
            "fold": int(f),
            "coefs": [float(c) for c in model.coef_],
        })
    return oof, fold_records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- meta-stack 9 residual-learner OOFs (LGBM / Ridge / SLSQP / NNLS)")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] y_unb shape = {y_unb.shape}")

    # Load 9 component OOFs.
    preds = {}
    standalone_rae = {}
    for tag, fname, ref in COMPONENT_FILES:
        p = DATA_PROCESSED / fname
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({tag})")
        v = np.load(p).astype(np.float64)
        if v.shape[0] != n_unb:
            raise ValueError(f"shape mismatch: {tag}={v.shape}")
        preds[tag] = v
        standalone_rae[tag] = float(rae(y_unb, v))
        print(f"   {tag:8s}  loaded {p.name:35s}  RAE={standalone_rae[tag]:.4f}  (ref {ref:.4f})")

    tags = [t for t, _, _ in COMPONENT_FILES]
    P = np.column_stack([preds[t] for t in tags])
    print(f"\n[stack] design matrix P shape = {P.shape}  (9 features, 253 rows)")

    # Pairwise pred correlation matrix snapshot.
    corr = np.corrcoef(P, rowvar=False)
    print("\n[diag] pairwise pred Pearson (lower=more orthogonal):")
    hdr = "         " + "  ".join(f"{t:>7s}" for t in tags)
    print(hdr)
    for i, t in enumerate(tags):
        row = "  ".join(f"{corr[i, j]:>7.4f}" for j in range(len(tags)))
        print(f"   {t:7s}  {row}")

    # ----- LGBM meta -----
    print("\n" + "-" * 78)
    print("  BLOCK: LGBM meta (Huber alpha=1.0, depth=2, leaves=3, n_est=50, lr=0.03)")
    print("-" * 78)
    lgbm_oof = _cross_fit_lgbm(P, y_unb)
    rae_lgbm = float(rae(y_unb, lgbm_oof))
    print(f"   pooled RAE(LGBM meta cross-fit) = {rae_lgbm:.4f}")

    # ----- Ridge meta -----
    print("\n" + "-" * 78)
    print("  BLOCK: Ridge meta (per-fold alpha CV over {0.01,0.1,1,10,100})")
    print("-" * 78)
    ridge_oof, ridge_alpha_pooled = _cross_fit_ridge(P, y_unb)
    rae_ridge = float(rae(y_unb, ridge_oof))
    print(f"   pooled alpha mode = {ridge_alpha_pooled}")
    print(f"   pooled RAE(Ridge meta cross-fit) = {rae_ridge:.4f}")

    # ----- SLSQP simplex meta -----
    print("\n" + "-" * 78)
    print("  BLOCK: SLSQP simplex meta (w >= 0, sum w = 1)")
    print("-" * 78)
    slsqp_oof, slsqp_folds = _cross_fit_slsqp(P, y_unb)
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    fold_w = np.array([r["weights"] for r in slsqp_folds])
    mean_w_slsqp = fold_w.mean(axis=0)
    print(f"   per-fold SLSQP weights:")
    for rec in slsqp_folds:
        ws = "  ".join(f"{t}={w:.3f}" for t, w in zip(tags, rec["weights"]))
        print(f"     fold {rec['fold']}: {ws}")
    print(f"   mean SLSQP weights:")
    for t, w in zip(tags, mean_w_slsqp):
        print(f"     {t}: {w:.4f}")
    print(f"   pooled RAE(SLSQP cross-fit) = {rae_slsqp:.4f}")

    # In-sample SLSQP for reporting.
    w_full_slsqp = _slsqp_simplex(P, y_unb)
    rae_full_slsqp = float(rae(y_unb, P @ w_full_slsqp))
    print(f"   in-sample SLSQP weights:")
    for t, w in zip(tags, w_full_slsqp):
        print(f"     {t}: {w:.4f}")
    print(f"   in-sample SLSQP RAE = {rae_full_slsqp:.4f}")

    # ----- NNLS (LinearRegression positive=True, no intercept) -----
    print("\n" + "-" * 78)
    print("  BLOCK: NNLS meta (LinearRegression positive=True, no intercept)")
    print("-" * 78)
    nnls_oof, nnls_folds = _cross_fit_nnls(P, y_unb)
    rae_nnls = float(rae(y_unb, nnls_oof))
    fold_c = np.array([r["coefs"] for r in nnls_folds])
    mean_c_nnls = fold_c.mean(axis=0)
    print(f"   mean NNLS coefs across folds:")
    for t, c in zip(tags, mean_c_nnls):
        print(f"     {t}: {c:.4f}")
    print(f"   pooled RAE(NNLS cross-fit) = {rae_nnls:.4f}")

    # In-sample NNLS for reporting.
    nnls_full = LinearRegression(positive=True, fit_intercept=False).fit(P, y_unb)
    nnls_full_coefs = [float(c) for c in nnls_full.coef_]
    rae_full_nnls = float(rae(y_unb, nnls_full.predict(P)))
    print(f"   in-sample NNLS coefs:")
    for t, c in zip(tags, nnls_full_coefs):
        print(f"     {t}: {c:.4f}")
    print(f"   in-sample NNLS RAE = {rae_full_nnls:.4f}")

    # ----- Verdict -----
    candidates = {
        "lgbm":  rae_lgbm,
        "ridge": rae_ridge,
        "slsqp": rae_slsqp,
        "nnls":  rae_nnls,
    }
    best_tag = min(candidates, key=candidates.get)
    best_rae = candidates[best_tag]

    beats_nb1251 = best_rae < NB1251_REF - MARGIN
    flat_nb1251 = abs(best_rae - NB1251_REF) < MARGIN

    if beats_nb1251:
        verdict = f"META_STACK_BEATS_NB1251 ({best_tag} @ {best_rae:.4f})"
    elif flat_nb1251:
        verdict = f"META_STACK_FLAT_VS_NB1251 ({best_tag} @ {best_rae:.4f})"
    else:
        verdict = f"META_STACK_HURTS_VS_NB1251 ({best_tag} @ {best_rae:.4f})"

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1251 ref blend (best_fixed_w) RAE = {NB1251_REF:.4f}")
    print(f"   meta-learner candidates (cross-fit pooled RAE):")
    for tag, val in sorted(candidates.items(), key=lambda kv: kv[1]):
        marker = "  <-- best" if tag == best_tag else ""
        print(f"     {tag:7s} = {val:.4f}{marker}")
    print(f"")
    print(f"   best meta-learner          : {best_tag} @ {best_rae:.4f}")
    print(f"   delta vs nb1251 (0.5394)   : {best_rae - NB1251_REF:+.4f}")
    print(f"   beats_nb1251 (>= {MARGIN})    : {beats_nb1251}")
    print(f"   verdict                    : {verdict}")

    # Persist artifacts.
    np.save(DATA_PROCESSED / f"{TAG}_lgbm_oof.npy", lgbm_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_ridge_oof.npy", ridge_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_slsqp_oof.npy", slsqp_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_nnls_oof.npy", nnls_oof.astype(np.float32))
    print(f"\n[save] {TAG}_lgbm_oof.npy / ridge / slsqp / nnls")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "component_tags": tags,
        "component_files": [f for _, f, _ in COMPONENT_FILES],
        "standalone_rae": standalone_rae,
        "pred_corr_matrix": [[float(corr[i, j]) for j in range(len(tags))]
                             for i in range(len(tags))],
        "lgbm_params": {k: v for k, v in LGBM_META.items()
                        if k not in ("random_state", "verbose", "n_jobs")},
        "rae_lgbm_cross_fit": rae_lgbm,
        "ridge_alpha_pooled_mode": ridge_alpha_pooled,
        "rae_ridge_cross_fit": rae_ridge,
        "slsqp_fold_records": slsqp_folds,
        "slsqp_mean_fold_weights": {t: float(w) for t, w in zip(tags, mean_w_slsqp)},
        "slsqp_in_sample_weights": {t: float(w) for t, w in zip(tags, w_full_slsqp)},
        "slsqp_in_sample_rae": rae_full_slsqp,
        "rae_slsqp_cross_fit": rae_slsqp,
        "nnls_fold_records": nnls_folds,
        "nnls_mean_fold_coefs": {t: float(c) for t, c in zip(tags, mean_c_nnls)},
        "nnls_in_sample_coefs": {t: float(c) for t, c in zip(tags, nnls_full_coefs)},
        "nnls_in_sample_rae": rae_full_nnls,
        "rae_nnls_cross_fit": rae_nnls,
        "candidate_rae_table": candidates,
        "best_meta_tag": best_tag,
        "best_meta_rae": best_rae,
        "nb1251_ref": NB1251_REF,
        "delta_best_vs_nb1251": best_rae - NB1251_REF,
        "beats_nb1251": bool(beats_nb1251),
        "flat_vs_nb1251": bool(flat_nb1251),
        "margin": MARGIN,
        "verdict": verdict,
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
    for k in ("standalone_rae",
              "rae_lgbm_cross_fit", "rae_ridge_cross_fit",
              "rae_slsqp_cross_fit", "rae_nnls_cross_fit",
              "slsqp_mean_fold_weights", "nnls_mean_fold_coefs",
              "slsqp_in_sample_weights", "nnls_in_sample_coefs",
              "best_meta_tag", "best_meta_rae",
              "delta_best_vs_nb1251", "beats_nb1251", "verdict"):
        print(f"  {k}: {res.get(k)}")
