"""nb1563 -- Meta-stack: gradient boost over PRE-unblind top OOFs as features.

PROTOCOL
    1. Load 8 PRE-unblind OOFs:
        - nb1472_mean_oof.npy            (0.5330)
        - nb1484_best_oof.npy            (0.5231)
        - nb1500_bob_mean_oof.npy        (0.5236)
        - nb1501_mean_bag_oof.npy        (0.5223)
        - nb1512_bob_mean_oof.npy        (0.5246)
        - nb1521_bob_mean_oof.npy        (0.5221)
        - nb1553_best_oof.npy            (0.5225)
        - chemprop_aux on 253            (0.6216)  <-- te_chemprop_aux[unb_idx]
    2. Feature matrix (253, 8). Target y_unb.
    3. 5-fold cross-fit shallow LGBM Huber
       (max_depth=2, n_estimators=40, learning_rate=0.03, min_child_samples=30,
        objective="huber", alpha=1.0).
    4. Pool RAE on the 5-fold cross-fit OOF.
    5. Also try:
        a. simple NNLS (least squares, w >= 0; no simplex constraint).
        b. SLSQP simplex (w >= 0, sum w = 1) -- 5-fold cross-fit.
        c. naive 1/K mean of 8.
    6. Verdict at 0.003 margin vs nb1521 (0.5221).

Outputs:
    scripts/nb1563_meta_stack_pre.py            (this file)
    data/processed/nb1563_summary.json
    data/processed/nb1563_best_oof.npy          (253,) float32 best variant
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
from scipy.optimize import minimize, nnls
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1563"

# 8 PRE-unblind OOF features, in declared order
FEATURE_FILES = [
    ("nb1472_mean",     "nb1472_mean_oof.npy",      0.5330),
    ("nb1484_best",     "nb1484_best_oof.npy",      0.5231),
    ("nb1500_bob_mean", "nb1500_bob_mean_oof.npy",  0.5236),
    ("nb1501_mean_bag", "nb1501_mean_bag_oof.npy",  0.5223),
    ("nb1512_bob_mean", "nb1512_bob_mean_oof.npy",  0.5246),
    ("nb1521_bob_mean", "nb1521_bob_mean_oof.npy",  0.5221),
    ("nb1553_best",     "nb1553_best_oof.npy",      0.5225),
    # 8th is chemprop_aux te[unb_idx] -- 0.6216 anchor
]
CHEMPROP_AUX_TE_FILE = "te_chemprop_aux.npy"
CHEMPROP_AUX_TAG = "chemprop_aux_unb"
CHEMPROP_AUX_REF = 0.6216

NB1521_REF = 0.5221
DECISION_MARGIN = 0.003

N_FOLDS = 5
SLSQP_SEED = 42
LGBM_SEED = 42


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.03,
        n_estimators=40,
        max_depth=2,
        num_leaves=4,
        min_child_samples=30,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _lgbm_cross_fit(X: np.ndarray, y: np.ndarray,
                    n_splits: int, seed: int) -> np.ndarray:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], y[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w: np.ndarray) -> float:
        r = y_tr - P_tr @ w
        return float(np.mean(r * r))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        _loss, w0, method="SLSQP",
        bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        return np.full(K, 1.0 / K)
    return w / s


def _slsqp_cross_fit(P: np.ndarray, y: np.ndarray,
                     n_splits: int, seed: int):
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w = _slsqp_blend_weights(P[tr_loc], y[tr_loc])
        oof[va_loc] = P[va_loc] @ w
        fold_records.append({
            "fold": int(f),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "w": [float(x) for x in w.tolist()],
        })
    return oof, fold_records


def _nnls_cross_fit(P: np.ndarray, y: np.ndarray,
                    n_splits: int, seed: int):
    """5-fold cross-fit NNLS (w >= 0, no simplex constraint)."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w, _ = nnls(P[tr_loc], y[tr_loc])
        oof[va_loc] = P[va_loc] @ w
        fold_records.append({
            "fold": int(f),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "w": [float(x) for x in w.tolist()],
        })
    return oof, fold_records


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- meta-stack over 8 PRE-unblind OOFs (LGBM Huber depth=2 / NNLS / SLSQP / mean)")
    print(f"          n_folds = {N_FOLDS}  decision_margin = {DECISION_MARGIN} vs nb1521 ({NB1521_REF})")
    print("=" * 78)

    # ---- Truth ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb = {n_unb}")

    # ---- Load 7 OOF features (on 253) ----
    cols = []
    feat_names = []
    feat_refs = []
    per_feat_rae = {}
    for tag, fname, ref in FEATURE_FILES:
        p = DATA_PROCESSED / fname
        if not p.exists():
            raise FileNotFoundError(f"missing OOF: {p}")
        a = np.load(p).astype(np.float64)
        if a.shape != (n_unb,):
            raise ValueError(f"{fname} shape mismatch: {a.shape} vs ({n_unb},)")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{fname} has non-finite values")
        r = float(rae(y_unb, a))
        per_feat_rae[tag] = r
        feat_names.append(tag)
        feat_refs.append(ref)
        cols.append(a)
        print(f"   [feat] {tag:<20s} pool RAE = {r:.4f}  (ref {ref:.4f})")

    # ---- 8th feature: chemprop_aux te[unb_idx] ----
    p_anchor = DATA_PROCESSED / CHEMPROP_AUX_TE_FILE
    if not p_anchor.exists():
        raise FileNotFoundError(f"missing chemprop_aux te: {p_anchor}")
    te_anchor_513 = np.load(p_anchor).astype(np.float64)
    if te_anchor_513.shape[0] < unb_idx.max() + 1:
        raise ValueError(f"chemprop_aux te too short: {te_anchor_513.shape}")
    a_anchor = te_anchor_513[unb_idx]
    if not np.all(np.isfinite(a_anchor)):
        raise ValueError("chemprop_aux te has non-finite values at unb_idx")
    r_anchor = float(rae(y_unb, a_anchor))
    per_feat_rae[CHEMPROP_AUX_TAG] = r_anchor
    feat_names.append(CHEMPROP_AUX_TAG)
    feat_refs.append(CHEMPROP_AUX_REF)
    cols.append(a_anchor)
    print(f"   [feat] {CHEMPROP_AUX_TAG:<20s} pool RAE = {r_anchor:.4f}  (ref {CHEMPROP_AUX_REF:.4f})")

    X = np.stack(cols, axis=1).astype(np.float64)  # (253, 8)
    K = X.shape[1]
    print(f"\n[matrix] X shape = {X.shape}  K = {K}")
    print(f"[matrix] feat col-means    = {[f'{m:.3f}' for m in X.mean(axis=0).tolist()]}")
    print(f"[matrix] feat col-stds     = {[f'{s:.3f}' for s in X.std(axis=0).tolist()]}")

    # ---- Method A: naive 1/K mean ----
    print("\n" + "-" * 78)
    print("METHOD A -- naive 1/K mean of 8 features")
    print("-" * 78)
    oof_mean = X.mean(axis=1)
    rae_mean = float(rae(y_unb, oof_mean))
    print(f"   rae_naive_mean        = {rae_mean:.4f}")

    # ---- Method B: 5-fold NNLS (cross-fit) ----
    print("\n" + "-" * 78)
    print(f"METHOD B -- 5-fold NNLS cross-fit (w >= 0, no simplex) seed={SLSQP_SEED}")
    print("-" * 78)
    oof_nnls, nnls_folds = _nnls_cross_fit(X, y_unb, n_splits=N_FOLDS, seed=SLSQP_SEED)
    rae_nnls = float(rae(y_unb, oof_nnls))
    W_nnls = np.array([f["w"] for f in nnls_folds])
    w_nnls_mean = W_nnls.mean(axis=0).tolist()
    w_nnls_std = W_nnls.std(axis=0).tolist()
    print(f"   mean w over folds     = [{', '.join(f'{w:.3f}' for w in w_nnls_mean)}]")
    print(f"   std  w over folds     = [{', '.join(f'{w:.3f}' for w in w_nnls_std)}]")
    print(f"   rae_nnls_crossfit     = {rae_nnls:.4f}")

    # ---- Method C: 5-fold SLSQP simplex cross-fit ----
    print("\n" + "-" * 78)
    print(f"METHOD C -- 5-fold SLSQP simplex cross-fit seed={SLSQP_SEED}")
    print("-" * 78)
    oof_slsqp, slsqp_folds = _slsqp_cross_fit(X, y_unb, n_splits=N_FOLDS, seed=SLSQP_SEED)
    rae_slsqp = float(rae(y_unb, oof_slsqp))
    W_slsqp = np.array([f["w"] for f in slsqp_folds])
    w_slsqp_mean = W_slsqp.mean(axis=0).tolist()
    w_slsqp_std = W_slsqp.std(axis=0).tolist()
    print(f"   mean w over folds     = [{', '.join(f'{w:.3f}' for w in w_slsqp_mean)}]")
    print(f"   std  w over folds     = [{', '.join(f'{w:.3f}' for w in w_slsqp_std)}]")
    print(f"   rae_slsqp_crossfit    = {rae_slsqp:.4f}")

    # ---- Method D: 5-fold LGBM Huber shallow cross-fit ----
    print("\n" + "-" * 78)
    print(f"METHOD D -- 5-fold shallow LGBM Huber cross-fit (depth=2, n_est=40, lr=0.03, "
          f"min_child=30) seed={LGBM_SEED}")
    print("-" * 78)
    oof_lgbm = _lgbm_cross_fit(X, y_unb, n_splits=N_FOLDS, seed=LGBM_SEED)
    rae_lgbm = float(rae(y_unb, oof_lgbm))
    print(f"   rae_lgbm_meta_stack   = {rae_lgbm:.4f}")

    # ---- Verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    methods = [
        ("naive_mean", rae_mean, oof_mean),
        ("nnls_crossfit", rae_nnls, oof_nnls),
        ("slsqp_crossfit", rae_slsqp, oof_slsqp),
        ("lgbm_meta_stack", rae_lgbm, oof_lgbm),
    ]
    methods_sorted = sorted(methods, key=lambda x: x[1])
    print("   ranking (best -> worst):")
    for name, r, _ in methods_sorted:
        print(f"     {name:<22s}  RAE = {r:.4f}")

    best_name, rae_best, best_oof = methods_sorted[0]

    delta_mean_vs_nb1521 = rae_mean - NB1521_REF
    delta_nnls_vs_nb1521 = rae_nnls - NB1521_REF
    delta_slsqp_vs_nb1521 = rae_slsqp - NB1521_REF
    delta_lgbm_vs_nb1521 = rae_lgbm - NB1521_REF
    delta_best_vs_nb1521 = rae_best - NB1521_REF

    beats_nb1521_mean = rae_mean < NB1521_REF - DECISION_MARGIN
    beats_nb1521_nnls = rae_nnls < NB1521_REF - DECISION_MARGIN
    beats_nb1521_slsqp = rae_slsqp < NB1521_REF - DECISION_MARGIN
    beats_nb1521_lgbm = rae_lgbm < NB1521_REF - DECISION_MARGIN
    beats_nb1521_best = rae_best < NB1521_REF - DECISION_MARGIN
    flat_vs_nb1521_best = abs(rae_best - NB1521_REF) < DECISION_MARGIN

    if beats_nb1521_best:
        verdict = f"META_STACK_BEATS_NB1521_NEW_PRIMARY_via_{best_name}"
    elif flat_vs_nb1521_best:
        verdict = f"META_STACK_FLAT_VS_NB1521_best_{best_name}"
    else:
        verdict = f"META_STACK_WORSE_THAN_NB1521_best_{best_name}"

    print(f"\n   d_mean_vs_nb1521      = {delta_mean_vs_nb1521:+.4f}")
    print(f"   d_nnls_vs_nb1521      = {delta_nnls_vs_nb1521:+.4f}")
    print(f"   d_slsqp_vs_nb1521     = {delta_slsqp_vs_nb1521:+.4f}")
    print(f"   d_lgbm_vs_nb1521      = {delta_lgbm_vs_nb1521:+.4f}")
    print(f"   d_best_vs_nb1521      = {delta_best_vs_nb1521:+.4f}   (best = {best_name})")
    print(f"   beats_nb1521_mean     = {beats_nb1521_mean}")
    print(f"   beats_nb1521_nnls     = {beats_nb1521_nnls}")
    print(f"   beats_nb1521_slsqp    = {beats_nb1521_slsqp}")
    print(f"   beats_nb1521_lgbm     = {beats_nb1521_lgbm}")
    print(f"   beats_nb1521_best     = {beats_nb1521_best}")
    print(f"   verdict               = {verdict}")
    print("=" * 78)

    # ---- Save ----
    out_oof = DATA_PROCESSED / f"{TAG}_best_oof.npy"
    np.save(out_oof, best_oof.astype(np.float32))
    print(f"\n[save] {out_oof}  (variant = {best_name})")

    summary = {
        "tag": TAG,
        "n_unb": int(n_unb),
        "n_features": int(K),
        "feature_names": feat_names,
        "feature_refs": feat_refs,
        "per_feature_rae": per_feat_rae,
        "n_folds": N_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "lgbm_seed": LGBM_SEED,
        "lgbm_params": _lgbm_params(LGBM_SEED),
        "rae_naive_mean": rae_mean,
        "rae_nnls_crossfit": rae_nnls,
        "rae_slsqp_crossfit": rae_slsqp,
        "rae_lgbm_meta_stack": rae_lgbm,
        "rae_best": rae_best,
        "best_variant": best_name,
        "method_ranking": [
            {"name": n, "rae": r} for n, r, _ in methods_sorted
        ],
        "nnls_fold_weights": nnls_folds,
        "nnls_w_mean_over_folds": w_nnls_mean,
        "nnls_w_std_over_folds": w_nnls_std,
        "slsqp_fold_weights": slsqp_folds,
        "slsqp_w_mean_over_folds": w_slsqp_mean,
        "slsqp_w_std_over_folds": w_slsqp_std,
        "delta_mean_vs_nb1521": delta_mean_vs_nb1521,
        "delta_nnls_vs_nb1521": delta_nnls_vs_nb1521,
        "delta_slsqp_vs_nb1521": delta_slsqp_vs_nb1521,
        "delta_lgbm_vs_nb1521": delta_lgbm_vs_nb1521,
        "delta_best_vs_nb1521": delta_best_vs_nb1521,
        "beats_nb1521_mean": bool(beats_nb1521_mean),
        "beats_nb1521_nnls": bool(beats_nb1521_nnls),
        "beats_nb1521_slsqp": bool(beats_nb1521_slsqp),
        "beats_nb1521_lgbm": bool(beats_nb1521_lgbm),
        "beats_nb1521_best": bool(beats_nb1521_best),
        "flat_vs_nb1521_best": bool(flat_vs_nb1521_best),
        "verdict": verdict,
        "nb1521_ref": NB1521_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
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
        "n_unb", "n_features",
        "per_feature_rae",
        "rae_naive_mean",
        "rae_nnls_crossfit",
        "rae_slsqp_crossfit",
        "rae_lgbm_meta_stack",
        "rae_best",
        "best_variant",
        "delta_best_vs_nb1521",
        "beats_nb1521_best",
        "flat_vs_nb1521_best",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
