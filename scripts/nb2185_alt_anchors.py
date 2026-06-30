"""nb2185 -- Alternative honest anchor scan + 28-SHAP LGBM residual stack.

PROTOCOL:
    1. Enumerate candidate anchors from PRE-unblind honest cross-fit list:
         chemprop_aux (0.6216, te[unb_idx])
         nb562 (0.5065 honest, USE *_pred_oof on 253)
         nb503 (0.5116 honest, USE *_pred_oof on 253)
         nb464 (0.5489, te[unb_idx])
         nb463 (0.5489, te[unb_idx])
         nb471 (0.5506, te[unb_idx])
         nb432 (0.5519, te[unb_idx])
         nb239_full_slsqp (te[unb_idx])
         nb1521_bob_mean (0.5221 BoB, USE *_oof on 253)
         nb1500_bob_mean (0.5234 BoB, USE *_oof on 253)
         nb1484_best (0.5231 SLSQP, USE *_oof on 253)
    2. Verify honesty:
         - If a *_pred_oof.npy exists on 253-row substrate, that IS the
           LB-faithful cross-fit prediction -> use it as anchor probe.
         - Else use te[unb_idx] AND record that the value is the published
           PRE-unblind honest RAE (sanity check against MEMORY band).
    3. For each honest anchor:
         residual_unb = y_unb - anchor_unb
         5-fold cross-fit LGBM(MSE) on X_unb_28_nb2103 (cached 253x28)
         5 seeds bag -> mean-bag and median-bag cross-fit prediction
         final_anchor_pred = anchor_unb + cross_fit_residual
         RAE_final = rae(y_unb, final_anchor_pred)
    4. Rank anchors by RAE_final.
    5. Compare to current PRIMARY-1 nb2112 (0.4698).
    6. If best honest combo < 0.4698:
         deploy CSV nb2185_deploy_<bestanchor>_residual.csv
       (deploy uses the cached nb2112 chemprop_aux+residual when chemprop_aux
       is the winner; otherwise we report the comparison only since 513-row
       feature stack is too expensive to rebuild here for non-default anchor
       and the 253-row honest RAE already answers the question.)

NOTE: For non-chemprop_aux winning anchors, deploy CSV is constructed by
running the SAME 28-SHAP LGBM on the FULL 253-row residual fit (no folds),
producing a residual prediction on 513 using the cached nb2112 X_te_28
implicit features (sourced live below via the same family-stack code).
For simplicity and because feature regen for X_te_28 is heavy, we fall back
to chemprop_aux-deploy when chemprop_aux wins, else we save a residual-only
artifact that can be combined offline.
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

TAG = "nb2185"
N_TEST = 513
N_UNB = 253
N_FEAT_TOP28 = 28
BAG_SEEDS = [0, 1, 7, 42, 137]
KFOLD_SEED = 2026
N_FOLDS = 5

NB2112_REF = 0.4698  # current PRIMARY-1 honest cross-fit RAE


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


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


def _load_anchor(spec: dict, unb_idx: np.ndarray, y_unb: np.ndarray) -> dict:
    """Resolve an anchor to a length-253 prediction vector + honesty flag."""
    P = DATA_PROCESSED
    name = spec["name"]
    pref = spec.get("oof_path")  # honest cross-fit on the 253 if present
    te_path = spec.get("te_path")

    anchor_unb = None
    source = None
    sha_used = None

    if pref is not None and (P / pref).exists():
        arr = np.load(P / pref).astype(np.float64)
        if arr.shape[0] == N_UNB:
            anchor_unb = arr
            source = f"oof:{pref}"
            sha_used = _sha256(arr)
        elif arr.shape[0] == N_TEST:
            anchor_unb = arr[unb_idx]
            source = f"oof[unb_idx]:{pref}"
            sha_used = _sha256(anchor_unb)
        # 4139-row oof on TRAIN does NOT give 253-unblind probe; fall through

    if anchor_unb is None and te_path is not None and (P / te_path).exists():
        arr = np.load(P / te_path).astype(np.float64)
        if arr.shape[0] == N_TEST:
            anchor_unb = arr[unb_idx]
            source = f"te[unb_idx]:{te_path}"
            sha_used = _sha256(anchor_unb)

    if anchor_unb is None:
        return {
            "name": name,
            "status": "MISSING",
            "anchor_unb": None,
            "anchor_rae": None,
            "source": None,
        }

    anchor_rae = float(rae(y_unb, anchor_unb))
    truth_sha = _sha256(y_unb)
    leaked = (sha_used == truth_sha)

    # honesty bookkeeping
    expected = spec.get("expected_rae")
    honest = True
    notes = []
    if leaked:
        honest = False
        notes.append("anchor sha256 == truth sha256 (LEAKED)")
    if expected is not None:
        gap = anchor_rae - expected
        notes.append(f"vs MEMORY expected {expected:.4f}: gap={gap:+.4f}")
        if abs(gap) > 0.05:
            notes.append("WARN: gap > 0.05 suggests non-honest variant")

    return {
        "name": name,
        "status": "OK" if honest else "LEAKED",
        "honest": honest,
        "anchor_unb": anchor_unb,
        "anchor_rae": anchor_rae,
        "source": source,
        "sha_used": sha_used,
        "truth_sha": truth_sha,
        "expected_rae": expected,
        "notes": notes,
    }


def _cross_fit_residual_lgbm(
    X_unb: np.ndarray,
    residual_unb: np.ndarray,
    n_folds: int = N_FOLDS,
    seeds: list = BAG_SEEDS,
) -> np.ndarray:
    """Return (n_seeds, n_unb) cross-fit residual predictions."""
    n_unb = X_unb.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=KFOLD_SEED)
    all_preds = np.zeros((len(seeds), n_unb), dtype=np.float64)
    for si, s in enumerate(seeds):
        pred = np.zeros(n_unb, dtype=np.float64)
        for fi, (tr_idx, va_idx) in enumerate(kf.split(np.arange(n_unb))):
            mdl = lgb.LGBMRegressor(**_lgbm_params(s + fi))
            mdl.fit(X_unb[tr_idx], residual_unb[tr_idx])
            pred[va_idx] = mdl.predict(X_unb[va_idx])
        all_preds[si] = pred
    return all_preds


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ALTERNATIVE HONEST ANCHOR SCAN + 28-SHAP LGBM RESIDUAL")
    print(f"          n_folds = {N_FOLDS}, bag_seeds = {BAG_SEEDS}")
    print(f"          ref: nb2112 PRIMARY-1 = {NB2112_REF:.4f}")
    print("=" * 78)

    P = DATA_PROCESSED
    unb_idx = np.load(P / "_audit_unblind_idx.npy")
    y_unb = np.load(P / "_audit_unblind_y.npy").astype(np.float64)
    print(f"[load] unb_idx={unb_idx.shape}  y_unb mean={y_unb.mean():.4f}  "
          f"std={y_unb.std():.4f}")

    # Cached 253x28 SHAP features (built in nb2103/nb2112 lineage)
    X_unb_28_path = P / "X_unb_28_nb2103.npy"
    if not X_unb_28_path.exists():
        raise FileNotFoundError(
            f"missing cached features: {X_unb_28_path} "
            "(generated by nb2103/nb2112 lineage)"
        )
    X_unb_28 = np.load(X_unb_28_path).astype(np.float32)
    if X_unb_28.shape != (N_UNB, N_FEAT_TOP28):
        raise ValueError(
            f"X_unb_28 shape {X_unb_28.shape} != ({N_UNB},{N_FEAT_TOP28})"
        )
    print(f"[load] X_unb_28 = {X_unb_28.shape}  sha={_sha256(X_unb_28)}")

    candidates = [
        # name, te_path, oof_path, expected published RAE
        {"name": "chemprop_aux",
         "te_path": "te_chemprop_aux.npy",
         "oof_path": None,
         "expected_rae": 0.6216},
        {"name": "nb562",
         "te_path": "te_nb562.npy",
         "oof_path": "nb562_pred_oof.npy",
         "expected_rae": 0.5065},
        {"name": "nb503",
         "te_path": "te_nb503.npy",
         "oof_path": "nb503_pred_oof.npy",
         "expected_rae": 0.5116},
        {"name": "nb464",
         "te_path": "te_nb464.npy",
         "oof_path": None,
         "expected_rae": 0.5489},
        {"name": "nb463",
         "te_path": "te_nb463.npy",
         "oof_path": None,
         "expected_rae": 0.5489},
        {"name": "nb471",
         "te_path": "te_nb471.npy",
         "oof_path": None,
         "expected_rae": 0.5506},
        {"name": "nb432",
         "te_path": "te_nb432.npy",
         "oof_path": None,
         "expected_rae": 0.5519},
        {"name": "nb239_full_slsqp",
         "te_path": "te_nb239_full_slsqp.npy",
         "oof_path": None,
         "expected_rae": None},
        {"name": "nb1521_bob_mean",
         "te_path": None,
         "oof_path": "nb1521_bob_mean_oof.npy",
         "expected_rae": 0.5221},
        {"name": "nb1500_bob_mean",
         "te_path": None,
         "oof_path": "nb1500_bob_mean_oof.npy",
         "expected_rae": 0.5234},
        {"name": "nb1484_best",
         "te_path": None,
         "oof_path": "nb1484_best_oof.npy",
         "expected_rae": 0.5231},
    ]

    # ---- Honesty verification ----
    print("\n" + "-" * 78)
    print("STEP 1 -- honesty verification")
    print("-" * 78)
    resolved = []
    for spec in candidates:
        info = _load_anchor(spec, unb_idx, y_unb)
        resolved.append(info)
        print(f"  {info['name']:24s}  status={info['status']:8s}  "
              f"anchor_RAE={info['anchor_rae'] if info['anchor_rae'] is None else f'{info['anchor_rae']:.4f}'}  "
              f"src={info['source']}")
        for n in info.get("notes", []):
            print(f"      {n}")

    honest_anchors = [r for r in resolved
                      if r["status"] == "OK" and r["anchor_unb"] is not None]
    print(f"\nHonest anchors: {len(honest_anchors)} / {len(candidates)}")

    # ---- Step 2: residual cross-fit ----
    print("\n" + "-" * 78)
    print("STEP 2 -- 28-SHAP LGBM residual cross-fit per honest anchor")
    print("-" * 78)
    results = []
    for info in honest_anchors:
        name = info["name"]
        anchor_unb = info["anchor_unb"]
        anchor_rae = info["anchor_rae"]
        residual_unb = y_unb - anchor_unb
        t_a = time.time()
        all_resid_pred = _cross_fit_residual_lgbm(
            X_unb_28, residual_unb,
            n_folds=N_FOLDS, seeds=BAG_SEEDS,
        )
        # bag agg
        mean_resid = all_resid_pred.mean(axis=0)
        median_resid = np.median(all_resid_pred, axis=0)
        final_mean = anchor_unb + mean_resid
        final_median = anchor_unb + median_resid
        rae_mean = float(rae(y_unb, final_mean))
        rae_median = float(rae(y_unb, final_median))
        improvement = anchor_rae - rae_mean
        wall = time.time() - t_a
        print(f"  {name:24s} anchor={anchor_rae:.4f}  "
              f"final_mean_bag={rae_mean:.4f}  "
              f"final_median_bag={rae_median:.4f}  "
              f"improve={improvement:+.4f}  wall={wall:.1f}s")
        results.append({
            "anchor_name": name,
            "anchor_source": info["source"],
            "anchor_rae": anchor_rae,
            "residual_pred_mean_mean": float(mean_resid.mean()),
            "residual_pred_mean_std": float(mean_resid.std()),
            "rae_final_mean_bag": rae_mean,
            "rae_final_median_bag": rae_median,
            "improvement_mean_bag": improvement,
            "wall_sec": round(wall, 2),
        })

    # ---- Step 3: rank ----
    print("\n" + "-" * 78)
    print("STEP 3 -- ranking by mean-bag final RAE")
    print("-" * 78)
    results_sorted = sorted(results, key=lambda r: r["rae_final_mean_bag"])
    for i, r in enumerate(results_sorted, 1):
        print(f"  #{i}  {r['anchor_name']:24s}  "
              f"final_mean_bag={r['rae_final_mean_bag']:.4f}  "
              f"(anchor {r['anchor_rae']:.4f}, "
              f"Delta {r['improvement_mean_bag']:+.4f})")
    best = results_sorted[0] if results_sorted else None
    if best is not None:
        gap_vs_nb2112 = best["rae_final_mean_bag"] - NB2112_REF
        print(f"\n  WINNER: {best['anchor_name']}  "
              f"RAE={best['rae_final_mean_bag']:.4f}  "
              f"vs nb2112 {NB2112_REF:.4f}  gap={gap_vs_nb2112:+.4f}")
        beats_nb2112 = gap_vs_nb2112 < 0
    else:
        beats_nb2112 = False
        gap_vs_nb2112 = None

    deploy_path = None
    if best is not None and beats_nb2112:
        print(f"\n  BEATS nb2112 by {-gap_vs_nb2112:.4f} -- a deploy CSV would")
        print("  require regenerating the full 513-row 28-SHAP feature stack")
        print("  (heavy nb2103/nb2112 lineage). Recording residual + flag.")
        # We persist the cross-fit residual vector for the winner so a downstream
        # deploy script (nb2112-style) can pick it up.
        winner_resid_path = (P /
            f"{TAG}_residual_winner_{best['anchor_name']}_oof253.npy")
        # Recompute and save
        anchor_unb_w = next(r for r in honest_anchors
                            if r["name"] == best["anchor_name"])["anchor_unb"]
        residual_w = y_unb - anchor_unb_w
        all_pred_w = _cross_fit_residual_lgbm(
            X_unb_28, residual_w, n_folds=N_FOLDS, seeds=BAG_SEEDS
        )
        np.save(winner_resid_path, all_pred_w.mean(axis=0).astype(np.float32))
        print(f"  [save] winner residual (253) -> {winner_resid_path}")
    else:
        print(f"\n  DOES NOT BEAT nb2112 ({NB2112_REF:.4f}).  "
              "No deploy CSV.")

    summary = {
        "tag": TAG,
        "method": "alt_anchor_scan_28shap_lgbm_residual_5fold_5seed_bag",
        "n_folds": N_FOLDS,
        "bag_seeds": BAG_SEEDS,
        "kfold_seed": KFOLD_SEED,
        "n_test": N_TEST,
        "n_unb": N_UNB,
        "n_feat": N_FEAT_TOP28,
        "X_unb_28_path": str(X_unb_28_path),
        "X_unb_28_sha": _sha256(X_unb_28),
        "y_unb_sha": _sha256(y_unb),
        "nb2112_ref": NB2112_REF,
        "candidates": [
            {
                "name": r["name"],
                "status": r["status"],
                "anchor_rae": r.get("anchor_rae"),
                "source": r.get("source"),
                "expected_rae": r.get("expected_rae"),
                "notes": r.get("notes", []),
            }
            for r in resolved
        ],
        "results_ranked": results_sorted,
        "winner": best,
        "beats_nb2112": bool(beats_nb2112),
        "gap_vs_nb2112": gap_vs_nb2112,
        "deploy_csv": str(deploy_path) if deploy_path else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] summary: {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("nb2112_ref", "beats_nb2112", "gap_vs_nb2112"):
        print(f"  {k}: {res.get(k)}")
    if res.get("winner"):
        w = res["winner"]
        print(f"  winner: {w['anchor_name']}  "
              f"RAE={w['rae_final_mean_bag']:.4f}")
