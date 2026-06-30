"""nb1471 -- Blend nb1462 (SHAP-pruned chemprop embed) + nb1373 (SHAP-pruned AtomPair).

HYPOTHESIS
    Deep encoder features (300-dim chemprop embedding, SHAP-pruned) and 2D
    fingerprint features (2048-bit AtomPair, SHAP-pruned to top-30 bits) are
    complementary representation spaces: the embedding captures learned
    distributed graph chemistry, the AtomPair bits encode explicit pairwise
    substructure topology. Pearson on residuals should be < 0.95 if they truly
    cover different failure modes. A linear blend may reduce variance neither
    family captures alone.

PROTOCOL
    1. Load nb1462_mean_bag_oof.npy (RAE ~ 0.5144, chemprop embed SHAP-pruned)
       and nb1373_mean_bag_oof.npy (RAE ~ 0.5095, AtomPair SHAP-pruned).
    2. Pearson on raw preds AND on residuals from truth.
    3. Grid sweep w in {0.0..1.0 step 0.05} for w*nb1462 + (1-w)*nb1373.
    4. 5-fold SLSQP cross-fit (simplex K=2).
    5. Gate vs anchors: nb1373 (0.5095) and nb1391 (0.5076) at margin 0.003.

OUTPUTS
    scripts/nb1471_blend_embed_atompair.py
    data/processed/nb1471_summary.json
    data/processed/nb1471_best_oof.npy   (253,) float32  best variant predictor
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
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1471"
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# Reference numbers (pooled RAE on 253 unblind) from prior notebook summaries.
NB1462_MEAN_REF = 0.5144   # chemprop embed SHAP-pruned mean-bag
NB1373_MEAN_REF = 0.5095   # AtomPair SHAP-pruned mean-bag
NB1391_REF      = 0.5076   # nb1373 + nb1352 best blend (current strongest pair)

# Gate against the best component AND the strongest 2-way pair already found.
ANCHOR_REF_COMPONENT = NB1373_MEAN_REF
ANCHOR_REF_PRIOR_BLEND = NB1391_REF
MARGIN = 0.003


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Argmin squared loss over the K-simplex (w_i >= 0, sum w_i = 1)."""
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
                     n_splits: int, seed: int
                     ) -> tuple[np.ndarray, list[dict]]:
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_records: list[dict] = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for f, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n))):
        w = _slsqp_blend_weights(P[tr_loc], y[tr_loc])
        oof[va_loc] = P[va_loc] @ w
        fold_records.append({
            "fold": int(f),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "weights": [float(x) for x in w],
        })
    return oof, fold_records


def _grid_2way(p_a: np.ndarray, p_b: np.ndarray, y: np.ndarray,
               step: float = 0.05
               ) -> tuple[list[dict], dict]:
    """Sweep w*p_a + (1-w)*p_b over w in {0, step, ..., 1}."""
    records: list[dict] = []
    ws = np.arange(0.0, 1.0 + step / 2, step)
    for w in ws:
        blend = w * p_a + (1.0 - w) * p_b
        r = float(rae(y, blend))
        records.append({"w_a": float(w), "rae": r})
    best = min(records, key=lambda r: r["rae"])
    return records, best


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 2-way blend  nb1462 (chemprop-embed) + nb1373 (AtomPair)")
    print(f"         grid sweep (step 0.05) + 5-fold SLSQP cross-fit")
    print(f"         gate vs nb1373 {ANCHOR_REF_COMPONENT:.4f} and "
          f"nb1391 {ANCHOR_REF_PRIOR_BLEND:.4f}  margin {MARGIN:.3f}")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    paths = {
        "nb1462_mean": DATA_PROCESSED / "nb1462_mean_bag_oof.npy",
        "nb1373_mean": DATA_PROCESSED / "nb1373_mean_bag_oof.npy",
    }
    for tag, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({tag}).")

    p_emb = np.load(paths["nb1462_mean"]).astype(np.float64)  # chemprop embed
    p_atp = np.load(paths["nb1373_mean"]).astype(np.float64)  # AtomPair

    for tag, arr in (("nb1462_mean", p_emb), ("nb1373_mean", p_atp)):
        if arr.shape[0] != n_unb:
            raise ValueError(
                f"shape mismatch: {tag}={arr.shape}, n_unb={n_unb}")

    rae_emb = float(rae(y_unb, p_emb))
    rae_atp = float(rae(y_unb, p_atp))
    print(f"[load] nb1462 mean (chemprop-embed, SHAP-pruned): "
          f"RAE = {rae_emb:.4f}  (ref {NB1462_MEAN_REF:.4f})")
    print(f"[load] nb1373 mean (AtomPair-30,    SHAP-pruned): "
          f"RAE = {rae_atp:.4f}  (ref {NB1373_MEAN_REF:.4f})")

    # Pearson -- raw preds AND residuals-from-truth.
    r_emb = p_emb - y_unb
    r_atp = p_atp - y_unb
    corr_pred = float(np.corrcoef(p_emb, p_atp)[0, 1])
    corr_resid = float(np.corrcoef(r_emb, r_atp)[0, 1])
    print(f"[diag] Pearson  pred(nb1462, nb1373)     = {corr_pred:+.4f}")
    print(f"[diag] Pearson  resid(nb1462, nb1373)    = {corr_resid:+.4f}")

    # (A) Grid sweep over w.
    print("\n" + "-" * 78)
    print("(A) Grid sweep  w*nb1462 + (1-w)*nb1373  (step 0.05)")
    print("-" * 78)
    grid, best_grid = _grid_2way(p_emb, p_atp, y_unb)
    print("    w_a (nb1462)   RAE")
    for r in grid:
        print(f"      {r['w_a']:5.2f}        {r['rae']:.4f}")
    top3 = sorted(grid, key=lambda r: r["rae"])[:3]
    print(f"  top-3:")
    for r in top3:
        print(f"     w_a={r['w_a']:.2f}   RAE={r['rae']:.4f}")
    print(f"  best grid  w_a={best_grid['w_a']:.2f}  RAE={best_grid['rae']:.4f}")

    # (B) SLSQP cross-fit (K=2 simplex).
    print("\n" + "-" * 78)
    print("(B) 5-fold SLSQP cross-fit (simplex w>=0, sum==1, K=2)")
    print("-" * 78)
    P = np.column_stack([p_emb, p_atp])  # (253, 2)
    slsqp_oof, fold_records = _slsqp_cross_fit(
        P, y_unb, n_splits=SLSQP_FOLDS, seed=SLSQP_SEED)
    rae_slsqp = float(rae(y_unb, slsqp_oof))
    fw = np.array([r["weights"] for r in fold_records])
    mean_w = fw.mean(axis=0).tolist()
    std_w = fw.std(axis=0).tolist()
    print(f"   SLSQP cross-fit  RAE = {rae_slsqp:.4f}")
    print(f"   mean fold weights:  w_nb1462={mean_w[0]:.4f}  "
          f"w_nb1373={mean_w[1]:.4f}  "
          f"(std {std_w[0]:.4f}/{std_w[1]:.4f})")

    # In-sample SLSQP (diagnostic only).
    w_full = _slsqp_blend_weights(P, y_unb)
    rae_full = float(rae(y_unb, P @ w_full))
    print(f"   in-sample SLSQP  w={w_full.tolist()}  RAE={rae_full:.4f}")

    # Choose best variant across (grid best, SLSQP cross-fit).
    candidates: list[dict] = []
    candidates.append({
        "name": "grid",
        "rae": best_grid["rae"], "w_a": best_grid["w_a"],
        "pred": best_grid["w_a"] * p_emb
                + (1.0 - best_grid["w_a"]) * p_atp,
    })
    candidates.append({
        "name": "slsqp_cross_fit",
        "rae": rae_slsqp, "w_a": None,
        "pred": slsqp_oof,
    })
    best = min(candidates, key=lambda c: c["rae"])
    best_rae = float(best["rae"])
    best_name = best["name"]

    beats_nb1373 = best_rae < ANCHOR_REF_COMPONENT - MARGIN
    flat_vs_nb1373 = abs(best_rae - ANCHOR_REF_COMPONENT) < MARGIN
    beats_nb1391 = best_rae < ANCHOR_REF_PRIOR_BLEND - MARGIN
    flat_vs_nb1391 = abs(best_rae - ANCHOR_REF_PRIOR_BLEND) < MARGIN

    if beats_nb1391:
        verdict = (f"NB1471_BEATS_NB1391_BY_MARGIN  (best={best_name})")
    elif beats_nb1373 and not beats_nb1391:
        verdict = (f"NB1471_BEATS_NB1373_BUT_NOT_NB1391  (best={best_name})")
    elif flat_vs_nb1373:
        verdict = f"NB1471_FLAT_VS_NB1373  (best={best_name})"
    else:
        verdict = f"NB1471_HURTS_VS_NB1373  (best={best_name})"

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1462 standalone         : {rae_emb:.4f}")
    print(f"   nb1373 standalone         : {rae_atp:.4f}")
    print(f"   grid best                 : {best_grid['rae']:.4f}  "
          f"@ w_nb1462={best_grid['w_a']:.2f}")
    print(f"   SLSQP cross-fit           : {rae_slsqp:.4f}")
    print(f"   anchor nb1373 (component) : {ANCHOR_REF_COMPONENT:.4f}")
    print(f"   anchor nb1391 (prior pair): {ANCHOR_REF_PRIOR_BLEND:.4f}")
    print(f"   best variant              : {best_name}  ({best_rae:.4f})")
    print(f"   delta vs nb1373           : "
          f"{best_rae - ANCHOR_REF_COMPONENT:+.4f}  (margin {MARGIN:.3f})")
    print(f"   delta vs nb1391           : "
          f"{best_rae - ANCHOR_REF_PRIOR_BLEND:+.4f}  (margin {MARGIN:.3f})")
    print(f"   beats_nb1373              : {beats_nb1373}")
    print(f"   beats_nb1391              : {beats_nb1391}")
    print(f"   verdict                   : {verdict}")

    best_oof = np.asarray(best["pred"], dtype=np.float32)
    np.save(DATA_PROCESSED / f"{TAG}_best_oof.npy", best_oof)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_best_oof.npy'}  "
          f"(variant={best_name})")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "components": [
            "nb1462_mean_bag_oof",
            "nb1373_mean_bag_oof",
        ],
        "rae_nb1462_standalone": rae_emb,
        "rae_nb1373_standalone": rae_atp,
        "pred_corr_pearson": corr_pred,
        "residual_corr_pearson": corr_resid,
        "grid": grid,
        "grid_top3": top3,
        "grid_best": best_grid,
        "rae_slsqp_cross_fit": rae_slsqp,
        "fold_records_slsqp": fold_records,
        "mean_fold_weights_slsqp": mean_w,
        "std_fold_weights_slsqp": std_w,
        "in_sample_slsqp_weights": [float(x) for x in w_full],
        "in_sample_slsqp_rae": rae_full,
        "best_variant": best_name,
        "best_rae": best_rae,
        "best_w_a_nb1462": best.get("w_a"),
        "delta_best_vs_nb1373": best_rae - ANCHOR_REF_COMPONENT,
        "delta_best_vs_nb1391": best_rae - ANCHOR_REF_PRIOR_BLEND,
        "beats_nb1373": bool(beats_nb1373),
        "flat_vs_nb1373": bool(flat_vs_nb1373),
        "beats_nb1391": bool(beats_nb1391),
        "flat_vs_nb1391": bool(flat_vs_nb1391),
        "verdict": verdict,
        "nb1462_mean_ref": NB1462_MEAN_REF,
        "nb1373_mean_ref": NB1373_MEAN_REF,
        "nb1391_ref": NB1391_REF,
        "margin": MARGIN,
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
        "rae_nb1462_standalone",
        "rae_nb1373_standalone",
        "pred_corr_pearson",
        "residual_corr_pearson",
        "grid_best",
        "rae_slsqp_cross_fit",
        "mean_fold_weights_slsqp",
        "best_variant", "best_rae", "best_w_a_nb1462",
        "delta_best_vs_nb1373", "delta_best_vs_nb1391",
        "beats_nb1373", "beats_nb1391", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
