"""nb1413 -- Direct 2-way blend nb1373 (AtomPair-30) + nb1364 (Mordred-30).

HYPOTHESIS
    nb1401 4-way blend (AtomPair + Mordred + MACCS + ChemBERTa) assigned
    Mordred ~30% weight. A direct AtomPair+Mordred 2-way may compete with
    nb1391 (AtomPair+MACCS) best-w 0.5076. The two feature families are
    orthogonal: AtomPair is hashed substructure topology, Mordred is
    physicochemical descriptors.

PROTOCOL
    1. Load nb1373_mean_bag_oof (RAE 0.5095) and nb1364_mean_bag_oof
       (RAE 0.5242). Also load *_median_bag_oof for diagnostic.
    2. Compute Pearson on raw preds AND residuals-from-truth.
    3. Grid-search w in {0.50..0.95 step 0.05} for w*nb1373 + (1-w)*nb1364
       on both (mean,mean) and (median,median) pairings.
    4. 5-fold SLSQP cross-fit on (mean,mean) and (median,median).
    5. Verdict at 0.003 margin vs nb1391 best (0.5076).

OUTPUTS
    data/processed/nb1413_summary.json
    data/processed/nb1413_best_oof.npy   (253,) float32 best variant predictor
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

TAG = "nb1413"
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# Reference numbers (pooled RAE on 253 unblind) from prior notebook summaries.
NB1373_MEAN_REF = 0.5095   # AtomPair-pruned mean-bag
NB1373_MEDIAN_REF = 0.5128 # AtomPair-pruned median-bag
NB1364_MEAN_REF = 0.5242   # Mordred-pruned   mean-bag
NB1364_MEDIAN_REF = 0.5282 # Mordred-pruned   median-bag
NB1391_BEST_REF = 0.5076   # nb1391 grid best (AtomPair+MACCS, w=0.85)

ANCHOR_REF = NB1391_BEST_REF   # gate against the existing 2-way best
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
               w_lo: float = 0.50, w_hi: float = 0.95, step: float = 0.05
               ) -> tuple[list[dict], dict]:
    """Sweep w*p_a + (1-w)*p_b over w in {w_lo, w_lo+step, ..., w_hi}."""
    records: list[dict] = []
    ws = np.arange(w_lo, w_hi + step / 2, step)
    for w in ws:
        blend = w * p_a + (1.0 - w) * p_b
        r = float(rae(y, blend))
        records.append({"w_a": float(w), "rae": r})
    best = min(records, key=lambda r: r["rae"])
    return records, best


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 2-way blend  nb1373 (AtomPair-30) + nb1364 (Mordred-30)")
    print(f"         grid sweep (w in [0.50..0.95] step 0.05) + 5f SLSQP")
    print(f"         gate vs nb1391_best {ANCHOR_REF:.4f}  margin {MARGIN:.3f}")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    paths = {
        "nb1373_mean":   DATA_PROCESSED / "nb1373_mean_bag_oof.npy",
        "nb1373_median": DATA_PROCESSED / "nb1373_median_bag_oof.npy",
        "nb1364_mean":   DATA_PROCESSED / "nb1364_mean_bag_oof.npy",
        "nb1364_median": DATA_PROCESSED / "nb1364_median_bag_oof.npy",
    }
    for tag, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({tag}).")

    p1m = np.load(paths["nb1373_mean"]).astype(np.float64)
    p1d = np.load(paths["nb1373_median"]).astype(np.float64)
    p2m = np.load(paths["nb1364_mean"]).astype(np.float64)
    p2d = np.load(paths["nb1364_median"]).astype(np.float64)

    for tag, arr in (
        ("nb1373_mean", p1m), ("nb1373_median", p1d),
        ("nb1364_mean", p2m), ("nb1364_median", p2d),
    ):
        if arr.shape[0] != n_unb:
            raise ValueError(
                f"shape mismatch: {tag}={arr.shape}, n_unb={n_unb}")

    rae_p1m = float(rae(y_unb, p1m))
    rae_p1d = float(rae(y_unb, p1d))
    rae_p2m = float(rae(y_unb, p2m))
    rae_p2d = float(rae(y_unb, p2d))
    print(f"[load] nb1373 mean   (AtomPair-30): RAE = {rae_p1m:.4f}  "
          f"(ref {NB1373_MEAN_REF:.4f})")
    print(f"[load] nb1373 median (AtomPair-30): RAE = {rae_p1d:.4f}  "
          f"(ref {NB1373_MEDIAN_REF:.4f})")
    print(f"[load] nb1364 mean   (Mordred-30) : RAE = {rae_p2m:.4f}  "
          f"(ref {NB1364_MEAN_REF:.4f})")
    print(f"[load] nb1364 median (Mordred-30) : RAE = {rae_p2d:.4f}  "
          f"(ref {NB1364_MEDIAN_REF:.4f})")

    # Pairwise correlations -- both raw preds and residuals-from-truth.
    preds = {"nb1373_mean": p1m, "nb1373_median": p1d,
             "nb1364_mean": p2m, "nb1364_median": p2d}
    resids = {k: v - y_unb for k, v in preds.items()}
    keys = list(preds.keys())
    corr_pred: dict[str, float] = {}
    corr_resid: dict[str, float] = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            corr_pred[f"{ki}__{kj}"] = float(
                np.corrcoef(preds[ki], preds[kj])[0, 1])
            corr_resid[f"{ki}__{kj}"] = float(
                np.corrcoef(resids[ki], resids[kj])[0, 1])
    print("[diag] pred Pearson:")
    for k, v in corr_pred.items():
        print(f"   {k:50s} {v:+.4f}")
    print("[diag] residual Pearson:")
    for k, v in corr_resid.items():
        print(f"   {k:50s} {v:+.4f}")

    # (A) Grid sweep -- both pairings.
    print("\n" + "-" * 78)
    print("(A) Grid sweep  w*nb1373 + (1-w)*nb1364  (w in [0.50..0.95] step 0.05)")
    print("-" * 78)
    grid_mm, best_mm = _grid_2way(p1m, p2m, y_unb)
    grid_dd, best_dd = _grid_2way(p1d, p2d, y_unb)
    # cross pairings (mean,median) and (median,mean) -- diagnostic only.
    grid_md, best_md = _grid_2way(p1m, p2d, y_unb)
    grid_dm, best_dm = _grid_2way(p1d, p2m, y_unb)

    def _print_full_grid(name: str, records: list[dict]) -> None:
        print(f"   {name}  full grid:")
        for r in records:
            print(f"     w={r['w_a']:.2f}   RAE={r['rae']:.4f}")

    _print_full_grid("(mean ,mean )", grid_mm)
    _print_full_grid("(median,median)", grid_dd)

    def _top3(name: str, records: list[dict]) -> list[dict]:
        top3 = sorted(records, key=lambda r: r["rae"])[:3]
        print(f"   {name}  top-3:")
        for r in top3:
            print(f"     w={r['w_a']:.2f}   RAE={r['rae']:.4f}")
        return top3

    top3_mm = _top3("(mean ,mean )", grid_mm)
    top3_dd = _top3("(median,median)", grid_dd)
    top3_md = _top3("(mean ,median)", grid_md)
    top3_dm = _top3("(median,mean )", grid_dm)

    # (B) SLSQP cross-fit on both pairings.
    print("\n" + "-" * 78)
    print("(B) 5-fold SLSQP cross-fit (simplex w>=0, sum==1, K=2)")
    print("-" * 78)
    P_mm = np.column_stack([p1m, p2m])  # (253, 2)
    P_dd = np.column_stack([p1d, p2d])
    slsqp_mm_oof, fold_mm = _slsqp_cross_fit(
        P_mm, y_unb, n_splits=SLSQP_FOLDS, seed=SLSQP_SEED)
    slsqp_dd_oof, fold_dd = _slsqp_cross_fit(
        P_dd, y_unb, n_splits=SLSQP_FOLDS, seed=SLSQP_SEED)
    rae_slsqp_mm = float(rae(y_unb, slsqp_mm_oof))
    rae_slsqp_dd = float(rae(y_unb, slsqp_dd_oof))
    print(f"   SLSQP cross-fit (mean ,mean )  RAE = {rae_slsqp_mm:.4f}")
    print(f"   SLSQP cross-fit (median,median) RAE = {rae_slsqp_dd:.4f}")

    fw_mm = np.array([r["weights"] for r in fold_mm])
    fw_dd = np.array([r["weights"] for r in fold_dd])
    mean_w_mm = fw_mm.mean(axis=0).tolist()
    std_w_mm = fw_mm.std(axis=0).tolist()
    mean_w_dd = fw_dd.mean(axis=0).tolist()
    std_w_dd = fw_dd.std(axis=0).tolist()
    print(f"   (mm) mean fold weights   w1373={mean_w_mm[0]:.4f}  "
          f"w1364={mean_w_mm[1]:.4f}  (std {std_w_mm[0]:.4f}/{std_w_mm[1]:.4f})")
    print(f"   (dd) mean fold weights   w1373={mean_w_dd[0]:.4f}  "
          f"w1364={mean_w_dd[1]:.4f}  (std {std_w_dd[0]:.4f}/{std_w_dd[1]:.4f})")

    # In-sample SLSQP (diagnostic only).
    w_full_mm = _slsqp_blend_weights(P_mm, y_unb)
    w_full_dd = _slsqp_blend_weights(P_dd, y_unb)
    p_full_mm = P_mm @ w_full_mm
    p_full_dd = P_dd @ w_full_dd
    rae_full_mm = float(rae(y_unb, p_full_mm))
    rae_full_dd = float(rae(y_unb, p_full_dd))
    print(f"   in-sample SLSQP (mm)  w={w_full_mm.tolist()}  "
          f"RAE={rae_full_mm:.4f}")
    print(f"   in-sample SLSQP (dd)  w={w_full_dd.tolist()}  "
          f"RAE={rae_full_dd:.4f}")

    # Choose best variant across (grid mm, grid dd, slsqp mm, slsqp dd).
    candidates: list[dict] = []
    candidates.append({
        "name": "grid_mean_mean",
        "rae": best_mm["rae"], "w_a": best_mm["w_a"],
        "pred": best_mm["w_a"] * p1m + (1.0 - best_mm["w_a"]) * p2m,
    })
    candidates.append({
        "name": "grid_median_median",
        "rae": best_dd["rae"], "w_a": best_dd["w_a"],
        "pred": best_dd["w_a"] * p1d + (1.0 - best_dd["w_a"]) * p2d,
    })
    candidates.append({
        "name": "slsqp_mean_mean", "rae": rae_slsqp_mm, "w_a": None,
        "pred": slsqp_mm_oof,
    })
    candidates.append({
        "name": "slsqp_median_median", "rae": rae_slsqp_dd, "w_a": None,
        "pred": slsqp_dd_oof,
    })
    best = min(candidates, key=lambda c: c["rae"])
    best_rae = float(best["rae"])
    best_name = best["name"]

    beats_nb1391 = best_rae < ANCHOR_REF - MARGIN
    flat_vs_nb1391 = abs(best_rae - ANCHOR_REF) < MARGIN

    if beats_nb1391:
        verdict = f"NB1413_BEATS_NB1391_BY_MARGIN  (best={best_name})"
    elif flat_vs_nb1391:
        verdict = f"NB1413_FLAT_VS_NB1391  (best={best_name})"
    else:
        verdict = f"NB1413_HURTS_VS_NB1391  (best={best_name})"

    # also diagnostic vs nb1373 standalone
    beats_nb1373 = best_rae < NB1373_MEAN_REF - MARGIN
    flat_vs_nb1373 = abs(best_rae - NB1373_MEAN_REF) < MARGIN

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1373 mean   standalone : {rae_p1m:.4f}")
    print(f"   nb1373 median standalone : {rae_p1d:.4f}")
    print(f"   nb1364 mean   standalone : {rae_p2m:.4f}")
    print(f"   nb1364 median standalone : {rae_p2d:.4f}")
    print(f"   grid best (mean,mean)    : {best_mm['rae']:.4f}  "
          f"@ w={best_mm['w_a']:.2f}")
    print(f"   grid best (median,median): {best_dd['rae']:.4f}  "
          f"@ w={best_dd['w_a']:.2f}")
    print(f"   SLSQP cross-fit (mean)   : {rae_slsqp_mm:.4f}")
    print(f"   SLSQP cross-fit (median) : {rae_slsqp_dd:.4f}")
    print(f"   anchor nb1391_best       : {ANCHOR_REF:.4f}")
    print(f"   best variant             : {best_name}  ({best_rae:.4f})")
    print(f"   margin vs nb1391_best    : "
          f"{best_rae - ANCHOR_REF:+.4f} (gate {MARGIN:.3f})")
    print(f"   beats_nb1391             : {beats_nb1391}")
    print(f"   beats_nb1373 (standalone): {beats_nb1373}")
    print(f"   verdict                  : {verdict}")

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
            "nb1373_mean_bag_oof",
            "nb1373_median_bag_oof",
            "nb1364_mean_bag_oof",
            "nb1364_median_bag_oof",
        ],
        "rae_nb1373_mean_standalone": rae_p1m,
        "rae_nb1373_median_standalone": rae_p1d,
        "rae_nb1364_mean_standalone": rae_p2m,
        "rae_nb1364_median_standalone": rae_p2d,
        "pred_corr_pearson": corr_pred,
        "residual_corr_pearson": corr_resid,
        "grid_full_mean_mean": grid_mm,
        "grid_full_median_median": grid_dd,
        "grid_top3_mean_mean": top3_mm,
        "grid_top3_median_median": top3_dd,
        "grid_top3_mean_median": top3_md,
        "grid_top3_median_mean": top3_dm,
        "grid_best_mean_mean": best_mm,
        "grid_best_median_median": best_dd,
        "grid_best_mean_median": best_md,
        "grid_best_median_mean": best_dm,
        "rae_slsqp_cross_fit_mean_mean": rae_slsqp_mm,
        "rae_slsqp_cross_fit_median_median": rae_slsqp_dd,
        "fold_records_slsqp_mean_mean": fold_mm,
        "fold_records_slsqp_median_median": fold_dd,
        "mean_fold_weights_slsqp_mean_mean": mean_w_mm,
        "std_fold_weights_slsqp_mean_mean": std_w_mm,
        "mean_fold_weights_slsqp_median_median": mean_w_dd,
        "std_fold_weights_slsqp_median_median": std_w_dd,
        "in_sample_slsqp_weights_mean_mean":
            [float(x) for x in w_full_mm],
        "in_sample_slsqp_rae_mean_mean": rae_full_mm,
        "in_sample_slsqp_weights_median_median":
            [float(x) for x in w_full_dd],
        "in_sample_slsqp_rae_median_median": rae_full_dd,
        "best_variant": best_name,
        "best_rae": best_rae,
        "best_w_a_nb1373": best.get("w_a"),
        "delta_best_vs_nb1391": best_rae - ANCHOR_REF,
        "delta_best_vs_nb1373_mean": best_rae - NB1373_MEAN_REF,
        "beats_nb1391": bool(beats_nb1391),
        "flat_vs_nb1391": bool(flat_vs_nb1391),
        "beats_nb1373": bool(beats_nb1373),
        "flat_vs_nb1373": bool(flat_vs_nb1373),
        "verdict": verdict,
        "nb1373_mean_ref": NB1373_MEAN_REF,
        "nb1373_median_ref": NB1373_MEDIAN_REF,
        "nb1364_mean_ref": NB1364_MEAN_REF,
        "nb1364_median_ref": NB1364_MEDIAN_REF,
        "nb1391_best_ref": NB1391_BEST_REF,
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
        "rae_nb1373_mean_standalone",
        "rae_nb1373_median_standalone",
        "rae_nb1364_mean_standalone",
        "rae_nb1364_median_standalone",
        "pred_corr_pearson",
        "residual_corr_pearson",
        "grid_best_mean_mean",
        "grid_best_median_median",
        "rae_slsqp_cross_fit_mean_mean",
        "rae_slsqp_cross_fit_median_median",
        "mean_fold_weights_slsqp_mean_mean",
        "mean_fold_weights_slsqp_median_median",
        "best_variant", "best_rae", "best_w_a_nb1373",
        "delta_best_vs_nb1391",
        "delta_best_vs_nb1373_mean",
        "beats_nb1391", "beats_nb1373", "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
