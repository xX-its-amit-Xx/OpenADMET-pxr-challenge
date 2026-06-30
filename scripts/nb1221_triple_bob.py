"""nb1221 -- Triple BoB blend: nb1190 BoB (triple-FP) + nb1200 BoB (MACCS) + 3rd-axis.

Hypothesis:
    nb1211 mean (2-way BoB blend nb1190+nb1200) currently sits at pooled RAE
    0.5451 on the 253 unblind. Both components are BoB ensembles -- the gain
    over their standalone numbers (0.5499 / 0.5495) is small because the two
    BoB streams already share variance (residual Pearson ~0.97).

    Adding a 3rd axis -- specifically a non-BoB single-bag with a distinct
    chemistry-feature family -- might widen the orthogonal-residual basin:
        * nb1153 mean_bag (Mordred residual, single-bag): standalone 0.5640.
          Lower quality than the two BoBs but uses a different feature family
          (Mordred 2D descriptor library), so residual-decorrelation may carry.
        * nb1172 mean_bag (AtomPair residual, single-bag): standalone 0.5659.
          Same idea -- distinct topology from Morgan/RDKit-desc/MACCS.

    Three variants tested:
        (A) [nb1190_bob, nb1200_bob, nb1153_mean_bag]    -- BoB + BoB + Mordred
        (B) [nb1190_bob, nb1200_bob, nb1172_mean_bag]    -- BoB + BoB + AtomPair
        (C) [nb1190_bob, nb1200_bob, nb1153_mean_bag, nb1172_mean_bag] -- 4-way

Protocol:
  1. Load nb1190_bob_mean_oof, nb1200_bob_mean_oof, nb1153_mean_bag_oof,
     nb1172_mean_bag_oof; all 253 unblind.
  2. For each variant (A/B/C):
       5-fold cross-fit SLSQP on simplex
       naive equal-weight mean
       naive median
       pooled RAE on each.
  3. Verdict per variant at 0.003 margin vs nb1211 mean 0.5451.
  4. Pairwise residual Pearson on all 4 components for diagnostic.

NO deploy (513) refit -- 253-only honest cross-fit diagnostic.

Outputs:
  data/processed/nb1221_A_slsqp_oof.npy   (253,) float32
  data/processed/nb1221_A_mean_oof.npy    (253,) float32
  data/processed/nb1221_A_median_oof.npy  (253,) float32
  data/processed/nb1221_B_slsqp_oof.npy   (253,) float32
  data/processed/nb1221_B_mean_oof.npy    (253,) float32
  data/processed/nb1221_B_median_oof.npy  (253,) float32
  data/processed/nb1221_C_slsqp_oof.npy   (253,) float32
  data/processed/nb1221_C_mean_oof.npy    (253,) float32
  data/processed/nb1221_C_median_oof.npy  (253,) float32
  data/processed/nb1221_summary.json
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

TAG = "nb1221"
SLSQP_FOLDS = 5
SLSQP_SEED = 42

# Reference numbers (pooled RAE on 253 unblind).
NB1190_BOB_MEAN_REF = 0.5499
NB1200_BOB_MEAN_REF = 0.5491
NB1153_MEAN_BAG_REF = 0.5640
NB1172_MEAN_BAG_REF = 0.5659
NB1211_MEAN_REF = 0.5451  # 2-way nb1190+nb1200 naive mean baseline


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    """Argmin MSE over the K-simplex (w_i >= 0, sum w_i = 1)."""
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w: np.ndarray) -> float:
        pred = P_tr @ w
        diff = y_tr - pred
        return float(np.mean(diff * diff))

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
                     n_splits: int, seed: int) -> tuple[np.ndarray, list[dict]]:
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


def _eval_block(label: str, P: np.ndarray, y: np.ndarray,
                comp_tags: list[str]) -> dict:
    """Run SLSQP cross-fit + mean + median on a stack P (n, K)."""
    print("\n" + "-" * 78)
    print(f"  BLOCK: {label}   K={P.shape[1]}  components={comp_tags}")
    print("-" * 78)

    slsqp_oof, fold_records = _slsqp_cross_fit(P, y, SLSQP_FOLDS, SLSQP_SEED)
    rae_slsqp = float(rae(y, slsqp_oof))

    fold_weights = np.array([r["weights"] for r in fold_records])
    mean_weights = fold_weights.mean(axis=0)
    print(f"   per-fold weights:")
    for rec in fold_records:
        w = rec["weights"]
        wstr = "  ".join(f"w[{comp_tags[i]}]={w[i]:.4f}" for i in range(len(w)))
        print(f"     fold {rec['fold']}: {wstr}")
    mstr = "  ".join(f"w[{comp_tags[i]}]={mean_weights[i]:.4f}"
                     for i in range(len(mean_weights)))
    print(f"   mean weights across folds:  {mstr}")
    print(f"   pooled RAE(SLSQP cross-fit) = {rae_slsqp:.4f}")

    w_full = _slsqp_blend_weights(P, y)
    p_full = P @ w_full
    rae_full = float(rae(y, p_full))
    fstr = "  ".join(f"w[{comp_tags[i]}]={w_full[i]:.4f}"
                     for i in range(len(w_full)))
    print(f"   in-sample SLSQP weights:    {fstr}   RAE = {rae_full:.4f}")

    mean_oof = P.mean(axis=1)
    rae_mean = float(rae(y, mean_oof))
    print(f"   pooled RAE(naive mean)   = {rae_mean:.4f}")

    median_oof = np.median(P, axis=1)
    rae_median = float(rae(y, median_oof))
    print(f"   pooled RAE(naive median) = {rae_median:.4f}")

    return {
        "label": label,
        "components": comp_tags,
        "fold_records": fold_records,
        "mean_fold_weights": [float(x) for x in mean_weights],
        "in_sample_slsqp_weights": [float(x) for x in w_full],
        "in_sample_slsqp_rae": rae_full,
        "rae_slsqp_cross_fit": rae_slsqp,
        "rae_naive_mean": rae_mean,
        "rae_naive_median": rae_median,
        "slsqp_oof": slsqp_oof,
        "mean_oof": mean_oof,
        "median_oof": median_oof,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- triple-BoB blend: nb1190 BoB + nb1200 BoB + 3rd axis")
    print(f"          (A) +nb1153 (Mordred)  (B) +nb1172 (AtomPair)  (C) 4-way")
    print(f"          5-fold SLSQP cross-fit + naive mean + naive median")
    print(f"          verdict vs nb1211 mean baseline 0.5451")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    paths = {
        "nb1190_bob":      DATA_PROCESSED / "nb1190_bob_mean_oof.npy",
        "nb1200_bob":      DATA_PROCESSED / "nb1200_bob_mean_oof.npy",
        "nb1153_mean_bag": DATA_PROCESSED / "nb1153_mean_bag_oof.npy",
        "nb1172_mean_bag": DATA_PROCESSED / "nb1172_mean_bag_oof.npy",
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({k})")

    preds = {k: np.load(p).astype(np.float64) for k, p in paths.items()}
    for k, v in preds.items():
        if v.shape[0] != n_unb:
            raise ValueError(f"shape mismatch: {k}={v.shape}, n_unb={n_unb}")

    standalone_rae = {k: float(rae(y_unb, v)) for k, v in preds.items()}
    print("\n[load] standalone pooled RAE on 253 unblind:")
    print(f"   nb1190 BoB      : {standalone_rae['nb1190_bob']:.4f}  "
          f"(ref {NB1190_BOB_MEAN_REF:.4f})")
    print(f"   nb1200 BoB      : {standalone_rae['nb1200_bob']:.4f}  "
          f"(ref {NB1200_BOB_MEAN_REF:.4f})")
    print(f"   nb1153 mean_bag : {standalone_rae['nb1153_mean_bag']:.4f}  "
          f"(ref {NB1153_MEAN_BAG_REF:.4f})")
    print(f"   nb1172 mean_bag : {standalone_rae['nb1172_mean_bag']:.4f}  "
          f"(ref {NB1172_MEAN_BAG_REF:.4f})")
    print(f"   nb1211 mean baseline : {NB1211_MEAN_REF:.4f}")

    # Pairwise residual Pearson on all 4 components.
    comp_keys = ["nb1190_bob", "nb1200_bob", "nb1153_mean_bag", "nb1172_mean_bag"]
    K = len(comp_keys)
    pred_corr = np.zeros((K, K), dtype=np.float64)
    resid_corr = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            pi = preds[comp_keys[i]]
            pj = preds[comp_keys[j]]
            pred_corr[i, j] = float(np.corrcoef(pi, pj)[0, 1])
            resid_corr[i, j] = float(np.corrcoef(pi - y_unb, pj - y_unb)[0, 1])
    print("\n[diag] pairwise PRED Pearson correlation:")
    print(f"            {'  '.join(f'{k:>16s}' for k in comp_keys)}")
    for i, ki in enumerate(comp_keys):
        row = "  ".join(f"{pred_corr[i, j]:16.4f}" for j in range(K))
        print(f"   {ki:>16s}  {row}")
    print("\n[diag] pairwise RESIDUAL Pearson correlation:")
    print(f"            {'  '.join(f'{k:>16s}' for k in comp_keys)}")
    for i, ki in enumerate(comp_keys):
        row = "  ".join(f"{resid_corr[i, j]:16.4f}" for j in range(K))
        print(f"   {ki:>16s}  {row}")

    # ---- Variant A: [nb1190, nb1200, nb1153] ----
    P_A = np.column_stack([preds["nb1190_bob"], preds["nb1200_bob"],
                           preds["nb1153_mean_bag"]])
    block_A = _eval_block(
        "VARIANT A: nb1190_bob + nb1200_bob + nb1153_mean_bag", P_A, y_unb,
        comp_tags=["nb1190_bob", "nb1200_bob", "nb1153_mean_bag"],
    )

    # ---- Variant B: [nb1190, nb1200, nb1172] ----
    P_B = np.column_stack([preds["nb1190_bob"], preds["nb1200_bob"],
                           preds["nb1172_mean_bag"]])
    block_B = _eval_block(
        "VARIANT B: nb1190_bob + nb1200_bob + nb1172_mean_bag", P_B, y_unb,
        comp_tags=["nb1190_bob", "nb1200_bob", "nb1172_mean_bag"],
    )

    # ---- Variant C: [nb1190, nb1200, nb1153, nb1172] (4-way) ----
    P_C = np.column_stack([preds["nb1190_bob"], preds["nb1200_bob"],
                           preds["nb1153_mean_bag"], preds["nb1172_mean_bag"]])
    block_C = _eval_block(
        "VARIANT C: nb1190_bob + nb1200_bob + nb1153_mean_bag + nb1172_mean_bag",
        P_C, y_unb,
        comp_tags=["nb1190_bob", "nb1200_bob", "nb1153_mean_bag", "nb1172_mean_bag"],
    )

    # ---- Save per-variant artifacts ----
    for tag, block in [("A", block_A), ("B", block_B), ("C", block_C)]:
        np.save(DATA_PROCESSED / f"{TAG}_{tag}_slsqp_oof.npy",
                block["slsqp_oof"].astype(np.float32))
        np.save(DATA_PROCESSED / f"{TAG}_{tag}_mean_oof.npy",
                block["mean_oof"].astype(np.float32))
        np.save(DATA_PROCESSED / f"{TAG}_{tag}_median_oof.npy",
                block["median_oof"].astype(np.float32))
        print(f"\n[save] {DATA_PROCESSED / f'{TAG}_{tag}_slsqp_oof.npy'}")
        print(f"[save] {DATA_PROCESSED / f'{TAG}_{tag}_mean_oof.npy'}")
        print(f"[save] {DATA_PROCESSED / f'{TAG}_{tag}_median_oof.npy'}")

    # ---- Verdict per variant vs nb1211 mean 0.5451 ----
    def _verdict_for(block: dict, label: str) -> tuple[str, float, str]:
        candidates = {
            f"{label}_slsqp":  block["rae_slsqp_cross_fit"],
            f"{label}_mean":   block["rae_naive_mean"],
            f"{label}_median": block["rae_naive_median"],
        }
        best_tag = min(candidates, key=candidates.get)
        best_rae = candidates[best_tag]
        delta = best_rae - NB1211_MEAN_REF
        if best_rae < NB1211_MEAN_REF - 0.003:
            verdict = f"VARIANT_{label}_BEATS_NB1211_MEAN ({best_tag} @ {best_rae:.4f})"
        elif abs(delta) < 0.003:
            verdict = f"VARIANT_{label}_FLAT_VS_NB1211_MEAN ({best_tag} @ {best_rae:.4f})"
        else:
            verdict = f"VARIANT_{label}_HURTS_VS_NB1211_MEAN ({best_tag} @ {best_rae:.4f})"
        return verdict, best_rae, best_tag

    verdict_A, best_A, best_tag_A = _verdict_for(block_A, "A")
    verdict_B, best_B, best_tag_B = _verdict_for(block_B, "B")
    verdict_C, best_C, best_tag_C = _verdict_for(block_C, "C")

    # All 9 candidates (3 variants x {slsqp, mean, median}).
    all_candidates = {
        "A_slsqp":  block_A["rae_slsqp_cross_fit"],
        "A_mean":   block_A["rae_naive_mean"],
        "A_median": block_A["rae_naive_median"],
        "B_slsqp":  block_B["rae_slsqp_cross_fit"],
        "B_mean":   block_B["rae_naive_mean"],
        "B_median": block_B["rae_naive_median"],
        "C_slsqp":  block_C["rae_slsqp_cross_fit"],
        "C_mean":   block_C["rae_naive_mean"],
        "C_median": block_C["rae_naive_median"],
    }
    best_overall_tag = min(all_candidates, key=all_candidates.get)
    best_overall_rae = all_candidates[best_overall_tag]
    if best_overall_rae < NB1211_MEAN_REF - 0.003:
        overall_verdict = (f"OVERALL_BEATS_NB1211_MEAN "
                           f"({best_overall_tag} @ {best_overall_rae:.4f})")
    elif abs(best_overall_rae - NB1211_MEAN_REF) < 0.003:
        overall_verdict = (f"OVERALL_FLAT_VS_NB1211_MEAN "
                           f"({best_overall_tag} @ {best_overall_rae:.4f})")
    else:
        overall_verdict = (f"OVERALL_HURTS_VS_NB1211_MEAN "
                           f"({best_overall_tag} @ {best_overall_rae:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1211 mean baseline       : {NB1211_MEAN_REF:.4f}")
    print(f"")
    print(f"   ---- Variant A (nb1190 + nb1200 + nb1153 Mordred) ----")
    print(f"     SLSQP   : {block_A['rae_slsqp_cross_fit']:.4f}")
    print(f"     mean    : {block_A['rae_naive_mean']:.4f}")
    print(f"     median  : {block_A['rae_naive_median']:.4f}")
    print(f"     verdict : {verdict_A}")
    print(f"")
    print(f"   ---- Variant B (nb1190 + nb1200 + nb1172 AtomPair) ----")
    print(f"     SLSQP   : {block_B['rae_slsqp_cross_fit']:.4f}")
    print(f"     mean    : {block_B['rae_naive_mean']:.4f}")
    print(f"     median  : {block_B['rae_naive_median']:.4f}")
    print(f"     verdict : {verdict_B}")
    print(f"")
    print(f"   ---- Variant C (4-way: nb1190 + nb1200 + nb1153 + nb1172) ----")
    print(f"     SLSQP   : {block_C['rae_slsqp_cross_fit']:.4f}")
    print(f"     mean    : {block_C['rae_naive_mean']:.4f}")
    print(f"     median  : {block_C['rae_naive_median']:.4f}")
    print(f"     verdict : {verdict_C}")
    print(f"")
    print(f"   ---- Overall best ----")
    print(f"     {best_overall_tag} @ {best_overall_rae:.4f}  "
          f"(delta vs nb1211 mean = {best_overall_rae - NB1211_MEAN_REF:+.4f})")
    print(f"     verdict : {overall_verdict}")

    def _block_summary(block: dict) -> dict:
        return {k: v for k, v in block.items()
                if k not in ("slsqp_oof", "mean_oof", "median_oof")}

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "slsqp_folds": SLSQP_FOLDS,
        "slsqp_seed": SLSQP_SEED,
        "components": comp_keys,
        "standalone_rae": standalone_rae,
        "pred_corr_matrix": pred_corr.tolist(),
        "residual_corr_matrix": resid_corr.tolist(),
        "pred_corr_keys": comp_keys,
        "block_A": _block_summary(block_A),
        "block_B": _block_summary(block_B),
        "block_C": _block_summary(block_C),
        "candidate_rae_table": all_candidates,
        "best_overall_tag": best_overall_tag,
        "best_overall_rae": best_overall_rae,
        "best_A_tag": best_tag_A,
        "best_A_rae": best_A,
        "best_B_tag": best_tag_B,
        "best_B_rae": best_B,
        "best_C_tag": best_tag_C,
        "best_C_rae": best_C,
        "nb1211_mean_ref": NB1211_MEAN_REF,
        "nb1190_bob_ref": NB1190_BOB_MEAN_REF,
        "nb1200_bob_ref": NB1200_BOB_MEAN_REF,
        "nb1153_mean_bag_ref": NB1153_MEAN_BAG_REF,
        "nb1172_mean_bag_ref": NB1172_MEAN_BAG_REF,
        "delta_A_best_vs_nb1211_mean": best_A - NB1211_MEAN_REF,
        "delta_B_best_vs_nb1211_mean": best_B - NB1211_MEAN_REF,
        "delta_C_best_vs_nb1211_mean": best_C - NB1211_MEAN_REF,
        "delta_overall_vs_nb1211_mean": best_overall_rae - NB1211_MEAN_REF,
        "verdict_A": verdict_A,
        "verdict_B": verdict_B,
        "verdict_C": verdict_C,
        "verdict_overall": overall_verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("standalone_rae",
              "candidate_rae_table",
              "best_A_tag", "best_A_rae",
              "best_B_tag", "best_B_rae",
              "best_C_tag", "best_C_rae",
              "best_overall_tag", "best_overall_rae",
              "delta_A_best_vs_nb1211_mean",
              "delta_B_best_vs_nb1211_mean",
              "delta_C_best_vs_nb1211_mean",
              "delta_overall_vs_nb1211_mean",
              "verdict_A", "verdict_B", "verdict_C", "verdict_overall"):
        print(f"  {k}: {res.get(k)}")
