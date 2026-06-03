"""nb1091 -- Chain nb562-style scalar stretch on top of nb1070 median bag.

Hypothesis: nb1070's per-quantile-bin stretch (5 seeds, median-bagged) lands at
honest cross-fit pooled RAE 0.5771. The per-bin grid is bounded {0.80..2.00}
and operates within-bin, but if the overall residual distribution after
per-quantile stretch still has mild global under- or over-dispersion, a final
single scalar stretch centered on the post-bin mean can squeeze further.

Procedure (mirrors nb562 cross-fit, on top of nb1070 OOF rather than nb503):
  1) Reconstruct nb1070 OOF on the 253 unblind by re-running its per-seed
     KFold cross-fit and median-bagging. The deploy te_nb1070.npy is the
     all-253 deploy (different from OOF). For honest cross-fit on top, we
     need the OOF vector.
     Implementation note: nb1070 didn't persist the OOF, so we recompute it
     here using identical seeds/grid/n_bins to keep the chain reproducible.
  2) Honest 5-fold cross-fit (KFold seed=0) on the resulting 253 nb1070-OOF:
       - For each fold's train subset, grid-pick s in {0.95, 1.00, 1.05, 1.10}
         minimising RAE around mu = mean(nb1070_oof_train).
       - Apply chosen s to the held-out fold.
     Pooled OOF RAE = honest cross-fit RAE for the chained stretch.
  3) Deploy: fit (mu, s_global) on ALL 253 nb1070-OOF, apply to te_nb1070.npy.
  4) Save te_nb1091.npy + submissions/nb1091_chain_nb1070.csv + summary.

Outputs:
  data/processed/nb1091_pred_oof.npy
  data/processed/te_nb1091.npy
  data/processed/nb1091_summary.json
  submissions/nb1091_chain_nb1070.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1091"
ANCHOR = "nb1070"
PARENT_ANCHOR = "nb1014"
N_BINS = 5
NB1070_STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
NB1070_SEEDS = [0, 1, 7, 42, 137]
N_FOLDS = 5
CHAIN_STRETCH_GRID = [0.95, 1.00, 1.05, 1.10]
CHAIN_SEED = 0

NB1070_BAG_MEDIAN_RAE = 0.5771


# ---------- nb1070 per-quantile-bin stretch primitives (verbatim) ----------

def _bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def _fit_per_bin(p_tr: np.ndarray, y_tr: np.ndarray, edges: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
    bins_tr = _bin_assign(p_tr, edges)
    mus = np.zeros(N_BINS, dtype=np.float64)
    ss = np.ones(N_BINS, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins_tr == b
        if int(mask.sum()) < 2:
            mus[b] = float(p_tr.mean())
            ss[b] = 1.0
            continue
        mu_b = float(p_tr[mask].mean())
        mus[b] = mu_b
        y_b = y_tr[mask]
        p_b = p_tr[mask]
        best_s, best_r = 1.0, float("inf")
        for s in NB1070_STRETCH_GRID:
            r = float(rae(y_b, mu_b + s * (p_b - mu_b)))
            if r < best_r:
                best_r = r
                best_s = float(s)
        ss[b] = best_s
    return mus, ss


def _apply_per_bin(p: np.ndarray, edges: np.ndarray,
                   mus: np.ndarray, ss: np.ndarray) -> np.ndarray:
    bins = _bin_assign(p, edges)
    out = np.empty_like(p, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins == b
        if not mask.any():
            continue
        out[mask] = mus[b] + ss[b] * (p[mask] - mus[b])
    return out


def _nb1070_oof_for_seed(seed: int, p_unb: np.ndarray, y_unb: np.ndarray
                         ) -> np.ndarray:
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(p_tr, qs)
        mus, ss = _fit_per_bin(p_tr, y_tr, edges)
        oof[va_loc] = _apply_per_bin(p_unb[va_loc], edges, mus, ss)
    return oof


# ---------- nb562-style scalar stretch primitives ----------

def _stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def _best_s_grid(p_tr: np.ndarray, y_tr: np.ndarray, grid) -> tuple[float, float]:
    mu = float(p_tr.mean())
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        r = float(rae(y_tr, _stretch(p_tr, mu, s)))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, mu


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- chain nb562-style scalar stretch on top of {ANCHOR}")
    print(f"          chain grid = {CHAIN_STRETCH_GRID}")
    print("=" * 78)

    # ---- Load anchors ----
    te = load_test()
    te_names = te["name"].values
    preds_513_parent = np.load(DATA_PROCESSED / f"te_{PARENT_ANCHOR}.npy"
                                ).astype(np.float64)
    te_nb1070_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy"
                             ).astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb_parent = preds_513_parent[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] te_{PARENT_ANCHOR} shape={preds_513_parent.shape}  "
          f"te_{ANCHOR} shape={te_nb1070_513.shape}  "
          f"y shape={y_unb.shape}")

    # ---- Step 1: reconstruct nb1070 OOF (median bag across 5 seeds) ----
    print("\n" + "-" * 78)
    print(f"RECONSTRUCT {ANCHOR} OOF (per-seed cross-fit then median bag)")
    print("-" * 78)
    oof_stack = np.zeros((len(NB1070_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    for i, seed in enumerate(NB1070_SEEDS):
        oof_i = _nb1070_oof_for_seed(seed, p_unb_parent, y_unb)
        oof_stack[i] = oof_i
        r_i = float(rae(y_unb, oof_i))
        per_seed_rae.append(r_i)
        print(f"   seed {seed:>3d}: pooled_RAE = {r_i:.4f}")
    nb1070_oof = np.median(oof_stack, axis=0)
    nb1070_oof_rae = float(rae(y_unb, nb1070_oof))
    print(f"\n[bag] median-bagged {ANCHOR} OOF RAE = {nb1070_oof_rae:.4f}  "
          f"(reference {NB1070_BAG_MEDIAN_RAE:.4f})")

    # ---- Step 2: chained scalar stretch -- honest 5-fold cross-fit ----
    print("\n" + "-" * 78)
    print(f"CHAIN SCALAR STRETCH  honest {N_FOLDS}-fold cross-fit (seed={CHAIN_SEED})")
    print("-" * 78)
    chain_oof = np.full(n_unb, np.nan, dtype=np.float64)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=CHAIN_SEED)
    per_fold: list[dict] = []
    for fold_i, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        p_tr = nb1070_oof[tr_loc]
        y_tr = y_unb[tr_loc]
        p_va = nb1070_oof[va_loc]
        best_s, mu_tr = _best_s_grid(p_tr, y_tr, CHAIN_STRETCH_GRID)
        chain_oof[va_loc] = _stretch(p_va, mu_tr, best_s)
        per_fold.append({
            "fold": fold_i, "mu": mu_tr, "s": best_s,
            "n_train": int(len(tr_loc)), "n_val": int(len(va_loc)),
        })
        print(f"   fold {fold_i}: mu={mu_tr:.4f}  best_s={best_s:.2f}  "
              f"n_tr={len(tr_loc)}  n_va={len(va_loc)}")

    chain_oof_rae = float(rae(y_unb, chain_oof))
    delta_vs_nb1070 = chain_oof_rae - nb1070_oof_rae
    beats_nb1070 = chain_oof_rae < nb1070_oof_rae
    print(f"\n[result] {TAG} chained OOF pooled RAE = {chain_oof_rae:.4f}  "
          f"(delta vs {ANCHOR} = {delta_vs_nb1070:+.4f})")
    print(f"[result] beats_{ANCHOR} = {beats_nb1070}")

    # ---- Per-grid diagnostic: pooled RAE if s were fixed globally ----
    mu_full = float(nb1070_oof.mean())
    fixed_s_rae = {}
    for s in CHAIN_STRETCH_GRID:
        fixed_s_rae[f"{s:.2f}"] = float(rae(y_unb,
                                              _stretch(nb1070_oof, mu_full, s)))
    print("\n[diag] in-sample pooled RAE under fixed global s:")
    for k, v in fixed_s_rae.items():
        print(f"   s = {k}: {v:.4f}")

    # ---- Step 3: deploy on 513 -- fit on ALL 253 nb1070-OOF, apply to te_nb1070 ----
    print("\n" + "-" * 78)
    print("DEPLOY  fit (mu, s) on all 253 nb1070-OOF, apply to te_nb1070 (513)")
    print("-" * 78)
    best_s_deploy, mu_deploy = _best_s_grid(nb1070_oof, y_unb, CHAIN_STRETCH_GRID)
    deploy_513 = _stretch(te_nb1070_513, mu_deploy, best_s_deploy
                          ).astype(np.float32)
    in_rae_deploy = float(rae(y_unb,
                                deploy_513[unb_idx].astype(np.float64)))
    print(f"   deploy mu={mu_deploy:.4f}  s={best_s_deploy:.2f}")
    print(f"   te_{ANCHOR} (513)  mean={te_nb1070_513.mean():.3f}  "
          f"std={te_nb1070_513.std():.3f}")
    print(f"   te_{TAG}   (513)  mean={deploy_513.mean():.3f}  "
          f"std={deploy_513.std():.3f}  "
          f"in-sample 253 RAE = {in_rae_deploy:.4f}")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy",
            chain_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_chain_nb1070.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] {TAG}_pred_oof.npy  te_{TAG}.npy  {plain}")

    if delta_vs_nb1070 <= -0.001:
        verdict = "CHAIN_HELPS"
    elif abs(delta_vs_nb1070) < 0.001:
        verdict = "CHAIN_NEUTRAL"
    else:
        verdict = "CHAIN_HURTS"

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "parent_anchor": PARENT_ANCHOR,
        "n_bins": N_BINS,
        "n_folds": N_FOLDS,
        "chain_seed": CHAIN_SEED,
        "chain_stretch_grid": CHAIN_STRETCH_GRID,
        "nb1070_seeds": NB1070_SEEDS,
        "per_seed_rae": per_seed_rae,
        "nb1070_oof_reconstructed_rae": nb1070_oof_rae,
        "nb1070_oof_reference_rae": NB1070_BAG_MEDIAN_RAE,
        "chain_oof_rae": chain_oof_rae,
        "delta_vs_nb1070": delta_vs_nb1070,
        "beats_nb1070": bool(beats_nb1070),
        "verdict": verdict,
        "fixed_global_s_in_sample_rae": fixed_s_rae,
        "per_fold": per_fold,
        "deploy_mu": mu_deploy,
        "deploy_s": best_s_deploy,
        "in_rae_deploy_on_253": in_rae_deploy,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "anchor_te_mean": float(te_nb1070_513.mean()),
        "anchor_te_std": float(te_nb1070_513.std()),
        "plain_submission": str(plain),
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
    for k in ("nb1070_oof_reconstructed_rae", "chain_oof_rae",
              "delta_vs_nb1070", "beats_nb1070", "verdict",
              "deploy_s", "deploy_mu", "in_rae_deploy_on_253",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
