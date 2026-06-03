"""nb1114 -- Conformal-QR-style uncertainty-gated shrinkage on nb1070.

Hypothesis: high-uncertainty test compounds benefit from shrinkage toward the
train median (4.65); narrow-interval (confident) compounds do not. We measure
per-compound uncertainty as the std across nb1070's 5-seed bag of OOF (or
deploy) predictions and gate a hard shrinkage rule on the top-quartile of
that std.

Procedure:
  1. Reproduce nb1070's per-seed honest cross-fit on the 253 unblind to obtain
     a (5, 253) OOF stack.  Per-compound std = std across seed axis.
  2. WIDE   = compounds with std >= q75 of std.
  3. NARROW = the rest.
  4. WIDE   -> shrunk = 0.3 * 4.65 + 0.7 * nb1070_oof
     NARROW -> keep nb1070_oof.
  5. Honest cross-fit RAE on 253 unblind.
  6. Same gating applied to the 513 deploy: per-compound std taken across
     the 5 per-seed deploy_513 predictions (re-uses nb1070's deploy stack
     protocol).

Outputs:
  data/processed/te_nb1114.npy
  data/processed/nb1114_summary.json
  submissions/nb1114_conformal_qr.csv
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

TAG = "nb1114"
ANCHOR = "nb1014"
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]

TRAIN_MEDIAN = 4.65
SHRINK_FACTOR = 0.3            # weight on TRAIN_MEDIAN for wide compounds
WIDE_QUANTILE = 0.75           # top quartile of per-compound std = WIDE

NB1070_BAG_MEDIAN_RAE = 0.5780  # reference, taken from nb1070 summary (anchor)


def bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def fit_per_bin_stretch(p_train: np.ndarray, y_train: np.ndarray,
                        edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins_tr = bin_assign(p_train, edges)
    mus = np.zeros(N_BINS, dtype=np.float64)
    ss = np.ones(N_BINS, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins_tr == b
        n_b = int(mask.sum())
        if n_b < 2:
            mus[b] = float(p_train.mean())
            ss[b] = 1.0
            continue
        mu_b = float(p_train[mask].mean())
        mus[b] = mu_b
        y_b = y_train[mask]
        p_b = p_train[mask]
        best_s, best_r = 1.0, float("inf")
        for s in STRETCH_GRID:
            stretched = mu_b + s * (p_b - mu_b)
            r = float(rae(y_b, stretched))
            if r < best_r:
                best_r = r
                best_s = float(s)
        ss[b] = best_s
    return mus, ss


def apply_per_bin_stretch(p: np.ndarray, edges: np.ndarray,
                          mus: np.ndarray, ss: np.ndarray) -> np.ndarray:
    bins = bin_assign(p, edges)
    out = np.empty_like(p, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins == b
        if not mask.any():
            continue
        out[mask] = mus[b] + ss[b] * (p[mask] - mus[b])
    return out


def run_one_seed_oof(seed: int, p_unb: np.ndarray, y_unb: np.ndarray
                     ) -> tuple[np.ndarray, list[list[float]]]:
    n = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    per_fold_ss: list[list[float]] = []
    for tr_loc, va_loc in kf.split(np.arange(n)):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        p_va = p_unb[va_loc]
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(p_tr, qs)
        mus, ss = fit_per_bin_stretch(p_tr, y_tr, edges)
        oof[va_loc] = apply_per_bin_stretch(p_va, edges, mus, ss)
        per_fold_ss.append(ss.tolist())
    return oof, per_fold_ss


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- uncertainty-gated shrink on nb1070 5-seed bag")
    print(f"          WIDE = top quartile std across {len(SEEDS)} seeds")
    print(f"          shrink: WIDE -> {SHRINK_FACTOR}*{TRAIN_MEDIAN} + "
          f"{1-SHRINK_FACTOR}*nb1070")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    preds_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    p_unb = preds_513[unb_idx]
    n_unb = len(y_unb)

    in_rae_anchor = float(rae(y_unb, p_unb))
    print(f"[baseline] in_RAE(te_{ANCHOR} on 253 unblind) = "
          f"{in_rae_anchor:.4f}")

    # ---- Per-seed OOF stack ----
    print("\n" + "-" * 78)
    print("PER-SEED OOF STACK (reproduces nb1070)")
    print("-" * 78)
    oof_stack = np.zeros((len(SEEDS), n_unb), dtype=np.float64)
    per_seed_records: list[dict] = []
    for i, seed in enumerate(SEEDS):
        oof, per_fold_ss = run_one_seed_oof(seed, p_unb, y_unb)
        oof_stack[i] = oof
        pooled = float(rae(y_unb, oof))
        per_seed_records.append({
            "seed": seed,
            "pooled_rae": pooled,
            "per_fold_ss": per_fold_ss,
        })
        print(f"   seed {seed:>3d}: pooled_RAE = {pooled:.4f}")

    nb1070_oof = np.median(oof_stack, axis=0)            # nb1070 baseline
    bag_median_rae = float(rae(y_unb, nb1070_oof))
    print(f"\n[nb1070] median-bag OOF RAE = {bag_median_rae:.4f}")

    # ---- Per-compound prediction-interval width = std across seeds ----
    row_std = oof_stack.std(axis=0)
    q75 = float(np.quantile(row_std, WIDE_QUANTILE))
    wide_mask = row_std >= q75
    narrow_mask = ~wide_mask
    n_wide = int(wide_mask.sum())
    n_narrow = int(narrow_mask.sum())
    print(f"\n[gate] per-compound across-seed std:")
    print(f"       min={row_std.min():.4f}  median={np.median(row_std):.4f}  "
          f"q75={q75:.4f}  max={row_std.max():.4f}")
    print(f"       WIDE (std >= q75)   = {n_wide:>3d}  "
          f"({100*n_wide/n_unb:.1f}%)")
    print(f"       NARROW              = {n_narrow:>3d}  "
          f"({100*n_narrow/n_unb:.1f}%)")

    # ---- Shrink wide toward train median ----
    shrunk_oof = nb1070_oof.copy()
    shrunk_oof[wide_mask] = (
        SHRINK_FACTOR * TRAIN_MEDIAN
        + (1.0 - SHRINK_FACTOR) * nb1070_oof[wide_mask]
    )

    shrunk_rae = float(rae(y_unb, shrunk_oof))
    delta_vs_nb1070 = shrunk_rae - bag_median_rae

    # Diagnostic: how does the shrink act on WIDE-only subset?
    wide_rae_before = float(rae(y_unb[wide_mask], nb1070_oof[wide_mask]))
    wide_rae_after = float(rae(y_unb[wide_mask], shrunk_oof[wide_mask]))
    narrow_rae = float(rae(y_unb[narrow_mask], nb1070_oof[narrow_mask]))

    print(f"\n[shrink] WIDE-only RAE   before={wide_rae_before:.4f}  "
          f"after={wide_rae_after:.4f}  "
          f"delta={wide_rae_after - wide_rae_before:+.4f}")
    print(f"[shrink] NARROW-only RAE = {narrow_rae:.4f}  (unchanged)")
    print(f"[shrink] OVERALL OOF RAE = {shrunk_rae:.4f}  "
          f"(vs nb1070 {bag_median_rae:.4f}, delta {delta_vs_nb1070:+.4f})")

    # =================================================================
    # Deploy: re-run the nb1070 5-seed deploy bag for the 513 to get a
    # (5, 513) stack and apply the same WIDE/NARROW gating with std on the
    # deploy stack itself. WIDE 513-rows are shrunk toward 4.65.
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY (per-seed fit on all 253, gate via deploy-stack std)")
    print("-" * 78)
    deploy_stack = np.zeros((len(SEEDS), preds_513.shape[0]),
                            dtype=np.float64)
    qs_all = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
    edges_all = np.quantile(p_unb, qs_all)
    mus_all, _ = fit_per_bin_stretch(p_unb, y_unb, edges_all)
    for i, seed in enumerate(SEEDS):
        rec = per_seed_records[i]
        per_fold_ss_arr = np.array(rec["per_fold_ss"])  # (5, N_BINS)
        ss_seed = per_fold_ss_arr.mean(axis=0)
        deploy_seed_513 = apply_per_bin_stretch(preds_513, edges_all,
                                                mus_all, ss_seed)
        deploy_stack[i] = deploy_seed_513

    deploy_median_513 = np.median(deploy_stack, axis=0)   # nb1070-equivalent

    deploy_row_std = deploy_stack.std(axis=0)
    q75_deploy = float(np.quantile(deploy_row_std, WIDE_QUANTILE))
    wide_mask_513 = deploy_row_std >= q75_deploy
    n_wide_513 = int(wide_mask_513.sum())
    print(f"[deploy-gate] 513 std: min={deploy_row_std.min():.4f}  "
          f"median={np.median(deploy_row_std):.4f}  "
          f"q75={q75_deploy:.4f}  max={deploy_row_std.max():.4f}")
    print(f"              WIDE (513) = {n_wide_513}  "
          f"({100*n_wide_513/len(deploy_row_std):.1f}%)")

    deploy_513 = deploy_median_513.copy()
    deploy_513[wide_mask_513] = (
        SHRINK_FACTOR * TRAIN_MEDIAN
        + (1.0 - SHRINK_FACTOR) * deploy_median_513[wide_mask_513]
    )
    deploy_513 = deploy_513.astype(np.float32)

    in_rae_deploy_native = float(rae(y_unb,
                                      deploy_median_513[unb_idx].astype(
                                          np.float64)))
    in_rae_deploy_shrunk = float(rae(y_unb,
                                      deploy_513[unb_idx].astype(np.float64)))
    print(f"   deploy_513 nb1070-equiv  in-sample 253 = {in_rae_deploy_native:.4f}")
    print(f"   deploy_513 nb1114 shrunk in-sample 253 = {in_rae_deploy_shrunk:.4f}")

    # ---- Save ----
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_conformal_qr.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    if delta_vs_nb1070 <= -0.001:
        verdict = "GATED_SHRINK_HELPS"
    elif abs(delta_vs_nb1070) < 0.001:
        verdict = "GATED_SHRINK_TIES"
    else:
        verdict = "GATED_SHRINK_HURTS"
    beats_nb1070 = bool(shrunk_rae < bag_median_rae)

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    print(f"   nb1070 median-bag OOF        = {bag_median_rae:.4f}")
    print(f"   nb1114 gated shrink OOF      = {shrunk_rae:.4f}  "
          f"(delta {delta_vs_nb1070:+.4f})")
    print(f"   beats_nb1070                 = {beats_nb1070}")
    print(f"   verdict                      = {verdict}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "seeds": SEEDS,
        "wide_quantile": WIDE_QUANTILE,
        "shrink_factor": SHRINK_FACTOR,
        "train_median": TRAIN_MEDIAN,
        "in_rae_anchor_on_253": in_rae_anchor,
        "nb1070_bag_median_rae": bag_median_rae,
        "shrunk_oof_rae": shrunk_rae,
        "delta_vs_nb1070": delta_vs_nb1070,
        "beats_nb1070": beats_nb1070,
        "verdict": verdict,
        "n_wide_253": n_wide,
        "n_narrow_253": n_narrow,
        "q75_std_253": q75,
        "row_std_min": float(row_std.min()),
        "row_std_median": float(np.median(row_std)),
        "row_std_max": float(row_std.max()),
        "wide_rae_before": wide_rae_before,
        "wide_rae_after": wide_rae_after,
        "narrow_rae": narrow_rae,
        "n_wide_513": n_wide_513,
        "q75_std_513": q75_deploy,
        "deploy_row_std_min": float(deploy_row_std.min()),
        "deploy_row_std_median": float(np.median(deploy_row_std)),
        "deploy_row_std_max": float(deploy_row_std.max()),
        "in_rae_deploy_nb1070_equiv": in_rae_deploy_native,
        "in_rae_deploy_shrunk_on_253": in_rae_deploy_shrunk,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "anchor_te_mean": float(preds_513.mean()),
        "anchor_te_std": float(preds_513.std()),
        "plain_submission": str(plain),
        "per_seed_records": [
            {"seed": r["seed"], "pooled_rae": r["pooled_rae"]}
            for r in per_seed_records
        ],
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
    for k in ("nb1070_bag_median_rae", "shrunk_oof_rae", "delta_vs_nb1070",
              "beats_nb1070", "verdict",
              "n_wide_253", "wide_rae_before", "wide_rae_after",
              "narrow_rae", "in_rae_deploy_shrunk_on_253",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
