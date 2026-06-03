"""nb1063 -- Per-quantile-bin stretch (nb1053 protocol) on alternative anchors.

nb1053 applied a 5-bin per-quantile rank-stretch to te_nb1014 and improved on
nb562's scalar stretch by handling tail-heavy variance compression. The
hypothesis here is that this calibration is universal: every variance-
compressed predictor (pred_std < truth_std on novel scaffolds) should gain.

We test 3 PRE-unblind anchors:
  - chemprop_aux           (te_chemprop_aux.npy, raw RAE 0.6216)
  - nb972 long-train       (te_nb972_long_train.npy, raw RAE 0.6898)
  - nb901 NR-multitask     (reconstructed from submissions/nb901_nr_multitask.csv,
                            raw RAE 0.6765)

Procedure per anchor (identical to nb1053):
  1. Load 513 deploy predictions (npy or aligned from CSV).
  2. Subset to 253 unblind indices via _audit_unblind_idx.npy.
  3. 5-fold KFold(seed=42) on the 253:
       - Train-only quantile edges (20/40/60/80 pct).
       - Per-bin (mu_b, s_b): mu = train-bin mean, s grid-fit on bin RAE.
       - Apply train-fold edges to val-fold predictions; no leakage.
  4. Pooled cross-fit RAE on the 253.
  5. Deploy: refit (edges, mu, s) on all 253 and apply to 513.

Each anchor emits:
  data/processed/te_nb1063_<tag>.npy
  submissions/nb1063_<tag>_perq_stretch.csv
Plus a combined data/processed/nb1063_summary.json with all 3 results.
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

TAG = "nb1063"
N_BINS = 5
STRETCH_GRID = np.round(np.arange(0.80, 2.001, 0.05), 2).tolist()
N_FOLDS = 5
SEED = 42

# Raw RAE references (PRE-unblind in-sample on 253, from prior runs).
RAW_RAE_REF = {
    "chemprop_aux": 0.6216,
    "nb972":        0.6898,
    "nb901":        0.6765,
}


def bin_assign(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, N_BINS - 1)


def fit_per_bin_stretch(p_train: np.ndarray, y_train: np.ndarray,
                        edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bins_tr = bin_assign(p_train, edges)
    mus = np.zeros(N_BINS, dtype=np.float64)
    ss = np.ones(N_BINS, dtype=np.float64)
    for b in range(N_BINS):
        mask = bins_tr == b
        if int(mask.sum()) < 2:
            mus[b] = float(p_train.mean())
            ss[b] = 1.0
            continue
        mu_b = float(p_train[mask].mean())
        mus[b] = mu_b
        y_b = y_train[mask]
        p_b = p_train[mask]
        best_s, best_r = 1.0, float("inf")
        for s in STRETCH_GRID:
            r = float(rae(y_b, mu_b + s * (p_b - mu_b)))
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


def load_anchor(tag: str, te_names: np.ndarray) -> np.ndarray:
    """Return 513-length float64 predictions in te_names order."""
    if tag == "chemprop_aux":
        return np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    if tag == "nb972":
        return np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    if tag == "nb901":
        csv = SUBMISSIONS / "nb901_nr_multitask.csv"
        df = pd.read_csv(csv)
        name_to_p = dict(zip(df["Molecule Name"].astype(str).values,
                             df["pEC50"].astype(float).values))
        missing = [n for n in te_names if n not in name_to_p]
        if missing:
            raise RuntimeError(f"nb901 missing {len(missing)} test names; first={missing[:3]}")
        return np.array([name_to_p[n] for n in te_names], dtype=np.float64)
    raise ValueError(f"unknown anchor: {tag}")


def run_one_anchor(tag: str, preds_513: np.ndarray, te_names: np.ndarray,
                   te_smiles: np.ndarray, unb_idx: np.ndarray,
                   y_unb: np.ndarray) -> dict:
    print("\n" + "=" * 78)
    print(f"{TAG} :: anchor={tag}")
    print("=" * 78)

    p_unb = preds_513[unb_idx]
    print(f"[load] preds_513 mean={preds_513.mean():.3f} std={preds_513.std():.3f}")
    print(f"[load] p_unb mean={p_unb.mean():.3f} std={p_unb.std():.3f}  "
          f"truth_std={y_unb.std():.3f}  compression={p_unb.std()/y_unb.std():.3f}")

    raw_rae = float(rae(y_unb, p_unb))
    print(f"[baseline] in_RAE raw on 253 = {raw_rae:.4f}  "
          f"(reference {RAW_RAE_REF[tag]:.4f})")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(y_unb), np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(len(y_unb)))):
        p_tr, y_tr = p_unb[tr_loc], y_unb[tr_loc]
        p_va, y_va = p_unb[va_loc], y_unb[va_loc]
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        edges = np.quantile(p_tr, qs)
        mus, ss = fit_per_bin_stretch(p_tr, y_tr, edges)
        pred_va = apply_per_bin_stretch(p_va, edges, mus, ss)
        oof[va_loc] = pred_va
        rae_va = float(rae(y_va, pred_va))
        bins_va = bin_assign(p_va, edges)
        bin_counts = [int((bins_va == b).sum()) for b in range(N_BINS)]
        folds.append({
            "fold": k,
            "edges": edges.tolist(),
            "mus": mus.tolist(),
            "ss": ss.tolist(),
            "val_rae": rae_va,
            "n_va": int(len(va_loc)),
            "bin_counts_va": bin_counts,
        })
        ss_str = ",".join(f"{s:.2f}" for s in ss)
        print(f"   fold {k}: val_RAE={rae_va:.4f}  ss=[{ss_str}]")

    pooled = float(rae(y_unb, oof))
    delta_vs_raw = pooled - raw_rae
    print(f"\n[honest] pooled cross-fit RAE = {pooled:.4f}  "
          f"(raw {raw_rae:.4f}, delta {delta_vs_raw:+.4f})")

    # Deploy: refit on all 253, apply to 513.
    qs_all = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
    deploy_edges = np.quantile(p_unb, qs_all)
    deploy_mus, deploy_ss = fit_per_bin_stretch(p_unb, y_unb, deploy_edges)
    deploy_513 = apply_per_bin_stretch(preds_513, deploy_edges,
                                       deploy_mus, deploy_ss).astype(np.float32)
    in_rae_deploy = float(rae(y_unb, deploy_513[unb_idx].astype(np.float64)))
    print(f"[deploy] in-sample 253 = {in_rae_deploy:.4f}  "
          f"deploy_ss={[round(float(x),2) for x in deploy_ss]}")
    print(f"[deploy] te(513) mean={deploy_513.mean():.3f} "
          f"std={deploy_513.std():.3f}")

    npy_path = DATA_PROCESSED / f"te_{TAG}_{tag}.npy"
    csv_path = SUBMISSIONS / f"{TAG}_{tag}_perq_stretch.csv"
    np.save(npy_path, deploy_513)
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(csv_path, index=False)
    print(f"[save] {npy_path}")
    print(f"[save] {csv_path}")

    per_bin_s_mean = [float(np.mean([f["ss"][b] for f in folds]))
                      for b in range(N_BINS)]
    per_bin_s_std = [float(np.std([f["ss"][b] for f in folds]))
                     for b in range(N_BINS)]

    if delta_vs_raw <= -0.005:
        verdict = "IMPROVES_RAW"
    elif abs(delta_vs_raw) < 0.005:
        verdict = "TIES_RAW"
    else:
        verdict = "WORSE_THAN_RAW"
    print(f"[verdict] {verdict} (delta vs raw {delta_vs_raw:+.4f})")

    return {
        "anchor": tag,
        "raw_rae_reference": RAW_RAE_REF[tag],
        "raw_rae_on_253": raw_rae,
        "pooled_cross_fit_rae": pooled,
        "in_rae_deploy_on_253": in_rae_deploy,
        "delta_vs_raw": delta_vs_raw,
        "verdict": verdict,
        "deploy_edges": deploy_edges.tolist(),
        "deploy_mus": deploy_mus.tolist(),
        "deploy_ss": deploy_ss.tolist(),
        "per_bin_s_mean_across_folds": per_bin_s_mean,
        "per_bin_s_std_across_folds": per_bin_s_std,
        "pred_std_raw": float(p_unb.std()),
        "truth_std": float(y_unb.std()),
        "compression_ratio": float(p_unb.std() / y_unb.std()),
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "anchor_te_mean": float(preds_513.mean()),
        "anchor_te_std": float(preds_513.std()),
        "npy_path": str(npy_path),
        "csv_path": str(csv_path),
        "folds": folds,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-quantile stretch on chemprop_aux / nb972 / nb901")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    print(f"[load] te has {len(te_names)} rows; unblind n={len(y_unb)}")

    results = {}
    for tag in ("chemprop_aux", "nb972", "nb901"):
        preds_513 = load_anchor(tag, te_names)
        results[tag] = run_one_anchor(tag, preds_513, te_names, te_smiles,
                                       unb_idx, y_unb)

    # ---- Combined summary ----
    summary = {
        "tag": TAG,
        "n_bins": N_BINS,
        "stretch_grid": STRETCH_GRID,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "results": results,
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} HEADLINE ===")
    print(f"   anchor          raw_RAE    poolCV      delta     verdict")
    for tag, r in results.items():
        print(f"   {tag:<14}  {r['raw_rae_on_253']:.4f}    "
              f"{r['pooled_cross_fit_rae']:.4f}    "
              f"{r['delta_vs_raw']:+.4f}   {r['verdict']}")
    print(f"   wall            = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for tag, r in res["results"].items():
        print(f"  {tag}: raw={r['raw_rae_on_253']:.4f}  "
              f"poolCV={r['pooled_cross_fit_rae']:.4f}  "
              f"delta={r['delta_vs_raw']:+.4f}  {r['verdict']}")
