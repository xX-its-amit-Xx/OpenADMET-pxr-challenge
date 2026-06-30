"""nb1374 -- Per-row sim-gated blend of nb1352 (SHAP-pruned) and nb1290 (best-w 2-way).

Hypothesis:
    nb1352 and nb1290 have residual Pearson ~0.99 but slightly different
    per-row behaviors. Per-row sim-routed blend may extract marginal gain.

Protocol:
    1. Load nb1352_median_bag_oof.npy and nb1290_bestw_oof.npy.
    2. Compute pred-pred Pearson, residual Pearson on 253.
    3. Per-row sim_to_train (Tanimoto top-1 vs train Morgan); reuse
       nb1301_sim_train_unb.npy if available, else recompute.
    4. For each row: gate = sigmoid(k*(sim_to_train - threshold)).
       Tune k in {2,5,10} and threshold in {0.4, 0.5, 0.6}.
       Convention: high sim_to_train -> trust nb1290 (TRAIN kNN-friendly anchor).
       Blend = gate * nb1290 + (1-gate) * nb1352.
    5. Pool RAE per (k, threshold). Best variant.
    6. Verdict at 0.003 margin vs nb1352 median (0.5315).

Outputs:
    scripts/nb1374_per_row_gated_1352_1290.py
    data/processed/nb1374_summary.json
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

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1374"

NB1352_REF = 0.5315  # nb1352 median-bag pooled RAE
NB1290_REF = 0.5390  # nb1290 best-fixed-w pooled RAE
MARGIN = 0.003

K_GRID = [2.0, 5.0, 10.0]
THR_GRID = [0.4, 0.5, 0.6]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _compute_sim_train_unb() -> np.ndarray:
    """Fallback: recompute top-1 Tanimoto vs train Morgan FPs on 253 unb rows."""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from pxr.chem import standardize, morgan_fp_batch
    from pxr.data import load_test, load_train

    def _safe_inchikey(m):
        try:
            return Chem.MolToInchiKey(m) if m is not None else None
        except Exception:
            return None

    def _safe_can(m):
        try:
            return Chem.MolToSmiles(m) if m is not None else None
        except Exception:
            return None

    te = load_test()
    test_smiles = te["smiles"].astype(str).tolist() \
        if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    test_mols = [standardize(s) for s in test_smiles]
    std_test = [_safe_can(m) or "" for m in test_mols]
    fp_test = morgan_fp_batch(std_test)

    tr = load_train()
    col = "smiles" if "smiles" in tr.columns else "SMILES"
    tr_smiles = tr[col].astype(str).tolist()
    tr_mols = [standardize(s) for s in tr_smiles]
    tr_iks = [_safe_inchikey(m) for m in tr_mols]
    tr_std = [_safe_can(m) or "" for m in tr_mols]
    seen, keep = set(), []
    for i, ik in enumerate(tr_iks):
        if ik is None or ik in seen:
            continue
        seen.add(ik)
        keep.append(i)
    fp_train = morgan_fp_batch([tr_std[i] for i in keep])
    fp_train = fp_train[fp_train.sum(axis=1) > 0]

    a = fp_test.astype(np.float32)
    b = fp_train.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    top1 = np.zeros(n_q, dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        top1[s:e] = sim.max(axis=1)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    return top1[unb_idx].astype(np.float32)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- per-row sim-gated blend of nb1352 (SHAP-pruned) and nb1290 (best-w)")
    print(f"          K grid     = {K_GRID}")
    print(f"          THR grid   = {THR_GRID}")
    print(f"          ref        = nb1352 median ({NB1352_REF:.4f})  margin = {MARGIN}")
    print("=" * 78)

    # ---- Load truth + predictions ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    p1352 = np.load(DATA_PROCESSED / "nb1352_median_bag_oof.npy").astype(np.float64)
    p1290 = np.load(DATA_PROCESSED / "nb1290_bestw_oof.npy").astype(np.float64)
    if p1352.shape[0] != n_unb or p1290.shape[0] != n_unb:
        raise ValueError(
            f"shape mismatch: nb1352={p1352.shape}, nb1290={p1290.shape}, n_unb={n_unb}"
        )

    rae_1352 = float(rae(y_unb, p1352))
    rae_1290 = float(rae(y_unb, p1290))
    print(f"[standalone] nb1352 median pooled RAE = {rae_1352:.4f}  (ref {NB1352_REF:.4f})")
    print(f"[standalone] nb1290 best-w pooled RAE = {rae_1290:.4f}  (ref {NB1290_REF:.4f})")

    # ---- Correlations ----
    pred_pearson = float(np.corrcoef(p1352, p1290)[0, 1])
    r_1352 = y_unb - p1352
    r_1290 = y_unb - p1290
    resid_pearson = float(np.corrcoef(r_1352, r_1290)[0, 1])
    print(f"[corr] pred-pred Pearson(1352, 1290)       = {pred_pearson:.4f}")
    print(f"[corr] residual Pearson(1352, 1290)         = {resid_pearson:.4f}")

    # ---- sim_to_train ----
    sim_path = DATA_PROCESSED / "nb1301_sim_train_unb.npy"
    if sim_path.exists():
        sim_train_unb = np.load(sim_path).astype(np.float64)
        print(f"[sim] reused {sim_path.name}  shape={sim_train_unb.shape}")
    else:
        print("[sim] recomputing top-1 Tanimoto vs train ...")
        sim_train_unb = _compute_sim_train_unb().astype(np.float64)
        print(f"[sim] computed  shape={sim_train_unb.shape}")
    if sim_train_unb.shape[0] != n_unb:
        raise ValueError(
            f"sim_train_unb shape {sim_train_unb.shape} != n_unb={n_unb}"
        )
    print(f"[sim] p10={np.percentile(sim_train_unb,10):.3f}  "
          f"p50={np.percentile(sim_train_unb,50):.3f}  "
          f"p90={np.percentile(sim_train_unb,90):.3f}  "
          f"mean={sim_train_unb.mean():.3f}")

    # ---- Sweep (k, threshold) ----
    print("\n" + "-" * 78)
    print("PER-ROW GATED BLEND SWEEP")
    print("   gate = sigmoid(k * (sim_to_train - threshold))")
    print("   blend = gate * nb1290 + (1 - gate) * nb1352")
    print("-" * 78)
    rows = []
    best = {"rae": float("inf")}
    for k in K_GRID:
        for thr in THR_GRID:
            gate = _sigmoid(k * (sim_train_unb - thr))
            blend = gate * p1290 + (1.0 - gate) * p1352
            r = float(rae(y_unb, blend))
            entry = {
                "k": float(k),
                "threshold": float(thr),
                "rae": r,
                "gate_mean": float(gate.mean()),
                "gate_std": float(gate.std()),
                "gate_p10": float(np.percentile(gate, 10)),
                "gate_p50": float(np.percentile(gate, 50)),
                "gate_p90": float(np.percentile(gate, 90)),
                "delta_vs_nb1352": r - rae_1352,
            }
            rows.append(entry)
            print(f"   k={k:5.2f}  thr={thr:.2f}  RAE={r:.4f}  "
                  f"gate mean={gate.mean():.3f}  std={gate.std():.3f}  "
                  f"d_vs_nb1352={r - rae_1352:+.4f}")
            if r < best["rae"]:
                best = entry

    # ---- Verdict ----
    best_rae = best["rae"]
    beats_nb1352 = best_rae < rae_1352 - MARGIN
    flat_vs_nb1352 = abs(best_rae - rae_1352) < MARGIN
    delta = best_rae - rae_1352
    if beats_nb1352:
        verdict = (f"PER_ROW_GATED_BEATS_NB1352 "
                   f"(k={best['k']:.1f}, thr={best['threshold']:.2f} -> {best_rae:.4f})")
    elif flat_vs_nb1352:
        verdict = (f"PER_ROW_GATED_FLAT_VS_NB1352 "
                   f"(k={best['k']:.1f}, thr={best['threshold']:.2f} -> {best_rae:.4f})")
    else:
        verdict = (f"PER_ROW_GATED_HURTS_VS_NB1352 "
                   f"(k={best['k']:.1f}, thr={best['threshold']:.2f} -> {best_rae:.4f})")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1352 standalone RAE   : {rae_1352:.4f}")
    print(f"   nb1290 standalone RAE   : {rae_1290:.4f}")
    print(f"   best gated blend RAE    : {best_rae:.4f}  "
          f"(k={best['k']:.1f}, thr={best['threshold']:.2f})")
    print(f"   delta vs nb1352         : {delta:+.4f}")
    print(f"   beats_nb1352            : {beats_nb1352}")
    print(f"   verdict                 : {verdict}")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "standalone": {
            "nb1352_median_bag": rae_1352,
            "nb1290_bestw":      rae_1290,
        },
        "pred_pearson_1352_1290":     pred_pearson,
        "residual_pearson_1352_1290": resid_pearson,
        "k_grid":   [float(x) for x in K_GRID],
        "thr_grid": [float(x) for x in THR_GRID],
        "sim_train_unb_stats": {
            "p10":  float(np.percentile(sim_train_unb, 10)),
            "p25":  float(np.percentile(sim_train_unb, 25)),
            "p50":  float(np.percentile(sim_train_unb, 50)),
            "p75":  float(np.percentile(sim_train_unb, 75)),
            "p90":  float(np.percentile(sim_train_unb, 90)),
            "mean": float(sim_train_unb.mean()),
            "std":  float(sim_train_unb.std()),
        },
        "grid_results":   rows,
        "best_variant":   best,
        "best_rae":       best_rae,
        "rae_nb1352":     rae_1352,
        "delta_vs_nb1352": delta,
        "beats_nb1352":   bool(beats_nb1352),
        "flat_vs_nb1352": bool(flat_vs_nb1352),
        "margin":         MARGIN,
        "verdict":        verdict,
        "wall_sec":       round(time.time() - t0, 2),
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
    for k in (
        "n_unb", "standalone",
        "pred_pearson_1352_1290", "residual_pearson_1352_1290",
        "sim_train_unb_stats",
        "best_variant", "best_rae",
        "rae_nb1352", "delta_vs_nb1352",
        "beats_nb1352", "flat_vs_nb1352",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
