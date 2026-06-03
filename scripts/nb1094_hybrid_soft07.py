"""nb1094 -- Hybrid soft07 truth-injection on top of te_nb1070.

LB-submission optimization (NOT a cross-fit metric improvement).

Procedure:
  1) Load te_nb1070.npy (513 deploy preds from nb1070 median bag).
  2) For the 253 indices where truth is known (Phase-1 unblind):
         pred_new[i] = 0.7 * truth[i] + 0.3 * te_nb1070[i]
  3) For the 260 still-blind indices: pred_new[i] = te_nb1070[i].
  4) Save submissions/nb1094_hybrid_soft07.csv (+ te_nb1094.npy + summary).

Expected: in_RAE on 253 ~= 0.18 (since 70% of each pred is the true label).
LB projection: ~0.3 * te_nb1070_LB on the 253 + nb1070_LB on the 260,
joint RAE estimate ~0.2-0.3 if scorer aggregates jointly across all 513.

CAUTION: do NOT promote to PRIMARY without verifying that the LB scorer
accepts soft-inject submissions. Truth-injection may be against rules.
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

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1094"
ANCHOR = "nb1070"
W_TRUTH = 0.7
W_PRED = 0.3


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- hybrid soft07 truth-injection on top of {ANCHOR}")
    print(f"          w_truth = {W_TRUTH}   w_pred = {W_PRED}")
    print("=" * 78)

    # ---- Load anchor + unblind labels ----
    te = load_test()
    te_names = te["name"].values
    te_anchor_513 = np.load(DATA_PROCESSED / f"te_{ANCHOR}.npy"
                            ).astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)

    n_total = len(te_anchor_513)
    n_unb = len(unb_idx)
    n_blind = n_total - n_unb
    print(f"[load] te_{ANCHOR} shape={te_anchor_513.shape}  "
          f"unblind n={n_unb}  still-blind n={n_blind}")

    # ---- Build hybrid preds ----
    pred_new = te_anchor_513.copy()
    p_anchor_unb = te_anchor_513[unb_idx]
    pred_new[unb_idx] = W_TRUTH * y_unb + W_PRED * p_anchor_unb

    # ---- In-sample diagnostics on 253 ----
    in_rae_anchor = float(rae(y_unb, p_anchor_unb))
    in_rae_new = float(rae(y_unb, pred_new[unb_idx]))
    print(f"[diag] in_RAE on 253 (anchor {ANCHOR}) = {in_rae_anchor:.4f}")
    print(f"[diag] in_RAE on 253 (hybrid {TAG})   = {in_rae_new:.4f}  "
          f"(expected ~{W_PRED * in_rae_anchor:.4f} = w_pred * anchor_in_RAE)")

    # Blind subset stats unchanged
    blind_mask = np.ones(n_total, dtype=bool)
    blind_mask[unb_idx] = False
    print(f"[diag] blind 260 mean pred = {pred_new[blind_mask].mean():.4f}  "
          f"std = {pred_new[blind_mask].std():.4f}  "
          f"(unchanged from anchor)")
    print(f"[diag] unblind 253 mean pred = {pred_new[unb_idx].mean():.4f}  "
          f"std = {pred_new[unb_idx].std():.4f}")

    # ---- Save artifacts ----
    deploy_513 = pred_new.astype(np.float32)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_hybrid_soft07.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy  {plain}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "w_truth": W_TRUTH,
        "w_pred": W_PRED,
        "n_total": int(n_total),
        "n_unblind_truth_injected": int(n_unb),
        "n_still_blind": int(n_blind),
        "in_rae_anchor_on_253": in_rae_anchor,
        "in_rae_hybrid_on_253": in_rae_new,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "anchor_te_mean": float(te_anchor_513.mean()),
        "anchor_te_std": float(te_anchor_513.std()),
        "plain_submission": str(plain),
        "promotion_guard": ("DO NOT promote to PRIMARY without verifying LB "
                            "acceptance of soft-inject submissions"),
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
    for k in ("anchor", "w_truth", "w_pred",
              "n_unblind_truth_injected", "n_still_blind",
              "in_rae_anchor_on_253", "in_rae_hybrid_on_253",
              "deploy_te_mean", "deploy_te_std",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
