"""nb2500 -- Rank-aligned hybrid prediction (DIFFERENT axis from additive blend).

CONTEXT:
    All recent post-hoc combinations of nb2240 (K=20 LGBM residual, OOF
    RAE=0.4630) with a counter-axis anchor (nb2490 clean counter, OOF
    RAE=0.5382) have been ADDITIVE convex blends (SLSQP / Huber / Ridge),
    which collapse to ~100% nb2240 weight because nb2490 is strictly worse
    standalone.

    This script tries a DIFFERENT operator: RANK-ALIGN. The nb2490 counter
    axis is biologically orthogonal -- it knows which compounds are
    counter-screen-active and therefore likely promiscuous (per
    feedback_phase2_prescriptions.md, F2 greasy-novel-inactive carries
    -0.11 RAE prize). If counter-axis is a BETTER RANKER than nb2240 even
    when its absolute magnitudes are worse, then re-ranking nb2240's
    magnitudes by counter-axis order should beat both standalone preds.

PROTOCOL (quantile-magnitude mapping):
    1. Sort rows by nb2490 (counter-axis) predictions ASCENDING:
         counter_rank[i] = position of row i in nb2490 sort order
    2. Sort nb2240 predictions ASCENDING: nb2240_sorted = np.sort(nb2240)
    3. Re-assign: output[i] = nb2240_sorted[counter_rank[i]]
       (i.e. the row with the j-th smallest counter-axis pred receives the
        j-th smallest nb2240 magnitude)
    4. Evaluate via 5-fold scaffold CV on 253 unblind (cross-fit the rank
       alignment per fold: rank fit on training rows, apply to held-out).
       Then full-sample for the deploy 513-row te.

GATE: mean_rae < 0.4570 -> PROMOTE
      mean_rae < 0.4601 -> MARGINAL_BEAT
      else FAIL.

Outputs:
    data/processed/nb2500_pred_oof.npy   (253,)  cross-fit rank-aligned OOF
    data/processed/te_nb2500.npy         (513,)  full-sample rank-aligned te
    data/processed/nb2500_summary.json
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2500"
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
NB2240_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
NB2490_OOF_PATH = DATA_PROCESSED / "nb2490_pred_oof.npy"
NB2490_TE_PATH = DATA_PROCESSED / "te_nb2490.npy"
UNBLIND_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601


def _fail_dep(msg: str) -> None:
    print(f"[FAIL_DEP] {msg}")
    summary = {"tag": TAG, "status": "FAIL_DEP", "reason": msg}
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    sys.exit(0)


def rank_align(magnitude_src: np.ndarray, order_src: np.ndarray) -> np.ndarray:
    """Rank-align: output[i] = sorted(magnitude_src)[rank_of(order_src[i])]

    The row that has the smallest `order_src` value receives the smallest
    `magnitude_src` value; the row with the largest `order_src` receives the
    largest `magnitude_src`. Preserves the magnitude distribution exactly,
    re-orders by `order_src`.
    """
    n = len(order_src)
    assert len(magnitude_src) == n
    # rank of each row in the order_src (0 = smallest)
    order_rank = np.argsort(np.argsort(order_src, kind="stable"), kind="stable")
    sorted_mag = np.sort(magnitude_src, kind="stable")
    return sorted_mag[order_rank]


def cross_fit_rank_align(
    mag_unb: np.ndarray,
    order_unb: np.ndarray,
    y_unb: np.ndarray,
    scaffolds: list,
    kf_seed: int,
) -> tuple[np.ndarray, float]:
    """5-fold scaffold cross-fit of rank-align operator.

    For each fold:
      - Build sorted-magnitude pool ONLY from training rows: sorted(mag_tr)
      - For each held-out row i, compute its rank within the FULL order_unb
        restricted to training+test rows (use empirical CDF on training
        order_src to assign rank position, then map to sorted_mag pool by
        proportional index).

    Returns: oof predictions (253,) + pooled RAE.
    """
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    oof = np.full_like(y_unb, np.nan, dtype=np.float64)
    for fold_idx, (tr_idx, te_idx) in enumerate(splits):
        mag_tr = mag_unb[tr_idx]
        order_tr = order_unb[tr_idx]
        order_te = order_unb[te_idx]
        sorted_mag = np.sort(mag_tr, kind="stable")
        n_tr = len(sorted_mag)
        # For each held-out row, find rank within training order_src distribution
        # via searchsorted, then map proportionally to sorted_mag pool
        ranks_in_tr = np.searchsorted(np.sort(order_tr, kind="stable"), order_te, side="left")
        # ranks_in_tr is 0..n_tr; map to sorted_mag index 0..n_tr-1
        # proportional remap: idx = floor(ranks_in_tr * n_tr / (n_tr+1)) clipped
        # simpler: clip to [0, n_tr-1]
        mag_idx = np.clip(
            np.floor(ranks_in_tr.astype(np.float64) * (n_tr - 1) / max(n_tr, 1)).astype(int),
            0, n_tr - 1,
        )
        oof[te_idx] = sorted_mag[mag_idx]
    assert not np.any(np.isnan(oof)), "oof has nan"
    return oof, float(rae(y_unb, oof))


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Rank-aligned hybrid (counter-rank as ordering, nb2240 as magnitudes)")
    print("=" * 78)

    # ---- Dependency check ----
    for p in (NB2240_OOF_PATH, NB2240_TE_PATH, NB2490_OOF_PATH, NB2490_TE_PATH,
              UNBLIND_IDX_PATH, UNBLIND_Y_PATH):
        if not p.exists():
            _fail_dep(f"missing {p}")

    # ---- Load ----
    nb2240_oof = np.load(NB2240_OOF_PATH).astype(np.float64)
    nb2240_te = np.load(NB2240_TE_PATH).astype(np.float64)
    nb2490_oof = np.load(NB2490_OOF_PATH).astype(np.float64)
    nb2490_te = np.load(NB2490_TE_PATH).astype(np.float64)
    unb_idx = np.load(UNBLIND_IDX_PATH)
    y_unb = np.load(UNBLIND_Y_PATH).astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}  nb2240_oof_RAE={rae(y_unb, nb2240_oof):.4f}  "
          f"nb2490_oof_RAE={rae(y_unb, nb2490_oof):.4f}")

    # ---- Scaffold-CV setup ----
    te_df = load_test()
    te_smiles = te_df["smiles"].astype(str).tolist() if "smiles" in te_df.columns \
        else te_df["SMILES"].astype(str).tolist()
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- In-sample sanity: rank-align on the full 253 ----
    align_full = rank_align(nb2240_oof, nb2490_oof)
    rae_full = float(rae(y_unb, align_full))
    print(f"[sanity] in-sample full-253 rank-align RAE = {rae_full:.4f}  "
          f"(diagnostic, NOT the gate decision number)")

    # ---- 5-fold scaffold cross-fit per seed ----
    per_seed = {}
    pooled_oof_seeds = []
    for s in KF_SEEDS:
        oof_s, r_s = cross_fit_rank_align(
            nb2240_oof, nb2490_oof, y_unb, unb_scaffolds, kf_seed=s,
        )
        per_seed[s] = r_s
        pooled_oof_seeds.append(oof_s)
        print(f"[cv  seed={s}] pooled cross-fit RAE = {r_s:.4f}")

    pooled_oof = np.mean(np.stack(pooled_oof_seeds, axis=0), axis=0)
    mean_rae = float(np.mean(list(per_seed.values())))
    std_rae = float(np.std(list(per_seed.values())))
    rae_mean_of_oofs = float(rae(y_unb, pooled_oof))
    print(f"\n[result] mean_rae across {len(KF_SEEDS)} kf-seeds = "
          f"{mean_rae:.4f} +- {std_rae:.4f}")
    print(f"[result] RAE of mean-of-seed-OOFs = {rae_mean_of_oofs:.4f}")

    # ---- Gate decision ----
    if mean_rae < GATE_PROMOTE:
        status = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        status = "MARGINAL_BEAT"
    else:
        status = "FAIL"
    delta_vs_nb2240 = mean_rae - float(rae(y_unb, nb2240_oof))
    print(f"\n[GATE] mean_rae={mean_rae:.4f}  vs PROMOTE<{GATE_PROMOTE}  "
          f"MARGINAL<{GATE_MARGINAL}  -> {status}")
    print(f"[GATE] delta vs nb2240 standalone = {delta_vs_nb2240:+.4f}")

    # ---- Deploy te: full-sample rank-align on 513 ----
    te_align = rank_align(nb2240_te, nb2490_te)

    # ---- Save outputs ----
    oof_out = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_out = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_out, pooled_oof.astype(np.float32))
    np.save(te_out, te_align.astype(np.float32))
    print(f"[save] {oof_out}")
    print(f"[save] {te_out}")

    summary = {
        "tag": TAG,
        "method": "rank-aligned hybrid (counter-rank ordering, nb2240 magnitudes via quantile-mapping)",
        "anchors": {
            "magnitude_src": "nb2240_mean_bag_oof_K20",
            "order_src": "nb2490 (counter-clean cycle-199)",
        },
        "n_unb": int(n_unb),
        "n_test": 513,
        "n_unique_scaffolds": int(n_unique_scaf),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "per_seed_rae": {str(k): float(v) for k, v in per_seed.items()},
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "rae_of_mean_of_seed_oofs": rae_mean_of_oofs,
        "rae_anchor_nb2240": float(rae(y_unb, nb2240_oof)),
        "rae_anchor_nb2490": float(rae(y_unb, nb2490_oof)),
        "rae_in_sample_full_align": rae_full,
        "delta_vs_nb2240": delta_vs_nb2240,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "status": status,
        "pred_oof_path": str(oof_out),
        "te_path": str(te_out),
        "elapsed_s": round(time.time() - t0, 2),
    }
    sumpath = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(sumpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {sumpath}")
    print(f"\n[done] {TAG} status={status}  mean_rae={mean_rae:.4f}  "
          f"elapsed={summary['elapsed_s']}s")
    return summary


if __name__ == "__main__":
    main()
