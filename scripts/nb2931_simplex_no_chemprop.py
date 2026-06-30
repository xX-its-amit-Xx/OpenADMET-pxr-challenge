"""nb2931 -- SLSQP simplex blend on a 3-anchor PRE-clean pool EXCLUDING chemprop_aux.

MOTIVATION:
    Prior SLSQP blends consistently zeroed chemprop_aux (its mass migrated to nb1191
    + nb2240_K20). Drop it entirely -> cleaner test of whether the 2-anchor cluster
    (nb2240_K20, nb1191) plus the counter-orthogonal axis (counter_clean) can
    extract any additional gain via simplex SLSQP at the per-fold level.

ANCHOR POOL (n=3, all PRE-clean):
    1. nb2240_K20     -> data/processed/nb2240_mean_bag_oof_K20.npy   (~0.4630)
    2. nb1191         -> data/processed/nb1191_pred_oof.npy           (~0.4697)
    3. counter_clean  -> data/processed/nb2490_pred_oof.npy           (~0.5382)
    chemprop_aux EXCLUDED.

PROTOCOL:
    - Scaffold 5-fold CV across 5 kf_seeds {42, 1, 7, 137, 1009}.
    - For each fold: SLSQP minimize fold-TRAIN RAE on the simplex (w>=0, sum w=1),
      8 multi-starts (uniform + 7 Dirichlet draws) for robustness.
    - Apply per-fold weights to held-out validation slice; concat for OOF.
    - Report per-kf_seed pooled OOS-concat RAE; mean across seeds.

GATE:
    mean_rae < 0.4570 -> PROMOTE
    mean_rae < 0.4598 -> MARGINAL_BEAT
    else              -> FAIL

DEPLOY:
    Refit SLSQP on full 253 -> single-pool simplex weight vector.
    Apply to (513, 3) stacked te-anchors -> nb2931_te_513 prediction.
    Also save mean-of-per-fold-weight deploy variant.

ARTIFACTS:
    data/processed/nb2931_summary.json
    data/processed/nb2931_pred_oof.npy      (253,) float32  -- seed-mean OOS-concat
    data/processed/te_nb2931.npy            (513,) float32  -- full-pool SLSQP deploy
    submissions/nb2931_simplex_no_chemprop.csv
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
import pandas as pd
from rdkit import RDLogger
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

TAG = "nb2931"
N_FOLDS = 5
KF_SEEDS = [42, 1, 7, 137, 1009]
N_STARTS_FOLD = 8
N_STARTS_FULL = 12

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598
DEGEN_MAX_W = 0.85

# PRE-clean 3-anchor pool (chemprop_aux deliberately excluded).
ANCHORS = [
    ("nb2240_K20",    DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
                      DATA_PROCESSED / "te_nb2240_K20.npy",
                      "PRE-clean K=20 RFE residual stack on chemprop_aux anchor"),
    ("nb1191",        DATA_PROCESSED / "nb1191_pred_oof.npy",
                      DATA_PROCESSED / "te_nb1191.npy",
                      "PRE-clean deep-30 pyramid (15-seed wide-verified)"),
    ("counter_clean", DATA_PROCESSED / "nb2490_pred_oof.npy",
                      DATA_PROCESSED / "te_nb2490.npy",
                      "PRE-clean counter-assay residual (nb730-free)"),
]


def _simplex_slsqp(P: np.ndarray, y: np.ndarray, n_starts: int = 8,
                   seed: int = 0) -> tuple[np.ndarray, float]:
    """Minimize RAE(y, P @ w) over the simplex (w>=0, sum w=1) with multi-start."""
    K = P.shape[1]
    rng = np.random.default_rng(seed)

    def loss(w: np.ndarray) -> float:
        return float(rae(y, P @ w))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K

    starts = [np.full(K, 1.0 / K)]
    for _ in range(max(0, n_starts - 1)):
        starts.append(rng.dirichlet(np.ones(K)))

    best_w, best_r = None, np.inf
    for x0 in starts:
        try:
            res = minimize(loss, x0, method="SLSQP", bounds=bnds, constraints=cons,
                           options={"maxiter": 300, "ftol": 1e-9})
            w = np.clip(res.x, 0.0, 1.0)
            s = float(w.sum())
            if s <= 0.0:
                continue
            w = w / s
            r = float(rae(y, P @ w))
            if r < best_r:
                best_r, best_w = r, w
        except Exception:
            continue
    if best_w is None:
        best_w = np.full(K, 1.0 / K)
        best_r = float(rae(y, P @ best_w))
    return best_w, best_r


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SLSQP simplex 3-anchor (no chemprop_aux): "
          "{nb2240_K20, nb1191, counter_clean}")
    print("=" * 78)

    # ---- Load ground truth + scaffolds ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # ---- Load anchors (all required) ----
    anchor_names: list[str] = []
    anchor_provenance: dict[str, str] = {}
    oof_cols, te_cols = [], []
    for name, oof_path, te_path, prov in ANCHORS:
        if not oof_path.exists():
            raise FileNotFoundError(f"{name}: pred_oof missing at {oof_path}")
        if not te_path.exists():
            raise FileNotFoundError(f"{name}: te missing at {te_path}")
        oof = np.load(oof_path).astype(np.float64)
        te_v = np.load(te_path).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{name}: oof shape {oof.shape} != ({n_unb},)")
        if te_v.shape[0] != n_test:
            raise ValueError(f"{name}: te shape {te_v.shape} != ({n_test},)")
        anchor_names.append(name)
        anchor_provenance[name] = prov
        oof_cols.append(oof)
        te_cols.append(te_v)
    K = len(anchor_names)
    P_unb = np.column_stack(oof_cols)  # (253, 3)
    P_te = np.column_stack(te_cols)    # (513, 3)

    anchor_in_rae = {k: round(float(rae(y_unb, P_unb[:, i])), 4)
                     for i, k in enumerate(anchor_names)}
    print(f"[anchors] K={K}")
    for k in anchor_names:
        print(f"   {k:14s}  unb_RAE={anchor_in_rae[k]:.4f}  [{anchor_provenance[k]}]")

    # Leak sanity: anchors should NOT be bit-equal to truth on >5% of rows.
    leak_flags = {}
    for i, k in enumerate(anchor_names):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")

    # ---- Scaffolds ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}  n_folds={N_FOLDS}  kf_seeds={KF_SEEDS}")

    # ---- Per-seed scaffold-CV SLSQP ----
    print("\n" + "-" * 78)
    print("SLSQP simplex CV (5 kf_seeds x 5-fold scaffold)")
    print("-" * 78)
    per_seed_pooled: list[float] = []
    per_seed_mean_fold: list[float] = []
    oof_seed_stack = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_fold_weights: dict[int, list[dict]] = {}
    per_seed_mean_w: list[np.ndarray] = []
    per_seed_max_w: list[float] = []
    any_seed_degenerate = False

    for s_idx, kf_seed in enumerate(KF_SEEDS):
        splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                        shuffle=True, seed=kf_seed)
        oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
        fold_rae_list: list[float] = []
        fold_w_list: list[np.ndarray] = []
        per_fold_rows: list[dict] = []
        for f_idx, (tr_loc, va_loc) in enumerate(splits):
            # SLSQP on TRAIN rows only -> simplex weights
            w, r_train = _simplex_slsqp(P_unb[tr_loc], y_unb[tr_loc],
                                        n_starts=N_STARTS_FOLD, seed=kf_seed * 11 + f_idx)
            val_pred = P_unb[va_loc] @ w
            oof_blend[va_loc] = val_pred
            r_val = float(rae(y_unb[va_loc], val_pred))
            fold_rae_list.append(r_val)
            fold_w_list.append(w)
            per_fold_rows.append({
                "fold": int(f_idx),
                "n_train": int(len(tr_loc)),
                "n_val": int(len(va_loc)),
                "weights": {anchor_names[k]: round(float(w[k]), 4) for k in range(K)},
                "train_rae": round(float(r_train), 4),
                "val_rae": round(r_val, 4),
                "max_w": round(float(w.max()), 4),
                "degenerate": bool(w.max() > DEGEN_MAX_W),
            })

        assert not np.any(np.isnan(oof_blend)), f"OOF coverage gap on kf_seed={kf_seed}"
        pooled = float(rae(y_unb, oof_blend))
        mean_fold = float(np.mean(fold_rae_list))
        per_seed_pooled.append(pooled)
        per_seed_mean_fold.append(mean_fold)
        oof_seed_stack[s_idx] = oof_blend

        w_stack = np.stack(fold_w_list, axis=0)  # (5, 3)
        w_mean = w_stack.mean(axis=0)
        w_mean = w_mean / w_mean.sum()
        per_seed_mean_w.append(w_mean)
        per_seed_max_w.append(float(w_stack.max(axis=1).max()))
        seed_degen = any(r["degenerate"] for r in per_fold_rows)
        any_seed_degenerate = any_seed_degenerate or seed_degen
        per_seed_fold_weights[kf_seed] = per_fold_rows

        print(f"   kf_seed={kf_seed:5d}  pooled={pooled:.4f}  mean_fold={mean_fold:.4f}  "
              f"mean_w={{ "
              + ", ".join(f"{anchor_names[k]}={w_mean[k]:.3f}" for k in range(K))
              + f" }}  any_fold_degen={seed_degen}")

    # ---- Aggregate across seeds ----
    mean_pooled = float(np.mean(per_seed_pooled))
    std_pooled = float(np.std(per_seed_pooled))
    oof_final = oof_seed_stack.mean(axis=0)
    pooled_on_seed_mean = float(rae(y_unb, oof_final))

    mean_w_across_seeds = np.stack(per_seed_mean_w, axis=0).mean(axis=0)
    mean_w_across_seeds = mean_w_across_seeds / mean_w_across_seeds.sum()

    print(f"\n[wide-seed] mean pooled = {mean_pooled:.4f} +/- {std_pooled:.4f} "
          f"(n_seeds={len(KF_SEEDS)})")
    print(f"[wide-seed] pooled on seed-mean OOF = {pooled_on_seed_mean:.4f}")
    print(f"[wide-seed] mean weights across seeds = "
          + ", ".join(f"{anchor_names[k]}={mean_w_across_seeds[k]:.4f}" for k in range(K)))

    # ---- Single-pool SLSQP (deploy weights) ----
    print("\n" + "-" * 78)
    print("DEPLOY: single-pool SLSQP on all 253")
    print("-" * 78)
    w_full, r_full = _simplex_slsqp(P_unb, y_unb, n_starts=N_STARTS_FULL, seed=0)
    full_pool_weights = {anchor_names[k]: round(float(w_full[k]), 4) for k in range(K)}
    full_pool_degen = bool(w_full.max() > DEGEN_MAX_W)
    print(f"   in-sample RAE = {r_full:.4f}  max_w={w_full.max():.4f}  "
          f"degen={full_pool_degen}")
    for k in range(K):
        flag = " (zeroed)" if w_full[k] < 1e-6 else ""
        print(f"     w[{anchor_names[k]:14s}] = {w_full[k]:+.4f}{flag}")

    # Deploy te = P_te @ w_full
    te_pred = (P_te @ w_full).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))

    # Mean-of-fold-weights variant (LB-honester alternative)
    te_pred_mean_w = (P_te @ mean_w_across_seeds).astype(np.float32)
    te_pred_mean_w = np.clip(te_pred_mean_w, 3.0, 9.0)
    te_unb_in_mean_w = float(rae(y_unb, te_pred_mean_w[unb_idx]))

    print(f"   te(full-pool) mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"in-sample unb RAE={te_unb_in:.4f}")
    print(f"   te(mean-fold)  mean={te_pred_mean_w.mean():.3f} std={te_pred_mean_w.std():.3f} "
          f"in-sample unb RAE={te_unb_in_mean_w:.4f}")

    # ---- Gate ----
    if mean_pooled < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_pooled < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_pooled {mean_pooled:.4f}  "
          f"(< {GATE_PROMOTE} PROMOTE / < {GATE_MARGINAL} MARGINAL)  ->  {verdict}")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_final.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_simplex_no_chemprop.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "slsqp_simplex_3anchor_no_chemprop",
        "anchor_pool": anchor_names,
        "anchor_provenance": anchor_provenance,
        "anchor_in_rae": anchor_in_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "chemprop_aux_excluded": True,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_starts_fold": N_STARTS_FOLD,
        "n_starts_full": N_STARTS_FULL,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "per_seed_pooled_rae": per_seed_pooled,
        "per_seed_mean_fold_rae": per_seed_mean_fold,
        "per_seed_max_w": per_seed_max_w,
        "per_seed_mean_w": [
            {anchor_names[k]: round(float(w[k]), 4) for k in range(K)}
            for w in per_seed_mean_w
        ],
        "per_seed_fold_weights": {str(k): v for k, v in per_seed_fold_weights.items()},
        "mean_pooled_rae": mean_pooled,
        "std_pooled_rae": std_pooled,
        "pooled_on_seed_mean_oof": pooled_on_seed_mean,
        "mean_rae": mean_pooled,
        "mean_w_across_seeds": {anchor_names[k]: round(float(mean_w_across_seeds[k]), 4)
                                for k in range(K)},
        "any_seed_degenerate": bool(any_seed_degenerate),
        "full_pool_slsqp": {
            "weights": full_pool_weights,
            "rae_in_sample": round(float(r_full), 4),
            "max_w": round(float(w_full.max()), 4),
            "degenerate": full_pool_degen,
        },
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "te_unb_in_sample_rae_full_pool": round(te_unb_in, 4),
        "te_unb_in_sample_rae_mean_fold": round(te_unb_in_mean_w, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    # Compact final echo
    print("\n" + "=" * 78)
    print(json.dumps({
        "tag": TAG,
        "anchor_in_rae": anchor_in_rae,
        "mean_pooled_rae": round(mean_pooled, 4),
        "std_pooled_rae": round(std_pooled, 4),
        "pooled_on_seed_mean_oof": round(pooled_on_seed_mean, 4),
        "full_pool_weights": full_pool_weights,
        "mean_w_across_seeds": summary["mean_w_across_seeds"],
        "any_seed_degenerate": summary["any_seed_degenerate"],
        "verdict": verdict,
        "wall_sec": summary["wall_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
