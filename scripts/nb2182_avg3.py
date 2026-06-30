"""nb2182 -- simple 3-way arithmetic average of {nb2171, nb2095, nb1191}.

Per nb2170 collapse memo: SLSQP on highly correlated stacks degenerates to a
single anchor (in-sample SSE minimum has no gradient room when the columns are
near-collinear). Arithmetic mean has zero fit parameters, so it CANNOT
collapse. Whatever Pearson the three stacks share, the average is fixed.

DESIGN:
- Anchors:
    nb2171  -- nb1162-style stack with nb730_honest SWAPPED to nb1191 (5 cols:
               nb2103_K28, chemprop_aux, nb1191, nb503, nb562); honest OOF ~0.4676.
    nb2095  -- pseudo-PRE pyramid (chemprop_aux, nb1150, nb1158_K32, nb2112_K28,
               nb1014); honest OOF 0.4720.
    nb1191  -- PRE-unblind pyramid (chemprop_aux, nb1150, nb1158_K32, nb2112_K28);
               honest OOF 0.4718.
- Stage 1 (OOF reconstruction): rebuild each stack's honest cross-fit OOF on
  the 253 unblind using its own anchor OOFs + SLSQP-simplex + rank-stretch
  under scaffold-CV (kf_seeds {1001..1005}, n_folds=5, stretch grid same as
  source notebooks). Mean across seeds -> single OOF vector per stack.
- Stage 2 (blend): arithmetic mean across the 3 stack OOFs per row. No fit.
- Stage 3 (deploy): arithmetic mean of cached te_nb2171.npy + te_nb2095.npy +
  te_nb1191.npy on the 513.

GATE: pooled scaffold-CV RAE on the average <= 0.4646 (beat nb2171 0.4676 by
>= 0.003).

Outputs:
    scripts/nb2182_avg3.py
    data/processed/nb2182_summary.json
    data/processed/te_nb2182.npy        (always saved)
    submissions/nb2182_avg3.csv         (only on gate pass)
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

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2182"
NB2171_REF_RAE = 0.4676    # current best stack OOF
GATE_OOF_MAX = NB2171_REF_RAE - 0.003   # 0.4646
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# ---------------------------------------------------------------------------
# nb1150 reconstruction (used by nb2095, nb1191; same as nb2170/nb2171)
# ---------------------------------------------------------------------------
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS_FULL_POOL = [0.0, 0.2942, 0.0, 0.7058]


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS_FULL_POOL, dtype=np.float64)
    return P @ w


# ---------------------------------------------------------------------------
# Stack anchor specifications (mirror source notebooks exactly)
# ---------------------------------------------------------------------------
STACKS = {
    "nb2171": [
        ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy"),
        ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
        ("nb1191",       "_RECONSTRUCT_nb1191_oof"),
        ("nb503",        "nb503_pred_oof.npy"),
        ("nb562",        "nb562_pred_oof.npy"),
    ],
    "nb2095": [
        ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
        ("nb1150",       "_RECONSTRUCT_nb1150_oof"),
        ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy"),
        ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy"),
        ("nb1014",       "nb1133_nb1014_pred_oof.npy"),
    ],
    "nb1191": [
        ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
        ("nb1150",       "_RECONSTRUCT_nb1150_oof"),
        ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy"),
        ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy"),
    ],
}

STACK_TE_DEPLOY = {
    "nb2171": "te_nb2171.npy",
    "nb2095": "te_nb2095.npy",
    "nb1191": "te_nb1191.npy",
}
STACK_INDIV_OOF_REPORTED = {
    "nb2171": 0.4676,
    "nb2095": 0.4720,
    "nb1191": 0.4718,
}


# ---------------------------------------------------------------------------
# Stack-rebuild helpers
# ---------------------------------------------------------------------------
def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def load_anchor_vec(rel: str, n_unb: int) -> np.ndarray:
    if rel == "_RECONSTRUCT_nb1150_oof":
        return reconstruct_nb1150_oof(n_unb)
    if rel == "_RECONSTRUCT_nb1191_oof":
        # nb1191 anchor: rebuild as its own SLSQP+stretch honest OOF mean
        # across seeds. We compute this once and cache below.
        raise RuntimeError(
            "_RECONSTRUCT_nb1191_oof must be supplied via cache, not loader"
        )
    p = DATA_PROCESSED / rel
    assert p.exists(), f"missing OOF {p}"
    v = np.load(p).astype(np.float64)
    assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
    return v


def stack_oof_for_splits(P, y_unb, splits):
    """Per-fold SLSQP + rank-stretch on anchor cols P[n_unb,K]. Returns
    (oof[n_unb], pooled_rae)."""
    n_unb = P.shape[0]
    oof = np.full(n_unb, np.nan)
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P[tr_loc], y_unb[tr_loc])
        blend_tr = P[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P[va_loc] @ w_f
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
    pooled = float(rae(y_unb, oof))
    return oof, pooled


def reconstruct_stack_oof_mean_seeds(anchor_cols_P, y_unb, unb_scaffolds):
    """Mean-across-kf_seeds honest cross-fit OOF for a single stack."""
    per_seed_oof = []
    per_seed_rae = []
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        oof, pooled = stack_oof_for_splits(anchor_cols_P, y_unb, splits)
        per_seed_oof.append(oof)
        per_seed_rae.append(pooled)
    mean_oof = np.mean(np.column_stack(per_seed_oof), axis=1)
    return mean_oof, per_seed_rae


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 3-way arithmetic mean of {{nb2171, nb2095, nb1191}}")
    print("    NON-SLSQP, NO FIT -- cannot collapse")
    print("=" * 78)

    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Build nb1191 OOF first (needed as an anchor for nb2171) ----
    print("\n[stage1a] reconstruct nb1191 OOF (anchor for nb2171)")
    nb1191_anchors_P = np.column_stack([
        load_anchor_vec(rel, n_unb) for _, rel in STACKS["nb1191"]
    ])
    nb1191_mean_oof, nb1191_per_seed_rae = reconstruct_stack_oof_mean_seeds(
        nb1191_anchors_P, y_unb, unb_scaffolds,
    )
    nb1191_pooled_mean = float(np.mean(nb1191_per_seed_rae))
    print(f"   nb1191 per-seed RAE = {[f'{r:.4f}' for r in nb1191_per_seed_rae]}")
    print(f"   nb1191 mean-seed RAE = {nb1191_pooled_mean:.4f}  "
          f"(reported {STACK_INDIV_OOF_REPORTED['nb1191']:.4f})")
    print(f"   RAE of mean-of-seed OOF = "
          f"{float(rae(y_unb, nb1191_mean_oof)):.4f}")

    # ---- Helper to resolve _RECONSTRUCT_nb1191_oof for nb2171 ----
    def resolve_vec(rel: str) -> np.ndarray:
        if rel == "_RECONSTRUCT_nb1191_oof":
            return nb1191_mean_oof
        return load_anchor_vec(rel, n_unb)

    # ---- Stage 1b: reconstruct OOFs for the three stacks ----
    stack_mean_oof = {}
    stack_pooled_mean = {}
    stack_per_seed_rae = {}
    print("\n[stage1b] reconstruct stack OOFs (mean across 5 kf_seeds)")
    for sname, spec in STACKS.items():
        if sname == "nb1191":
            stack_mean_oof[sname] = nb1191_mean_oof
            stack_per_seed_rae[sname] = nb1191_per_seed_rae
            stack_pooled_mean[sname] = nb1191_pooled_mean
        else:
            cols = [resolve_vec(rel) for _, rel in spec]
            P = np.column_stack(cols)
            mean_oof, per_seed_rae = reconstruct_stack_oof_mean_seeds(
                P, y_unb, unb_scaffolds,
            )
            stack_mean_oof[sname] = mean_oof
            stack_per_seed_rae[sname] = per_seed_rae
            stack_pooled_mean[sname] = float(np.mean(per_seed_rae))
        rep = STACK_INDIV_OOF_REPORTED[sname]
        delta = stack_pooled_mean[sname] - rep
        print(f"   {sname:8s}  per-seed={[f'{r:.4f}' for r in stack_per_seed_rae[sname]]}  "
              f"mean={stack_pooled_mean[sname]:.4f}  (rep~{rep:.4f}  delta={delta:+.4f})")

    # ---- Stage 2: arithmetic mean across the 3 stack OOFs ----
    stack_order = ["nb2171", "nb2095", "nb1191"]
    oof_matrix = np.column_stack([stack_mean_oof[s] for s in stack_order])
    avg_oof = oof_matrix.mean(axis=1)
    avg_rae = float(rae(y_unb, avg_oof))
    print(f"\n[stage2] arithmetic mean of 3 stack OOFs:")
    print(f"   pred mean/std = {avg_oof.mean():.4f} / {avg_oof.std():.4f}")
    print(f"   truth mean/std = {y_unb.mean():.4f} / {y_unb.std():.4f}")
    print(f"   pooled scaffold-CV RAE (mean of stack mean-seed OOFs) = "
          f"{avg_rae:.4f}")

    # Also compute per-seed avg-RAE: for each kf_seed, mean the per-seed OOFs.
    # Need to recompute per-seed OOFs explicitly. Build per-seed avg.
    print("\n[stage2 detail] per-seed pooled RAE of avg-of-3 OOFs")
    per_seed_avg_rae = []
    per_seed_stack_oofs_all = {s: [] for s in stack_order}
    for kf_seed in KF_SEEDS:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
        )
        seed_stack_oofs = {}
        for sname, spec in STACKS.items():
            if sname == "nb1191":
                cols = [resolve_vec(rel) for _, rel in spec]
            else:
                cols = [resolve_vec(rel) for _, rel in spec]
            P = np.column_stack(cols)
            oof, _ = stack_oof_for_splits(P, y_unb, splits)
            seed_stack_oofs[sname] = oof
            per_seed_stack_oofs_all[sname].append(oof)
        avg_seed = np.column_stack(
            [seed_stack_oofs[s] for s in stack_order]
        ).mean(axis=1)
        r_s = float(rae(y_unb, avg_seed))
        per_seed_avg_rae.append(r_s)
        print(f"   seed={kf_seed}  avg_RAE={r_s:.4f}")
    avg_rae_mean_seeds = float(np.mean(per_seed_avg_rae))
    avg_rae_std_seeds = float(np.std(per_seed_avg_rae))
    print(f"   mean across seeds = {avg_rae_mean_seeds:.4f}  "
          f"(+/- {avg_rae_std_seeds:.4f})")

    # ---- Pairwise Pearson across the 3 stack mean-seed OOFs ----
    print("\n[pearson] pairwise on mean-seed stack OOFs:")
    names_div = stack_order
    R_div = np.eye(3)
    pearson_pairs = {}
    for i, j in combinations(range(3), 2):
        r, _ = pearsonr(stack_mean_oof[names_div[i]], stack_mean_oof[names_div[j]])
        R_div[i, j] = r
        R_div[j, i] = r
        key = f"{names_div[i]}|{names_div[j]}"
        pearson_pairs[key] = float(r)
        print(f"   {key:20s}  Pearson={r:+.4f}")
    max_pearson = max(pearson_pairs.values())
    min_pearson = min(pearson_pairs.values())
    print(f"   min={min_pearson:.4f}  max={max_pearson:.4f}")

    # Pearson of each stack vs the average (sanity)
    print("[pearson] each stack vs avg:")
    for s in stack_order:
        r, _ = pearsonr(stack_mean_oof[s], avg_oof)
        print(f"   {s:8s}  Pearson(stack, avg) = {r:+.4f}")

    # ---- Deploy: arithmetic mean of the 3 cached te files on the 513 ----
    print("\n[stage3] deploy = arithmetic mean of cached te files (513)")
    te_cols = []
    for s in stack_order:
        p = DATA_PROCESSED / STACK_TE_DEPLOY[s]
        assert p.exists(), f"missing te {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_te,), f"{s} te shape {v.shape}"
        te_cols.append(v)
        print(f"   {s:8s}  te mean/std = {v.mean():.4f} / {v.std():.4f}  "
              f"te[unb] RAE = {float(rae(y_unb, v[unb_idx])):.4f}")
    deploy_te = np.column_stack(te_cols).mean(axis=1).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    lb_band_est = LB_W_OOF * avg_rae_mean_seeds + LB_W_TE * te_unb_rae
    print(f"   deploy_te mean/std = {deploy_te.mean():.4f} / {deploy_te.std():.4f}")
    print(f"   deploy_te[unb_idx] RAE (in-sample) = {te_unb_rae:.4f}")
    print(f"   LB-band estimate = {LB_W_OOF}*OOF({avg_rae_mean_seeds:.4f}) + "
          f"{LB_W_TE}*te_unb({te_unb_rae:.4f}) = {lb_band_est:.4f}")

    # ---- Gate ----
    gate_a = avg_rae_mean_seeds <= GATE_OOF_MAX
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   avg_RAE (mean-seeds) = {avg_rae_mean_seeds:.4f}  "
          f"<= {GATE_OOF_MAX:.4f} ?  -> {'PASS' if gate_a else 'FAIL'}")
    print(f"   delta vs nb2171 (0.4676)        = "
          f"{avg_rae_mean_seeds - NB2171_REF_RAE:+.4f}  "
          f"(need <= -0.003 to PASS)")

    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_avg3.csv"
    if gate_a:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate PASSED)")
    else:
        print(f"[skip] gate FAILED -- no submission CSV written")
        sub_csv_path = None

    # ---- Comparisons ----
    print("\n[compare] vs individual stack OOFs:")
    for s in stack_order:
        marker = "  <- avg" if False else ""
        print(f"   {s:8s}  reported={STACK_INDIV_OOF_REPORTED[s]:.4f}  "
              f"rebuilt_mean={stack_pooled_mean[s]:.4f}")
    print(f"   {'avg3':8s}  rebuilt_mean={avg_rae_mean_seeds:.4f}  <- this build")
    print(f"   delta vs best (nb2171 0.4676) = "
          f"{avg_rae_mean_seeds - NB2171_REF_RAE:+.4f}")

    summary = {
        "tag": TAG,
        "method": "arithmetic_mean_of_3_stacks_no_fit",
        "stacks": stack_order,
        "stack_indiv_oof_reported": STACK_INDIV_OOF_REPORTED,
        "stack_te_deploy_paths": STACK_TE_DEPLOY,
        "nb2171_ref_rae": NB2171_REF_RAE,
        "gate_oof_max_target": GATE_OOF_MAX,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "stack_per_seed_rae": {
            s: [float(r) for r in stack_per_seed_rae[s]] for s in stack_order
        },
        "stack_pooled_rae_mean_seeds": {
            s: float(stack_pooled_mean[s]) for s in stack_order
        },
        "per_seed_avg_pooled_rae": [float(r) for r in per_seed_avg_rae],
        "avg_pooled_rae_mean_seeds": avg_rae_mean_seeds,
        "avg_pooled_rae_std_seeds": avg_rae_std_seeds,
        "rae_of_mean_of_seed_oof_then_average": avg_rae,
        "pairwise_pearson_stack_oofs": pearson_pairs,
        "max_pairwise_pearson": float(max_pearson),
        "min_pairwise_pearson": float(min_pearson),
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "delta_vs_nb2171": avg_rae_mean_seeds - NB2171_REF_RAE,
        "delta_vs_required_to_pass_gate": (
            avg_rae_mean_seeds - NB2171_REF_RAE
        ) - (-0.003),
        "gate_pass": bool(gate_a),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if sub_csv_path else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   avg3 pooled scaffold-CV RAE (mean-seeds) = "
          f"{avg_rae_mean_seeds:.4f}  (+/- {avg_rae_std_seeds:.4f})")
    print(f"   delta vs nb2171 (0.4676)                  = "
          f"{avg_rae_mean_seeds - NB2171_REF_RAE:+.4f}")
    print(f"   max pairwise Pearson                      = {max_pearson:.4f}")
    print(f"   gate_pass                                 = {gate_a}")
    print(f"   wall                                      = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "avg_pooled_rae_mean_seeds",
        "avg_pooled_rae_std_seeds",
        "delta_vs_nb2171",
        "max_pairwise_pearson",
        "min_pairwise_pearson",
        "pairwise_pearson_stack_oofs",
        "te_unb_rae_in_sample",
        "lb_band_estimate",
        "gate_pass",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
