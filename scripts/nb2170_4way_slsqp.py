"""nb2170 -- 4-way SLSQP simplex on top of PRIMARY-1 stack candidates.

Hypothesis test: does ANY convex blend of the 4 PRIMARY-1 stacks
{nb1162, nb2095, nb1191, nb2060} beat nb2095 alone (0.4720 deep-30)?

Per memory feedback_stack_overfitting + feedback_lb_two_regime_calibration:
- nb1162 carries 88.7% weight on nb730_honest (POST-unblind anchor); its
  honest cross-fit OOF (0.4206) is in the POST-leak regime and LB-fragile.
- nb2095/nb1191/nb2060 are PRE/pseudo-PRE pyramids; cross-fits 0.4720/0.4718/0.4720.
- Expectation: meta-SLSQP collapses to nb1162 weight=1 (in-sample minimum)
  OR to nb2095 weight=1 (LB-honest minimum). Documented for the record.

DESIGN -- honest cross-fit at META layer:
Each of the 4 stacks is itself a per-fold SLSQP + rank-stretch on its own
anchor set. To produce stack-level OOFs on the 253 that share a SINGLE set
of scaffold splits (so meta-SLSQP is well-defined), we REBUILD each stack
per meta-fold using the SAME splits:
  For each kf_seed in {1001..1005}:
    splits = scaffold_kfold_indices(unb_scaffolds, 5, seed=kf_seed)
    For each fold (tr, va):
      For each stack s in {nb1162, nb2095, nb1191, nb2060}:
        w_s = slsqp_simplex(anchors_s[tr], y[tr])
        s_s = best_stretch(anchors_s[tr] @ w_s, y[tr])
        stack_oof_s[va] = mu + s_s * (anchors_s[va] @ w_s - mu)
      w_meta = slsqp_simplex(stack_cols[tr], y[tr])
      s_meta = best_stretch(stack_cols[tr] @ w_meta, y[tr])
      meta_oof[va] = mu_m + s_m * (stack_cols[va] @ w_meta - mu_m)

The 4 stacks share several anchor OOFs (chemprop_aux, nb2103, nb503, nb562),
which means meta diversity is structurally bounded -- the diversity gate
(<=0.95 pairwise Pearson) will document this.

DEPLOY: refit per-stack SLSQP on FULL 253; meta-SLSQP on the 4 stack-level
in-sample (full-pool) predictions; apply to the 513 via each stack's cached
te file (which already encodes its own deploy-weight blend + stretch).

GATE A: meta-SLSQP pooled OOF (mean of 5 seeds) <= 0.4690 (beat nb2095 by
    >= 0.003)
GATE B: meta-SLSQP non-degenerate (max single weight < 0.95)

If GATE A passes -> save deploy te_npy and submission CSV; flag deep-30
verify needed before promote.

Always saves data/processed/nb2170_summary.json.
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

TAG = "nb2170"
GATE_OOF_MAX = 0.4690        # must beat nb2095 (0.4720 deep-30) by >= 0.003
GATE_DIVERSITY_MAX_PEARSON = 0.95
GATE_DEGEN_MAX_WEIGHT = 0.95
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# ---------------------------------------------------------------------------
# Stack definitions -- mirror each PRIMARY-1's anchor spec exactly.
# Each entry: (display, [(anchor_disp, oof_rel_or_RECONSTRUCT, te_rel), ...])
# ---------------------------------------------------------------------------

# nb1150 reconstructed from cached SLSQP4 weights (same recipe as nb2060/nb2095)
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


STACKS = {
    "nb1162": [
        ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_chemprop_aux.npy"),
        ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
        ("nb730_honest", "nb730_honest_pred_oof.npy",         "te_nb730_honest.npy"),
        ("nb503",        "nb503_pred_oof.npy",                "te_nb503.npy"),
        ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
    ],
    "nb2095": [
        ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
        ("nb1150",       "_RECONSTRUCT_nb1150_oof",           "te_nb1150.npy"),
        ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",       "te_nb1158.npy"),
        ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_nb2112.npy"),
        ("nb1014",       "nb1133_nb1014_pred_oof.npy",        "te_nb1014.npy"),
    ],
    "nb1191": [
        ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
        ("nb1150",       "_RECONSTRUCT_nb1150_oof",           "te_nb1150.npy"),
        ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",       "te_nb1158.npy"),
        ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_nb2112.npy"),
    ],
    "nb2060": [
        ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
        ("nb1150",       "_RECONSTRUCT_nb1150_oof",           "te_nb1150.npy"),
        ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",       "te_nb1158.npy"),
        ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_nb2112.npy"),
        ("nb503",        "nb503_pred_oof.npy",                "te_nb503.npy"),
        ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
    ],
}

STACK_TE_DEPLOY = {
    "nb1162": "te_nb1162.npy",
    "nb2095": "te_nb2095.npy",
    "nb1191": "te_nb1191.npy",
    "nb2060": "te_nb2060.npy",
}
STACK_INDIV_OOF_REPORTED = {
    "nb1162": 0.4206,
    "nb2095": 0.4720,
    "nb1191": 0.4718,
    "nb2060": 0.4720,
}


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


def load_stack_anchors(stack_name: str, n_unb: int):
    """Return list of (disp, oof_vec[n_unb]) tuples for a stack."""
    cols = []
    for disp, oof_rel, _te_rel in STACKS[stack_name]:
        if oof_rel == "_RECONSTRUCT_nb1150_oof":
            v = reconstruct_nb1150_oof(n_unb)
        else:
            p = DATA_PROCESSED / oof_rel
            assert p.exists(), f"{stack_name}: missing OOF {p}"
            v = np.load(p).astype(np.float64)
            assert v.shape == (n_unb,), f"{stack_name} {disp} shape {v.shape}"
        cols.append((disp, v))
    return cols


def stack_oof_for_splits(stack_name, anchor_cols, y_unb, splits):
    """Reconstruct one stack's honest cross-fit OOF for a given fold split.

    Per-fold SLSQP + rank-stretch on stack's anchor OOFs. Returns
    (oof_blend[n_unb], pooled_rae, fold_w_list, fold_s_list).
    """
    P = np.column_stack([v for _, v in anchor_cols])
    n_unb = P.shape[0]
    oof = np.full(n_unb, np.nan)
    fold_w, fold_s = [], []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P[tr_loc], y_unb[tr_loc])
        blend_tr = P[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P[va_loc] @ w_f
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    pooled = float(rae(y_unb, oof))
    return oof, pooled, fold_w, fold_s


def meta_cv_for_seed(stack_anchors_dict, y_unb, unb_scaffolds, kf_seed):
    """One scaffold-fold sweep that rebuilds all 4 stacks AND meta-blends them.

    Returns:
      stack_oofs_dict[name] -> n_unb honest OOF for this seed
      meta_oof -> n_unb meta-SLSQP honest OOF
      stack_pooled_rae[name] -> float
      meta_pooled_rae -> float
      meta_fold_w  -> list of [w_nb1162, w_nb2095, w_nb1191, w_nb2060]
      meta_fold_s  -> list of stretch
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    # ---- Stage 1: rebuild each stack's honest OOF on these splits ----
    stack_oofs, stack_pooled = {}, {}
    for sname, anchors in stack_anchors_dict.items():
        oof, pooled, _, _ = stack_oof_for_splits(sname, anchors, y_unb, splits)
        stack_oofs[sname] = oof
        stack_pooled[sname] = pooled

    # ---- Stage 2: meta-SLSQP on same splits using stack OOFs as columns ----
    stack_order = list(stack_anchors_dict.keys())
    P_meta = np.column_stack([stack_oofs[s] for s in stack_order])
    meta_oof = np.full(n_unb, np.nan)
    meta_fold_w, meta_fold_s = [], []
    for tr_loc, va_loc in splits:
        w_m = slsqp_simplex(P_meta[tr_loc], y_unb[tr_loc])
        blend_tr = P_meta[tr_loc] @ w_m
        mu_tr = float(blend_tr.mean())
        s_m, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_meta[va_loc] @ w_m
        meta_oof[va_loc] = mu_tr + s_m * (blend_va - mu_tr)
        meta_fold_w.append(w_m)
        meta_fold_s.append(s_m)
    meta_pooled = float(rae(y_unb, meta_oof))
    return (stack_oofs, meta_oof, stack_pooled, meta_pooled,
            meta_fold_w, meta_fold_s, stack_order)


def pearson_matrix(cols_dict):
    names = list(cols_dict.keys())
    n = len(names)
    R = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = pearsonr(cols_dict[names[i]], cols_dict[names[j]])
            R[i, j] = r
            R[j, i] = r
    return names, R


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 4-way SLSQP simplex on PRIMARY-1 stacks")
    print(f"   stacks: {list(STACKS.keys())}")
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

    # ---- Load each stack's anchor OOFs once ----
    print("\n[load anchors per stack]")
    stack_anchors_dict = {}
    for sname in STACKS:
        anchors = load_stack_anchors(sname, n_unb)
        stack_anchors_dict[sname] = anchors
        print(f"   {sname:8s}  K_anchors={len(anchors)}  "
              f"anchors=[{', '.join(d for d, _ in anchors)}]")

    # ---- Sanity: each stack's first-seed honest OOF roughly matches reported ----
    # (use kf_seed=1001 + same splits as nb2095/nb1191/nb2060)
    splits_check = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=1001,
    )
    print("\n[sanity] stack OOF (kf_seed=1001) vs reported:")
    for sname, anchors in stack_anchors_dict.items():
        _, pooled, _, _ = stack_oof_for_splits(sname, anchors, y_unb, splits_check)
        rep = STACK_INDIV_OOF_REPORTED[sname]
        print(f"   {sname:8s}  pooled={pooled:.4f}  (reported~{rep:.4f}  "
              f"delta={pooled - rep:+.4f})")

    # ---- Stage 1 + 2: meta-CV across all 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"META SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}")
    print("-" * 78)

    per_seed = []
    all_meta_oofs = []
    all_stack_oofs = {sn: [] for sn in STACKS}
    stack_order_persistent = list(STACKS.keys())

    for kf_seed in KF_SEEDS:
        (stack_oofs_seed, meta_oof, stack_pooled, meta_pooled,
         meta_fw, meta_fs, stack_order) = meta_cv_for_seed(
            stack_anchors_dict, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "stack_pooled_rae": {k: float(v) for k, v in stack_pooled.items()},
            "meta_pooled_rae": meta_pooled,
            "meta_fold_s": [float(x) for x in meta_fs],
            "meta_fold_w_mean": [
                float(x) for x in np.mean(meta_fw, axis=0)
            ],
            "stack_order": stack_order,
        })
        all_meta_oofs.append(meta_oof)
        for sn in stack_order_persistent:
            all_stack_oofs[sn].append(stack_oofs_seed[sn])
        print(f"   seed={kf_seed}  meta_RAE={meta_pooled:.4f}  "
              f"mean_s={np.mean(meta_fs):.3f}  "
              f"w_mean(nb1162,nb2095,nb1191,nb2060)="
              f"{np.round(np.mean(meta_fw, axis=0), 3).tolist()}")

    mean_meta_oof = np.mean(np.column_stack(all_meta_oofs), axis=1)
    meta_rae_mean_seeds = float(
        np.mean([r["meta_pooled_rae"] for r in per_seed])
    )
    meta_rae_std_seeds = float(
        np.std([r["meta_pooled_rae"] for r in per_seed])
    )
    rae_of_mean_meta = float(rae(y_unb, mean_meta_oof))
    print(f"\n[cv] meta pooled RAE mean across seeds = "
          f"{meta_rae_mean_seeds:.4f} (+/- {meta_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed meta OOFs     = {rae_of_mean_meta:.4f}")

    # ---- Diversity gate: pairwise Pearson on mean-of-seed stack OOFs ----
    mean_stack_oofs = {
        sn: np.mean(np.column_stack(all_stack_oofs[sn]), axis=1)
        for sn in stack_order_persistent
    }
    names_div, R_div = pearson_matrix(mean_stack_oofs)
    print("\n[diversity] pairwise Pearson on mean-of-seed stack OOFs:")
    print(f"   stacks: {names_div}")
    for i in range(len(names_div)):
        row = "   " + " ".join(f"{R_div[i, j]:+.3f}" for j in range(len(names_div)))
        print(f"   {names_div[i]:8s} {row}")
    pearson_upper = [(names_div[i], names_div[j], float(R_div[i, j]))
                     for i, j in combinations(range(len(names_div)), 2)]
    max_pearson = max(p for _, _, p in pearson_upper)
    print(f"   max pairwise Pearson = {max_pearson:.4f}  "
          f"(gate <= {GATE_DIVERSITY_MAX_PEARSON})")

    # ---- Deploy: full-pool refit ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit each stack on FULL 253, then meta-SLSQP on stack columns)")
    print("-" * 78)

    # Stage 1 deploy: each stack's full-pool SLSQP + stretch -> in-sample preds
    # on the 253 (these are in-sample/overfit; they serve as the stack columns
    # for the meta deploy weights). Stretch s = mean across all per-seed
    # folds for that stack.
    stack_full_pool_unb = {}
    stack_full_pool_stretch_s = {}
    for sname, anchors in stack_anchors_dict.items():
        P_s = np.column_stack([v for _, v in anchors])
        w_s = slsqp_simplex(P_s, y_unb)
        blend = P_s @ w_s
        mu_s = float(blend.mean())
        # Use mean per-fold stretch from this stack's per-seed folds
        per_stack_fold_s = []
        for kf_seed in KF_SEEDS:
            splits = scaffold_kfold_indices(
                unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
            )
            _, _, _, fold_s = stack_oof_for_splits(sname, anchors, y_unb, splits)
            per_stack_fold_s.extend(fold_s)
        s_s = float(np.mean(per_stack_fold_s))
        stack_full_pool_unb[sname] = mu_s + s_s * (blend - mu_s)
        stack_full_pool_stretch_s[sname] = s_s
        print(f"   stage1 {sname:8s}  w={np.round(w_s, 3).tolist()}  "
              f"s={s_s:.3f}  mu={mu_s:.3f}")

    P_meta_deploy = np.column_stack(
        [stack_full_pool_unb[sn] for sn in stack_order_persistent]
    )
    w_meta_deploy = slsqp_simplex(P_meta_deploy, y_unb)
    blend_meta_unb = P_meta_deploy @ w_meta_deploy
    mu_meta = float(blend_meta_unb.mean())
    # mean meta stretch across all seeds * folds
    s_meta_all = [s for r in per_seed for s in r["meta_fold_s"]]
    s_meta = float(np.mean(s_meta_all))
    in_meta_rae = float(rae(
        y_unb, mu_meta + s_meta * (blend_meta_unb - mu_meta)
    ))
    print(f"   meta deploy w={np.round(w_meta_deploy, 4).tolist()}  "
          f"s={s_meta:.3f}  mu={mu_meta:.3f}")
    print(f"   in-sample meta RAE (253) = {in_meta_rae:.4f}  (overfit upper bound)")

    # ---- Deploy fan: blend the 4 stacks' cached te[513] files with w_meta ----
    te_stack_cols = []
    for sn in stack_order_persistent:
        p = DATA_PROCESSED / STACK_TE_DEPLOY[sn]
        assert p.exists(), f"missing stack te: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_te,), f"{sn} te shape {v.shape}"
        te_stack_cols.append(v)
    P_te = np.column_stack(te_stack_cols)
    blend_te = P_te @ w_meta_deploy
    deploy_te = (mu_meta + s_meta * (blend_te - mu_meta)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))
    lb_band_est = LB_W_OOF * meta_rae_mean_seeds + LB_W_TE * te_unb_rae
    print(f"   deploy te(513) mean/std = {deploy_te.mean():.3f} / "
          f"{deploy_te.std():.3f}")
    print(f"   te[unb_idx] RAE (in-sample) = {te_unb_rae:.4f}")
    print(f"   LB-band estimate = {LB_W_OOF}*OOF({meta_rae_mean_seeds:.4f}) + "
          f"{LB_W_TE}*te_unb({te_unb_rae:.4f}) = {lb_band_est:.4f}")

    # ---- Gates ----
    gate_a = meta_rae_mean_seeds <= GATE_OOF_MAX
    max_w_deploy = float(np.max(w_meta_deploy))
    gate_b_nondegen = max_w_deploy < GATE_DEGEN_MAX_WEIGHT
    gate_div = max_pearson <= GATE_DIVERSITY_MAX_PEARSON
    gate_pass = gate_a and gate_b_nondegen and gate_div

    # Detect degeneracy collapse (per memory expectation: w_nb1162=1)
    deploy_w_dict = {
        sn: float(w_meta_deploy[i])
        for i, sn in enumerate(stack_order_persistent)
    }
    argmax_stack = max(deploy_w_dict, key=deploy_w_dict.get)
    is_degenerate = deploy_w_dict[argmax_stack] >= GATE_DEGEN_MAX_WEIGHT
    expected_collapse_per_memo = (
        argmax_stack == "nb1162" and is_degenerate
    )

    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   gate A (OOF beat): meta={meta_rae_mean_seeds:.4f} <= "
          f"{GATE_OOF_MAX:.4f}  -> {'PASS' if gate_a else 'FAIL'}")
    print(f"   gate B (non-degenerate): max_w={max_w_deploy:.4f} < "
          f"{GATE_DEGEN_MAX_WEIGHT}  -> "
          f"{'PASS' if gate_b_nondegen else 'FAIL'}")
    print(f"   gate diversity: max_Pearson={max_pearson:.4f} <= "
          f"{GATE_DIVERSITY_MAX_PEARSON}  -> "
          f"{'PASS' if gate_div else 'FAIL'}")
    print(f"   overall: {'PASS' if gate_pass else 'FAIL'}")
    if is_degenerate:
        print(f"   degeneracy detected -> collapsed to {argmax_stack} "
              f"(w={deploy_w_dict[argmax_stack]:.4f})  "
              f"expected_per_memo={expected_collapse_per_memo}")

    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_4way_meta_slsqp.csv"
    if gate_pass:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate PASSED -- needs deep-30 verify)")
    else:
        print(f"[skip] gate FAILED -- no submission CSV written")
        sub_csv_path = None

    # ---- Comparisons ----
    cmp_table = {sn: STACK_INDIV_OOF_REPORTED[sn] for sn in STACKS}
    cmp_table["meta_nb2170"] = meta_rae_mean_seeds
    print(f"\n[compare] vs reported indiv OOF:")
    for k, v in cmp_table.items():
        marker = "  <- this build" if k == "meta_nb2170" else ""
        print(f"   {k:14s}  RAE={v:.4f}{marker}")
    nb2095_ref = STACK_INDIV_OOF_REPORTED["nb2095"]
    print(f"   delta meta vs nb2095 (0.4720)  = "
          f"{meta_rae_mean_seeds - nb2095_ref:+.4f}  "
          f"(need <= {-0.003:.4f} to beat by 0.003)")

    summary = {
        "tag": TAG,
        "method": "4way_SLSQP_simplex_meta_on_PRIMARY1_stacks",
        "stacks_meta": list(STACKS.keys()),
        "stack_anchor_oof_paths": {
            sn: [a[1] for a in STACKS[sn]] for sn in STACKS
        },
        "stack_te_deploy_paths": STACK_TE_DEPLOY,
        "stack_indiv_oof_reported": STACK_INDIV_OOF_REPORTED,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed,
        "meta_pooled_rae_mean_seeds": meta_rae_mean_seeds,
        "meta_pooled_rae_std_seeds": meta_rae_std_seeds,
        "rae_of_mean_of_seed_meta_oofs": rae_of_mean_meta,
        "pairwise_pearson_stack_oofs": {
            f"{a}|{b}": p for a, b, p in pearson_upper
        },
        "max_pairwise_pearson": max_pearson,
        "stack_full_pool_stretch_s": stack_full_pool_stretch_s,
        "deploy_weights": [
            {"name": sn, "w": float(w_meta_deploy[i])}
            for i, sn in enumerate(stack_order_persistent)
        ],
        "deploy_mu_blend": mu_meta,
        "deploy_s": s_meta,
        "in_sample_meta_rae_overfit_bound": in_meta_rae,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "lb_band_w_oof": LB_W_OOF,
        "lb_band_w_te": LB_W_TE,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "gate_oof_max_target": GATE_OOF_MAX,
        "gate_degeneracy_max_w_target": GATE_DEGEN_MAX_WEIGHT,
        "gate_diversity_max_pearson_target": GATE_DIVERSITY_MAX_PEARSON,
        "gate_a_oof_beat_pass": bool(gate_a),
        "gate_b_nondegenerate_pass": bool(gate_b_nondegen),
        "gate_diversity_pass": bool(gate_div),
        "gate_pass": bool(gate_pass),
        "is_degenerate_collapse": bool(is_degenerate),
        "degeneracy_argmax_stack": argmax_stack,
        "degeneracy_max_weight": deploy_w_dict[argmax_stack],
        "expected_collapse_to_nb1162_per_memo": bool(expected_collapse_per_memo),
        "delta_vs_nb2095_alone": meta_rae_mean_seeds - nb2095_ref,
        "delta_vs_required": (meta_rae_mean_seeds - nb2095_ref) - (-0.003),
        "deep30_verify_required_if_gate_pass": True,
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
    print(f"   meta pooled scaffold-CV RAE (mean of seeds) = "
          f"{meta_rae_mean_seeds:.4f}  (+/- {meta_rae_std_seeds:.4f})")
    print(f"   delta vs nb2095 alone (0.4720)              = "
          f"{meta_rae_mean_seeds - nb2095_ref:+.4f}")
    print(f"   max pairwise Pearson (diversity)            = {max_pearson:.4f}")
    print(f"   deploy weights = {[(sn, round(deploy_w_dict[sn], 4)) for sn in stack_order_persistent]}")
    print(f"   degenerate                                  = {is_degenerate} "
          f"(argmax={argmax_stack})")
    print(f"   gate_pass                                   = {gate_pass}")
    print(f"   wall                                        = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "meta_pooled_rae_mean_seeds",
        "meta_pooled_rae_std_seeds",
        "delta_vs_nb2095_alone",
        "max_pairwise_pearson",
        "is_degenerate_collapse",
        "degeneracy_argmax_stack",
        "expected_collapse_to_nb1162_per_memo",
        "gate_pass",
        "deploy_weights",
        "deploy_s",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
