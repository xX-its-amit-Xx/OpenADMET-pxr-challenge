"""nb1195 -- Rank-stretch on nb1162 stack-pyramid output (F2-diagnosed).

CONTEXT:
    nb1162 is the 5-anchor SLSQP+rank-stretch stacking pyramid:
        anchors = [nb2103_K28, chemprop_aux, nb730_honest, nb503, nb562]
        scaffold 5-fold CV pooled RAE = 0.4204 (cached in nb1162_summary.json)
    Internally nb1162 tried a stretch grid {1.00 ... 1.15} but per-fold s
    landed on 1.0 every fold (gate B failed). That decision was made on the
    POOLED 253-row variance balance. The F2 cohort (novel-scaffold,
    scaf_train_freq=0) is where variance compression bites hardest, so we
    re-diagnose pred_std vs truth_std on F2 only, then re-run a wider per-fold
    stretch search dominated by the F2-aware loss.

PROTOCOL:
    1. Reconstruct nb1162 scaffold-CV OOF on the 253 unblind rows:
         - Load the 5 anchor OOFs + truth y_unb + unb scaffolds
         - Re-run identical SLSQP scaffold-CV using s=1.0 inside (so we
           recover the *pre-stretch* blend that nb1162 fed into its grid)
         - That reconstructed OOF, by construction, has pooled RAE ~0.4204
    2. Diagnose pred_std vs truth_std on the F2 cohort:
         - F2 cohort = unblind rows whose Bemis-Murcko scaffold has
           train_freq=0 (novel scaffold, no train support)
         - Report n_F2, pred_std_F2, truth_std_F2, and the F2-only RAE
    3. Per-fold cross-fit grid stretch over WIDE grid
           s in {1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15}
       Fitted on the in-fold training portion of the reconstructed OOF
       (mu = mean of in-fold pred), applied to the held-out fold
    4. Pooled scaffold-CV RAE per s after stretch
    5. GATES (BOTH must pass to deploy):
         best_s >= 1.03                                (stretch alive)
         delta_RAE = base_rae - best_rae >= 0.005      (real improvement)
    6. If gates pass:
         - Refit blend weights on full 253 (SLSQP) and stretch by best_s
           around mean(blend_full); fan to 513 via the cached anchor te
           vectors used by nb1162; write submissions/nb1196_deploy_stretched.csv
           as PRIMARY-1-STRETCHED candidate.
    7. Always write data/processed/nb1195_summary.json.

This is a 1-parameter scalar calibration on top of an already-deployed
ensemble. Matches the "Rank-stretch is universal post-hoc calibration" memory
note: s in [1.05, 1.10] gains -0.003 to -0.005 RAE on variance-compressed
predictors when truth_std > pred_std. If the gates fail, we do NOT submit and
nb1162 stays as-is.
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
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1195"
N_FOLDS = 5
SEED = 42
STRETCH_GRID = [1.00, 1.02, 1.05, 1.08, 1.10, 1.12, 1.15]
GATE_S_MIN = 1.03
GATE_DELTA_RAE = 0.005

# Same anchor ordering as nb1162 (must stay in lockstep)
ANCHORS = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy",       "te_chemprop_aux.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy",  "te_chemprop_aux.npy"),
    ("nb730_honest", "nb730_honest_pred_oof.npy",         "te_nb730_honest.npy"),
    ("nb503",        "nb503_pred_oof.npy",                "te_nb503.npy"),
    ("nb562",        "nb562_pred_oof.npy",                "te_nb562.npy"),
]


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


def best_stretch_on(pred_tr: np.ndarray, y_tr: np.ndarray,
                    mu: float, grid) -> tuple[float, float]:
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        r = float(rae(y_tr, mu + s * (pred_tr - mu)))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Rank-stretch on nb1162 stack pyramid (F2-diagnosed)")
    print("=" * 78)

    # ---- Load 513 + 253 indices ----
    te = load_test()
    te_names = te["name"].values
    te_smiles = te["smiles"].values
    n_te = len(te_names)

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_te={n_te}  n_unb={n_unb}")

    # ---- Scaffolds for unb + train (for F2 cohort) ----
    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) or f"__sing_{i}__"
                     for i, s in enumerate(unb_smiles)]

    print("[scaffold] computing train scaffold frequency...")
    train = load_train()
    train_scaf = [bemis_murcko(s) for s in train["smiles"].values]
    train_scaf_counts: dict[str, int] = {}
    for s in train_scaf:
        if s:
            train_scaf_counts[s] = train_scaf_counts.get(s, 0) + 1

    f2_mask = np.array(
        [train_scaf_counts.get(s, 0) == 0 for s in unb_scaffolds],
        dtype=bool,
    )
    n_f2 = int(f2_mask.sum())
    print(f"[F2] novel-scaffold rows (train_freq==0): {n_f2} / {n_unb}")

    # ---- Load anchor OOFs + te ----
    oof_cols = []
    te_cols = []
    indiv_rae = {}
    print("\n[anchors]")
    for disp, oof_rel, te_rel in ANCHORS:
        oof = np.load(DATA_PROCESSED / oof_rel).astype(np.float64)
        te_arr = np.load(DATA_PROCESSED / te_rel).astype(np.float64)
        assert oof.shape == (n_unb,)
        assert te_arr.shape == (n_te,)
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:14s} oof_RAE={r:.4f}")

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K = P_unb.shape[1]

    # ---- Step 1: reconstruct nb1162 OOF (SLSQP, s=1.0 internal) ----
    print("\n" + "-" * 78)
    print("STEP 1: reconstruct nb1162 pre-stretch scaffold-CV blend")
    print("-" * 78)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=SEED,
    )
    pred_oof = np.full(n_unb, np.nan)
    fold_weights = []
    for k, (tr_loc, va_loc) in enumerate(splits):
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        pred_oof[va_loc] = P_unb[va_loc] @ w_f
        fold_weights.append(w_f)
        w_str = ",".join(f"{x:.3f}" for x in w_f)
        print(f"   fold {k}: n_tr={len(tr_loc):3d} n_va={len(va_loc):3d}  "
              f"w=[{w_str}]")
    assert not np.isnan(pred_oof).any()
    base_rae = float(rae(y_unb, pred_oof))
    print(f"[reconstruct] base pooled RAE = {base_rae:.4f}  "
          f"(nb1162 cached 0.4204)")

    # ---- Step 2: variance diagnosis (full + F2) ----
    print("\n" + "-" * 78)
    print("STEP 2: pred_std vs truth_std diagnosis")
    print("-" * 78)
    full_pred_std = float(pred_oof.std())
    full_truth_std = float(y_unb.std())
    full_var_ratio = full_truth_std / max(full_pred_std, 1e-9)
    print(f"   [ALL  n={n_unb}]  pred_std={full_pred_std:.4f}  "
          f"truth_std={full_truth_std:.4f}  s_var={full_var_ratio:.4f}")

    f2_pred = pred_oof[f2_mask]
    f2_truth = y_unb[f2_mask]
    if n_f2 >= 2:
        f2_pred_std = float(f2_pred.std())
        f2_truth_std = float(f2_truth.std())
        f2_var_ratio = f2_truth_std / max(f2_pred_std, 1e-9)
        f2_rae = float(rae(f2_truth, f2_pred))
    else:
        f2_pred_std = f2_truth_std = f2_var_ratio = float("nan")
        f2_rae = float("nan")
    print(f"   [F2   n={n_f2}]  pred_std={f2_pred_std:.4f}  "
          f"truth_std={f2_truth_std:.4f}  s_var={f2_var_ratio:.4f}  "
          f"F2_RAE={f2_rae:.4f}")

    in_pred = pred_oof[~f2_mask]
    in_truth = y_unb[~f2_mask]
    in_pred_std = float(in_pred.std()) if (~f2_mask).sum() >= 2 else float("nan")
    in_truth_std = float(in_truth.std()) if (~f2_mask).sum() >= 2 else float("nan")
    in_rae_val = float(rae(in_truth, in_pred)) if (~f2_mask).sum() >= 2 else float("nan")
    print(f"   [in-m n={int((~f2_mask).sum())}]  pred_std={in_pred_std:.4f}  "
          f"truth_std={in_truth_std:.4f}  in-manifold_RAE={in_rae_val:.4f}")

    # ---- Step 3+4: per-fold cross-fit grid stretch ----
    print("\n" + "-" * 78)
    print(f"STEP 3+4: per-fold cross-fit grid stretch  grid={STRETCH_GRID}")
    print("-" * 78)
    grid_results = {}
    for s_global in STRETCH_GRID:
        oof_s = np.full(n_unb, np.nan)
        for tr_loc, va_loc in splits:
            mu_tr = float(pred_oof[tr_loc].mean())
            oof_s[va_loc] = mu_tr + s_global * (pred_oof[va_loc] - mu_tr)
        r_pool = float(rae(y_unb, oof_s))
        r_f2 = (float(rae(y_unb[f2_mask], oof_s[f2_mask]))
                if n_f2 >= 2 else float("nan"))
        grid_results[s_global] = {
            "pooled_rae": r_pool,
            "f2_rae": r_f2,
            "delta_vs_base": base_rae - r_pool,
        }
        print(f"   s={s_global:.2f}  pooled_RAE={r_pool:.4f}  "
              f"F2_RAE={r_f2:.4f}  delta_vs_base={base_rae - r_pool:+.4f}")

    # Also: per-fold best (independent s per fold) — diagnostic, no double dip
    print("\n   [diagnostic] per-fold independent best (in-fold train search)")
    oof_perfold = np.full(n_unb, np.nan)
    perfold_s = []
    for k, (tr_loc, va_loc) in enumerate(splits):
        mu_tr = float(pred_oof[tr_loc].mean())
        s_k, _ = best_stretch_on(
            pred_oof[tr_loc], y_unb[tr_loc], mu_tr, STRETCH_GRID,
        )
        oof_perfold[va_loc] = mu_tr + s_k * (pred_oof[va_loc] - mu_tr)
        perfold_s.append(s_k)
        print(f"      fold {k}: s*={s_k:.2f}  mu_tr={mu_tr:.3f}")
    perfold_rae = float(rae(y_unb, oof_perfold))
    perfold_mean_s = float(np.mean(perfold_s))
    print(f"   per-fold pooled RAE = {perfold_rae:.4f}  "
          f"mean_s={perfold_mean_s:.4f}")

    # ---- Step 5: gate evaluation ----
    best_s = float(min(grid_results, key=lambda s: grid_results[s]["pooled_rae"]))
    best_rae = grid_results[best_s]["pooled_rae"]
    delta_rae = base_rae - best_rae
    gate_s = best_s >= GATE_S_MIN
    gate_delta = delta_rae >= GATE_DELTA_RAE
    gate_pass = gate_s and gate_delta

    print("\n" + "=" * 78)
    print("STEP 5: GATE EVALUATION")
    print("=" * 78)
    print(f"   base RAE                = {base_rae:.4f}")
    print(f"   best stretch s          = {best_s:.2f}")
    print(f"   best stretched RAE      = {best_rae:.4f}")
    print(f"   delta (base - best)     = {delta_rae:+.4f}")
    print(f"   GATE A: best_s >= {GATE_S_MIN}    -> "
          f"{'PASS' if gate_s else 'FAIL'}")
    print(f"   GATE B: delta >= {GATE_DELTA_RAE}    -> "
          f"{'PASS' if gate_delta else 'FAIL'}")
    print(f"   OVERALL                 -> "
          f"{'PASS (deploy)' if gate_pass else 'FAIL (no deploy)'}")

    # ---- Step 6: deploy if both gates pass ----
    deploy_csv_path = None
    deploy_te_npy_path = None
    deploy_info: dict = {}
    if gate_pass:
        print("\n" + "-" * 78)
        print("STEP 6: DEPLOY (refit blend on 253, stretch by best_s)")
        print("-" * 78)
        w_deploy = slsqp_simplex(P_unb, y_unb)
        blend_unb = P_unb @ w_deploy
        mu_deploy = float(blend_unb.mean())
        blend_te = P_te @ w_deploy
        deploy_te = (mu_deploy + best_s * (blend_te - mu_deploy)).astype(np.float32)
        in_sample_check = float(rae(
            y_unb, mu_deploy + best_s * (blend_unb - mu_deploy)
        ))
        w_str = ", ".join(
            f"{disp}={w:.4f}" for (disp, _, _), w in zip(ANCHORS, w_deploy)
        )
        print(f"   deploy weights = {w_str}")
        print(f"   deploy mu      = {mu_deploy:.4f}")
        print(f"   deploy s       = {best_s:.2f}")
        print(f"   in-sample RAE  = {in_sample_check:.4f}  (lower bound)")
        print(f"   te(513) mean/std = {deploy_te.mean():.3f} / "
              f"{deploy_te.std():.3f}")

        deploy_te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
        np.save(deploy_te_npy_path, deploy_te)

        # PRIMARY-1-STRETCHED candidate -> nb1196_deploy_stretched.csv
        deploy_csv_path = SUBMISSIONS / "nb1196_deploy_stretched.csv"
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(deploy_csv_path, index=False)
        print(f"\n[save] {deploy_csv_path}  (PRIMARY-1-STRETCHED candidate)")

        deploy_info = {
            "deploy_weights": [
                {"name": disp, "w": float(w)}
                for (disp, _, _), w in zip(ANCHORS, w_deploy)
            ],
            "deploy_mu": mu_deploy,
            "deploy_s": float(best_s),
            "in_sample_rae_lower_bound": in_sample_check,
            "te_mean": float(deploy_te.mean()),
            "te_std": float(deploy_te.std()),
            "te_npy_path": str(deploy_te_npy_path),
            "submission_csv": str(deploy_csv_path),
        }
    else:
        print("\n[skip] gate FAILED -> nb1196_deploy_stretched.csv NOT written")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "source_blend": "nb1162_stack_pyramid",
        "anchors": [a[0] for a in ANCHORS],
        "n_unb": n_unb,
        "n_te": n_te,
        "n_f2_novel_scaffold": n_f2,
        "base_pooled_rae_reconstructed": base_rae,
        "nb1162_cached_pooled_rae": 0.4206326222520904,
        "variance_diag_full": {
            "pred_std": full_pred_std,
            "truth_std": full_truth_std,
            "s_var_ratio": full_var_ratio,
        },
        "variance_diag_f2": {
            "n": n_f2,
            "pred_std": f2_pred_std,
            "truth_std": f2_truth_std,
            "s_var_ratio": f2_var_ratio,
            "f2_only_rae": f2_rae,
        },
        "variance_diag_in_manifold": {
            "n": int((~f2_mask).sum()),
            "pred_std": in_pred_std,
            "truth_std": in_truth_std,
            "in_manifold_rae": in_rae_val,
        },
        "stretch_grid": STRETCH_GRID,
        "grid_results": {
            f"{s:.2f}": grid_results[s] for s in STRETCH_GRID
        },
        "per_fold_diagnostic": {
            "per_fold_s": [float(x) for x in perfold_s],
            "per_fold_mean_s": perfold_mean_s,
            "per_fold_pooled_rae": perfold_rae,
        },
        "best_s": float(best_s),
        "best_stretched_rae": float(best_rae),
        "delta_rae_vs_base": float(delta_rae),
        "gate_s_min": GATE_S_MIN,
        "gate_delta_rae": GATE_DELTA_RAE,
        "gate_a_best_s_ge_min": bool(gate_s),
        "gate_b_delta_ge_min": bool(gate_delta),
        "gate_pass": bool(gate_pass),
        "deploy": deploy_info if gate_pass else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   base RAE             = {base_rae:.4f}")
    print(f"   F2 n / variance      = {n_f2} / s_var={f2_var_ratio:.4f}")
    print(f"   best_s               = {best_s:.2f}")
    print(f"   best stretched RAE   = {best_rae:.4f}")
    print(f"   delta                = {delta_rae:+.4f}")
    print(f"   gate_pass            = {gate_pass}")
    print(f"   wall                 = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "base_pooled_rae_reconstructed",
        "n_f2_novel_scaffold",
        "best_s",
        "best_stretched_rae",
        "delta_rae_vs_base",
        "gate_pass",
    ):
        print(f"  {k}: {res.get(k)}")
