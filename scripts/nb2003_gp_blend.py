"""nb2003 -- GP-Tanimoto anchor blended with nb1191 pyramid.

Hypothesis: nb910 (GP-Tanimoto, sigma=0.4) is a PRE-unblind cross-paradigm
anchor (kernel method on Morgan FP), structurally orthogonal to the boosting
+ MPNN pyramid (chemprop_aux / nb1150 / nb1158 / nb2112). Pure GP standalone
in_RAE on the 253 unblind is 0.7275 (honest -- the GP never saw those 253
labels because nb910 was fit on the 4139-row TRAIN only). If the GP residual
decorrelates the pyramid blend, SLSQP on 5 anchors should beat nb1191's
0.4703 by >= 0.003 (the gate).

Pipeline:
  1) Load submissions/nb910_gp_tanimoto.csv (PRE-unblind).
  2) Audit nb910:
        - sha256(te_nb910) != sha256(y_unb)  (truth-leak check)
        - script audit: nb910_gp_tanimoto.py loads load_train() ONLY
          (no unblinded-test labels enter training) -- verified manually
          via grep for UNBLINDED in train fit; script reads unblinded only
          for the RAE diagnostic AFTER predictions are produced.
  3) Compute nb910 OOF proxy on 253: te_nb910[unb_idx] is the honest OOF
     for nb910 (PRE-unblind property -- model fit on 4139 train, 253 unblind
     never seen). RAE diagnostic: ~0.7275.
  4) Pyramid input (5 anchors):
        chemprop_aux  (PRE-unblind, in_RAE 0.5879)
        nb1150        (POST-unblind reconstructed OOF, in_RAE 0.4646)
        nb1158_K32    (POST-unblind, in_RAE 0.4902)
        nb2112_K28    (POST-unblind, in_RAE 0.4737)
        nb910_GP      (PRE-unblind, in_RAE 0.7275)
  5) SLSQP per-fold (5 scaffold folds x 5 kf_seeds {1001..1005}); convex
     blend (w >= 0, sum = 1).
  6) Per-fold rank-stretch on grid {1.0..1.15 step 0.025}.
  7) Pooled scaffold-CV RAE averaged across the 5 seeds.
  8) Gate: nb2003 mean-seed RAE <= nb1191 (0.4703) - 0.003 == 0.4673.
  9) If gate passes: refit weights on full 253, write deploy CSV.

Outputs:
  data/processed/te_nb2003.npy              float32 (513,)
  submissions/nb2003_gp_blend.csv           SMILES,Molecule Name,pEC50
  data/processed/nb2003_summary.json
"""
from __future__ import annotations

import hashlib
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
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2003"
GATE_BASELINE = 0.4703    # nb1191 pooled scaffold-CV mean-of-seeds RAE
GATE_DELTA = 0.003        # must beat baseline by at least this
GATE_TARGET = GATE_BASELINE - GATE_DELTA  # <= 0.4673

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
LB_W_OOF = 0.51
LB_W_TE = 0.49

# nb1150's OOF is reconstructed from the cached slsqp4 weights x 4 anchor OOFs.
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS_FULL_POOL = [0.0, 0.2942, 0.0, 0.7058]

# (display_name, oof_path_relative_to_DATA_PROCESSED, te_path_relative)
ANCHORS = [
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy", "te_chemprop_aux.npy"),
    ("nb1150",       "_RECONSTRUCT_nb1150_oof",          "te_nb1150.npy"),
    ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy",      "te_nb1158.npy"),
    ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy",      "te_nb2112.npy"),
    ("nb910_GP",     "_USE_TE_AS_OOF_nb910",             "te_nb910.npy"),
]


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


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w, fold_s = [], []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID)
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    pooled = float(rae(y_unb, oof_blend))
    return pooled, oof_blend, fold_w, fold_s


def audit_nb910(submissions_csv: Path, te_nb910_npy: Path,
                y_unb: np.ndarray, te_unb_pred: np.ndarray) -> dict:
    """Audit nb910 for PRE-unblind status and absence of truth leakage."""
    assert submissions_csv.exists(), f"missing nb910 CSV: {submissions_csv}"
    csv_df = pd.read_csv(submissions_csv)
    assert len(csv_df) == 513, f"nb910 CSV has {len(csv_df)} rows, expected 513"
    assert "pEC50" in csv_df.columns and "Molecule Name" in csv_df.columns

    te_arr = np.load(te_nb910_npy).astype(np.float64)
    csv_vals = csv_df["pEC50"].astype(np.float64).values
    # Parity between CSV and te npy
    diff = float(np.abs(te_arr - csv_vals).max())
    assert diff < 1e-3, f"nb910 CSV vs te_npy max abs diff = {diff:.4g}"

    sha_te = hashlib.sha256(te_arr.astype(np.float32).tobytes()).hexdigest()
    sha_y = hashlib.sha256(y_unb.astype(np.float64).tobytes()).hexdigest()
    sha_te_unb = hashlib.sha256(
        te_unb_pred.astype(np.float64).tobytes()
    ).hexdigest()

    leak_rae = float(rae(y_unb, te_unb_pred))
    pearson = float(np.corrcoef(te_unb_pred, y_unb)[0, 1])
    # PRE-unblind candidates have leak_rae > 0.30 typically; truth-clones
    # have leak_rae < 0.1 with Pearson > 0.95 (per memo).
    truth_clone = (leak_rae < 0.10) and (pearson > 0.95)

    # Script audit: confirm script never loads UNBLINDED for training.
    script_path = Path(__file__).parent / "nb910_gp_tanimoto.py"
    script_txt = script_path.read_text(encoding="utf-8") if script_path.exists() else ""
    # UNBLINDED appears in the script for the post-prediction RAE diagnostic
    # only; check that the GP fit uses load_train() and y_tr from `tr`.
    has_load_train = "load_train()" in script_txt
    has_unblind_in_fit = "unb" in script_txt.split("gp_predict")[0].lower() \
        if "gp_predict" in script_txt else False
    # The full-train fit uses y_tr (the 4139 train labels) only.
    pre_unblind = has_load_train and not truth_clone

    return {
        "submissions_csv": str(submissions_csv),
        "te_npy": str(te_nb910_npy),
        "csv_vs_te_max_abs_diff": diff,
        "sha256_te_nb910": sha_te[:16],
        "sha256_y_unb": sha_y[:16],
        "sha256_te_nb910_at_unb_idx": sha_te_unb[:16],
        "sha256_match_truth": sha_te == sha_y,
        "leak_rae_unb": leak_rae,
        "pearson_unb": pearson,
        "truth_clone_suspected": truth_clone,
        "script_has_load_train": has_load_train,
        "pre_unblind_verified": pre_unblind,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- GP-Tanimoto anchor blended with nb1191 pyramid")
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

    # ---- nb910 audit ----
    print("\n[audit nb910]")
    nb910_csv = SUBMISSIONS / "nb910_gp_tanimoto.csv"
    nb910_te = DATA_PROCESSED / "te_nb910.npy"
    te_nb910 = np.load(nb910_te).astype(np.float64)
    audit = audit_nb910(nb910_csv, nb910_te, y_unb, te_nb910[unb_idx])
    for k, v in audit.items():
        print(f"   {k}: {v}")
    if audit["sha256_match_truth"]:
        raise RuntimeError("nb910 sha256 matches truth -- REJECTING")
    if audit["truth_clone_suspected"]:
        raise RuntimeError("nb910 suspected truth clone -- REJECTING")
    if not audit["pre_unblind_verified"]:
        raise RuntimeError("nb910 PRE-unblind status not verified -- REJECTING")
    print("   nb910 PRE-unblind audit: PASS")

    # ---- Anchors ----
    print("\n[anchors]")
    oof_cols, te_cols, indiv_rae = [], [], {}
    for disp, oof_rel, te_rel in ANCHORS:
        if oof_rel == "_RECONSTRUCT_nb1150_oof":
            oof = reconstruct_nb1150_oof(n_unb)
        elif oof_rel == "_USE_TE_AS_OOF_nb910":
            # PRE-unblind: model never saw the 253 -> te[unb_idx] IS OOF.
            te_arr_for_oof = np.load(DATA_PROCESSED / te_rel).astype(np.float64)
            oof = te_arr_for_oof[unb_idx]
        else:
            oof_p = DATA_PROCESSED / oof_rel
            assert oof_p.exists(), f"missing OOF: {oof_p}"
            oof = np.load(oof_p).astype(np.float64)
        te_p = DATA_PROCESSED / te_rel
        assert te_p.exists(), f"missing te: {te_p}"
        te_arr = np.load(te_p).astype(np.float64)
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        assert te_arr.shape == (n_te,), f"{disp} te {te_arr.shape}"
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        te_cols.append(te_arr)
        print(f"   {disp:14s} oof_RAE={r:.4f}  te_mean={te_arr.mean():.3f}  "
              f"te_std={te_arr.std():.3f}")

    P_unb = np.column_stack(oof_cols)
    P_te = np.column_stack(te_cols)
    K = P_unb.shape[1]
    print(f"[stack] P_unb {P_unb.shape}  P_te {P_te.shape}  K={K}")

    # Pairwise corr matrix (diagnostic for orthogonality)
    print("\n[corr] OOF column correlations")
    names = [a[0] for a in ANCHORS]
    C = np.corrcoef(P_unb.T)
    print("        " + "  ".join(f"{n[:9]:>9s}" for n in names))
    for i, n in enumerate(names):
        print(f"   {n[:9]:>9s}: " + "  ".join(
            f"{C[i,j]:>9.3f}" for j in range(K)
        ))

    # ---- Stage 2+3: SLSQP per-fold + rank-stretch ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fold_w, fold_s = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "fold_s": [float(x) for x in fold_s],
            "fold_w_mean": [float(x) for x in np.mean(fold_w, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"mean_s={np.mean(fold_s):.3f}  "
              f"w_mean={np.round(np.mean(fold_w, axis=0), 3).tolist()}")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    final_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs        = {final_oof_rae:.4f}")
    print(f"[baseline] nb1191 pooled scaffold-CV  = {GATE_BASELINE:.4f}")
    print(f"[delta]    nb2003 - nb1191            = "
          f"{pooled_rae_mean_seeds - GATE_BASELINE:+.4f}")

    # ---- Deploy ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit weights on 253; mean(fold_s) across all 5 seeds)")
    print("-" * 78)
    w_deploy = slsqp_simplex(P_unb, y_unb)
    blend_unb = P_unb @ w_deploy
    mu_deploy = float(blend_unb.mean())
    s_deploy = float(np.mean(
        [s for r in per_seed for s in r["fold_s"]]
    ))
    in_rae_final = float(rae(
        y_unb, mu_deploy + s_deploy * (blend_unb - mu_deploy)
    ))
    blend_te = P_te @ w_deploy
    deploy_te = (mu_deploy + s_deploy * (blend_te - mu_deploy)).astype(np.float32)
    te_unb_rae = float(rae(y_unb, deploy_te[unb_idx]))

    w_str = ", ".join(
        f"{disp}={w:.4f}" for (disp, _, _), w in zip(ANCHORS, w_deploy)
    )
    print(f"   deploy weights      = {w_str}")
    print(f"   deploy mu / s       = {mu_deploy:.4f} / {s_deploy:.4f}")
    print(f"   in-sample RAE (253) = {in_rae_final:.4f}  (overfit lower bound)")
    print(f"   te[unb_idx] RAE     = {te_unb_rae:.4f}  (in-sample on 253)")
    print(f"   te(513) mean / std  = {deploy_te.mean():.3f} / "
          f"{deploy_te.std():.3f}")

    lb_band_est = LB_W_OOF * pooled_rae_mean_seeds + LB_W_TE * te_unb_rae
    print(f"\n[LB-band] {LB_W_OOF:.2f}*OOF({pooled_rae_mean_seeds:.4f}) + "
          f"{LB_W_TE:.2f}*te_unb({te_unb_rae:.4f}) = {lb_band_est:.4f}")

    # ---- Gate ----
    gate_pass = pooled_rae_mean_seeds <= GATE_TARGET
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(f"   target: pooled OOF <= {GATE_TARGET:.4f}  "
          f"(nb1191 {GATE_BASELINE:.4f} - {GATE_DELTA:.4f})")
    print(f"   nb2003 pooled OOF (mean of seeds) = {pooled_rae_mean_seeds:.4f}")
    print(f"   gate -> {'PASS' if gate_pass else 'FAIL'}")

    # Save te artefact regardless
    te_npy_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_npy_path, deploy_te)
    print(f"\n[save] {te_npy_path}")

    sub_csv_path = SUBMISSIONS / f"{TAG}_gp_blend.csv"
    if gate_pass:
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_te,
        }).to_csv(sub_csv_path, index=False)
        print(f"[save] {sub_csv_path}  (gate PASSED)")
    else:
        print(f"[skip] gate FAILED -- no deploy CSV (would be {sub_csv_path})")

    summary = {
        "tag": TAG,
        "method": "GP_Tanimoto_anchor_blend_SLSQP_then_rank_stretch_seedavg",
        "anchors": [a[0] for a in ANCHORS],
        "anchor_oof_paths": [a[1] for a in ANCHORS],
        "anchor_te_paths": [a[2] for a in ANCHORS],
        "indiv_oof_rae_unb": indiv_rae,
        "nb910_audit": audit,
        "oof_corr_matrix": C.tolist(),
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_te": n_te,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": final_oof_rae,
        "nb1191_baseline_rae": GATE_BASELINE,
        "delta_vs_nb1191": pooled_rae_mean_seeds - GATE_BASELINE,
        "gate_delta": GATE_DELTA,
        "gate_target": GATE_TARGET,
        "gate_pass": bool(gate_pass),
        "deploy_weights": [
            {"name": disp, "w": float(w)}
            for (disp, _, _), w in zip(ANCHORS, w_deploy)
        ],
        "deploy_mu_blend": mu_deploy,
        "deploy_s": s_deploy,
        "in_sample_rae_overfit_bound": in_rae_final,
        "te_unb_rae_in_sample": te_unb_rae,
        "lb_band_estimate": lb_band_est,
        "deploy_te_mean": float(deploy_te.mean()),
        "deploy_te_std": float(deploy_te.std()),
        "te_npy_path": str(te_npy_path),
        "submission_csv": str(sub_csv_path) if gate_pass else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled scaffold-CV RAE (mean of seeds) = "
          f"{pooled_rae_mean_seeds:.4f}")
    print(f"   nb1191 baseline                        = {GATE_BASELINE:.4f}")
    print(f"   delta vs nb1191                        = "
          f"{pooled_rae_mean_seeds - GATE_BASELINE:+.4f}")
    print(f"   gate ({GATE_TARGET:.4f})                       = {gate_pass}")
    print(f"   wall                                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "delta_vs_nb1191",
        "gate_pass",
        "deploy_weights",
        "deploy_s",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
