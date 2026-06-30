"""nb2770 -- Polynomial Chaos Expansion (PCE) on K=20 substrate.

NEW PARADIGM (orthogonal polynomial basis in function space):
    Polynomial Chaos Expansion (PCE) approximates a stochastic response
    f(x) by an orthogonal polynomial decomposition in a probabilistic
    basis tied to the input distribution.  For Gaussian-distributed
    inputs the canonical basis is Hermite (the Wiener-Hermite chaos);
    `chaospy.generate_expansion` with `chaospy.Normal(0, 1)` constructs
    this exact basis.

    Orthogonality of the polynomial basis under the input measure means
    least-squares Ridge/LinearRegression on the projected
    Phi(x) = [psi_alpha(x)] features yields a sparse decomposition in
    function space: the coefficients are Galerkin projections onto each
    Hermite polynomial, capturing both univariate non-linear response
    AND low-order cross-feature interactions (degree-2 includes all
    pairwise psi_i(x_i) * psi_j(x_j) terms).

    Why this is a NEW paradigm versus cycles 134-167 exhausted axes:
    * Cycle-134 LGBM tree-split substrate exhausted.
    * Cycle-136 aux models all fail to beat nb2103 on chemprop_aux
      residual.
    * Cycles 167-169 post-hoc-blend ceiling 0.4682 deep-30 confirmed.
    * PCE on Hermite basis is a CLOSED-FORM kernel approximation to
      smooth response surfaces that LGBM axis-aligned splits cannot
      represent natively.  Linear regression on a degree-2 Hermite
      basis over a K=20 standardized substrate yields 231 basis
      features (1 + 20 + C(20,2) + 20 = 231 for the symmetric
      isotropic Hermite expansion via chaospy), which is < n=253 so
      regression is well-conditioned.

    Hypothesis: if the chemprop_aux residual surface w.r.t. the K=20
    RFE-survivor columns has a smooth low-order polynomial structure
    (rather than tree-step structure), Hermite-basis Ridge will yield
    a lower scaffold-CV RAE than nb2240 K=20 LGBM (0.4630) and may
    approach the deep-30 ceiling band (0.4682-0.4720).

PROTOCOL:
    1. Load X_117_unb + X_117_te substrate -> slice K=20 columns from
       nb2240 summary (k20_surviving_idx_in_117).
    2. residual = y_unb - chemprop_aux_te[unb_idx] (only PRE-clean anchor).
    3. Per scaffold fold: StandardScaler.fit on train_K20 -> transform
       val_K20 (and the full te slice for deploy).
    4. chaospy.generate_expansion(order=2,
                                  dist=chaospy.J(*[chaospy.Normal(0,1)
                                                    for _ in range(20)]))
       -> 231-element orthogonal Hermite polynomial basis.
    5. Evaluate basis on each scaled X-row to get Phi(N, 231).
    6. LinearRegression on Phi(train) vs residual(train), predict
       Phi(val).
    7. 5-fold scaffold CV (`scaffold_kfold_indices`), 5 kf_seeds
       {1001..1005}.
    8. Deploy: refit StandardScaler + PCE + LinearRegression on full 253
       per seed -> predict 513 te residual; mean-bag aggregate.

GATE (mean-bag corrected RAE):
    mean_rae < 0.4570 -> "PROMOTE"
    mean_rae < 0.4598 -> "MARGINAL_BEAT"
    else              -> "FAIL"

OUTPUTS:
    scripts/nb2770_polynomial_chaos.py
    data/processed/nb2770_summary.json
    data/processed/nb2770_pred_oof.npy   (253,) float32 mean-bag CORRECTED
    data/processed/te_nb2770.npy         (513,) float32 deploy refit
    submissions/nb2770_polynomial_chaos.csv
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

# Guarded chaospy import -- per task spec, if install/import fails, dump
# INSTALL_FAILED summary and exit clean (no crash).
try:
    import chaospy
    _CHAOSPY_OK = True
    _CHAOSPY_ERR = None
except Exception as _e:  # pragma: no cover
    _CHAOSPY_OK = False
    _CHAOSPY_ERR = repr(_e)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2770"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
NB2240_SUMMARY = DATA_PROCESSED / "nb2240_summary.json"

# PCE config
PCE_ORDER = 2
PCE_DIMS = 20

# CV config
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gate thresholds (mean-bag RAE)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

CHEMPROP_AUX_REF = 0.6216
NB2240_K20_REF = 0.4630  # StandardScaler+LGBM K=20 baseline


def _make_pce_basis(dims: int = PCE_DIMS, order: int = PCE_ORDER):
    """Build PCE (Hermite) expansion via chaospy.

    chaospy 4.x requires unique distribution INSTANCES inside `J(...)`
    (passing the same Normal(0,1) object N times raises
    StochasticallyDependentError), so construct fresh Normals.
    """
    if not _CHAOSPY_OK:
        raise RuntimeError(f"chaospy import failed: {_CHAOSPY_ERR}")
    dist = chaospy.J(*[chaospy.Normal(0, 1) for _ in range(dims)])
    expansion = chaospy.generate_expansion(order=order, dist=dist)
    return expansion


def _eval_pce(expansion, X: np.ndarray) -> np.ndarray:
    """Evaluate PCE basis on each row of X -> Phi (n_samples, n_basis)."""
    # chaospy basis-callable signature is (x_dim0, x_dim1, ..., x_dimD-1)
    # each a 1D array of length n_samples.  Output shape -> (n_basis, n_samples).
    Phi = np.asarray(expansion(*X.T), dtype=np.float64)
    # Shape (n_basis, n_samples) -> transpose to (n_samples, n_basis)
    Phi = Phi.T
    Phi = np.ascontiguousarray(Phi, dtype=np.float64)
    return Phi


def _scaffold_cv_one_seed(
    X_unb_K: np.ndarray,
    residual: np.ndarray,
    unb_scaffolds: list,
    kf_seed: int,
    expansion,
) -> tuple[np.ndarray, list[dict]]:
    """One scaffold-CV pass: per-fold StandardScaler + PCE + LinearRegression."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n = len(residual)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_diags = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(X_unb_K[tr_loc]).astype(np.float64)
        Xva_s = sc.transform(X_unb_K[va_loc]).astype(np.float64)
        Phi_tr = _eval_pce(expansion, Xtr_s)
        Phi_va = _eval_pce(expansion, Xva_s)
        mdl = LinearRegression()
        mdl.fit(Phi_tr, residual[tr_loc])
        oof[va_loc] = mdl.predict(Phi_va)
        fold_diags.append({
            "fold": fold_i,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "phi_dim": int(Phi_tr.shape[1]),
            "lin_intercept": float(mdl.intercept_),
            "lin_coef_abs_mean": float(np.mean(np.abs(mdl.coef_))),
            "lin_coef_abs_max": float(np.max(np.abs(mdl.coef_))),
        })
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof, fold_diags


def _deploy_te_one_seed(
    X_unb_K: np.ndarray,
    residual: np.ndarray,
    X_te_K: np.ndarray,
    seed: int,
    expansion,
) -> tuple[np.ndarray, dict]:
    """Fit Scaler + PCE-Linear on full 253; predict 513 te residual."""
    sc = StandardScaler()
    Xunb_s = sc.fit_transform(X_unb_K).astype(np.float64)
    Xte_s = sc.transform(X_te_K).astype(np.float64)
    Phi_unb = _eval_pce(expansion, Xunb_s)
    Phi_te = _eval_pce(expansion, Xte_s)
    mdl = LinearRegression()
    mdl.fit(Phi_unb, residual)
    pred = mdl.predict(Phi_te).astype(np.float32)
    diag = {
        "seed": int(seed),
        "phi_dim": int(Phi_unb.shape[1]),
        "lin_intercept": float(mdl.intercept_),
        "lin_coef_abs_mean": float(np.mean(np.abs(mdl.coef_))),
        "lin_coef_abs_max": float(np.max(np.abs(mdl.coef_))),
    }
    return pred, diag


def _install_failed_dump(reason: str) -> dict:
    """Per task spec: if chaospy install/import fails, dump INSTALL_FAILED
    summary so the cron caller can detect-and-skip without an exception.
    """
    summary = {
        "tag": TAG,
        "method": "polynomial_chaos_expansion_hermite_order2_dim20",
        "verdict": "INSTALL_FAILED",
        "install_error": reason,
        "anchor": ANCHOR,
        "anchor_pre_unblind": True,
        "wall_sec": 0.0,
        "pre_unblind_clean": True,
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INSTALL_FAILED] {reason}")
    print(f"[save] {out_path}")
    return summary


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Polynomial Chaos Expansion (Hermite, order={PCE_ORDER}, "
          f"dim={PCE_DIMS}) + LinearRegression")
    print(f"        anchor = {ANCHOR}  scaffold-CV {N_FOLDS}-fold  "
          f"kf_seeds={KF_SEEDS}")
    print(f"        substrate = nb2240 K=20 RFE survivors (idx_in_117)")
    print(f"        ref nb2240 K=20 StandardScaler+LGBM = {NB2240_K20_REF:.4f}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    if not _CHAOSPY_OK:
        return _install_failed_dump(_CHAOSPY_ERR or "chaospy import failed")

    # ---- Load truth + anchor + smiles ----
    te = load_test()
    n_test = len(te)
    test_smiles_raw = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values
        if "name" in te.columns
        else te["Molecule Name"].values
    )

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"anchor te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(f"anchor te shape {te_anchor_513.shape}")
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- Load substrate K=20 ----
    if not NB2240_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2240_SUMMARY}")
    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"missing pyramid feature matrices: {X117_UNB_PATH}, {X117_TE_PATH}"
        )
    with open(NB2240_SUMMARY) as f:
        sum_2240 = json.load(f)
    k20_idx = np.array(sum_2240["k20_surviving_idx_in_117"], dtype=int)
    k20_names = list(sum_2240["k20_surviving_names"])
    if len(k20_idx) != PCE_DIMS:
        raise ValueError(
            f"expected {PCE_DIMS} K=20 surviving cols, got {len(k20_idx)}"
        )
    X117_unb = np.load(X117_UNB_PATH).astype(np.float32)
    X117_te = np.load(X117_TE_PATH).astype(np.float32)
    if X117_unb.shape[0] != n_unb:
        raise ValueError(f"X117_unb rows {X117_unb.shape} != n_unb {n_unb}")
    if X117_te.shape[0] != n_test:
        raise ValueError(f"X117_te rows {X117_te.shape} != n_test {n_test}")
    X_unb_K = X117_unb[:, k20_idx].astype(np.float64)
    X_te_K = X117_te[:, k20_idx].astype(np.float64)
    print(f"[feat] X_unb_K = {X_unb_K.shape}  X_te_K = {X_te_K.shape}")
    print(f"[feat] K=20 families: {sum_2240.get('k20_family_counts', {})}")

    # ---- Build PCE basis (shared across folds + seeds; basis depends
    #      only on input measure + order, not on fold data).
    ts = time.time()
    expansion = _make_pce_basis(dims=PCE_DIMS, order=PCE_ORDER)
    n_basis = len(expansion)
    print(f"[pce] basis: chaospy.Hermite order={PCE_ORDER} dim={PCE_DIMS} "
          f"-> {n_basis} basis polynomials  (gen wall {time.time()-ts:.2f}s)")
    expected_basis = 1 + PCE_DIMS + (PCE_DIMS * (PCE_DIMS - 1)) // 2 + PCE_DIMS
    # Symmetric isotropic Hermite total-degree-2 basis size:
    #   1 (constant) + 20 (linear) + 20 (squares) + 190 (pairwise) = 231
    print(f"[pce] expected basis size (total-degree-2 in 20 dims) = "
          f"{expected_basis}")

    # ---- Scaffolds ----
    unb_smiles = [test_smiles_raw[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique unb scaffolds = {n_unique_scaf}")

    # ---- 5-seed scaffold CV ----
    print("\n" + "-" * 78)
    print(f"5-SEED SCAFFOLD-CV  seeds={KF_SEEDS}  folds={N_FOLDS}  "
          f"input_dim={PCE_DIMS}  pce_basis_dim={n_basis}")
    print("-" * 78)
    per_seed_oof_resid = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_seed_te_resid = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_fold_diags: list[list[dict]] = []
    per_seed_deploy_diag: list[dict] = []

    for i, seed in enumerate(KF_SEEDS):
        ts = time.time()
        resid_oof, fold_diags = _scaffold_cv_one_seed(
            X_unb_K, residual, unb_scaffolds, seed, expansion,
        )
        per_seed_oof_resid[i] = resid_oof
        per_seed_fold_diags.append(fold_diags)
        te_resid, deploy_diag = _deploy_te_one_seed(
            X_unb_K, residual, X_te_K, seed, expansion,
        )
        per_seed_te_resid[i] = te_resid
        per_seed_deploy_diag.append(deploy_diag)
        pred_corr = anchor + resid_oof
        rae_s = float(rae(y_unb, pred_corr))
        per_seed_rae.append(rae_s)
        print(f"   seed={seed}  rae_corr={rae_s:.4f}  "
              f"d_vs_anchor={rae_s - rae_anchor:+.4f}  "
              f"resid_std={resid_oof.std():.4f}  "
              f"wall={time.time() - ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    mean_bag_resid = per_seed_oof_resid.mean(axis=0)
    median_bag_resid = np.median(per_seed_oof_resid, axis=0)
    rae_mean_bag = float(rae(y_unb, anchor + mean_bag_resid))
    rae_median_bag = float(rae(y_unb, anchor + median_bag_resid))

    print("\n[cv] per_seed_mean RAE = "
          f"{per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag   RAE = {rae_mean_bag:.4f}")
    print(f"[cv] median_bag RAE = {rae_median_bag:.4f}")
    print(f"[cv] anchor     RAE = {rae_anchor:.4f}  "
          f"(d_mean_bag = {rae_mean_bag - rae_anchor:+.4f})")
    print(f"[cv] reference  nb2240 K=20 StandardScaler+LGBM = "
          f"{NB2240_K20_REF:.4f}  "
          f"(d = {rae_mean_bag - NB2240_K20_REF:+.4f})")

    # ---- Deploy te (mean-bag corrected) ----
    mean_bag_te_resid = per_seed_te_resid.mean(axis=0)
    te_deploy = (te_anchor_513 + mean_bag_te_resid).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, te_deploy[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = "
          f"{te_deploy.mean():.3f}/{te_deploy.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}  "
          f"(deploy refit, in-sample optimism expected)")

    # ---- Save artefacts ----
    pred_oof_corrected = (anchor + mean_bag_resid).astype(np.float32)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_corrected)
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_polynomial_chaos.csv"
    pd.DataFrame({
        "SMILES": test_smiles_raw,
        "Molecule Name": te_names,
        "pEC50": te_deploy.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if rae_mean_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_mean_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_bag_rae        = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = "
          f"{rae_mean_bag < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{rae_mean_bag < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": ("polynomial_chaos_expansion_hermite_order2_dim20_"
                   "on_K20_substrate_on_chemprop_aux_residual"),
        "rationale": (
            "Hermite polynomial chaos basis is orthogonal under the N(0,1) "
            "input measure of StandardScaler-projected K=20 substrate; "
            "LinearRegression on the degree-2 expansion gives a sparse "
            "closed-form decomposition in function space, capturing smooth "
            "univariate non-linearity + low-order pairwise interactions "
            "that LGBM axis-aligned tree splits represent only via greedy "
            "piecewise approximation.  New paradigm vs cycle-134 LGBM "
            "exhaustion."
        ),
        "anchor": ANCHOR,
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": True,
        "rae_anchor_chemprop_aux": rae_anchor,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "feature_source": "nb2240_K20_RFE_surviving_idx_in_117",
        "feat_dim_input": int(PCE_DIMS),
        "k20_surviving_idx_in_117": k20_idx.tolist(),
        "k20_surviving_names": k20_names,
        "k20_family_counts": sum_2240.get("k20_family_counts", {}),
        "pce_order": int(PCE_ORDER),
        "pce_dims": int(PCE_DIMS),
        "pce_basis_dim_actual": int(n_basis),
        "pce_basis_dim_expected_total_degree_2": int(expected_basis),
        "pce_distribution": "chaospy.Normal(0,1) x 20  (Wiener-Hermite chaos)",
        "scaler": "StandardScaler (per-fold fit on train only)",
        "regressor": "sklearn.linear_model.LinearRegression",
        "fit_policy": "per-fold-on-train (Scaler + PCE-eval + LinReg on train)",
        "n_unb": n_unb,
        "n_test": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "model_class": ("chaospy.generate_expansion(order=2, "
                        "dist=J(Normal(0,1)^20)) -> "
                        "sklearn.linear_model.LinearRegression"),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "median_bag_rae": rae_median_bag,
        "mean_rae": rae_mean_bag,  # alias for gate consumers
        "delta_vs_anchor": rae_mean_bag - rae_anchor,
        "delta_vs_nb2240_K20_lgbm": rae_mean_bag - NB2240_K20_REF,
        "nb2240_K20_lgbm_ref": NB2240_K20_REF,
        "per_seed_fold_diagnostics": per_seed_fold_diags,
        "per_seed_deploy_diagnostics": per_seed_deploy_diag,
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(te_deploy.mean()),
        "te_deploy_std": float(te_deploy.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "verdict": verdict,
        "pre_unblind_clean": True,
        "chaospy_version": getattr(chaospy, "__version__", "unknown"),
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
    for k in (
        "pce_order",
        "pce_dims",
        "pce_basis_dim_actual",
        "per_seed_rae",
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "mean_bag_rae",
        "median_bag_rae",
        "delta_vs_anchor",
        "delta_vs_nb2240_K20_lgbm",
        "te_unb_in_sample_rae",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
