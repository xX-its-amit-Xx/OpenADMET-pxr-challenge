"""nb1193 -- Chemprop K=32 multi-seed deploy (cross-paradigm to nb1158 LGBM K=32).

GOAL:
    Build a true GNN multi-seed deploy candidate that is paradigm-orthogonal to
    nb1158 (LGBM K=32). Gate requires:
        OOF RAE <= 0.55 AND Pearson residual corr to nb1158 <= 0.85
    If passes -> emit deploy CSV and flag as PRIMARY-2-MPNN.

DESIGN CHOICE (Chemprop vs LGBM fallback):
    A real Chemprop 5-fold scaffold-CV at K=16 seeds is approx
       16 seeds x 5 folds x 5-10 min/fold = 7-13 hours on CPU.
    Per feedback_resource_limits.md this is too heavy and contends with the
    LB-logger / activity / structure crons. The task explicitly authorizes the
    LGBM-32-seed fallback "with perturbed hyperparams". We take that path here
    and CLEARLY document it as a fallback, not a chemprop GNN result. The
    paradigm-orthogonality angle is still preserved by:
      (a) anchoring on `chemprop_aux` (the true GNN PRIMARY-1 output) as the
          distillation target / feature, AND
      (b) using a DIFFERENT objective (Huber alpha=0.9) and feature stack
          (Morgan 2048 + RDKit 217 + chemprop_aux distillation feature, 2266 d)
          from nb1158 (which uses a 117-d 5-way SHAP-pruned matrix + MSE on
          residuals only). Two LGBM models trained on disjoint feature spaces
          with different objectives ARE different enough to be a useful
          orthogonality check; whether it passes the rho<=0.85 gate is empirical.

    If user wants a true Chemprop run, schedule it via the Kaggle P100 path
    (feedback_kaggle_p100_cuda.md) -- this script is the in-band fallback.

INPUTS:
    data/processed/oof_chemprop_aux.npy             (4139,) anchor OOF
    data/processed/te_chemprop_aux.npy              (513,)  anchor test
    data/processed/_audit_unblind_idx.npy           (253,)  unblind indices into 513
    data/processed/_audit_unblind_y.npy             (253,)  unblind truth
    data/processed/nb1158_mean_bag_oof_K32.npy      (253,)  nb1158 OOF on unblind

PROTOCOL:
    Features: combined Morgan(2048) + RDKit(217) + chemprop_aux as 2266-d.
    For each of K=32 LGBM seeds (perturbed n_estimators, num_leaves, lr):
        Scaffold 5-fold CV on 4139 train -> 4139-length OOF.
        Refit on full 4139, predict 513.
    Mean-bag across 32 seeds. Compute:
        OOF RAE on full 4139 (scaffold-CV, LB-faithful proxy).
        OOF RAE on 253 unblind slice.
        Pearson residual corr with nb1158 OOF on 253 (orthogonality).

GATES (must pass both for PRIMARY-2-MPNN flag):
    G1: OOF RAE on 253 unblind <= 0.55
    G2: Pearson(residual_bag - chemprop_aux, residual_nb1158 - chemprop_aux) on 253 <= 0.85

OUTPUTS:
    data/processed/nb1193_oof_4139.npy
    data/processed/nb1193_mean_bag_oof_253.npy
    data/processed/te_nb1193.npy
    submissions/nb1193_chemprop_k32_fallback.csv  (if any candidate at all)
    data/processed/nb1193_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import psutil

from pxr.chem import bemis_murcko, standardize
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1193"
K = 32
N_FOLDS = 5
SEEDS = list(range(2000, 2000 + K))  # disjoint from nb1158 (uses 0/1/7/42/137/1001-1010)

# Gates
GATE_OOF_RAE_253 = 0.55
GATE_RHO_NB1158 = 0.85

# CPU safety
MAX_CPU_PCT = 80.0
CPU_CHECK_INTERVAL = 1.0

SUB_DIR = ROOT / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)


def check_cpu_safe(label: str = "") -> None:
    """Raise if system CPU is over MAX_CPU_PCT."""
    cur = psutil.cpu_percent(interval=CPU_CHECK_INTERVAL)
    avail_gb = psutil.virtual_memory().available / 1e9
    print(f"   [cpu-check {label}] cpu%={cur:.1f}  mem_avail={avail_gb:.1f} GB")
    if cur > MAX_CPU_PCT:
        raise RuntimeError(
            f"CPU at {cur:.1f}% > {MAX_CPU_PCT}% threshold -- aborting "
            f"to avoid contention with crons (feedback_resource_limits)"
        )


def make_lgbm(seed: int, idx: int) -> lgb.LGBMRegressor:
    """Build a Huber LGBM with perturbed hyperparams per index.

    Perturbations introduce structural diversity across the 32 seeds beyond
    just the random state -- gives us a real seed-bag, not 32 near-duplicates.
    """
    rng = np.random.default_rng(seed)
    n_est = int(rng.choice([400, 500, 600, 700]))
    nl = int(rng.choice([32, 48, 64, 96]))
    lr = float(rng.choice([0.03, 0.05, 0.07]))
    ff = float(rng.choice([0.7, 0.8, 0.9]))
    bf = float(rng.choice([0.7, 0.8, 0.9]))
    min_data = int(rng.choice([10, 20, 30]))
    return lgb.LGBMRegressor(
        objective="huber",
        alpha=0.9,
        n_estimators=n_est,
        num_leaves=nl,
        learning_rate=lr,
        feature_fraction=ff,
        bagging_fraction=bf,
        bagging_freq=1,
        min_data_in_leaf=min_data,
        random_state=seed,
        n_jobs=2,  # cap per-model jobs; we run seeds serially to leave headroom
        verbose=-1,
    )


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- K={K} chemprop-anchored distillation seed-bag (LGBM fallback)")
    print(f"   anchor: chemprop_aux  |  paradigm-orthogonal to nb1158 LGBM K=32")
    print("=" * 78)

    # ------------ 1) CPU safety gate ------------
    print("\n[gate-0] checking CPU and memory before spawning work ...")
    check_cpu_safe("init")

    # ------------ 2) Load anchors ------------
    tr = load_train()
    te = load_test()
    n_tr, n_te = len(tr), len(te)
    print(f"[load] train rows = {n_tr}  test rows = {n_te}")

    oof_ca = np.load(DATA_PROCESSED / "oof_chemprop_aux.npy").astype(np.float64)
    te_ca = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    nb1158_oof_253 = np.load(
        DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy"
    ).astype(np.float64)
    n_unb = len(y_unb)
    assert oof_ca.shape == (n_tr,)
    assert te_ca.shape == (n_te,)
    assert nb1158_oof_253.shape == (n_unb,)
    print(f"[load] anchors:  oof_ca mean={oof_ca.mean():.3f}  "
          f"te_ca mean={te_ca.mean():.3f}")
    print(f"[load] nb1158 OOF(253): mean={nb1158_oof_253.mean():.3f}  "
          f"std={nb1158_oof_253.std():.3f}  RAE={rae(y_unb, nb1158_oof_253):.4f}")

    # Reference scores
    anchor_unb = te_ca[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[ref] chemprop_aux in_RAE on 253 = {rae_anchor:.4f}")

    # ------------ 3) Features (Morgan + RDKit + chemprop_aux distillation) ------------
    print("\n[feat] computing combined Morgan+RDKit features ...")
    smi_tr = tr["smiles"].apply(standardize).tolist()
    smi_te = te["smiles"].apply(standardize).tolist()
    X_tr_base = impute(combined(smi_tr))
    X_te_base = impute(combined(smi_te))
    # Distillation feature: anchor prediction
    X_tr = np.column_stack([X_tr_base, oof_ca.astype(np.float32)])
    X_te = np.column_stack([X_te_base, te_ca.astype(np.float32)])
    print(f"[feat] X_tr={X_tr.shape}  X_te={X_te.shape}  (+chemprop_aux as feature)")

    y_tr = tr["pec50"].values.astype(np.float64)

    # ------------ 4) Scaffold folds ------------
    scaffolds = [bemis_murcko(s) or "" for s in smi_tr]
    folds = scaffold_kfold_indices(scaffolds, n_splits=N_FOLDS,
                                   shuffle=True, seed=42)
    fold_sizes = [len(va) for _, va in folds]
    print(f"[cv] scaffold {N_FOLDS}-fold sizes: {fold_sizes}")

    # ------------ 5) Per-seed scaffold-CV OOF + 513 refit ------------
    print("\n" + "-" * 78)
    print(f"TRAINING K={K} LGBM Huber seeds (perturbed hyperparams)")
    print("-" * 78)
    oof_stack = np.zeros((K, n_tr), dtype=np.float64)
    te_stack = np.zeros((K, n_te), dtype=np.float64)
    per_seed_oof_rae_4139 = []
    per_seed_oof_rae_253 = []
    for i, seed in enumerate(SEEDS):
        t_seed = time.time()
        # quick CPU gate every 4 seeds to keep system safe
        if i > 0 and i % 4 == 0:
            check_cpu_safe(f"seed#{i}")
        oof = np.zeros(n_tr, dtype=np.float64)
        for tr_idx, va_idx in folds:
            model = make_lgbm(seed, i)
            model.fit(X_tr[tr_idx], y_tr[tr_idx])
            oof[va_idx] = model.predict(X_tr[va_idx])
        oof_stack[i] = oof
        # Refit on full 4139, predict 513
        full = make_lgbm(seed + 10000, i)  # different seed for refit ensemble
        full.fit(X_tr, y_tr)
        te_stack[i] = full.predict(X_te)
        r_4139 = float(rae(y_tr, oof))
        r_253 = float(rae(y_unb, te_stack[i][unb_idx]))
        per_seed_oof_rae_4139.append(r_4139)
        per_seed_oof_rae_253.append(r_253)
        if i < 4 or i % 8 == 0 or i == K - 1:
            print(f"   seed {seed:>4d} [{i+1:>2d}/{K}]: "
                  f"OOF_4139={r_4139:.4f}  te253_insample={r_253:.4f}  "
                  f"[{time.time()-t_seed:.1f}s]")

    # ------------ 6) Mean-bag ------------
    oof_bag_4139 = oof_stack.mean(axis=0)
    te_bag = te_stack.mean(axis=0)
    bag_oof_rae_4139 = float(rae(y_tr, oof_bag_4139))
    # Honest cross-fit RAE on 253: use scaffold-CV OOF, sliced to unblind
    # (For 253 in unblind set, their scaffold-CV OOF on 4139 IS LB-faithful.)
    # But unb_idx is into 513, not 4139. We don't have a direct mapping --
    # use the te_bag[unb_idx] which is "model trained on ALL 4139, predicting
    # 513" -- this is the deploy path; cross-fit honest number is bag_oof_rae_4139.
    bag_te_rae_253 = float(rae(y_unb, te_bag[unb_idx]))
    print(f"\n[bag] mean across {K} seeds:")
    print(f"   scaffold-CV OOF RAE (4139)   = {bag_oof_rae_4139:.4f}")
    print(f"   te(513)[unb_idx] RAE (253)   = {bag_te_rae_253:.4f}  (in-sample optimistic)")
    print(f"   per-seed OOF_4139 mean       = {np.mean(per_seed_oof_rae_4139):.4f}  "
          f"std={np.std(per_seed_oof_rae_4139):.4f}")

    # ------------ 7) Orthogonality: residual corr with nb1158 ------------
    pred_bag_253 = te_bag[unb_idx]  # deploy path predictions on 253
    resid_bag_vs_anchor = pred_bag_253 - anchor_unb
    resid_nb1158_vs_anchor = nb1158_oof_253 - anchor_unb
    # Pearson corr
    if resid_bag_vs_anchor.std() > 0 and resid_nb1158_vs_anchor.std() > 0:
        rho_resid = float(np.corrcoef(resid_bag_vs_anchor,
                                      resid_nb1158_vs_anchor)[0, 1])
    else:
        rho_resid = float("nan")
    # Raw pred corr too (sanity)
    rho_pred = float(np.corrcoef(pred_bag_253, nb1158_oof_253)[0, 1])
    print(f"\n[ortho] residual-vs-anchor Pearson(nb1193_bag, nb1158) = {rho_resid:.4f}")
    print(f"[ortho] raw pred Pearson(nb1193_bag, nb1158) = {rho_pred:.4f}")

    # ------------ 8) Gates ------------
    # G1: OOF RAE on 253. Use cross-fit-faithful proxy = the 253 unblind compounds'
    #     scaffold-CV OOF predictions on 4139 are not directly available because
    #     unb_idx points into 513 (test), and 253 unblind compounds are POST-train
    #     -- they are in the 513 test set, refit-only. So the LB-faithful number
    #     is the per-seed bag_oof_rae_4139. We additionally report bag_te_rae_253
    #     as the in-sample diagnostic for the deploy refit.
    #     Per feedback_lb_two_regime_calibration: PRE-unblind te (trained on 4139
    #     only) has in_RAE ~= LB + 0.003. nb1193 trains ONLY on 4139 (no 253 leak)
    #     -- so bag_te_rae_253 IS a valid PRE-unblind estimate of LB.
    lb_estimate = bag_te_rae_253
    g1_pass = lb_estimate <= GATE_OOF_RAE_253
    g2_pass = (not np.isnan(rho_resid)) and (rho_resid <= GATE_RHO_NB1158)
    overall_pass = g1_pass and g2_pass
    print("\n" + "=" * 78)
    print("GATES")
    print("=" * 78)
    print(f"  G1 OOF RAE on 253 (LB estimate)  : {lb_estimate:.4f}  "
          f"<= {GATE_OOF_RAE_253:.2f}  -> {'PASS' if g1_pass else 'FAIL'}")
    print(f"  G2 residual rho vs nb1158        : {rho_resid:.4f}  "
          f"<= {GATE_RHO_NB1158:.2f}  -> {'PASS' if g2_pass else 'FAIL'}")
    print(f"  OVERALL                          : {'PASS' if overall_pass else 'FAIL'}")
    flag = "PRIMARY-2-MPNN-CANDIDATE" if overall_pass else "REJECTED"
    print(f"  flag                             : {flag}")

    # ------------ 9) Save artifacts (always; CSV only if pass) ------------
    np.save(DATA_PROCESSED / f"{TAG}_oof_4139.npy", oof_bag_4139.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_253.npy",
            pred_bag_253.astype(np.float32))
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_bag.astype(np.float32))
    print(f"\n[save] data/processed/{TAG}_oof_4139.npy")
    print(f"[save] data/processed/{TAG}_mean_bag_oof_253.npy")
    print(f"[save] data/processed/te_{TAG}.npy")

    deploy_csv = SUB_DIR / f"{TAG}_chemprop_k32_fallback.csv"
    if overall_pass:
        name_col = "name" if "name" in te.columns else "Molecule Name"
        names = te[name_col].astype(str).tolist()
        smis_full = (te["smiles"].astype(str).tolist()
                     if "smiles" in te.columns
                     else te["SMILES"].astype(str).tolist())
        sub_df = pd.DataFrame({
            "Molecule Name": names,
            "SMILES": smis_full,
            "pEC50": te_bag.astype(np.float64),
        })
        sub_df.to_csv(deploy_csv, index=False)
        print(f"[save] {deploy_csv}  rows={len(sub_df)}  [PASS]")
    else:
        print(f"[skip] {deploy_csv}  (gates failed; no deploy CSV emitted)")

    # ------------ 10) Summary ------------
    summary = {
        "tag": TAG,
        "method": "lgbm_huber_distillation_K32_chemprop_anchored_FALLBACK",
        "fallback_used": True,
        "fallback_reason": (
            "Real Chemprop 5-fold scaffold-CV K=16 estimated 7-13h on CPU; "
            "contends with active crons (feedback_resource_limits). LGBM "
            "Huber on Morgan+RDKit+chemprop_aux-distillation feature with "
            "32 perturbed-hyperparam seeds is the authorized fallback."
        ),
        "anchor": "chemprop_aux",
        "K": K,
        "n_folds": N_FOLDS,
        "seeds_diagnostic": SEEDS,
        "n_tr": int(n_tr),
        "n_te": int(n_te),
        "n_unb": int(n_unb),
        "feat_dim": int(X_tr.shape[1]),
        "rae_anchor_chemprop_aux_253": rae_anchor,
        "bag_oof_rae_4139": bag_oof_rae_4139,
        "bag_te_rae_253_insample": bag_te_rae_253,
        "lb_estimate_253": lb_estimate,
        "per_seed_oof_rae_4139_mean": float(np.mean(per_seed_oof_rae_4139)),
        "per_seed_oof_rae_4139_std": float(np.std(per_seed_oof_rae_4139)),
        "per_seed_oof_rae_4139": per_seed_oof_rae_4139,
        "per_seed_te_rae_253_insample": per_seed_oof_rae_253,
        "rho_residual_vs_nb1158_253": rho_resid,
        "rho_raw_pred_vs_nb1158_253": rho_pred,
        "nb1158_lb_estimate_253": float(rae(y_unb, nb1158_oof_253)),
        "gate_oof_rae_253_threshold": GATE_OOF_RAE_253,
        "gate_rho_nb1158_threshold": GATE_RHO_NB1158,
        "g1_oof_rae_pass": bool(g1_pass),
        "g2_rho_pass": bool(g2_pass),
        "overall_pass": bool(overall_pass),
        "flag": flag,
        "submission_path": str(deploy_csv) if overall_pass else None,
        "te_path": str(DATA_PROCESSED / f"te_{TAG}.npy"),
        "oof_path_4139": str(DATA_PROCESSED / f"{TAG}_oof_4139.npy"),
        "oof_path_253": str(DATA_PROCESSED / f"{TAG}_mean_bag_oof_253.npy"),
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "FALLBACK from real Chemprop. Trains only on 4139 (no 253 leak) so "
            "bag_te_rae_253 IS a PRE-unblind LB-faithful estimate. Cross-paradigm "
            "to nb1158 via different objective (Huber alpha=0.9 vs MSE), different "
            "feature stack (2266-d Morgan+RDKit+chemprop_aux vs 117-d SHAP-pruned), "
            "different anchor anchoring strategy (chemprop_aux as feature vs as "
            "residual baseline)."
        ),
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
        "K", "fallback_used", "feat_dim",
        "rae_anchor_chemprop_aux_253",
        "bag_oof_rae_4139", "bag_te_rae_253_insample", "lb_estimate_253",
        "per_seed_oof_rae_4139_mean", "per_seed_oof_rae_4139_std",
        "rho_residual_vs_nb1158_253", "rho_raw_pred_vs_nb1158_253",
        "g1_oof_rae_pass", "g2_rho_pass", "overall_pass", "flag",
        "submission_path",
    ):
        print(f"  {k}: {res.get(k)}")
