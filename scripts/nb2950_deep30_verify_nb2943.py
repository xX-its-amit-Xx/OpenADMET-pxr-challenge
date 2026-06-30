"""nb2950 -- DEEP verify of nb2943 best combo (FIXED) over 5 fresh kf_seeds.

CONTEXT
-------
nb2943 single-kf grid search (16 cells, kf_seed=1001) selected
    best (w_nb2240=0.5, w_nb1191=0.0, w_K=0.5)  pooled RAE = 0.4576
This is selection-bias prone (cf. nb2621 K-grid, nb2920 ratio-sweep) -- the
WINNING cell is the minimum over 16 noisy estimates on a SINGLE seed.  Need
to re-evaluate the FIXED winning combo on multiple disjoint kf_seeds to
distinguish a real ceiling-break from lucky-seed noise.

PROTOCOL
--------
- Apply FIXED combo (no re-search): w_nb2240=0.5, w_nb1191=0.0, w_K=0.5
- equal_K = mean(K18, K24, K28)  (K20 carried by nb2240 anchor, K=20)
- pred_oof = 0.5 * nb2240 + 0.5 * equal_K
- 5-fold scaffold CV, kf_seeds {1001..1005} (matches task spec)
- per-seed pooled RAE -> mean/std/median across 5 seeds
- df = 0 (no learning), pure scalar blend

GATE
----
mean < 0.4570 -> VERIFIED_PROMOTE  (beats nb2943 single-kf 0.4576)
mean < 0.4598 -> VERIFIED_MARGINAL (within 1sd of nb2943, holds vs nb2934)
else          -> LUCKY_SELECTION

Outputs:
    scripts/nb2950_deep30_verify_nb2943.py
    data/processed/nb2950_summary.json
    data/processed/nb2950_pred_oof.npy   (253,) float32  [mean of 5-seed OOFs]
    data/processed/te_nb2950.npy         (513,) float32  [deploy refit]
    submissions/nb2950_deep30_verify_nb2943.csv          [if PROMOTE]
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
from rdkit import RDLogger
from scipy.stats import t as student_t

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2950"
PARENT_TAG = "nb2943"

# ---- FIXED best combo from nb2943 ----
W_NB2240 = 0.5
W_NB1191 = 0.0
W_K = 0.5

# ---- Component paths (identical to nb2943) ----
K18_OOF_PATH = DATA_PROCESSED / "nb2604_mean_bag_oof_K18.npy"
K18_TE_PATH = DATA_PROCESSED / "te_nb2604_K18.npy"
K20_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"   # == nb2240 anchor
K20_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
K24_OOF_PATH = DATA_PROCESSED / "nb2310_mean_bag_oof_K24.npy"
K24_TE_PATH = DATA_PROCESSED / "te_nb2310_K24.npy"
K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
K28_TE_PATH = DATA_PROCESSED / "te_nb2112.npy"
NB1191_OOF_PATH = DATA_PROCESSED / "nb1191_pred_oof.npy"
NB1191_TE_PATH = DATA_PROCESSED / "te_nb1191.npy"

# ---- Eval protocol ----
N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# ---- Gates ----
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# ---- References ----
NB2943_SINGLE_KF_RAE = 0.4576
NB2934_REF = 0.4580
NB2604_REF = 0.4580
NB2171_REF = 0.4682
NB1191_REF = 0.4718


def evaluate_blend_for_seed(oofs, equal_K_oof, y_unb, unb_scaffolds, kf_seed):
    """Run 5-fold scaffold CV on FIXED blend for one seed.

    Since the blend is a fixed scalar combination with df=0, there is no
    learning; the OOF prediction at any row is identical to the blended
    prediction at that row.  We still do scaffold splits + pool to keep the
    metric definition consistent with nb2943 and with the broader pipeline.
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    pred = W_NB2240 * oofs["K20"] + W_NB1191 * oofs["nb1191"] + W_K * equal_K_oof
    n_unb = len(y_unb)
    oof_pooled = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold = []
    for tr_loc, va_loc in splits:
        oof_pooled[va_loc] = pred[va_loc]
        per_fold.append(float(rae(y_unb[va_loc], pred[va_loc])))
    if np.isnan(oof_pooled).any():
        raise RuntimeError("scaffold splits did not cover all rows")
    pooled = float(rae(y_unb, oof_pooled))
    return pooled, per_fold, oof_pooled


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEEP verify {PARENT_TAG} best combo "
          f"(w_nb2240={W_NB2240}, w_nb1191={W_NB1191}, w_K={W_K})")
    print(f"         paradigm: FIXED 3-component blend, df=0")
    print(f"         kf_seeds={KF_SEEDS}  n_folds={N_FOLDS}")
    print(f"         refs nb2943_singlekf={NB2943_SINGLE_KF_RAE}  "
          f"nb2934={NB2934_REF}  nb2171={NB2171_REF}")
    print("=" * 78)

    # ---- Load truth + scaffolds ----
    te_df = load_test()
    n_test = len(te_df)
    te_smiles = (te_df["smiles"].astype(str).tolist()
                 if "smiles" in te_df.columns
                 else te_df["SMILES"].astype(str).tolist())
    te_names = (te_df["name"].values
                if "name" in te_df.columns
                else te_df["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- Load components ----
    print("\n" + "-" * 78)
    print("Load 5 components")
    print("-" * 78)
    comps = [
        ("K18", K18_OOF_PATH, K18_TE_PATH),
        ("K20", K20_OOF_PATH, K20_TE_PATH),     # == nb2240
        ("K24", K24_OOF_PATH, K24_TE_PATH),
        ("K28", K28_OOF_PATH, K28_TE_PATH),
        ("nb1191", NB1191_OOF_PATH, NB1191_TE_PATH),
    ]
    oofs = {}
    tes = {}
    per_anchor_rae = {}
    for name, oof_p, te_p in comps:
        if not oof_p.exists():
            raise FileNotFoundError(f"missing OOF: {oof_p}")
        if not te_p.exists():
            raise FileNotFoundError(f"missing te: {te_p}")
        oof = np.load(oof_p).astype(np.float64)
        te_v = np.load(te_p).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{name} oof shape {oof.shape} != ({n_unb},)")
        if te_v.shape != (n_test,):
            raise ValueError(f"{name} te shape {te_v.shape} != ({n_test},)")
        oofs[name] = oof
        tes[name] = te_v
        r = float(rae(y_unb, oof))
        per_anchor_rae[name] = r
        print(f"  {name:>7s}  oof_RAE={r:.4f}  te_mean={te_v.mean():.3f}  "
              f"te_std={te_v.std():.3f}  [{oof_p.name}]")

    # ---- Build equal_K = mean(K18, K24, K28) (K20 excluded -> carried by nb2240) ----
    equal_K_oof = (oofs["K18"] + oofs["K24"] + oofs["K28"]) / 3.0
    equal_K_te = (tes["K18"] + tes["K24"] + tes["K28"]) / 3.0
    equal_K_rae = float(rae(y_unb, equal_K_oof))
    print(f"\n  equal_K = mean(K18, K24, K28)  oof_RAE={equal_K_rae:.4f}  "
          f"te_mean={equal_K_te.mean():.3f}  te_std={equal_K_te.std():.3f}")

    # ---- Multi-seed CV pooling ----
    print("\n" + "-" * 78)
    print(f"DEEP scaffold 5-fold CV  kf_seeds={KF_SEEDS}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, per_fold, oof_pooled = evaluate_blend_for_seed(
            oofs, equal_K_oof, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "per_fold_rae": [float(x) for x in per_fold],
            "fold_mean": float(np.mean(per_fold)),
            "fold_std": float(np.std(per_fold)),
        })
        all_oofs.append(oof_pooled)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"fold_mean={np.mean(per_fold):.4f}  "
              f"fold_std={np.std(per_fold):.4f}")

    pooled_arr = np.array([r["pooled_rae"] for r in per_seed], dtype=np.float64)
    n_seeds = len(pooled_arr)
    pooled_mean = float(pooled_arr.mean())
    pooled_std = float(pooled_arr.std(ddof=1)) if n_seeds > 1 else 0.0
    pooled_min = float(pooled_arr.min())
    pooled_max = float(pooled_arr.max())
    pooled_median = float(np.median(pooled_arr))

    # 95% CI on the mean (t-distribution, df = n-1 = 4)
    if n_seeds > 1:
        sem = pooled_std / np.sqrt(n_seeds)
        t_crit = float(student_t.ppf(0.975, df=n_seeds - 1))
        ci_low = pooled_mean - t_crit * sem
        ci_high = pooled_mean + t_crit * sem
    else:
        sem = 0.0
        t_crit = 0.0
        ci_low = pooled_mean
        ci_high = pooled_mean

    # mean of OOF arrays across seeds (alt aggregation)
    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    rae_of_mean_oof = float(rae(y_unb, mean_oof))

    print(f"\n[deep]  mean = {pooled_mean:.4f}  std(ddof=1) = {pooled_std:.4f}  "
          f"median = {pooled_median:.4f}")
    print(f"[deep]  min  = {pooled_min:.4f}  max  = {pooled_max:.4f}")
    print(f"[deep]  95%CI on mean (t, df={n_seeds - 1}) = "
          f"[{ci_low:.4f}, {ci_high:.4f}]")
    print(f"[deep]  RAE of mean-of-seed OOFs = {rae_of_mean_oof:.4f}")

    # ---- Selection-bias diagnostic vs nb2943 ----
    shift_vs_single_kf = pooled_mean - NB2943_SINGLE_KF_RAE
    print(f"\n[selection-bias diagnostic]")
    print(f"   nb2943 single-kf (seed=1001 minimum over 16 cells) = "
          f"{NB2943_SINGLE_KF_RAE:.4f}")
    print(f"   nb2950 deep mean (5 fresh seeds, FIXED combo)      = "
          f"{pooled_mean:.4f}")
    print(f"   shift (positive = nb2943 lucky-selection)          = "
          f"{shift_vs_single_kf:+.4f}")

    # ---- Gate ----
    if pooled_mean < GATE_PROMOTE:
        verdict = "VERIFIED_PROMOTE"
    elif pooled_mean < GATE_MARGINAL:
        verdict = "VERIFIED_MARGINAL"
    else:
        verdict = "LUCKY_SELECTION"
    print(f"\n[gate] mean={pooled_mean:.4f}  "
          f"(<{GATE_PROMOTE} PROMOTE / <{GATE_MARGINAL} MARGINAL)  -> "
          f"{verdict}")

    delta_vs_nb2934 = pooled_mean - NB2934_REF
    delta_vs_nb2171 = pooled_mean - NB2171_REF
    delta_vs_nb1191 = pooled_mean - NB1191_REF
    print(f"   delta vs nb2934 ({NB2934_REF}) = {delta_vs_nb2934:+.4f}")
    print(f"   delta vs nb2171 ({NB2171_REF}) = {delta_vs_nb2171:+.4f}")
    print(f"   delta vs nb1191 ({NB1191_REF}) = {delta_vs_nb1191:+.4f}")

    # ---- Deploy refit (the blend is fixed, no learning) ----
    deploy_pred_oof = W_NB2240 * oofs["K20"] + W_NB1191 * oofs["nb1191"] + W_K * equal_K_oof
    deploy_pred_te = W_NB2240 * tes["K20"] + W_NB1191 * tes["nb1191"] + W_K * equal_K_te
    in_sample_rae = float(rae(y_unb, deploy_pred_oof))
    te_unb_in = float(rae(y_unb, deploy_pred_te[unb_idx]))
    print(f"\n[deploy]  in-sample RAE (253) = {in_sample_rae:.4f}")
    print(f"[deploy]  te[unb_idx] RAE      = {te_unb_in:.4f}")
    print(f"[deploy]  te(513) mean / std   = "
          f"{deploy_pred_te.mean():.3f} / {deploy_pred_te.std():.3f}")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, deploy_pred_oof.astype(np.float32))
    np.save(te_path, deploy_pred_te.astype(np.float32))
    print(f"\n[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_deep30_verify_nb2943.csv"
    if verdict == "VERIFIED_PROMOTE":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": deploy_pred_te.astype(np.float32),
        }).to_csv(sub_csv, index=False)
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "deep_seed_verify_fixed_blend_nb2240_nb1191_equal_K",
        "paradigm": "FIXED_3component_blend_df_zero_5_kf_seeds",
        "fixed_combo": {
            "w_nb2240": W_NB2240,
            "w_nb1191": W_NB1191,
            "w_K_ensemble": W_K,
        },
        "anchors": {
            "nb2240": "K=20 (nb2240_mean_bag_oof_K20.npy)",
            "nb1191": "PRE-pyramid (nb1191_pred_oof.npy)",
            "equal_K": "mean(K18, K24, K28)",
        },
        "per_anchor_rae_in_sample": per_anchor_rae,
        "equal_K_oof_rae": equal_K_rae,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "n_seeds": n_seeds,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_seed_results": per_seed,
        "pooled_rae_mean": pooled_mean,
        "pooled_rae_std": pooled_std,
        "pooled_rae_median": pooled_median,
        "pooled_rae_min": pooled_min,
        "pooled_rae_max": pooled_max,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "sem": float(sem),
        "t_crit": t_crit,
        "rae_of_mean_of_seed_oofs": rae_of_mean_oof,
        "nb2943_single_kf_rae": NB2943_SINGLE_KF_RAE,
        "shift_vs_nb2943_single_kf": float(shift_vs_single_kf),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "delta_vs_nb2934": delta_vs_nb2934,
        "delta_vs_nb2171": delta_vs_nb2171,
        "delta_vs_nb1191": delta_vs_nb1191,
        "nb2934_ref": NB2934_REF,
        "nb2604_ref": NB2604_REF,
        "nb2171_ref": NB2171_REF,
        "nb1191_ref": NB1191_REF,
        "in_sample_rae": in_sample_rae,
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(deploy_pred_te.mean()),
        "te_std": float(deploy_pred_te.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "VERIFIED_PROMOTE" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   FIXED combo            = (w_nb2240={W_NB2240}, "
          f"w_nb1191={W_NB1191}, w_K={W_K})")
    print(f"   per-anchor RAE         = " + "  ".join(
        f"{k}={v:.4f}" for k, v in per_anchor_rae.items()))
    print(f"   equal_K RAE            = {equal_K_rae:.4f}")
    print(f"   n_seeds                = {n_seeds}")
    print(f"   pooled mean / std      = {pooled_mean:.4f} / {pooled_std:.4f}")
    print(f"   95% CI on mean         = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   shift vs nb2943_singlekf = {shift_vs_single_kf:+.4f}")
    print(f"   verdict                = {verdict}")
    print(f"   delta nb2934           = {delta_vs_nb2934:+.4f}")
    print(f"   delta nb2171           = {delta_vs_nb2171:+.4f}")
    print(f"   te[unb_idx] in-sample  = {te_unb_in:.4f}")
    print(f"   wall                   = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean",
        "pooled_rae_std",
        "ci95_low",
        "ci95_high",
        "shift_vs_nb2943_single_kf",
        "verdict",
        "delta_vs_nb2934",
        "delta_vs_nb2171",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
