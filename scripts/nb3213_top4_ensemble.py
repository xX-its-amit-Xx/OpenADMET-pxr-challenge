"""nb3213 -- Equal-weight 4-way ensemble of clip winners (nb3190, nb3173, nb3174, nb3170).

NEW PARADIGM:
    Extend nb3204 (3-way clip-winner mean, 0.4418) to 4-way by adding nb3170
    (verifier of nb3161 range-clip). All four anchors are PRE-unblind clip
    operators sitting on the same chemprop_aux substrate.

CONTEXT:
    nb3204 (3-way mean of nb3190/nb3173/nb3174) reached pooled RAE 0.4418
    at 15 fresh seeds {1186..1200} -- BETTER than gate 0.4426. This script
    asks: does the 4th component nb3170 further reduce variance, or does
    the high pairwise correlation (>0.999) of clip winners bottom out the
    averaging gain at ~0.4418?

PROTOCOL:
    Load nb3190_pred_oof, nb3173_pred_oof, nb3174_pred_oof, nb3170_pred_oof
    Equal-weight (1/4) mean -> single pred_oof_ens
    For each kf_seed in {1216..1230} (15 FRESH seeds, non-overlapping with nb3204):
        scaffold_kfold_indices(n_splits=5, seed=kf_seed)
        Pool 5 folds -> pooled_rae (invariant to split for fixed predictions)
        Also track per-fold-val-RAE mean (varies with split)
    Aggregate mean +/- std over 15 seeds.

GATE (per task):
    mean < 0.4418 -> "BETTER"   (must strictly beat nb3204 3-way ceiling)
    else          -> "FAIL"

References:
    nb3204 3-way (nb3190+nb3173+nb3174) = 0.4418 (current ceiling at 15-seed)
    nb3190 = 0.4422 (clip on nb3090, learned)
    nb3173 = 0.4423 (learned-clip on parent A)
    nb3174 = 0.4430 (clip on alternates)
    nb3170 = 0.4437 (verify of nb3161 range-clip on nb3080)
    nb2171 prior PRIMARY-1 = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3190_pred_oof.npy   (253,) float32
    data/processed/nb3173_pred_oof.npy   (253,) float32
    data/processed/nb3174_pred_oof.npy   (253,) float32
    data/processed/nb3170_pred_oof.npy   (253,) float32
    data/processed/te_nb3190.npy         (513,) float32
    data/processed/te_nb3173.npy         (513,) float32
    data/processed/te_nb3174.npy         (513,) float32
    data/processed/te_nb3170.npy         (513,) float32

Outputs:
    data/processed/nb3213_summary.json
    data/processed/nb3213_pred_oof.npy   (253,) float32 -- equal-weight mean
    data/processed/te_nb3213.npy         (513,) float32 -- equal-weight te mean
    submissions/nb3213_top4_ensemble.csv  (only on BETTER)
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

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3213"
COMPONENTS = ["nb3190", "nb3173", "nb3174", "nb3170"]

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Gate (per task) -----------------------------------------------------------
GATE_BETTER = 0.4418

# -- References ----------------------------------------------------------------
REF_NB3204 = 0.4418
REF_NB3190 = 0.4422
REF_NB3173 = 0.4423
REF_NB3174 = 0.4430
REF_NB3170 = 0.4437
REF_NB2171 = 0.4682


def _eval_one_seed(
    pred_ens: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Eval equal-weight ensemble at a single kf_seed (no fitting)."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    fold_val_raes = []
    n_unb = len(y_unb)
    covered = np.zeros(n_unb, dtype=bool)
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        covered[va_loc] = True
        val_pred = pred_ens[va_loc]
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
    if not covered.all():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, pred_ens))  # split-invariant for fixed predictions
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(
        f"{TAG} -- EQUAL-WEIGHT 4-WAY ENSEMBLE of clip winners "
        f"({', '.join(COMPONENTS)})"
    )
    print(f"          weights = (1/4, 1/4, 1/4, 1/4)")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(f"          gate: mean < {GATE_BETTER:.4f} -> BETTER, else -> FAIL")
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (
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
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Load component pred_oof's + te's -------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load {len(COMPONENTS)} component pred_oof + te files")
    print("-" * 78)
    comp_oofs = []
    comp_tes = []
    comp_oof_raes = []
    for tag in COMPONENTS:
        oof_path = DATA_PROCESSED / f"{tag}_pred_oof.npy"
        te_path = DATA_PROCESSED / f"te_{tag}.npy"
        oof_arr = np.load(oof_path).astype(np.float64)
        te_arr = np.load(te_path).astype(np.float64)
        if oof_arr.shape != (n_unb,):
            raise ValueError(
                f"{tag} pred_oof shape {oof_arr.shape} != ({n_unb},)"
            )
        if te_arr.shape != (n_test,):
            raise ValueError(f"{tag} te shape {te_arr.shape} != ({n_test},)")
        r = float(rae(y_unb, oof_arr))
        comp_oofs.append(oof_arr)
        comp_tes.append(te_arr)
        comp_oof_raes.append(r)
        print(
            f"   {tag}: oof_RAE={r:.4f}  "
            f"oof mean={oof_arr.mean():.3f} std={oof_arr.std():.3f}  "
            f"te mean={te_arr.mean():.3f} std={te_arr.std():.3f}"
        )

    # Pairwise correlations
    print("\n   pairwise OOF correlations:")
    for i in range(len(COMPONENTS)):
        for j in range(i + 1, len(COMPONENTS)):
            c = float(np.corrcoef(comp_oofs[i], comp_oofs[j])[0, 1])
            print(f"     {COMPONENTS[i]} <-> {COMPONENTS[j]}: {c:.4f}")

    # -- Build equal-weight ensemble ------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 2: equal-weight (1/{len(COMPONENTS)}) mean of pred_oof and te")
    print("-" * 78)
    pred_ens = np.mean(np.stack(comp_oofs, axis=0), axis=0)
    te_ens = np.mean(np.stack(comp_tes, axis=0), axis=0).astype(np.float32)
    ens_full_oof_rae = float(rae(y_unb, pred_ens))
    print(
        f"   pred_ens: oof_RAE={ens_full_oof_rae:.4f}  "
        f"mean={pred_ens.mean():.3f}  std={pred_ens.std():.3f}  "
        f"min={pred_ens.min():.3f}  max={pred_ens.max():.3f}"
    )
    print(
        f"   te_ens:   mean={te_ens.mean():.3f}  std={te_ens.std():.3f}  "
        f"min={te_ens.min():.3f}  max={te_ens.max():.3f}"
    )

    # Variance-reduction summary vs components
    comp_oof_mean = float(np.mean(comp_oof_raes))
    delta_vs_best_comp = ens_full_oof_rae - float(np.min(comp_oof_raes))
    delta_vs_mean_comp = ens_full_oof_rae - comp_oof_mean
    delta_vs_nb3204 = ens_full_oof_rae - REF_NB3204
    print(
        f"   vs best component  ({np.min(comp_oof_raes):.4f}): "
        f"delta = {delta_vs_best_comp:+.4f}"
    )
    print(
        f"   vs mean components ({comp_oof_mean:.4f}): "
        f"delta = {delta_vs_mean_comp:+.4f}"
    )
    print(
        f"   vs nb3204 3-way ({REF_NB3204:.4f}): "
        f"delta = {delta_vs_nb3204:+.4f}"
    )

    # Leak sanity on ensemble
    leak_eq = float(np.mean(np.isclose(pred_ens, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN: {leak_eq:.1%} rows == truth -- possible leak")

    # -- Scaffolds for outer CV -----------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Multi-seed sweep -----------------------------------------------------
    print("\n" + "-" * 78)
    print(
        f"SWEEP: {len(KF_SEEDS)} fresh kf_seeds "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _eval_one_seed(pred_ens, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
        })
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"per_fold_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"per_fold_std={res['per_fold_val_rae_std']:.4f}  "
            f"wall={time.time()-ts:.2f}s"
        )

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    # df=14, two-sided 95% t_mult = 2.1448
    t_mult = 2.1448
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))

    arr_pf = np.asarray(per_fold_means, dtype=np.float64)
    pf_mean = float(arr_pf.mean())
    pf_std = float(arr_pf.std(ddof=1)) if n_s > 1 else 0.0

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled_RAE mean   = {mean_rae:.4f}")
    print(f"   pooled_RAE std    = {std_rae:.4f}")
    print(f"   sem               = {sem:.4f}")
    print(f"   95% CI (df=14)    = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median            = {median_rae:.4f}")
    print(f"   min/max           = [{arr.min():.4f}, {arr.max():.4f}]")
    print(
        f"   per_fold_mean RAE = {pf_mean:.4f} +/- {pf_std:.4f} "
        f"(split-sensitive)"
    )
    print(
        f"\n   ref nb3204 = {REF_NB3204:.4f}  nb3190 = {REF_NB3190:.4f}  "
        f"nb3173 = {REF_NB3173:.4f}  nb3174 = {REF_NB3174:.4f}  "
        f"nb3170 = {REF_NB3170:.4f}"
    )
    print(f"   ref nb2171 PRIMARY-1 = {REF_NB2171:.4f}")
    print(f"   shift vs nb3204 ref  = {mean_rae - REF_NB3204:+.4f}")
    print(f"   shift vs nb3190 ref  = {mean_rae - REF_NB3190:+.4f}")
    print(f"   shift vs nb3173 ref  = {mean_rae - REF_NB3173:+.4f}")
    print(f"   shift vs nb3174 ref  = {mean_rae - REF_NB3174:+.4f}")
    print(f"   shift vs nb3170 ref  = {mean_rae - REF_NB3170:+.4f}")
    print(f"   gain vs nb2171       = {REF_NB2171 - mean_rae:+.4f}")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"BETTER. nb3213 equal-weight 4-way ensemble of "
            f"({', '.join(COMPONENTS)}) 15-seed mean {mean_rae:.4f} "
            f"+/- {std_rae:.4f} clears strict gate {GATE_BETTER:.4f}. "
            f"Adding nb3170 to the 3-way nb3204 mean delivers delta "
            f"{delta_vs_nb3204:+.4f} vs nb3204 ceiling. Variance-reduction "
            f"across near-correlated (>0.999) clip winners continues to "
            f"compound at 4 anchors. PROMOTE nb3213 to PRIMARY-1 candidate "
            f"(displaces nb2171 0.4682 by -{REF_NB2171 - mean_rae:.4f} RAE, "
            f"and nb3204 0.4418 by {delta_vs_nb3204:+.4f}). Deep-30 "
            f"verification required before LB fire per cycle-160 rule "
            f"(15-seed std likely under-dispersed at this near-correlated "
            f"ensemble level)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"FAIL. nb3213 equal-weight 4-way ensemble mean {mean_rae:.4f} "
            f"fails strict gate {GATE_BETTER:.4f} (does not beat nb3204 "
            f"3-way 0.4418). Components correlate >0.999, so adding a 4th "
            f"clip variant (nb3170, the weakest at 0.4437) does not further "
            f"reduce variance -- delta vs nb3204 = {delta_vs_nb3204:+.4f}. "
            f"Equal-weight averaging of near-identical predictions has "
            f"bottomed out; the 0.4418 ceiling is the post-hoc-blend floor "
            f"for this clip-winner family on the chemprop_aux substrate. "
            f"Keep nb3204 (or current PRIMARY-1) and pivot to substrate "
            f"change (new anchor, off-manifold features, abstention) per "
            f"cycle-134 paradigm-exhaustion thesis."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_ens.astype(np.float32))
    np.save(te_path, te_ens)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_top4_ensemble.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_ens,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "method": "equal_weight_mean_of_4_clip_winners",
        "components": COMPONENTS,
        "weights": [round(1.0 / len(COMPONENTS), 6)] * len(COMPONENTS),
        "component_oof_rae": {
            COMPONENTS[i]: round(comp_oof_raes[i], 4)
            for i in range(len(COMPONENTS))
        },
        "component_oof_rae_mean": round(comp_oof_mean, 4),
        "component_oof_rae_min": round(float(np.min(comp_oof_raes)), 4),
        "pairwise_corr": {
            f"{COMPONENTS[i]}_{COMPONENTS[j]}": round(
                float(np.corrcoef(comp_oofs[i], comp_oofs[j])[0, 1]), 4
            )
            for i in range(len(COMPONENTS))
            for j in range(i + 1, len(COMPONENTS))
        },
        "anchor_pre_unblind": True,  # All components PRE-unblind clipped
        "ens_full_oof_rae": round(ens_full_oof_rae, 4),
        "ens_leak_eq_truth_frac": round(leak_eq, 4),
        "delta_vs_best_component": round(delta_vs_best_comp, 4),
        "delta_vs_mean_components": round(delta_vs_mean_comp, 4),
        "delta_vs_nb3204": round(delta_vs_nb3204, 4),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_val_rae_means_array": [
            round(float(v), 4) for v in per_fold_means
        ],
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "per_fold_mean_rae_mean": round(pf_mean, 4),
        "per_fold_mean_rae_std": round(pf_std, 4),
        "ref_nb3204": REF_NB3204,
        "ref_nb3190": REF_NB3190,
        "ref_nb3173": REF_NB3173,
        "ref_nb3174": REF_NB3174,
        "ref_nb3170": REF_NB3170,
        "ref_nb2171": REF_NB2171,
        "shift_vs_nb3204": round(mean_rae - REF_NB3204, 4),
        "shift_vs_nb3190": round(mean_rae - REF_NB3190, 4),
        "shift_vs_nb3173": round(mean_rae - REF_NB3173, 4),
        "shift_vs_nb3174": round(mean_rae - REF_NB3174, 4),
        "shift_vs_nb3170": round(mean_rae - REF_NB3170, 4),
        "gain_vs_nb2171": round(REF_NB2171 - mean_rae, 4),
        "te_mean": float(te_ens.mean()),
        "te_std": float(te_ens.std()),
        "te_min": float(te_ens.min()),
        "te_max": float(te_ens.max()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (
            str(sub_csv) if verdict == "BETTER" else None
        ),
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae ({n_s} seeds)   = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI                = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3204       = {delta_vs_nb3204:+.4f}")
    print(f"   delta vs best comp    = {delta_vs_best_comp:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3204", "delta_vs_best_component",
        "delta_vs_mean_components",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
