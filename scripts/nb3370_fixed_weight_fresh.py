"""nb3370 -- FIXED-weight blend {nb3200, fresh chemprop v3} (rule out SLSQP-overfit).

CONTEXT (cycle-167+ substrate-change thread):
    nb3351 folded the FRESH chemprop v3 anchor (nb3350) into nb3200 via a
    per-fold SLSQP simplex on {nb3200, fresh_K18_v3} + learned clip. SLSQP
    *over-shrank*: every fold drove w(fresh)=1.0 on the K18-LIFTED fresh anchor
    in-fold-train, but the per-fold-mean was 0.5500 -- it overfit the K18-lift's
    fold-train residual rather than exploiting the fresh anchor's orthogonal
    error. Net verdict FAIL (+0.1077 vs the 0.4423 wall).

    This script removes BOTH degrees of freedom that could overfit:
        (a) NO SLSQP -- the blend weight is FIXED on a coarse grid, never fit.
        (b) NO K18 residual-lift -- the RAW fresh anchor is blended directly.
    If a small FIXED weight on the RAW fresh anchor helps, the 6% decorrelation
    (Pearson(fresh,nb3200)=0.837 on the 253) carries usable orthogonal signal
    and the nb3351 failure was an SLSQP-overfit artifact, not a substrate dead
    end. If even the best fixed w fails the 0.4423 wall, the fresh anchor's
    orthogonal component is simply not aligned with the frozen-anchor error and
    the substrate-change route is closed at this anchor strength.

PROTOCOL (task spec, verbatim):
    pred = (1 - w) * nb3200 + w * fresh_v3        # RAW fresh anchor, NOT K18-lift
    w in {0.00, 0.05, 0.10, 0.15, 0.20}
    15 fresh kf_seeds {1216..1230}, honest 5-fold scaffold split on 253 unblind.
    Per-fold-mean reported. The blend is GLOBALLY FIXED (no fold-train fitting):
    for a fixed w the prediction of any compound is identical regardless of which
    fold it lands in, so the scaffold CV here only partitions the 253 into folds
    to report a per-fold-averaged RAE that is directly comparable to nb3200's
    cycle-160 deep-30 per-fold-mean reference (0.4424). Averaging that per-fold
    RAE over 15 fresh seeds + reporting std follows the cycle-160 deep-30 rule
    (15-seed = hypothesis-grade; a BETTER verdict here would need deep-30
    re-verification before any PRIMARY move).

GATE (on the BEST-w 15-seed per-fold-mean):
    best_w per_fold_mean < 0.4423 -> "BETTER"
    else                          -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3200_pred_oof.npy        data/processed/te_nb3200.npy
    data/processed/nb3350_chemprop_v3_oof.npy data/processed/te_nb3350_chemprop_v3.npy
    data/processed/te_chemprop_aux.npy        (frozen anchor, decorrel diagnostic)

Outputs:
    data/processed/nb3370_summary.json   (always)
    data/processed/nb3370_pred_oof.npy   (253,) float32 -- best-w fixed-blend OOF
    data/processed/te_nb3370.npy         (513,) float32 -- best-w fixed-blend te
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
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb3370"

# -- Frozen-chain anchor (nb3200 = learned clip on nb3090) -------------------
NB3200_OOF_PATH = DATA_PROCESSED / "nb3200_pred_oof.npy"
NB3200_TE_PATH = DATA_PROCESSED / "te_nb3200.npy"

# -- Fresh chemprop v3 anchor (nb3350) -- RAW, no K18 lift -------------------
FRESH_OOF_PATH = DATA_PROCESSED / "nb3350_chemprop_v3_oof.npy"
FRESH_TE_PATH = DATA_PROCESSED / "te_nb3350_chemprop_v3.npy"

# -- Frozen chemprop_aux (decorrelation diagnostic only) ---------------------
FROZEN_AUX_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

# -- Fixed weight grid (task spec) -------------------------------------------
W_GRID = [0.00, 0.05, 0.10, 0.15, 0.20]

# -- CV protocol -------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Promotion gate ----------------------------------------------------------
GATE_BETTER = 0.4423        # best-w per-fold-mean < this -> BETTER

# -- References --------------------------------------------------------------
REF_NB3200 = 0.4424         # cycle-160 deep-30 mean (PRIMARY-1 candidate)
REF_NB3200_STD = 0.0023
REF_NB3090 = 0.4472
REF_NB2171 = 0.4682
REF_NB1191 = 0.4718
CHEMPROP_AUX_REF = 0.6216
NB3351_PERFOLD_MEAN = 0.5500   # SLSQP-on-K18-lift per-fold-mean (the thing we rule out)


def _per_fold_mean_for_w(pred_full, y_unb, splits):
    """Per-fold-mean RAE of a GLOBALLY-FIXED prediction over one fold partition.

    pred_full is already the final per-compound prediction for ALL 253 rows
    (fixed blend, no fold dependence). We only evaluate RAE on each fold's VAL
    rows and average across folds -- exactly the metric nb3200's deep-30
    reference reports, so the comparison is apples-to-apples.
    """
    fold_raes = [float(rae(y_unb[va], pred_full[va])) for _, va in splits]
    return float(np.mean(fold_raes)), float(np.std(fold_raes, ddof=1)), fold_raes


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- FIXED-weight blend {{nb3200, RAW fresh chemprop v3 (nb3350)}}")
    print(f"          (1-w)*nb3200 + w*fresh_v3   [NO SLSQP, NO K18 lift]")
    print(f"          w grid    : {W_GRID}")
    print(f"          kf_seeds  : {len(KF_SEEDS)} FRESH {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          gate      : best-w per_fold_mean < {GATE_BETTER:.4f} "
          f"-> BETTER else FAIL")
    print("=" * 78)

    # -- Load anchors --------------------------------------------------------
    for p in (NB3200_OOF_PATH, NB3200_TE_PATH, FRESH_OOF_PATH, FRESH_TE_PATH):
        if not p.exists():
            raise FileNotFoundError(f"missing required input: {p}")

    nb3200_oof = np.load(NB3200_OOF_PATH).astype(np.float64)   # (253,)
    nb3200_te = np.load(NB3200_TE_PATH).astype(np.float64)     # (513,)
    fresh_oof = np.load(FRESH_OOF_PATH).astype(np.float64)     # (253,) RAW
    fresh_te = np.load(FRESH_TE_PATH).astype(np.float64)       # (513,) RAW

    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns
                 else te["SMILES"].astype(str).tolist())
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    assert nb3200_oof.shape == (n_unb,), f"nb3200 oof {nb3200_oof.shape}"
    assert nb3200_te.shape == (n_test,), f"nb3200 te {nb3200_te.shape}"
    assert fresh_oof.shape == (n_unb,), f"fresh oof {fresh_oof.shape}"
    assert fresh_te.shape == (n_test,), f"fresh te {fresh_te.shape}"

    # -- Anchor diagnostics --------------------------------------------------
    rae_nb3200 = float(rae(y_unb, nb3200_oof))
    rae_fresh = float(rae(y_unb, fresh_oof))
    nb3200_leak = float(np.mean(np.isclose(nb3200_oof, y_unb, atol=1e-6)))
    fresh_leak = float(np.mean(np.isclose(fresh_oof, y_unb, atol=1e-6)))
    corr_fresh_nb3200 = float(np.corrcoef(fresh_oof, nb3200_oof)[0, 1])
    pearson_vs_frozen_aux = None
    if FROZEN_AUX_TE_PATH.exists():
        frozen_aux_te = np.load(FROZEN_AUX_TE_PATH).astype(np.float64)
        if frozen_aux_te.shape == (n_test,):
            pearson_vs_frozen_aux = float(
                np.corrcoef(fresh_oof, frozen_aux_te[unb_idx])[0, 1]
            )
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")
    print(f"[nb3200] oof RAE = {rae_nb3200:.4f} (ref {REF_NB3200:.4f})  "
          f"leak_eq={nb3200_leak:.2%}  mean={nb3200_oof.mean():.3f} "
          f"std={nb3200_oof.std():.3f}")
    print(f"[fresh ] oof RAE = {rae_fresh:.4f}  leak_eq={fresh_leak:.2%}  "
          f"mean={fresh_oof.mean():.3f} std={fresh_oof.std():.3f}")
    print(f"[corr  ] Pearson(fresh, nb3200) on 253       = {corr_fresh_nb3200:.4f}  "
          f"(decorrel = {1.0 - corr_fresh_nb3200:.1%})")
    print(f"[corr  ] Pearson(fresh, frozen chemprop_aux) = {pearson_vs_frozen_aux}")

    # -- Scaffolds for honest CV ---------------------------------------------
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique = {n_unique_scaf}")

    # -- Precompute fold partitions once per seed (shared across all w) -------
    seed_splits = {
        s: scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                  shuffle=True, seed=s)
        for s in KF_SEEDS
    }
    # sanity: every row covered exactly once per seed
    for s, splits in seed_splits.items():
        cover = np.zeros(n_unb, dtype=int)
        for _, va in splits:
            cover[va] += 1
        if not np.all(cover == 1):
            raise RuntimeError(f"kf={s}: scaffold splits do not cover all rows once")

    # -- Sweep w: per-w 15-seed per-fold-mean --------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP w over {W_GRID}  x  {len(KF_SEEDS)} fresh seeds "
          f"(fixed blend, per-fold-mean)")
    print("-" * 78)
    w_records = []
    oof_by_w = {}
    for w in W_GRID:
        pred_full = (1.0 - w) * nb3200_oof + w * fresh_oof   # (253,) fixed
        oof_by_w[w] = pred_full.astype(np.float32)
        pooled_w = float(rae(y_unb, pred_full))
        per_fold_means = []
        per_fold_stds = []
        for s in KF_SEEDS:
            pf_mean, pf_std, _ = _per_fold_mean_for_w(pred_full, y_unb,
                                                      seed_splits[s])
            per_fold_means.append(pf_mean)
            per_fold_stds.append(pf_std)
        pf_arr = np.asarray(per_fold_means, dtype=np.float64)
        n_s = len(pf_arr)
        mean_pf = float(pf_arr.mean())
        std_pf = float(pf_arr.std(ddof=1)) if n_s > 1 else 0.0
        sem_pf = std_pf / np.sqrt(n_s) if n_s > 1 else 0.0
        t_mult = 2.145  # df=14, two-sided 95%
        rec = {
            "w_fresh": round(w, 3),
            "pooled_rae": round(pooled_w, 4),
            "per_fold_mean": round(mean_pf, 4),
            "per_fold_std": round(std_pf, 4),
            "per_fold_sem": round(sem_pf, 4),
            "ci95_low": round(mean_pf - t_mult * sem_pf, 4),
            "ci95_high": round(mean_pf + t_mult * sem_pf, 4),
            "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
            "delta_vs_nb3200": round(mean_pf - REF_NB3200, 4),
            "delta_vs_gate": round(mean_pf - GATE_BETTER, 4),
        }
        w_records.append(rec)
        print(f"   w={w:.2f}: pooled={pooled_w:.4f}  perfold_mean={mean_pf:.4f} "
              f"+/- {std_pf:.4f}  CI95=[{rec['ci95_low']:.4f},"
              f"{rec['ci95_high']:.4f}]  d(nb3200)={mean_pf-REF_NB3200:+.4f}")

    # -- Identify best-w (min per-fold-mean) ---------------------------------
    pf_means = np.array([r["per_fold_mean"] for r in w_records])
    best_i = int(np.argmin(pf_means))
    best_w = W_GRID[best_i]
    best_pf = float(pf_means[best_i])
    best_rec = w_records[best_i]
    # w=0.0 is the pure-nb3200 control; report whether ANY positive w beat it
    pf_at_w0 = float([r["per_fold_mean"] for r in w_records
                      if r["w_fresh"] == 0.0][0])
    pos_w_recs = [r for r in w_records if r["w_fresh"] > 0.0]
    best_pos_pf = min(r["per_fold_mean"] for r in pos_w_recs)
    best_pos_w = [r["w_fresh"] for r in pos_w_recs
                  if r["per_fold_mean"] == best_pos_pf][0]
    fresh_helps = best_pos_pf < pf_at_w0 - 1e-9

    print("\n" + "-" * 78)
    print("BEST-w SELECTION")
    print("-" * 78)
    print(f"   best w (min per-fold-mean) = {best_w:.2f}  -> {best_pf:.4f}")
    print(f"   pure nb3200 (w=0.00)       = {pf_at_w0:.4f}")
    print(f"   best POSITIVE w            = {best_pos_w:.2f} -> {best_pos_pf:.4f}  "
          f"(fresh {'HELPS' if fresh_helps else 'does NOT help'} vs w=0: "
          f"{best_pos_pf - pf_at_w0:+.4f})")
    print(f"   ref nb3200 deep-30         = {REF_NB3200:.4f} +/- {REF_NB3200_STD:.4f}")
    print(f"   nb3351 SLSQP-on-K18 perfold= {NB3351_PERFOLD_MEAN:.4f} "
          f"(the over-shrink we rule out)")
    print(f"   delta best vs gate         = {best_pf - GATE_BETTER:+.4f}")

    # -- Deploy te at best-w (mirror best_w fixed blend on 513) --------------
    te_pred = ((1.0 - best_w) * nb3200_te + best_w * fresh_te).astype(np.float32)
    oof_for_save = oof_by_w[best_w]
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print("\n" + "-" * 78)
    print(f"DEPLOY te at best w={best_w:.2f}")
    print("-" * 78)
    print(f"   te(513) mean={te_pred.mean():.3f} std={te_pred.std():.3f} "
          f"min={te_pred.min():.3f} max={te_pred.max():.3f}")
    print(f"   te[unb] RAE (deploy on 513, NOT in-sample since fixed blend) "
          f"= {te_unb_in_rae:.4f}")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if best_pf < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE (deep-30 re-verify REQUIRED). nb3370 best fixed "
            f"w={best_w:.2f} 15-seed per-fold-mean {best_pf:.4f} beats the "
            f"{GATE_BETTER:.4f} wall ({best_pf - GATE_BETTER:+.4f}) and nb3200 "
            f"deep-30 {REF_NB3200:.4f} ({best_pf - REF_NB3200:+.4f}). A FIXED "
            f"small weight on the RAW fresh chemprop v3 anchor "
            f"(Pearson(fresh,nb3200)={corr_fresh_nb3200:.3f}, {1-corr_fresh_nb3200:.1%} "
            f"decorrelated) helps WITHOUT SLSQP and WITHOUT a K18 residual-lift -- "
            f"so the nb3351 over-shrink (per-fold-mean {NB3351_PERFOLD_MEAN:.4f}, "
            f"SLSQP drove w(fresh_K18)=1.0) was an optimizer/lift overfit artifact, "
            f"not a substrate dead end. The 6% decorrelation carries usable "
            f"orthogonal error structure. MUST re-verify at deep-30 (>=30 seeds) "
            f"before any PRIMARY-1 swap: 15-seed is hypothesis-grade (cycle-160 "
            f"deep-30 rule; prior ceiling candidates showed 3-5x under-dispersion)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3370 best fixed w={best_w:.2f} 15-seed per-fold-mean "
            f"{best_pf:.4f} fails the {GATE_BETTER:.4f} wall "
            f"({best_pf - GATE_BETTER:+.4f}; delta vs nb3200 deep-30 "
            f"{REF_NB3200:.4f} = {best_pf - REF_NB3200:+.4f}). The fixed-weight "
            f"control "
            + (f"selected a POSITIVE w={best_pos_w:.2f} that beat pure nb3200 by "
               f"{best_pos_pf - pf_at_w0:+.4f} but still did not clear the wall"
               if fresh_helps else
               f"could not improve on pure nb3200 (w=0.00 at {pf_at_w0:.4f}) at "
               f"ANY positive w (best positive {best_pos_w:.2f} -> {best_pos_pf:.4f}, "
               f"{best_pos_pf - pf_at_w0:+.4f})")
            + f". This confirms the nb3351 result was NOT merely SLSQP-overfit: even "
            f"the overfit-free fixed blend cannot break 0.4423, so the RAW fresh "
            f"chemprop v3 anchor's orthogonal component (RAE {rae_fresh:.4f}, "
            f"Pearson {corr_fresh_nb3200:.3f} vs nb3200) is not aligned with the "
            f"frozen-anchor error residual at this strength. The fresh-anchor "
            f"substrate-change route remains closed; a stronger / more decorrelated "
            f"fresh anchor (deeper chemprop, different architecture/seed) is required."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts ------------------------------------------------------
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"\n   [save] {oof_path}  (best-w fixed-blend OOF, 253,)")
    print(f"   [save] {te_path}   (best-w fixed-blend te, 513,)")

    summary = {
        "tag": TAG,
        "method": ("fixed_weight_blend_nb3200_with_RAW_fresh_chemprop_v3_nb3350_"
                   "no_slsqp_no_K18_lift_honest_5fold_scaffold_cv_15_fresh_seeds_"
                   "rule_out_slsqp_overfit"),
        "depends_on": ["nb3350", "nb3200"],
        "anchor_pre_unblind": True,
        # anchors
        "nb3200_oof_path": str(NB3200_OOF_PATH),
        "nb3200_te_path": str(NB3200_TE_PATH),
        "nb3200_oof_rae": round(rae_nb3200, 4),
        "nb3200_leak_eq_truth_frac": round(nb3200_leak, 4),
        "fresh_oof_path": str(FRESH_OOF_PATH),
        "fresh_te_path": str(FRESH_TE_PATH),
        "fresh_anchor_unblind_RAE": round(rae_fresh, 4),
        "fresh_leak_eq_truth_frac": round(fresh_leak, 4),
        "fresh_anchor_raw_used": True,
        "K18_lift_used": False,
        "slsqp_used": False,
        "corr_fresh_vs_nb3200_253": round(corr_fresh_nb3200, 4),
        "decorrel_frac_vs_nb3200": round(1.0 - corr_fresh_nb3200, 4),
        "pearson_fresh_vs_frozen_aux_253": (
            round(pearson_vs_frozen_aux, 4)
            if pearson_vs_frozen_aux is not None else None
        ),
        # CV config
        "w_grid": W_GRID,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        # per-w results
        "w_records": w_records,
        # best-w
        "best_w": round(best_w, 3),
        "best_per_fold_mean": round(best_pf, 4),
        "best_per_fold_std": best_rec["per_fold_std"],
        "best_ci95_low": best_rec["ci95_low"],
        "best_ci95_high": best_rec["ci95_high"],
        "pure_nb3200_w0_per_fold_mean": round(pf_at_w0, 4),
        "best_positive_w": round(best_pos_w, 3),
        "best_positive_w_per_fold_mean": round(best_pos_pf, 4),
        "fresh_helps_vs_w0": bool(fresh_helps),
        "fresh_help_delta_vs_w0": round(best_pos_pf - pf_at_w0, 4),
        # refs
        "ref_nb3200_deep30_mean": REF_NB3200,
        "ref_nb3200_deep30_std": REF_NB3200_STD,
        "ref_nb3090": REF_NB3090,
        "ref_nb2171": REF_NB2171,
        "ref_nb1191": REF_NB1191,
        "ref_chemprop_aux": CHEMPROP_AUX_REF,
        "ref_nb3351_slsqp_K18_per_fold_mean": NB3351_PERFOLD_MEAN,
        "delta_best_vs_nb3200_deep30": round(best_pf - REF_NB3200, 4),
        "delta_best_vs_gate": round(best_pf - GATE_BETTER, 4),
        # deploy te
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_deploy_rae": round(te_unb_in_rae, 4),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": None,
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
    print(f"   best w                = {best_w:.2f}")
    print(f"   best per-fold-mean    = {best_pf:.4f} +/- {best_rec['per_fold_std']:.4f}"
          f"  (15 seeds)")
    print(f"   95% CI                = [{best_rec['ci95_low']:.4f}, "
          f"{best_rec['ci95_high']:.4f}]")
    print(f"   pure nb3200 (w=0)     = {pf_at_w0:.4f}")
    print(f"   delta best vs nb3200  = {best_pf - REF_NB3200:+.4f}")
    print(f"   delta best vs gate    = {best_pf - GATE_BETTER:+.4f}")
    print(f"   fresh helps vs w=0    = {fresh_helps} "
          f"({best_pos_pf - pf_at_w0:+.4f} at best positive w={best_pos_w:.2f})")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_w", "best_per_fold_mean", "best_per_fold_std",
        "best_ci95_low", "best_ci95_high",
        "pure_nb3200_w0_per_fold_mean",
        "best_positive_w", "best_positive_w_per_fold_mean",
        "fresh_helps_vs_w0", "fresh_help_delta_vs_w0",
        "delta_best_vs_nb3200_deep30", "delta_best_vs_gate",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
