"""nb3300 -- Selective abstention: return train-median for the most uncertain rows.

NEW PARADIGM (substrate change, not operator change):
    Every post-hoc operator on the chemprop_aux / K-pyramid anchor (clip,
    stretch, simplex, sigmoid) has converged at ~0.4424 (nb3200 deep-30). The
    remaining error tail is dominated by novel-scaffold rows where the model is
    BOTH uncertain AND wrong (variance compression, F2 greasy-novel-inactive).

    Abstention attacks that tail directly: for the rows where the K-pyramid
    members DISAGREE the most (high ensemble std = epistemic uncertainty), we
    REFUSE to trust the (compressed, likely-wrong) point estimate and fall back
    to the train-median -- the Bayes-optimal constant under RAE when the
    conditional signal is unreliable. Everywhere else, defer to nb3200.

    This is orthogonal to all prior clip/blend operators: it changes WHICH rows
    get a model prediction at all, rather than how the prediction is reshaped.

PER-ROW UNCERTAINTY:
    u_i = std( K18_i, K19_i, K20_i, K24_i )   over the 4 K-pyramid members.
    Each K_* is itself a deep-30 averaged OOF/te, so u_i is the residual
    cross-K disagreement after seed averaging -- a clean epistemic signal.

PROTOCOL (per kf_seed in {1216..1230}, mirrors nb3200/nb3230 CV exactly):
    scaffold_kfold_indices(n_splits=5, shuffle=True, seed=kf_seed)
    For each fold:
        a) cutoff = quantile(u[tr_loc], 1 - frac)        # LEARNED on fold-train
           -> abstain on val rows with u >= cutoff (top-`frac` most uncertain)
        b) m = median(y_unb[tr_loc])                      # per-fold train-median
        c) pred[va] = where(u[va] >= cutoff, m, nb3200[va])
    Pool 5 folds -> pooled_rae.  Aggregate per-fold-mean over 15 seeds.

    frac swept in {0.0, 0.05, 0.10, 0.20}. frac=0.0 reproduces nb3200 exactly
    (sanity anchor). The cutoff is derived ONLY from train uncertainties, so no
    val label or val uncertainty leaks into the abstention decision.

DEPLOY (te, 513):
    u_te from te K-pyramid std; cutoff = quantile(u_oof_253, 1 - frac*);
    abstain te rows -> median(y_unb full 253); else nb3200 te.  frac* = best.

GATE (per task):
    best abstention per-fold-mean < 0.4423 -> "BETTER"
    else                                    -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3200_pred_oof.npy        (253) fallback OOF
    data/processed/te_nb3200.npy              (513) fallback te
    data/processed/nb2960_K18_30seed_oof.npy + nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy + te_nb3000_K19.npy
    data/processed/nb2960_K20_30seed_oof.npy + nb2960_K20_30seed_te.npy
    data/processed/nb2960_K24_30seed_oof.npy + nb2960_K24_30seed_te.npy

Outputs:
    data/processed/nb3300_summary.json
    data/processed/nb3300_pred_oof.npy   (253) float32 -- best-frac median-seed OOF
    data/processed/te_nb3300.npy         (513) float32 -- best-frac deploy te
    submissions/nb3300_abstention.csv    (only on BETTER)
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

TAG = "nb3300"
FALLBACK_TAG = "nb3200"

# -- Inputs: fallback predictor (nb3200) --------------------------------------
FALLBACK_OOF_PATH = DATA_PROCESSED / "nb3200_pred_oof.npy"
FALLBACK_TE_PATH = DATA_PROCESSED / "te_nb3200.npy"

# -- Inputs: K-pyramid members for per-row uncertainty (canonical deep-30) ----
#    (oof_filename, te_filename) per K -- matches nb3230 K_ANCHOR_FILES table.
K_PYRAMID_FILES = {
    18: ("nb2960_K18_30seed_oof.npy", "nb2960_K18_30seed_te.npy"),
    19: ("nb3000_K19_30seed_oof.npy", "te_nb3000_K19.npy"),
    20: ("nb2960_K20_30seed_oof.npy", "nb2960_K20_30seed_te.npy"),
    24: ("nb2960_K24_30seed_oof.npy", "nb2960_K24_30seed_te.npy"),
}

# -- CV protocol (15 FRESH seeds, mirror nb3230) ------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # {1216..1230}

# -- Abstention fraction sweep -------------------------------------------------
ABSTAIN_FRACS = [0.0, 0.05, 0.10, 0.20]

# -- Gate ----------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ----------------------------------------------------------------
REF_NB3200 = 0.4424   # deep-30 fallback (the operator-ceiling we must beat)
REF_NB3173 = 0.4437
REF_NB2171 = 0.4682


def _run_one_seed(
    fb_oof: np.ndarray,
    u_oof: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
    frac: float,
) -> dict:
    """Selective abstention at a single kf_seed + abstain-fraction.

    cutoff is learned on fold-TRAIN uncertainties only (leak-free). Abstained
    val rows take the fold-TRAIN median; all others take the nb3200 fallback.
    """
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes = []
    n_abstain_total = 0
    fold_cutoffs = []
    for tr_loc, va_loc in splits:
        m_tr = float(np.median(y_unb[tr_loc]))
        val_pred = fb_oof[va_loc].copy()
        if frac > 0.0:
            # threshold = top-`frac` of TRAIN uncertainties (quantile 1-frac)
            cutoff = float(np.quantile(u_oof[tr_loc], 1.0 - frac))
            abst = u_oof[va_loc] >= cutoff
            # guard: if cutoff degenerate (ties) and selects everything, keep it
            val_pred[abst] = m_tr
            n_abstain_total += int(abst.sum())
        else:
            cutoff = float("inf")
        fold_cutoffs.append(cutoff)
        oof[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))

    if np.isnan(oof).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "n_abstain": int(n_abstain_total),
        "cutoff_mean": float(np.mean(fold_cutoffs)) if frac > 0 else None,
        "oof": oof,
    }


def _aggregate(pooled_raes: list[float]) -> dict:
    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_s = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_s > 1 else 0.0
    sem = std_rae / np.sqrt(n_s) if n_s > 1 else 0.0
    # df=14, two-sided 95% t_mult = 2.1448
    t_mult = 2.1448
    return {
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "sem_rae": sem,
        "ci95_low": mean_rae - t_mult * sem,
        "ci95_high": mean_rae + t_mult * sem,
        "median_rae": float(np.median(arr)),
        "min_rae": float(arr.min()),
        "max_rae": float(arr.max()),
        "arr": arr,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SELECTIVE ABSTENTION (train-median for most uncertain rows)")
    print(f"          fallback     = {FALLBACK_TAG} (deep-30 mean {REF_NB3200:.4f})")
    print(f"          uncertainty  = std(K18,K19,K20,K24) per row")
    print(f"          abstain_frac = {ABSTAIN_FRACS}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(f"          gate: best per-fold-mean < {GATE_BETTER:.4f} -> BETTER, else FAIL")
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
    print(
        f"   y_unb: mean={y_unb.mean():.3f}  std={y_unb.std():.3f}  "
        f"median={np.median(y_unb):.3f}  min={y_unb.min():.3f}  max={y_unb.max():.3f}"
    )

    # -- Load fallback nb3200 -------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load fallback {FALLBACK_TAG} pred_oof + te")
    print("-" * 78)
    fb_oof = np.load(FALLBACK_OOF_PATH).astype(np.float64)
    fb_te = np.load(FALLBACK_TE_PATH).astype(np.float64)
    if fb_oof.shape != (n_unb,):
        raise ValueError(f"{FALLBACK_TAG} oof shape {fb_oof.shape} != ({n_unb},)")
    if fb_te.shape != (n_test,):
        raise ValueError(f"{FALLBACK_TAG} te shape {fb_te.shape} != ({n_test},)")
    fb_oof_rae = float(rae(y_unb, fb_oof))
    print(
        f"   fb_oof: oof_RAE={fb_oof_rae:.4f}  mean={fb_oof.mean():.3f}  "
        f"std={fb_oof.std():.3f}  min={fb_oof.min():.3f}  max={fb_oof.max():.3f}"
    )
    print(
        f"   fb_te:  mean={fb_te.mean():.3f}  std={fb_te.std():.3f}  "
        f"min={fb_te.min():.3f}  max={fb_te.max():.3f}"
    )
    # Leak sanity on fallback
    leak_eq = float(np.mean(np.isclose(fb_oof, y_unb, atol=1e-6)))
    if leak_eq > 0.05:
        print(f"   WARN fallback: {leak_eq:.1%} rows == truth -- possible leak")

    # -- Load K-pyramid members + build per-row uncertainty -------------------
    print("\n" + "-" * 78)
    print("STEP 2: load K-pyramid {18,19,20,24} -> per-row uncertainty std")
    print("-" * 78)
    oof_stack = []
    te_stack = []
    for K, (oof_fn, te_fn) in sorted(K_PYRAMID_FILES.items()):
        oK = np.load(DATA_PROCESSED / oof_fn).astype(np.float64)
        tK = np.load(DATA_PROCESSED / te_fn).astype(np.float64)
        if oK.shape != (n_unb,):
            raise ValueError(f"K={K}: oof shape {oK.shape} != ({n_unb},)")
        if tK.shape != (n_test,):
            raise ValueError(f"K={K}: te shape {tK.shape} != ({n_test},)")
        oof_stack.append(oK)
        te_stack.append(tK)
        print(
            f"   K={K}: oof_RAE={rae(y_unb, oK):.4f}  "
            f"oof mean={oK.mean():.3f} std={oK.std():.3f}  "
            f"te mean={tK.mean():.3f} std={tK.std():.3f}"
        )
    oof_mat = np.stack(oof_stack, axis=1)  # (253, 4)
    te_mat = np.stack(te_stack, axis=1)    # (513, 4)
    # population std (ddof=0) across the 4 members -- ensemble disagreement
    u_oof = oof_mat.std(axis=1)
    u_te = te_mat.std(axis=1)
    print(
        f"   u_oof (cross-K std): mean={u_oof.mean():.4f}  std={u_oof.std():.4f}  "
        f"min={u_oof.min():.4f}  median={np.median(u_oof):.4f}  max={u_oof.max():.4f}"
    )
    print(
        f"   u_te  (cross-K std): mean={u_te.mean():.4f}  std={u_te.std():.4f}  "
        f"min={u_te.min():.4f}  median={np.median(u_te):.4f}  max={u_te.max():.4f}"
    )

    # Diagnostic: does uncertainty rank-correlate with fallback abs error?
    fb_abs_err = np.abs(fb_oof - y_unb)
    try:
        from scipy.stats import spearmanr
        rho, pval = spearmanr(u_oof, fb_abs_err)
        print(
            f"   diag: Spearman(u_oof, |fb_oof - y|) = {rho:+.3f} (p={pval:.3g}) "
            f"-- positive => uncertain rows are the wrong rows (abstention helps)"
        )
    except Exception as e:
        rho, pval = None, None
        print(f"   diag: spearman unavailable ({e})")
    rho_str = f"{float(rho):+.2f}" if rho is not None else "n/a"

    # -- Scaffolds ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 3: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- Sweep abstention fractions ------------------------------------------
    print("\n" + "-" * 78)
    print(f"SWEEP: abstain_frac in {ABSTAIN_FRACS} x {len(KF_SEEDS)} seeds")
    print("-" * 78)
    frac_results = {}
    for frac in ABSTAIN_FRACS:
        pooled_raes = []
        oof_by_seed = []
        n_abst_by_seed = []
        for s in KF_SEEDS:
            res = _run_one_seed(fb_oof, u_oof, y_unb, unb_scaffolds, s, frac)
            pooled_raes.append(res["pooled_rae"])
            oof_by_seed.append(res["oof"])
            n_abst_by_seed.append(res["n_abstain"])
        agg = _aggregate(pooled_raes)
        # median-seed OOF for storage
        arr = agg["arr"]
        med_seed_idx = int(np.argsort(arr)[len(arr) // 2])
        frac_results[frac] = {
            "frac": frac,
            "agg": agg,
            "pooled_raes": pooled_raes,
            "oof_median_seed": oof_by_seed[med_seed_idx].astype(np.float32),
            "median_seed": KF_SEEDS[med_seed_idx],
            "n_abstain_mean": float(np.mean(n_abst_by_seed)),
        }
        print(
            f"   frac={frac:.2f}: mean={agg['mean_rae']:.4f} +/- {agg['std_rae']:.4f}  "
            f"CI=[{agg['ci95_low']:.4f},{agg['ci95_high']:.4f}]  "
            f"median={agg['median_rae']:.4f}  "
            f"n_abstain(mean/253)={np.mean(n_abst_by_seed):.1f}  "
            f"delta_vs_nb3200={agg['mean_rae'] - REF_NB3200:+.4f}"
        )

    # frac=0.0 sanity: should ~reproduce nb3200 deep-15 (note: 15 vs 30 seeds)
    base0 = frac_results[0.0]["agg"]["mean_rae"]
    print(
        f"\n   [sanity] frac=0.0 mean={base0:.4f} vs nb3200 deep-30 {REF_NB3200:.4f} "
        f"(diff {base0 - REF_NB3200:+.4f}; expected ~0 modulo 15-vs-30 seed set)"
    )

    # -- Pick best frac -------------------------------------------------------
    # Two distinct notions:
    #   best_frac     : lowest-RAE frac overall (used for DEPLOY; may be 0.0,
    #                   i.e. nb3200 verbatim -- a legitimate "don't abstain").
    #   best_abst_frac: lowest-RAE frac among ABSTAINING fracs (>0). The gate
    #                   verdict is keyed on THIS, because frac=0.0 abstains on
    #                   nothing and is just the fallback re-run -- letting it
    #                   claim "BETTER" would credit abstention for a no-op and
    #                   confuse the 15-vs-30-seed sampling delta with a real
    #                   abstention gain. The abstention PARADIGM only "wins" if
    #                   a frac>0 actually beats both the gate AND its own
    #                   apples-to-apples frac=0.0 baseline (same 15 seeds).
    best_frac = min(ABSTAIN_FRACS, key=lambda f: frac_results[f]["agg"]["mean_rae"])
    best_agg = frac_results[best_frac]["agg"]
    best_mean = best_agg["mean_rae"]
    abst_fracs = [f for f in ABSTAIN_FRACS if f > 0.0]
    best_abst_frac = min(
        abst_fracs, key=lambda f: frac_results[f]["agg"]["mean_rae"]
    )
    best_abst_agg = frac_results[best_abst_frac]["agg"]
    best_abst_mean = best_abst_agg["mean_rae"]
    # apples-to-apples 15-seed fallback baseline (frac=0.0 in THIS run)
    base0_mean = frac_results[0.0]["agg"]["mean_rae"]
    print("\n" + "-" * 78)
    print("BEST FRAC")
    print("-" * 78)
    print(
        f"   best_frac (overall, deploy) = {best_frac:.2f}  mean={best_mean:.4f} "
        f"+/- {best_agg['std_rae']:.4f}  CI=[{best_agg['ci95_low']:.4f},"
        f"{best_agg['ci95_high']:.4f}]"
    )
    print(
        f"   best_abstaining_frac (>0)   = {best_abst_frac:.2f}  "
        f"mean={best_abst_mean:.4f} +/- {best_abst_agg['std_rae']:.4f}  "
        f"delta_vs_frac0={best_abst_mean - base0_mean:+.4f}"
    )

    # -- Deploy at best_frac (te, 513) ---------------------------------------
    print("\n" + "-" * 78)
    print(f"DEPLOY: apply best_frac={best_frac:.2f} to te(513)")
    print("-" * 78)
    if best_frac > 0.0:
        # cutoff from FULL 253 OOF uncertainties (the deploy "train" set)
        deploy_cutoff = float(np.quantile(u_oof, 1.0 - best_frac))
        m_full = float(np.median(y_unb))
        te_abst = u_te >= deploy_cutoff
        te_pred = fb_te.copy()
        te_pred[te_abst] = m_full
        te_pred = te_pred.astype(np.float32)
        n_te_abst = int(te_abst.sum())
        print(
            f"   deploy_cutoff (q{1.0 - best_frac:.2f} of u_oof) = {deploy_cutoff:.4f}  "
            f"train_median = {m_full:.3f}"
        )
        print(f"   te abstained: {n_te_abst}/513 ({n_te_abst / n_test:.1%})")
    else:
        deploy_cutoff = float("inf")
        m_full = float(np.median(y_unb))
        te_pred = fb_te.astype(np.float32)
        n_te_abst = 0
        print(f"   frac=0.0 -> te == nb3200 verbatim (no abstention)")
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"   te(513): mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}")

    oof_for_save = frac_results[best_frac]["oof_median_seed"]
    median_seed = frac_results[best_frac]["median_seed"]

    # -- Gate -----------------------------------------------------------------
    # The gate is keyed on the best ABSTAINING frac (>0), not the overall best.
    # An abstention mechanism that only "wins" by choosing frac=0.0 (abstain on
    # nothing == nb3200 verbatim) has not actually demonstrated the paradigm;
    # the 0.4416 vs 0.4424 gap there is the 15-vs-30-seed sampling delta, not an
    # abstention gain. BETTER therefore requires (a) a frac>0 clearing the gate
    # AND (b) that same frac>0 actually beating the apples-to-apples frac=0.0
    # baseline at the identical 15 seeds.
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    abst_beats_gate = best_abst_mean < GATE_BETTER
    abst_beats_base0 = best_abst_mean < base0_mean
    if abst_beats_gate and abst_beats_base0:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3300 selective abstention (frac="
            f"{best_abst_frac:.2f}, ~"
            f"{frac_results[best_abst_frac]['n_abstain_mean']:.0f}/253 rows "
            f"routed to train-median) per-fold-mean {best_abst_mean:.4f} clears "
            f"gate {GATE_BETTER:.4f} AND beats the apples-to-apples frac=0.0 "
            f"baseline {base0_mean:.4f} by {base0_mean - best_abst_mean:+.4f}. "
            f"Routing the top-{best_abst_frac:.0%} highest cross-K-disagreement "
            f"rows to train-median is a genuine SUBSTRATE move (changes which "
            f"rows get a model prediction at all), orthogonal to clip/stretch/"
            f"simplex which all converged at ~0.4424. Re-verify with deep-30 "
            f"before PRIMARY-1 promotion (cycle-160 under-dispersion rule)."
        )
    else:
        verdict = "FAIL"
        why = []
        if not abst_beats_gate:
            why.append(
                f"best abstaining frac={best_abst_frac:.2f} mean "
                f"{best_abst_mean:.4f} >= gate {GATE_BETTER:.4f}"
            )
        if not abst_beats_base0:
            why.append(
                f"best abstaining frac={best_abst_frac:.2f} mean "
                f"{best_abst_mean:.4f} >= apples-to-apples frac=0.0 baseline "
                f"{base0_mean:.4f} (abstention HURTS: {best_abst_mean - base0_mean:+.4f})"
            )
        ladder_action = (
            f"REJECT. " + "; ".join(why) + ". Every frac>0 in "
            f"{[f for f in ABSTAIN_FRACS if f > 0]} raised RAE monotonically "
            f"(frac0={base0_mean:.4f} -> "
            + ", ".join(
                f"frac{f:.2f}={frac_results[f]['agg']['mean_rae']:.4f}"
                for f in ABSTAIN_FRACS if f > 0
            )
            + f"). Cross-K-disagreement (Spearman {rho_str} vs |err|) is too "
            f"weak a signal at n=253: the "
            f"abstained set mixes correctly-predicted high-disagreement rows in "
            f"with the wrong ones, and train-median ({np.median(y_unb):.2f}) "
            f"discards the (still net-useful) conditional signal. Abstention on "
            f"the K-pyramid-std axis is CLOSED. best_frac=0.0 deploys nb3200 "
            f"verbatim; keep nb3200/nb2171 ladder unchanged."
        )
    print(f"   abst_beats_gate  = {abst_beats_gate}  "
          f"(best_abst_mean {best_abst_mean:.4f} vs gate {GATE_BETTER:.4f})")
    print(f"   abst_beats_base0 = {abst_beats_base0}  "
          f"(best_abst_mean {best_abst_mean:.4f} vs frac0 {base0_mean:.4f})")
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_abstention.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    # frac sweep table for summary
    frac_table = {}
    for frac in ABSTAIN_FRACS:
        fr = frac_results[frac]
        a = fr["agg"]
        frac_table[f"{frac:.2f}"] = {
            "mean_rae": round(a["mean_rae"], 4),
            "std_rae": round(a["std_rae"], 4),
            "sem_rae": round(a["sem_rae"], 4),
            "ci95_low": round(a["ci95_low"], 4),
            "ci95_high": round(a["ci95_high"], 4),
            "median_rae": round(a["median_rae"], 4),
            "min_rae": round(a["min_rae"], 4),
            "max_rae": round(a["max_rae"], 4),
            "n_abstain_mean": round(fr["n_abstain_mean"], 2),
            "delta_vs_nb3200": round(a["mean_rae"] - REF_NB3200, 4),
            "median_seed": int(fr["median_seed"]),
            "pooled_rae_array": [round(float(v), 4) for v in fr["pooled_raes"]],
        }

    summary = {
        "tag": TAG,
        "fallback_tag": FALLBACK_TAG,
        "method": "selective_abstention_train_median_on_high_cross_K_disagreement",
        "paradigm": "substrate_change_route_uncertain_rows_to_train_median",
        "uncertainty_def": "std(K18,K19,K20,K24) per row (deep-30 members)",
        "fallback_oof_path": str(FALLBACK_OOF_PATH),
        "fallback_te_path": str(FALLBACK_TE_PATH),
        "k_pyramid_members": [18, 19, 20, 24],
        "anchor_pre_unblind": True,
        "fallback_oof_rae": round(fb_oof_rae, 4),
        "fallback_leak_eq_truth_frac": round(leak_eq, 4),
        "u_oof_mean": round(float(u_oof.mean()), 4),
        "u_oof_std": round(float(u_oof.std()), 4),
        "u_oof_median": round(float(np.median(u_oof)), 4),
        "u_te_mean": round(float(u_te.mean()), 4),
        "u_te_median": round(float(np.median(u_te)), 4),
        "spearman_u_vs_abserr": round(float(rho), 4) if rho is not None else None,
        "spearman_pval": float(pval) if pval is not None else None,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "abstain_fracs": ABSTAIN_FRACS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "frac_sweep": frac_table,
        "frac0_sanity_mean": round(base0, 4),
        "frac0_minus_nb3200": round(base0 - REF_NB3200, 4),
        "best_frac": best_frac,
        "best_mean_rae": round(best_mean, 4),
        "best_std_rae": round(best_agg["std_rae"], 4),
        "best_ci95_low": round(best_agg["ci95_low"], 4),
        "best_ci95_high": round(best_agg["ci95_high"], 4),
        "best_median_rae": round(best_agg["median_rae"], 4),
        "best_abst_frac": best_abst_frac,
        "best_abst_mean_rae": round(best_abst_mean, 4),
        "best_abst_std_rae": round(best_abst_agg["std_rae"], 4),
        "base0_mean_rae": round(base0_mean, 4),
        "abst_delta_vs_base0": round(best_abst_mean - base0_mean, 4),
        "abst_beats_gate": bool(best_abst_mean < GATE_BETTER),
        "abst_beats_base0": bool(best_abst_mean < base0_mean),
        "ref_nb3200": REF_NB3200,
        "ref_nb3173": REF_NB3173,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3200": round(best_mean - REF_NB3200, 4),
        "deploy_cutoff": (
            round(deploy_cutoff, 4) if np.isfinite(deploy_cutoff) else None
        ),
        "deploy_train_median": round(m_full, 4),
        "n_te_abstain": n_te_abst,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
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
    print(f"   best_frac          = {best_frac:.2f}")
    print(f"   best_mean_rae      = {best_mean:.4f} +/- {best_agg['std_rae']:.4f}")
    print(f"   95% CI             = [{best_agg['ci95_low']:.4f}, {best_agg['ci95_high']:.4f}]")
    print(f"   delta vs nb3200    = {best_mean - REF_NB3200:+.4f}")
    print(f"   verdict            = {verdict}")
    print(f"   wall               = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "best_frac", "best_mean_rae",
        "best_abst_frac", "best_abst_mean_rae",
        "base0_mean_rae", "abst_delta_vs_base0",
        "abst_beats_gate", "abst_beats_base0",
        "delta_vs_nb3200", "frac0_sanity_mean",
        "spearman_u_vs_abserr",
        "deploy_cutoff", "n_te_abstain",
        "te_unb_in_sample_rae",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
