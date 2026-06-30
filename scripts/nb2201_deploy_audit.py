"""nb2201 -- DEPLOY AUDIT for nb2189 hybrid mismatch.

CONTEXT (cycle 126/127):
    nb2189 K=20 honest 5-fold cross-fit RAE 0.4556 used nb562_pred_oof as
    anchor on 253-unblind substrate.  But the deploy CSV uses
    te_chemprop_aux as the 513-row anchor PLUS a residual LGBM trained
    against the nb562 residual.  Hybrid mismatch.

PROTOCOL:
    1. Load nb562_pred_oof.npy (253,), te_chemprop_aux.npy (513,),
       submissions/nb2189_deploy_truly_honest.csv (513,).
    2. Verify the deploy CSV's 253-piece (rows where Molecule Name in
       unb_idx) equals:  te_chemprop_aux[unb_idx] + LGBM_residual_te[unb_idx].
       Reconstruct the LGBM residual from te_nb2189.npy - te_chemprop_aux.
    3. Compute the implied 253-RAE of the deploy CSV (in-sample, since
       the residual LGBM was refit on ALL 253 residuals).
    4. Compare to:
         (a) nb2189 K=20 cross-fit median-bag RAE = 0.4556
         (b) nb2103 K=28 median-bag                = 0.4698
         (c) chemprop_aux alone on [unb_idx]
    5. Key question: is the deploy's 253-piece closer to honest 0.4556
       or a contaminated baseline (~0.42 in-sample optimism)?
    6. Anchor drift:
         nb562_pred_oof.mean() = 4.6862
         te_chemprop_aux[unb_idx].mean() = 4.7967
         drift = +0.1105
       What is the implied per-row offset error from this drift?
    7. VERDICT: HYBRID_VALID | HYBRID_BROKEN | NEEDS_REBUILD.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2201"
N_TEST = 513
N_UNB = 253

# Reference numbers from nb2189 summary (verified)
NB2189_CROSSFIT_RAE = 0.45558505824626144   # K=20 median-bag honest
NB2103_K28_MEDIAN_REF = 0.4698              # honest gate
NB562_OOF_RAE = 0.5065                      # anchor alone (honest)
NB2189_BEATS_GATE_DELTA = -0.014215

# Tolerance for VALID verdict
RECONSTRUCTION_TOL = 1e-4          # te=anchor+resid arithmetic
DRIFT_OFFSET_TOL = 0.005           # implied per-row offset on RAE


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def main() -> dict:
    P = DATA_PROCESSED
    print("=" * 78)
    print(f"{TAG} -- DEPLOY AUDIT: nb2189 hybrid (chemprop_aux + nb562-resid)")
    print("=" * 78)

    # ---- Load anchors ----
    nb562_oof = np.load(P / "nb562_pred_oof.npy").astype(np.float64)
    te_chemprop = np.load(P / "te_chemprop_aux.npy").astype(np.float64)
    te_nb2189 = np.load(P / "te_nb2189.npy").astype(np.float64)
    unb_idx = np.load(P / "_audit_unblind_idx.npy")
    y_unb = np.load(P / "_audit_unblind_y.npy").astype(np.float64)

    assert nb562_oof.shape == (N_UNB,), nb562_oof.shape
    assert te_chemprop.shape == (N_TEST,), te_chemprop.shape
    assert te_nb2189.shape == (N_TEST,), te_nb2189.shape
    assert unb_idx.shape == (N_UNB,), unb_idx.shape
    assert y_unb.shape == (N_UNB,), y_unb.shape

    print(f"[load] nb562_pred_oof  sha={_sha256(nb562_oof)}  "
          f"RAE={rae(y_unb, nb562_oof):.4f}")
    print(f"[load] te_chemprop_aux sha={_sha256(te_chemprop)}")
    print(f"[load] te_nb2189       sha={_sha256(te_nb2189)}")
    print(f"[load] y_unb           sha={_sha256(y_unb)}")

    # ---- Load deploy CSV ----
    deploy_csv = (Path(__file__).resolve().parents[1] /
                  "submissions" / "nb2189_deploy_truly_honest.csv")
    df = pd.read_csv(deploy_csv)
    print(f"[load] CSV rows={len(df)}  cols={df.columns.tolist()}")
    assert len(df) == N_TEST, len(df)

    # CSV pEC50 column on 513
    csv_pred_513 = df["pEC50"].to_numpy(dtype=np.float64)

    # Reconstruction check #1: CSV should equal te_nb2189 exactly
    recon_diff = np.abs(csv_pred_513 - te_nb2189)
    recon_max = float(recon_diff.max())
    recon_mean = float(recon_diff.mean())
    print(f"\n[recon-csv-vs-te_nb2189] max|d|={recon_max:.2e}  "
          f"mean|d|={recon_mean:.2e}")

    # ---- Decompose te_nb2189 = te_chemprop_aux + residual_LGBM_te ----
    resid_lgbm_te = te_nb2189 - te_chemprop
    print(f"\n[decomp] resid_lgbm_te:  mean={resid_lgbm_te.mean():+.4f}  "
          f"std={resid_lgbm_te.std():.4f}  "
          f"range=[{resid_lgbm_te.min():+.3f}, {resid_lgbm_te.max():+.3f}]")

    # ---- 253-piece of deploy CSV ----
    csv_pred_253 = csv_pred_513[unb_idx]
    chemprop_253 = te_chemprop[unb_idx]
    resid_te_253 = resid_lgbm_te[unb_idx]

    # Verify additive decomposition on 253
    recon_253_diff = np.abs(csv_pred_253 - (chemprop_253 + resid_te_253))
    recon_253_max = float(recon_253_diff.max())
    print(f"[decomp] csv[unb] = chemprop[unb] + resid_te[unb]: "
          f"max|d|={recon_253_max:.2e}")

    # ---- Compute implied 253-RAE of deploy ----
    rae_csv_253 = float(rae(y_unb, csv_pred_253))
    rae_chemprop_253 = float(rae(y_unb, chemprop_253))
    rae_nb562_oof = float(rae(y_unb, nb562_oof))

    print(f"\n[rae-253] csv[unb_idx]         = {rae_csv_253:.4f}  "
          f"(IN-SAMPLE: LGBM refit on all 253)")
    print(f"[rae-253] te_chemprop[unb_idx] = {rae_chemprop_253:.4f}  "
          f"(PRE-unblind, honest LB)")
    print(f"[rae-253] nb562_pred_oof       = {rae_nb562_oof:.4f}  "
          f"(honest 5-fold)")

    # ---- Comparisons ----
    print("\n" + "-" * 78)
    print("COMPARISON TO HONEST BENCHMARKS")
    print("-" * 78)
    gap_vs_2189_crossfit = rae_csv_253 - NB2189_CROSSFIT_RAE
    gap_vs_2103 = rae_csv_253 - NB2103_K28_MEDIAN_REF
    gap_vs_562 = rae_csv_253 - NB562_OOF_RAE
    gap_vs_chemprop = rae_csv_253 - rae_chemprop_253

    print(f"  deploy CSV [unb_idx] RAE             = {rae_csv_253:.4f}")
    print(f"  nb2189 K=20 honest cross-fit (a)     = {NB2189_CROSSFIT_RAE:.4f}  "
          f"gap = {gap_vs_2189_crossfit:+.4f}")
    print(f"  nb2103 K=28 median-bag (b)           = {NB2103_K28_MEDIAN_REF:.4f}  "
          f"gap = {gap_vs_2103:+.4f}")
    print(f"  nb562_pred_oof anchor                = {NB562_OOF_RAE:.4f}  "
          f"gap = {gap_vs_562:+.4f}")
    print(f"  chemprop_aux alone [unb_idx]         = {rae_chemprop_253:.4f}  "
          f"gap = {gap_vs_chemprop:+.4f}")

    # closer-to verdict
    if abs(gap_vs_2189_crossfit) < abs(gap_vs_chemprop):
        closer_to = "honest_2189_crossfit"
    else:
        closer_to = "chemprop_aux_baseline"
    print(f"\n  CSV [unb_idx] is CLOSER TO: {closer_to}")

    # ---- Anchor drift analysis ----
    print("\n" + "-" * 78)
    print("ANCHOR DRIFT")
    print("-" * 78)
    mean_nb562_oof = float(nb562_oof.mean())
    mean_chemprop_253 = float(chemprop_253.mean())
    anchor_drift = mean_chemprop_253 - mean_nb562_oof
    print(f"  nb562_pred_oof.mean()             = {mean_nb562_oof:.4f}")
    print(f"  te_chemprop_aux[unb_idx].mean()   = {mean_chemprop_253:.4f}")
    print(f"  drift = chemprop - nb562          = {anchor_drift:+.4f}")

    # Implied per-row offset: if residual LGBM expects anchor ~ 4.6862
    # but is applied on top of anchor ~ 4.7967, every CSV pred is shifted
    # by +drift relative to what nb2189 cross-fit estimated.
    # Per-row offset propagates into RAE as a mean-error term.
    mean_y_unb = float(y_unb.mean())
    mean_residual = float((y_unb - nb562_oof).mean())
    mean_resid_lgbm_te_unb = float(resid_te_253.mean())

    # In the cross-fit world (nb2189): pred = nb562_oof + resid_oof,
    #   mean(pred) approx mean(nb562_oof) + mean(resid_oof) approx mean(y_unb)
    # In deploy: pred = chemprop + resid_te = chemprop + (similar magnitude
    #   residual) -> mean approx mean(chemprop) + mean(resid_te)
    mean_csv_253 = float(csv_pred_253.mean())
    drift_pred_minus_truth = mean_csv_253 - mean_y_unb
    print(f"  mean(y_unb)                        = {mean_y_unb:.4f}")
    print(f"  mean(residual = y_unb - nb562_oof) = {mean_residual:+.4f}")
    print(f"  mean(resid_lgbm_te[unb_idx])       = "
          f"{mean_resid_lgbm_te_unb:+.4f}")
    print(f"  mean(csv_pred_253)                 = {mean_csv_253:.4f}")
    print(f"  mean(csv_pred_253) - mean(y_unb)   = "
          f"{drift_pred_minus_truth:+.4f}  "
          f"<-- implied per-row mean offset error")

    # RAE impact estimate: an additive shift `b` adds approximately
    # |b| / mean(|y - median(y)|) to RAE
    mad_y = float(np.mean(np.abs(y_unb - np.median(y_unb))))
    implied_rae_from_drift = abs(drift_pred_minus_truth) / mad_y
    print(f"  MAD(y_unb)                         = {mad_y:.4f}")
    print(f"  implied dRAE from shift            = "
          f"{implied_rae_from_drift:.4f}  (|drift| / MAD)")

    # ---- Counterfactual: what RAE would deploy have if anchor were nb562 ----
    # We CAN'T reconstruct what LGBM would predict for nb562's 253-row pos
    # without re-fitting, but we can apply the same te_resid on nb562:
    # pred_if_anchor_were_nb562 = nb562_oof + resid_lgbm_te[unb_idx]
    pred_nb562_plus_resid_te = nb562_oof + resid_te_253
    rae_nb562_plus_resid_te = float(rae(y_unb, pred_nb562_plus_resid_te))
    print(f"\n  counterfactual: nb562_oof + resid_te[unb_idx]    = "
          f"{rae_nb562_plus_resid_te:.4f}")
    print(f"  (this is what the deploy *would* score if the deploy"
          f"\n   anchor matched the training-time anchor exactly)")

    # ---- VERDICT ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  reconstruction (CSV = te_nb2189): max|d| = "
          f"{recon_max:.2e}  ok? {recon_max < RECONSTRUCTION_TOL}")
    print(f"  additive (te = chemprop + resid):  max|d| = "
          f"{recon_253_max:.2e}  ok? {recon_253_max < RECONSTRUCTION_TOL}")
    print(f"  closer-to                          : {closer_to}")
    print(f"  in-sample csv-vs-truth shift       : "
          f"{drift_pred_minus_truth:+.4f}")
    print(f"  implied RAE penalty from drift     : "
          f"{implied_rae_from_drift:.4f}")
    print(f"  honest 2189 crossfit RAE           : "
          f"{NB2189_CROSSFIT_RAE:.4f}")
    print(f"  IN-SAMPLE deploy CSV RAE [unb]     : {rae_csv_253:.4f}")

    # decision logic
    # HYBRID_VALID  : in-sample RAE matches what we'd expect from training
    #                 on full 253 (should be LOWER than honest cross-fit,
    #                 ~ -0.05 to -0.10), AND drift offset is small.
    # HYBRID_BROKEN : in-sample RAE is WORSE than honest 0.4556 by >+0.02,
    #                 indicating anchor-drift mis-aligns the residual.
    # NEEDS_REBUILD : in-sample matches honest closely (no gain), suggesting
    #                 anchor swap nullified the LGBM correction.
    if recon_max >= RECONSTRUCTION_TOL or recon_253_max >= RECONSTRUCTION_TOL:
        verdict = "NEEDS_REBUILD"
        reason = "reconstruction broken -- decomposition mismatch"
    elif rae_csv_253 > NB2189_CROSSFIT_RAE + 0.02:
        verdict = "HYBRID_BROKEN"
        reason = (f"in-sample CSV RAE {rae_csv_253:.4f} is WORSE than "
                  f"honest cross-fit {NB2189_CROSSFIT_RAE:.4f} by "
                  f"+{rae_csv_253 - NB2189_CROSSFIT_RAE:.4f} -- "
                  f"anchor drift {anchor_drift:+.4f} mis-aligns residual")
    elif abs(rae_csv_253 - NB2189_CROSSFIT_RAE) < 0.01 and \
            abs(rae_csv_253 - rae_chemprop_253) < 0.02:
        verdict = "NEEDS_REBUILD"
        reason = (f"in-sample CSV RAE {rae_csv_253:.4f} approx "
                  f"chemprop alone {rae_chemprop_253:.4f} -- "
                  f"residual LGBM contributing little new signal")
    elif rae_csv_253 < NB2189_CROSSFIT_RAE - 0.02:
        # in-sample optimism, but check drift impact
        if implied_rae_from_drift > DRIFT_OFFSET_TOL * 3:
            verdict = "HYBRID_BROKEN"
            reason = (f"in-sample looks great ({rae_csv_253:.4f}) BUT "
                      f"implied drift penalty {implied_rae_from_drift:.4f} "
                      f"will bite on LB -- mean offset "
                      f"{drift_pred_minus_truth:+.4f}")
        else:
            verdict = "HYBRID_VALID"
            reason = (f"in-sample optimism expected; drift impact "
                      f"{implied_rae_from_drift:.4f} below tolerance; "
                      f"counterfactual nb562+resid = "
                      f"{rae_nb562_plus_resid_te:.4f} aligns w/ honest")
    else:
        # roughly tracks honest 0.4556 +/- 0.02
        if implied_rae_from_drift > DRIFT_OFFSET_TOL * 3:
            verdict = "HYBRID_BROKEN"
            reason = (f"in-sample {rae_csv_253:.4f} tracks honest "
                      f"{NB2189_CROSSFIT_RAE:.4f} but drift penalty "
                      f"{implied_rae_from_drift:.4f} is large -- LB will "
                      f"shift up by ~{implied_rae_from_drift:.3f}")
        else:
            verdict = "HYBRID_VALID"
            reason = (f"in-sample {rae_csv_253:.4f} tracks honest "
                      f"{NB2189_CROSSFIT_RAE:.4f}; drift small "
                      f"({implied_rae_from_drift:.4f})")

    print(f"\n  ===> VERDICT: {verdict}")
    print(f"       reason: {reason}")

    # Predicted LB band: chemprop_aux PRE-unblind LB = 0.6246 (memory).
    # If hybrid valid, LB approx 0.6246 + (rae_csv_253 - rae_chemprop_253).
    # If hybrid broken (anchor drift), add implied_rae_from_drift.
    chemprop_aux_lb = 0.6246
    delta_resid_on_253 = rae_csv_253 - rae_chemprop_253
    lb_estimate_valid = chemprop_aux_lb + delta_resid_on_253
    lb_estimate_broken = lb_estimate_valid + implied_rae_from_drift
    print(f"\n  LB ESTIMATES:")
    print(f"    chemprop_aux alone (memory)          = {chemprop_aux_lb:.4f}")
    print(f"    residual delta on 253                = "
          f"{delta_resid_on_253:+.4f}")
    print(f"    LB if HYBRID_VALID                   = "
          f"{lb_estimate_valid:.4f}")
    print(f"    LB if HYBRID_BROKEN (+drift)         = "
          f"{lb_estimate_broken:.4f}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "audited_csv": str(deploy_csv),
        "audited_te_path": str(P / "te_nb2189.npy"),
        "anchor_oof_path": str(P / "nb562_pred_oof.npy"),
        "anchor_te_path": str(P / "te_chemprop_aux.npy"),
        "n_unb": N_UNB,
        "n_test": N_TEST,
        "shas": {
            "nb562_oof": _sha256(nb562_oof),
            "te_chemprop": _sha256(te_chemprop),
            "te_nb2189": _sha256(te_nb2189),
            "y_unb": _sha256(y_unb),
        },
        "reconstruction": {
            "csv_eq_te_nb2189_max_abs_diff": recon_max,
            "csv_eq_te_nb2189_mean_abs_diff": recon_mean,
            "additive_te_eq_chemprop_plus_resid_max": recon_253_max,
            "tolerance": RECONSTRUCTION_TOL,
        },
        "residual_lgbm_te_stats": {
            "mean_513": float(resid_lgbm_te.mean()),
            "std_513": float(resid_lgbm_te.std()),
            "min_513": float(resid_lgbm_te.min()),
            "max_513": float(resid_lgbm_te.max()),
            "mean_unb": mean_resid_lgbm_te_unb,
            "std_unb": float(resid_te_253.std()),
        },
        "rae_253": {
            "csv_unb_idx_IN_SAMPLE": rae_csv_253,
            "chemprop_aux_unb_idx": rae_chemprop_253,
            "nb562_pred_oof": rae_nb562_oof,
            "counterfactual_nb562_plus_resid_te": rae_nb562_plus_resid_te,
        },
        "comparisons_vs_csv_253": {
            "gap_vs_nb2189_crossfit_0_4556": gap_vs_2189_crossfit,
            "gap_vs_nb2103_K28_median_0_4698": gap_vs_2103,
            "gap_vs_nb562_oof_0_5065": gap_vs_562,
            "gap_vs_chemprop_aux_unb": gap_vs_chemprop,
            "closer_to": closer_to,
        },
        "anchor_drift": {
            "mean_nb562_pred_oof": mean_nb562_oof,
            "mean_chemprop_unb": mean_chemprop_253,
            "drift_chemprop_minus_nb562": anchor_drift,
            "mean_y_unb": mean_y_unb,
            "mean_csv_pred_253": mean_csv_253,
            "mean_csv_minus_y": drift_pred_minus_truth,
            "MAD_y_unb": mad_y,
            "implied_dRAE_from_drift": implied_rae_from_drift,
            "drift_tolerance": DRIFT_OFFSET_TOL,
        },
        "lb_estimates": {
            "chemprop_aux_lb_reference": chemprop_aux_lb,
            "delta_resid_on_unb": delta_resid_on_253,
            "lb_if_hybrid_valid": lb_estimate_valid,
            "lb_if_hybrid_broken_with_drift": lb_estimate_broken,
        },
        "references": {
            "nb2189_K20_median_crossfit": NB2189_CROSSFIT_RAE,
            "nb2103_K28_median": NB2103_K28_MEDIAN_REF,
            "nb562_pred_oof_RAE": NB562_OOF_RAE,
        },
        "verdict": verdict,
        "verdict_reason": reason,
    }
    out_path = P / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] summary -> {out_path}")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== FINAL ====")
    for k in ("verdict", "verdict_reason"):
        print(f"  {k}: {res.get(k)}")
