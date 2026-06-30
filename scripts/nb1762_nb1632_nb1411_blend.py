"""nb1762 -- Try nb1632 BoB + nb1411 POST blend with small w_post.

CONTEXT
    The 3-way blend nb1742 (nb1632 + nb1571 + nb1561) stayed within margin of
    nb1632 BoB (0.5107). nb1411 was previously declined because the SLSQP
    cross-fit 3-way (AtomPair-30 + MACCS-20 + Mordred-30) only reached 0.5045
    -- below the 0.003 sub-margin gate vs nb1632.

    Here we re-test the smallest possible mix: anchor at nb1632 BoB mean
    (0.5107) and try only small post-component weights w_post in
    {0.10, 0.15, 0.20, 0.25, 0.30}. Goal: catch a decompression/orthogonality
    gain that earlier sub-margin 3-way analysis missed.

COMPONENTS (both 253-row honest cross-fit OOFs)
    p_anchor = nb1632_bob_mean_oof.npy        RAE ~ 0.5107  (PRE-unblind)
    p_post   = nb1411_best_oof.npy            RAE ~ 0.5045  (POST-unblind)

    NOTE: p_post is the SLSQP-5fold-cross-fit OOF from nb1411 (best variant
    saved). The naive 1/3 referenced in the prompt has RAE ~ 0.5037 but is
    NOT separately saved; the saved best is the SLSQP cross-fit. Both are
    POST-unblind variants of the same AtomPair/MACCS/Mordred recipe.

PROTOCOL
    blend = (1 - w_post) * p_anchor  +  w_post * p_post
    w_post grid : {0.10, 0.15, 0.20, 0.25, 0.30}
    Verdict at margin 0.003 vs nb1632 BoB mean (0.5107).

POST-UNBLIND LB TRANSFER RISK
    nb1411 is trained against the 253 unblind labels (POST-unblind). Past
    rule (feedback_lb_two_regime_calibration): POST-unblind OOFs do NOT
    transfer 1:1 to LB; in-sample RAE is optimistic relative to actual LB.
    Any blend that depends on POST-unblind weight contributes commensurate
    LB-shift risk. Even at w_post = 0.30 the contribution to LB is
    speculative: best case roughly tracks the cross-fit RAE; worst case adds
    +0.05 to +0.10 LB shift on the post component. This script is a
    cross-fit-only diagnostic; LB deployment is gated downstream.

OUTPUTS
    data/processed/nb1762_summary.json
    (no submission, no te_*.npy -- diagnostic only)
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

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1762"

NB1632_BOB_REF = 0.5107
MARGIN = 0.003
W_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- nb1632 BoB anchor + nb1411 POST blend  (small w_post grid)")
    print(f"         w_post in {W_GRID}")
    print(f"         target margin {MARGIN:.3f} vs nb1632 BoB {NB1632_BOB_REF:.4f}")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    p_anchor_path = DATA_PROCESSED / "nb1632_bob_mean_oof.npy"
    p_post_path = DATA_PROCESSED / "nb1411_best_oof.npy"

    for p in (p_anchor_path, p_post_path):
        if not p.exists():
            raise FileNotFoundError(p)

    p_anchor = np.load(p_anchor_path).astype(np.float64)
    p_post = np.load(p_post_path).astype(np.float64)

    if p_anchor.shape[0] != n_unb:
        raise ValueError(
            f"nb1632 shape mismatch: {p_anchor.shape}, n_unb={n_unb}"
        )
    if p_post.shape[0] != n_unb:
        raise ValueError(
            f"nb1411 shape mismatch: {p_post.shape}, n_unb={n_unb}"
        )

    rae_anchor = float(rae(y_unb, p_anchor))
    rae_post = float(rae(y_unb, p_post))
    print(f"[load] nb1632 BoB mean   : RAE = {rae_anchor:.4f}  "
          f"(ref {NB1632_BOB_REF:.4f})")
    print(f"[load] nb1411 best       : RAE = {rae_post:.4f}  "
          f"(POST-unblind, LB-transfer risk)")

    e_anchor = p_anchor - y_unb
    e_post = p_post - y_unb
    corr_pred = float(np.corrcoef(p_anchor, p_post)[0, 1])
    corr_resid = float(np.corrcoef(e_anchor, e_post)[0, 1])
    print(f"[diag] pred  Pearson(nb1632, nb1411) = {corr_pred:.4f}")
    print(f"[diag] resid Pearson(nb1632, nb1411) = {corr_resid:.4f}")

    # w_post grid sweep.
    print("\n" + "-" * 78)
    print("w_post grid sweep")
    print("-" * 78)
    grid_rows = []
    for w_post in W_GRID:
        blend = (1.0 - w_post) * p_anchor + w_post * p_post
        rae_blend = float(rae(y_unb, blend))
        delta = rae_blend - NB1632_BOB_REF
        beats = rae_blend < NB1632_BOB_REF - MARGIN
        grid_rows.append({
            "w_post": w_post,
            "w_anchor": 1.0 - w_post,
            "rae": rae_blend,
            "delta_vs_nb1632": delta,
            "beats_nb1632_margin": bool(beats),
        })
        flag = "BEATS_MARGIN" if beats else ("FLAT" if abs(delta) < MARGIN
                                              else "HURTS")
        print(f"   w_post = {w_post:.2f}  RAE = {rae_blend:.4f}  "
              f"d_vs_nb1632 = {delta:+.4f}  [{flag}]")

    # Best w in the small grid.
    best_idx = int(np.argmin([r["rae"] for r in grid_rows]))
    best_w_post = grid_rows[best_idx]["w_post"]
    best_rae = grid_rows[best_idx]["rae"]
    best_delta = grid_rows[best_idx]["delta_vs_nb1632"]
    beats_nb1632 = best_rae < NB1632_BOB_REF - MARGIN
    flat_vs_nb1632 = abs(best_rae - NB1632_BOB_REF) < MARGIN

    if beats_nb1632:
        verdict = (f"NB1762_BLEND_BEATS_NB1632_BY_MARGIN  "
                   f"(best w_post = {best_w_post:.2f})")
    elif flat_vs_nb1632:
        verdict = (f"NB1762_BLEND_FLAT_VS_NB1632  "
                   f"(best w_post = {best_w_post:.2f})")
    else:
        verdict = (f"NB1762_BLEND_HURTS_VS_NB1632  "
                   f"(best w_post = {best_w_post:.2f})")

    # Verdict.
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1632 BoB mean standalone   : {rae_anchor:.4f}")
    print(f"   nb1411 best  standalone      : {rae_post:.4f}  (POST-unblind)")
    print(f"   best w_post                  : {best_w_post:.2f}")
    print(f"   best RAE                     : {best_rae:.4f}")
    print(f"   delta vs nb1632 BoB          : {best_delta:+.4f}  "
          f"(gate {MARGIN:.3f})")
    print(f"   beats_nb1632                 : {beats_nb1632}")
    print(f"   flat_vs_nb1632               : {flat_vs_nb1632}")
    print(f"   verdict                      : {verdict}")
    print("\n   NOTE: nb1411 is POST-unblind. Even with best w_post, LB transfer")
    print("         from this blend carries non-zero shift risk on the post")
    print("         contribution (per feedback_lb_two_regime_calibration).")
    print("         Do not deploy without a separate POST-unblind LB sanity")
    print("         check; this script is cross-fit-diagnostic only.")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "anchor": "nb1632_bob_mean_oof",
        "post_component": "nb1411_best_oof",
        "rae_nb1632_anchor": rae_anchor,
        "rae_nb1411_post": rae_post,
        "pred_pearson_anchor_post": corr_pred,
        "residual_pearson_anchor_post": corr_resid,
        "w_grid": list(W_GRID),
        "grid_rows": grid_rows,
        "best_w_post": float(best_w_post),
        "best_rae": float(best_rae),
        "delta_best_vs_nb1632": float(best_delta),
        "beats_nb1632": bool(beats_nb1632),
        "flat_vs_nb1632": bool(flat_vs_nb1632),
        "verdict": verdict,
        "nb1632_bob_ref": NB1632_BOB_REF,
        "margin": MARGIN,
        "post_unblind_lb_transfer_risk": (
            "nb1411 trained against 253 unblind labels; in-sample / cross-fit "
            "RAE is optimistic relative to LB. Best-case LB tracks cross-fit; "
            "worst-case adds +0.05 to +0.10 LB shift on the post contribution. "
            "Use cross-fit verdict as diagnostic only, not as LB estimate."
        ),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("rae_nb1632_anchor", "rae_nb1411_post",
              "pred_pearson_anchor_post", "residual_pearson_anchor_post",
              "grid_rows",
              "best_w_post", "best_rae",
              "delta_best_vs_nb1632",
              "beats_nb1632", "flat_vs_nb1632",
              "verdict"):
        print(f"  {k}: {res.get(k)}")
