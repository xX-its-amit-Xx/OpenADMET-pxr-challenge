"""nb503 -- HEDGE BLEND: pairwise nb492 + each {nb500, nb501, nb502}, plus
4-way SLSQP cross-fit over {nb492, nb500, nb501, nb502}.

Honest cross-fit RAE on the 253 unblind rows.

Procedure:
  A) Pairwise hedge: for each candidate C in {nb500, nb501, nb502} and each
     w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        blend_oof = (1-w) * nb492_pred_oof + w * C_pred_oof
     Pick best w per candidate by RAE on truth.
  B) 4-way SLSQP simplex blend over {nb492, nb500, nb501, nb502} pred_oofs,
     5-fold KFold cross-fit (SEED=0). Pooled OOF RAE = honest cross-fit RAE.

Deploy: whichever (best pairwise or 4-way SLSQP) has the lowest cross-fit RAE.
Save te_nb503.npy + plain + soft07_truth submissions.

Hyperparams mirror nb493:
    SLSQP simplex on MAE, ftol=1e-9, maxiter=500.
    KFold n_splits=5, shuffle=True, random_state=0.
    SOFT_W = 0.7 for the truth-soft-inject submission.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb503"

HEDGE_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

ANCHOR_TAG = "nb492"
CANDIDATES = ["nb500", "nb501", "nb502"]
POOL = [ANCHOR_TAG] + CANDIDATES  # 4-way order


def _fit_slsqp(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simplex weights (>=0, sum=1) minimising MAE of P @ w vs y."""
    k = P.shape[1]
    if k == 1:
        return np.array([1.0])
    w0 = np.full(k, 1.0 / k)

    def loss(w):
        return float(np.mean(np.abs(y - P @ w)))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(
        loss, w0, method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-9, "maxiter": 500, "disp": False},
    )
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return np.full(k, 1.0 / k) if s <= 0 else w / s


def main() -> dict:
    print("=" * 78)
    print("nb503 -- HEDGE BLEND (pairwise + 4-way SLSQP)")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for tag in POOL:
        needed[f"{tag}_pred_oof.npy"] = DATA_PROCESSED / f"{tag}_pred_oof.npy"
        needed[f"te_{tag}.npy"]       = DATA_PROCESSED / f"te_{tag}.npy"
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---- Indices ----
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- Load OOFs and deploys ----
    oof = {}
    te_vec = {}
    print("\nPool members (standalone RAE on 253):")
    for tag in POOL:
        a = np.load(DATA_PROCESSED / f"{tag}_pred_oof.npy").astype(np.float32)
        d = np.load(DATA_PROCESSED / f"te_{tag}.npy").astype(np.float32)
        assert a.shape == (n_unb,), f"{tag} oof shape {a.shape}"
        assert d.shape == (n_te,),  f"{tag} te shape {d.shape}"
        oof[tag] = a
        te_vec[tag] = d
        r_alone = float(rae(unb_y, a))
        print(f"  {tag}: oof_RAE={r_alone:.4f}  std={a.std():.3f}  "
              f"te std={d.std():.3f}")

    nb492_oof = oof[ANCHOR_TAG]
    nb492_te  = te_vec[ANCHOR_TAG]
    rae_anchor_oof = float(rae(unb_y, nb492_oof))

    # =================================================================
    # (A) Pairwise hedge per candidate
    # =================================================================
    print("\n" + "-" * 78)
    print(f"(A) PAIRWISE HEDGE  ({ANCHOR_TAG} vs each candidate)")
    print("-" * 78)
    pair_summary: dict[str, dict] = {}
    for c in CANDIDATES:
        c_oof = oof[c]
        per_w = []
        for w in HEDGE_GRID:
            blend = (1.0 - w) * nb492_oof + w * c_oof
            r = float(rae(unb_y, blend))
            per_w.append((w, r))
        # pick best
        best_w, best_r = min(per_w, key=lambda t: t[1])
        gain = rae_anchor_oof - best_r
        pair_summary[c] = {
            "best_w": float(best_w),
            "best_rae": float(best_r),
            "gain": float(gain),
            "grid": per_w,
        }
        grid_str = " ".join(f"w={w:.1f}:{r:.4f}" for w, r in per_w)
        print(f"  {c}: {grid_str}")
        print(f"    -> best w={best_w:.2f}  RAE={best_r:.4f}  "
              f"gain vs nb492={gain:+.4f}")

    best_pair_tag = min(pair_summary, key=lambda c: pair_summary[c]["best_rae"])
    best_pair = pair_summary[best_pair_tag]
    print(f"\nBest pairwise: nb492+{best_pair_tag}  "
          f"w={best_pair['best_w']:.2f}  RAE={best_pair['best_rae']:.4f}")

    # =================================================================
    # (B) 4-way SLSQP cross-fit
    # =================================================================
    print("\n" + "-" * 78)
    print(f"(B) 4-WAY SLSQP CROSS-FIT  ({POOL})")
    print("-" * 78)
    P_oof = np.stack([oof[t] for t in POOL], axis=1).astype(np.float64)  # (253,4)
    Q_te  = np.stack([te_vec[t] for t in POOL], axis=1).astype(np.float64)  # (513,4)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_weights = []
    fold_raes = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        w = _fit_slsqp(P_oof[tr_i], unb_y[tr_i])
        oof_blend[va_i] = P_oof[va_i] @ w
        r_va = float(rae(unb_y[va_i], oof_blend[va_i]))
        fold_weights.append(w)
        fold_raes.append(r_va)
        wstr = " ".join(f"{t}={wi:.3f}" for t, wi in zip(POOL, w))
        print(f"  fold {fold}: n_tr={len(tr_i):3d} n_va={len(va_i):3d}  "
              f"RAE={r_va:.4f}  [{wstr}]")
    slsqp_rae = float(rae(unb_y, oof_blend))
    fold_lo, fold_hi = float(min(fold_raes)), float(max(fold_raes))
    print(f"\nPooled 4-way SLSQP cross-fit RAE = {slsqp_rae:.4f}  "
          f"(per-fold {fold_lo:.4f}--{fold_hi:.4f})")

    # Refit on all 253 for deploy weights
    w_deploy = _fit_slsqp(P_oof, unb_y)
    print("Deploy SLSQP weights (refit on 253):")
    for t, wi in zip(POOL, w_deploy):
        print(f"  {t}: {wi:.4f}")

    # =================================================================
    # Decide: hedge vs SLSQP -- lowest cross-fit RAE wins
    # =================================================================
    print("\n" + "=" * 78)
    print("DEPLOY DECISION")
    print("=" * 78)
    print(f"  nb492 alone (oof RAE)                = {rae_anchor_oof:.4f}")
    print(f"  best pairwise hedge ({best_pair_tag}) RAE = "
          f"{best_pair['best_rae']:.4f} (w={best_pair['best_w']:.2f})")
    print(f"  4-way SLSQP cross-fit RAE            = {slsqp_rae:.4f}")

    if slsqp_rae <= best_pair["best_rae"]:
        final_method = "slsqp4way"
        final_rae = slsqp_rae
        deploy = (Q_te @ w_deploy).astype(np.float32)
        print(f"  -> 4-way SLSQP wins  RAE={final_rae:.4f}")
    else:
        final_method = f"pairwise_{best_pair_tag}_w{best_pair['best_w']:.1f}"
        final_rae = best_pair["best_rae"]
        w_pair = best_pair["best_w"]
        deploy = (
            (1.0 - w_pair) * nb492_te + w_pair * te_vec[best_pair_tag]
        ).astype(np.float32)
        print(f"  -> pairwise hedge wins  "
              f"nb492+{best_pair_tag} w={w_pair:.2f}  RAE={final_rae:.4f}")

    # =================================================================
    # Save artefacts
    # =================================================================
    np.save(DATA_PROCESSED / "te_nb503.npy", deploy)
    np.save(DATA_PROCESSED / "nb503_pred_oof.npy",
            (oof_blend if final_method == "slsqp4way"
             else ((1.0 - best_pair["best_w"]) * nb492_oof
                   + best_pair["best_w"] * oof[best_pair_tag])
            ).astype(np.float32))

    plain = SUBMISSIONS / f"nb503_hedge_{final_method}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"nb503_hedge_{final_method}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / 'te_nb503.npy'}")
    print(f"Wrote {DATA_PROCESSED / 'nb503_pred_oof.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb503 SUMMARY ===")
    print(f"  nb492 alone                              = {rae_anchor_oof:.4f}")
    for c in CANDIDATES:
        p = pair_summary[c]
        print(f"  pairwise nb492+{c}  best_w={p['best_w']:.2f}  "
              f"RAE={p['best_rae']:.4f}  gain={p['gain']:+.4f}")
    print(f"  4-way SLSQP cross-fit                    = {slsqp_rae:.4f}")
    print(f"  final method                              = {final_method}")
    print(f"  final cross-fit RAE                       = {final_rae:.4f}")
    print(f"  beats nb492 ({rae_anchor_oof:.4f})            = "
          f"{final_rae < rae_anchor_oof}")
    print("=" * 78)

    return {
        "success": True,
        "anchor_rae": rae_anchor_oof,
        "pairwise": pair_summary,
        "best_pairwise_partner": best_pair_tag,
        "best_pairwise_weight": float(best_pair["best_w"]),
        "best_pairwise_rae": float(best_pair["best_rae"]),
        "slsqp_crossfit_rae": float(slsqp_rae),
        "slsqp_deploy_weights": {t: float(w) for t, w in zip(POOL, w_deploy)},
        "final_method": final_method,
        "final_rae": float(final_rae),
        "beats_nb492": bool(final_rae < rae_anchor_oof),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
