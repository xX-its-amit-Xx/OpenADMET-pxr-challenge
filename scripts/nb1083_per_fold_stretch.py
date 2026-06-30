"""nb1083 -- PER-FOLD ADAPTIVE RANK STRETCH on nb2103 K=28 mean-bag OOF.

Memory feedback `feedback_rank_stretch_universal`: nb562's universal s=1.07
gained -0.003 to -0.005 RAE on every variance-compressed predictor we tried.
nb2103 K=28 mean-bag OOF (RAE 0.4737) and nb2112 median deploy (cross-fit
~0.4698) are FAR stronger anchors than nb503/nb562 (0.5116/0.5065). This
script tests whether a PER-FOLD adaptive s_k can extract more than the
universal scalar -- the hypothesis is no (single d.o.f. already saturates),
but we need a hard number before locking the ladder.

Protocol
--------
KFold(n_splits=5, shuffle=True, random_state=0) over the 253 unblind rows.
For each fold k:
    train fold = 253 \ held-out
    mu_k = mean(pred[train_fold])
    Two stretch variants:
        VARMATCH:  s_k = std(y[train_fold])     / std(pred[train_fold])
        MAD:       s_k = MAD(y[train_fold])     / MAD(pred[train_fold])
                   (MAD = median(|x - median(x)|), robust to tails)
    Apply to held-out:  stretched[va] = mu_k + s_k * (pred[va] - mu_k)

Aggregate pooled OOF -> RAE. Also try a GRID-search per fold (chooses s* on
the train fold) for completeness. Compare to nb562 universal s=1.07 and to
nb2103 baseline.

Decision
--------
Per task spec: must beat BOTH
    - nb562 univ-s on its own anchor (RAE 0.5065)  -- low bar, anchor is weaker
    - nb2112 median deploy cross-fit (RAE 0.4698)  -- this is the live ceiling
margin 0.003. If winning variant beats nb2112 (0.4698) by >= 0.003 we deploy.
Otherwise we record the negative result.

Deploy
------
If winning, refit (mu_global, s_global) on all 253 and apply to te_nb2112
(513 deploy preds from nb2103 K=28 median bag). Save deploy CSV.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
TAG = "nb1083"
GRID = [1.00, 1.025, 1.05, 1.075, 1.10, 1.125, 1.15, 1.20, 1.25, 1.30]
NB562_REF = 0.5065  # universal s=1.07 cross-fit on nb503/562 anchor
NB2112_MEDIAN_REF = 0.4698  # nb2103 K=28 median-bag cross-fit
NB2103_MEAN_REF = 0.4737  # nb2103 K=28 mean-bag cross-fit
DECISION_MARGIN = 0.003


def _stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def _mad(a: np.ndarray) -> float:
    return float(np.median(np.abs(a - np.median(a))))


def _cross_fit(pred: np.ndarray, y: np.ndarray, kf: KFold, mode: str) -> tuple:
    """Per-fold s_k cross-fit; returns (pooled_oof, per_fold_s, per_fold_rae)."""
    n = len(pred)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_s, fold_rae, fold_mu = [], [], []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n))):
        p_tr, y_tr = pred[tr_i], y[tr_i]
        mu_k = float(p_tr.mean())
        if mode == "varmatch":
            s_k = float(y_tr.std()) / max(float(p_tr.std()), 1e-9)
        elif mode == "mad":
            s_k = _mad(y_tr) / max(_mad(p_tr), 1e-9)
        elif mode == "grid":
            best_s, best_r = 1.0, float("inf")
            for s in GRID:
                r = float(rae(y_tr, _stretch(p_tr, mu_k, s)))
                if r < best_r:
                    best_r = r
                    best_s = float(s)
            s_k = best_s
        else:
            raise ValueError(mode)
        oof[va_i] = _stretch(pred[va_i], mu_k, s_k)
        r_va = float(rae(y[va_i], oof[va_i]))
        fold_s.append(float(s_k))
        fold_rae.append(r_va)
        fold_mu.append(mu_k)
    return oof, fold_s, fold_rae, fold_mu


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- PER-FOLD STRETCH on nb2103 K=28 mean-bag OOF")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "nb2103_oof":   DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy",
        "te_nb2112":    DATA_PROCESSED / "te_nb2112.npy",
    }
    miss = [k for k, p in needed.items() if not Path(p).exists()]
    if miss:
        print("MISSING:", miss)
        return {"success": False, "missing": miss}

    te_df = pd.read_csv(needed["TEST_BLINDED"])
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    y_unb = unb["pEC50"].astype(float).values.astype(np.float64)
    n_te = len(te_df)
    n_unb = len(unb_idx)

    pred = np.load(needed["nb2103_oof"]).astype(np.float64)
    te_deploy = np.load(needed["te_nb2112"]).astype(np.float64)
    assert pred.shape == (n_unb,), f"oof shape {pred.shape}"
    assert te_deploy.shape == (n_te,), f"te shape {te_deploy.shape}"

    rae_base = float(rae(y_unb, pred))
    truth_std = float(y_unb.std())
    pred_std = float(pred.std())
    s_var_global = truth_std / max(pred_std, 1e-9)
    s_mad_global = _mad(y_unb) / max(_mad(pred), 1e-9)
    print(f"\nn_unb={n_unb}  n_te={n_te}")
    print(f"nb2103 K=28 OOF RAE = {rae_base:.4f}  (refs mean={NB2103_MEAN_REF})")
    print(f"truth mean/std/MAD  = {y_unb.mean():.3f}/{truth_std:.3f}/{_mad(y_unb):.3f}")
    print(f"pred  mean/std/MAD  = {pred.mean():.3f}/{pred_std:.3f}/{_mad(pred):.3f}")
    print(f"global s_var        = {s_var_global:.4f}")
    print(f"global s_mad        = {s_mad_global:.4f}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    results = {}
    for mode in ("varmatch", "mad", "grid"):
        print(f"\n--- mode={mode} ---")
        oof, fold_s, fold_rae, fold_mu = _cross_fit(pred, y_unb, kf, mode)
        pooled = float(rae(y_unb, oof))
        for fold, (s, r, mu) in enumerate(zip(fold_s, fold_rae, fold_mu)):
            print(f"  fold {fold}: s={s:.4f}  mu={mu:.3f}  RAE={r:.4f}")
        print(f"  pooled cross-fit RAE = {pooled:.4f}  "
              f"(per-fold {min(fold_rae):.4f}-{max(fold_rae):.4f})")
        results[mode] = {
            "pooled_rae": pooled,
            "fold_s": fold_s,
            "fold_rae": fold_rae,
            "fold_mu": fold_mu,
            "oof": oof,
        }

    # Combo: take MIN per-fold across (varmatch, mad, grid) on TRAIN-fold RAE
    # (oracle-on-train choice; honest cross-fit because choice uses train only)
    print("\n--- mode=combo_train_oracle ---")
    oof_c = np.full(n_unb, np.nan, dtype=np.float64)
    combo_choices = []
    combo_fold_rae = []
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        p_tr, y_tr = pred[tr_i], y_unb[tr_i]
        mu_k = float(p_tr.mean())
        cand = {}
        for mode in ("varmatch", "mad", "grid"):
            if mode == "varmatch":
                s_k = float(y_tr.std()) / max(float(p_tr.std()), 1e-9)
            elif mode == "mad":
                s_k = _mad(y_tr) / max(_mad(p_tr), 1e-9)
            else:
                best_s, best_r = 1.0, float("inf")
                for s in GRID:
                    r = float(rae(y_tr, _stretch(p_tr, mu_k, s)))
                    if r < best_r:
                        best_r = r
                        best_s = float(s)
                s_k = best_s
            train_rae = float(rae(y_tr, _stretch(p_tr, mu_k, s_k)))
            cand[mode] = (s_k, train_rae)
        best_mode = min(cand, key=lambda m: cand[m][1])
        s_k = cand[best_mode][0]
        oof_c[va_i] = _stretch(pred[va_i], mu_k, s_k)
        r_va = float(rae(y_unb[va_i], oof_c[va_i]))
        combo_choices.append({"fold": fold, "mode": best_mode,
                              "s": float(s_k), "mu": mu_k, "va_rae": r_va,
                              "train_rae": cand[best_mode][1]})
        combo_fold_rae.append(r_va)
        print(f"  fold {fold}: chose {best_mode}  s={s_k:.4f}  "
              f"train_RAE={cand[best_mode][1]:.4f}  va_RAE={r_va:.4f}")
    pooled_combo = float(rae(y_unb, oof_c))
    print(f"  pooled combo cross-fit RAE = {pooled_combo:.4f}")
    results["combo_train_oracle"] = {
        "pooled_rae": pooled_combo,
        "fold_choices": combo_choices,
        "fold_rae": combo_fold_rae,
        "oof": oof_c,
    }

    # ---- Decision ----
    print("\n" + "=" * 78)
    print("DECISION TABLE")
    print("=" * 78)
    print(f"  baseline nb2103 K=28 OOF       = {rae_base:.4f}")
    print(f"  nb2112 median cross-fit (ref)  = {NB2112_MEDIAN_REF:.4f}")
    print(f"  nb562 univ s=1.07 (ref)        = {NB562_REF:.4f}")
    print()
    for mode, r in results.items():
        delta_nb2112 = r["pooled_rae"] - NB2112_MEDIAN_REF
        delta_base = r["pooled_rae"] - rae_base
        beats_nb2112 = r["pooled_rae"] < NB2112_MEDIAN_REF - DECISION_MARGIN
        beats_nb562 = r["pooled_rae"] < NB562_REF - DECISION_MARGIN
        print(f"  {mode:24s} RAE={r['pooled_rae']:.4f}  "
              f"vs nb2103={delta_base:+.4f}  vs nb2112={delta_nb2112:+.4f}  "
              f"beats_nb2112={beats_nb2112}  beats_nb562={beats_nb562}")

    winner_mode = min(results, key=lambda m: results[m]["pooled_rae"])
    winner_rae = results[winner_mode]["pooled_rae"]
    must_beat_nb562 = winner_rae < NB562_REF - DECISION_MARGIN
    must_beat_nb2112 = winner_rae < NB2112_MEDIAN_REF - DECISION_MARGIN
    deploy_ok = must_beat_nb562 and must_beat_nb2112

    print(f"\n  winner = {winner_mode}  RAE={winner_rae:.4f}")
    print(f"  beats nb562 (>= 0.003)?  {must_beat_nb562}")
    print(f"  beats nb2112 (>= 0.003)? {must_beat_nb2112}")
    print(f"  deploy?                  {deploy_ok}")

    # ---- Deploy ----
    deploy_path = None
    deploy_te_path = None
    deploy_oof_path = None
    deploy_s_global = None
    deploy_mu_global = None
    deploy_mode = None
    if deploy_ok:
        deploy_mode = winner_mode
        mu_global = float(pred.mean())
        if winner_mode == "varmatch":
            s_global = s_var_global
        elif winner_mode == "mad":
            s_global = s_mad_global
        elif winner_mode == "grid":
            best_s, best_r = 1.0, float("inf")
            for s in GRID:
                r = float(rae(y_unb, _stretch(pred, mu_global, s)))
                if r < best_r:
                    best_r = r
                    best_s = float(s)
            s_global = best_s
        else:  # combo: refit grid as a safe default (combo is unstable globally)
            best_s, best_r = 1.0, float("inf")
            for s in GRID:
                r = float(rae(y_unb, _stretch(pred, mu_global, s)))
                if r < best_r:
                    best_r = r
                    best_s = float(s)
            s_global = best_s
        deploy_s_global = float(s_global)
        deploy_mu_global = float(mu_global)
        te_stretched = _stretch(te_deploy, mu_global, s_global).astype(np.float32)
        in_sample_rae = float(rae(y_unb, te_stretched[unb_idx]))
        deploy_te_path = DATA_PROCESSED / "te_nb1083_per_fold_stretch.npy"
        deploy_oof_path = DATA_PROCESSED / "nb1083_pred_oof.npy"
        np.save(deploy_te_path, te_stretched)
        np.save(deploy_oof_path,
                results[winner_mode]["oof"].astype(np.float32))
        deploy_path = (SUBMISSIONS /
                       f"nb1083_deploy_per_fold_stretch.csv")
        pd.DataFrame({
            "Molecule Name": te_df["Molecule Name"],
            "SMILES": te_df["SMILES"],
            "pEC50": te_stretched,
        }).to_csv(deploy_path, index=False)
        print(f"\n  DEPLOY: mode={winner_mode}  s_global={s_global:.4f}  "
              f"mu_global={mu_global:.3f}")
        print(f"  te stretched mean/std = "
              f"{te_stretched.mean():.3f}/{te_stretched.std():.3f}")
        print(f"  te[unb_idx] in-sample RAE = {in_sample_rae:.4f}")
        print(f"  wrote {deploy_path}")
    else:
        print("\n  NO DEPLOY: best variant fails margin.")

    summary = {
        "tag": TAG,
        "method": "per_fold_adaptive_rank_stretch_on_nb2103_K28",
        "anchor_oof": "nb2103_mean_bag_oof_K28.npy",
        "anchor_te": "te_nb2112.npy",
        "n_unb": int(n_unb),
        "n_te": int(n_te),
        "kfold_seed": SEED,
        "kfold_n_splits": N_FOLDS,
        "stretch_grid": GRID,
        "decision_margin": DECISION_MARGIN,
        "nb2103_K28_OOF_rae_base": rae_base,
        "nb2112_median_ref": NB2112_MEDIAN_REF,
        "nb562_univ_ref": NB562_REF,
        "truth_mean": float(y_unb.mean()),
        "truth_std": truth_std,
        "truth_mad": float(_mad(y_unb)),
        "pred_mean": float(pred.mean()),
        "pred_std": pred_std,
        "pred_mad": float(_mad(pred)),
        "s_var_global": float(s_var_global),
        "s_mad_global": float(s_mad_global),
        "results": {
            mode: {
                "pooled_rae": r["pooled_rae"],
                "fold_s": r.get("fold_s"),
                "fold_mu": r.get("fold_mu"),
                "fold_rae": r["fold_rae"],
                "fold_choices": r.get("fold_choices"),
                "delta_vs_nb2112": r["pooled_rae"] - NB2112_MEDIAN_REF,
                "delta_vs_nb2103_base": r["pooled_rae"] - rae_base,
                "delta_vs_nb562": r["pooled_rae"] - NB562_REF,
                "beats_nb2112": r["pooled_rae"] < NB2112_MEDIAN_REF - DECISION_MARGIN,
                "beats_nb562":  r["pooled_rae"] < NB562_REF - DECISION_MARGIN,
            }
            for mode, r in results.items()
        },
        "winner_mode": winner_mode,
        "winner_rae": winner_rae,
        "beats_nb2112_margin": bool(must_beat_nb2112),
        "beats_nb562_margin": bool(must_beat_nb562),
        "deploy_ok": bool(deploy_ok),
        "deploy_mode": deploy_mode,
        "deploy_s_global": deploy_s_global,
        "deploy_mu_global": deploy_mu_global,
        "deploy_te_path": str(deploy_te_path) if deploy_te_path else None,
        "deploy_oof_path": str(deploy_oof_path) if deploy_oof_path else None,
        "deploy_csv": str(deploy_path) if deploy_path else None,
        "verdict": (
            "DEPLOY_PER_FOLD_STRETCH_BEATS_NB2112"
            if deploy_ok else
            "PER_FOLD_STRETCH_FAILS_MARGIN_VS_NB2112"
        ),
    }
    out = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nwrote {out}")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  winner_mode      : {res.get('winner_mode')}")
    print(f"  winner_rae       : {res.get('winner_rae')}")
    print(f"  beats nb2112     : {res.get('beats_nb2112_margin')}")
    print(f"  beats nb562      : {res.get('beats_nb562_margin')}")
    print(f"  deploy_ok        : {res.get('deploy_ok')}")
    print(f"  verdict          : {res.get('verdict')}")
