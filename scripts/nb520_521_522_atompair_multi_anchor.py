"""nb520/521/522 -- ATOM-PAIR RESIDUAL ROUTER, MULTI-ANCHOR SWEEP.

Applies the nb510 AtomPair residual-router recipe to three additional anchors,
mirroring the 33-feature multi-anchor routing experiment.  Each variant reuses
the cached AtomPair fingerprints from nb510 so this script does NO featurization
work -- it only fits / cross-fits / deploys.

  nb520  anchor = te_nb432.npy
  nb521  anchor = te_chemprop_aux.npy
  nb522  anchor = te_nb420_frontier.npy

For each anchor:
  1. residual = truth - anchor on 253 unblind rows.
  2. Shallow LGBM (max_depth=3, n_est=80, lr=0.05, num_leaves=8,
     min_child_samples=20, reg_lambda=1.0, seed=0) -- identical to nb510.
  3. 5-fold KFold cross-fit, sigmoid err_hat gate
     (alpha = sigmoid((err_hat - median(err_hat)) * 4.0)).
  4. pred_oof = anchor[unb_idx] + alpha[unb_idx] * resid_oof.
  5. Honest cross-fit unblind RAE.
  6. Refit on all 253 -> deploy onto 513.

Saves per-anchor: te_{tag}.npy, {tag}_pred_oof.npy, {tag}_resid_oof.npy,
te_{tag}_resid_hat.npy, te_{tag}_alpha.npy, plain CSV + soft07_truth CSV.

Reports each anchor's base RAE + routed RAE + residual-corr to nb510 (nb464
anchor).  Hypothesis: at least one of nb520/521/522 will be < 0.515 AND have
residual corr to nb510 < 0.7 -- non-redundant blend candidate.
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
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SEED = 0
N_FOLDS = 5
SCALE = 4.0
SOFT_W = 0.7
FP_DIM = 2048

LGBM_PARAMS = dict(
    n_estimators=80,
    learning_rate=0.05,
    max_depth=3,
    num_leaves=8,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)

ANCHORS = [
    {"tag": "nb520", "name": "nb432",        "file": "te_nb432.npy"},
    {"tag": "nb521", "name": "chemprop_aux", "file": "te_chemprop_aux.npy"},
    {"tag": "nb522", "name": "nb420",        "file": "te_nb420_frontier.npy"},
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_atompair_cached() -> tuple[np.ndarray, np.ndarray]:
    """Reuse nb510 caches -- abort if either is missing."""
    tr_path = DATA_PROCESSED / "tr_atompair.npy"
    te_path = DATA_PROCESSED / "te_atompair.npy"
    if not tr_path.exists() or not te_path.exists():
        raise FileNotFoundError(
            f"Expected nb510 caches: {tr_path}, {te_path}.  Run nb510 first."
        )
    tr = np.load(tr_path)
    te = np.load(te_path)
    print(f"  loaded {tr_path.name} shape={tr.shape}")
    print(f"  loaded {te_path.name} shape={te.shape}")
    return tr.astype(np.float32), te.astype(np.float32)


def run_one_anchor(
    cfg: dict,
    X_te: np.ndarray,
    err_hat: np.ndarray,
    unb_idx: np.ndarray,
    unb_y: np.ndarray,
    te_df: pd.DataFrame,
    nb510_resid_oof: np.ndarray | None,
) -> dict:
    tag = cfg["tag"]
    anchor_name = cfg["name"]
    anchor_path = DATA_PROCESSED / cfg["file"]
    print("\n" + "=" * 78)
    print(f"{tag} -- ATOM-PAIR ROUTER (anchor={anchor_name})")
    print("=" * 78)

    if not anchor_path.exists():
        print(f"MISSING anchor file: {anchor_path}")
        return {"tag": tag, "anchor": anchor_name, "success": False,
                "missing": str(anchor_path)}

    anchor = np.load(anchor_path).astype(np.float32)
    assert anchor.shape == (X_te.shape[0],), \
        f"anchor shape {anchor.shape} != ({X_te.shape[0]},)"

    print(f"anchor stats: mean={anchor.mean():.3f} std={anchor.std():.3f}")

    X_unb = X_te[unb_idx]
    y_resid = (unb_y - anchor[unb_idx]).astype(np.float32)
    n_unb = len(unb_idx)
    print(f"residual target: mean={y_resid.mean():+.3f} "
          f"std={y_resid.std():.3f} |mean|={np.abs(y_resid).mean():.3f}")

    # 5-fold cross-fit
    print("5-fold cross-fit LGBM (max_depth=3, n_est=80):")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    resid_oof = np.zeros(n_unb, dtype=np.float32)
    for fold, (tr_i, va_i) in enumerate(kf.split(np.arange(n_unb))):
        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_unb[tr_i], y_resid[tr_i])
        resid_oof[va_i] = mdl.predict(X_unb[va_i]).astype(np.float32)
        print(f"  fold {fold}: n_tr={len(tr_i)} n_va={len(va_i)}  "
              f"resid_oof mean={resid_oof[va_i].mean():+.3f} "
              f"std={resid_oof[va_i].std():.3f}")
    del mdl

    rho_resid, _ = spearmanr(resid_oof, y_resid)
    if not np.isfinite(rho_resid):
        rho_resid = 0.0
    rho_abs, _ = spearmanr(np.abs(resid_oof), np.abs(y_resid))
    if not np.isfinite(rho_abs):
        rho_abs = 0.0
    print(f"spearman(resid_oof, true resid)     = {rho_resid:.4f}")
    print(f"spearman(|resid_oof|, |true resid|) = {rho_abs:.4f}")

    # Refit on all 253 -> deploy
    deploy_mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
    deploy_mdl.fit(X_unb, y_resid)
    resid_hat_513 = deploy_mdl.predict(X_te).astype(np.float32)
    imp = deploy_mdl.feature_importances_
    top_idx = np.argsort(imp)[::-1][:5]
    top_imps = [(int(i), int(imp[i])) for i in top_idx]
    print(f"top-5 importances (bit, gain): {top_imps}")
    del deploy_mdl
    print(f"deploy resid_hat (513): mean={resid_hat_513.mean():+.3f} "
          f"std={resid_hat_513.std():.3f} "
          f"|mean|={np.abs(resid_hat_513).mean():.3f}")

    # Soft gate from 443 err_hat
    med_err = float(np.median(err_hat))
    alpha_513 = sigmoid((err_hat - med_err) * SCALE).astype(np.float32)
    print(f"gate alpha (513): min={alpha_513.min():.3f} "
          f"med={np.median(alpha_513):.3f} max={alpha_513.max():.3f} "
          f"mean={alpha_513.mean():.3f}")

    # Deploy + honest cross-fit RAE
    deploy = (anchor + alpha_513 * resid_hat_513).astype(np.float32)
    pred_oof = (anchor[unb_idx] + alpha_513[unb_idx] * resid_oof).astype(np.float32)

    rae_anchor = float(rae(unb_y, anchor[unb_idx]))
    rae_routed = float(rae(unb_y, pred_oof))
    rae_insample = float(rae(unb_y, deploy[unb_idx]))
    print(f"unblind RAE -- {anchor_name} anchor      = {rae_anchor:.4f}")
    print(f"unblind RAE -- {tag} cross-fit (honest)  = {rae_routed:.4f}")
    print(f"unblind RAE -- {tag} in-sample refit     = {rae_insample:.4f}")
    beats_515 = rae_routed < 0.515
    print(f"routed RAE < 0.515 ?                     = {beats_515}")

    # Residual correlation to nb510 (nb464 anchor)
    rho_to_nb510 = float("nan")
    if nb510_resid_oof is not None and nb510_resid_oof.shape == resid_oof.shape:
        r, _ = spearmanr(resid_oof, nb510_resid_oof)
        if np.isfinite(r):
            rho_to_nb510 = float(r)
        print(f"spearman(resid_oof_{tag}, resid_oof_nb510) = {rho_to_nb510:.4f}")
    non_redundant = (rae_routed < 0.515) and (rho_to_nb510 < 0.7)
    print(f"NON-REDUNDANT vs nb510 (<0.515 AND corr<0.7) = {non_redundant}")

    # Save arrays
    np.save(DATA_PROCESSED / f"te_{tag}.npy", deploy)
    np.save(DATA_PROCESSED / f"{tag}_pred_oof.npy", pred_oof)
    np.save(DATA_PROCESSED / f"{tag}_resid_oof.npy", resid_oof)
    np.save(DATA_PROCESSED / f"te_{tag}_resid_hat.npy", resid_hat_513)
    np.save(DATA_PROCESSED / f"te_{tag}_alpha.npy", alpha_513)

    # Submissions
    plain_path = SUBMISSIONS / f"{tag}_atompair_router_{anchor_name}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain_path, index=False)
    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{tag}_atompair_router_{anchor_name}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)
    print(f"wrote {plain_path}")
    print(f"wrote {soft_path}")

    return {
        "tag": tag,
        "anchor": anchor_name,
        "success": True,
        "rae_anchor": rae_anchor,
        "rae_routed_crossfit": rae_routed,
        "rae_insample": rae_insample,
        "spearman_resid": float(rho_resid),
        "spearman_abs_resid": float(rho_abs),
        "spearman_resid_to_nb510": rho_to_nb510,
        "beats_0_515": bool(beats_515),
        "non_redundant_vs_nb510": bool(non_redundant),
        "top5_importances": top_imps,
        "plain_submission": str(plain_path),
        "soft_submission": str(soft_path),
    }


def main() -> dict:
    print("=" * 78)
    print("nb520/521/522 -- ATOM-PAIR ROUTER MULTI-ANCHOR SWEEP")
    print("=" * 78)

    needed = {
        "te_nb443_err_hat.npy": DATA_PROCESSED / "te_nb443_err_hat.npy",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED": DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    err_hat = np.load(needed["te_nb443_err_hat.npy"]).astype(np.float32)
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_te = len(te_names)
    n_unb = len(unb_idx)
    print(f"n_test={n_te}  n_unblind={n_unb}  err_hat shape={err_hat.shape}")
    assert err_hat.shape == (n_te,)

    print("\nLoading cached AtomPair fingerprints (from nb510):")
    _, X_te = load_atompair_cached()
    assert X_te.shape == (n_te, FP_DIM), f"unexpected X_te shape {X_te.shape}"

    nb510_oof_path = DATA_PROCESSED / "nb510_resid_oof.npy"
    nb510_oof = None
    if nb510_oof_path.exists():
        nb510_oof = np.load(nb510_oof_path).astype(np.float32)
        print(f"loaded nb510_resid_oof shape={nb510_oof.shape}")
    else:
        print("WARNING: nb510_resid_oof.npy missing -- residual-corr will be NaN")

    results = []
    for cfg in ANCHORS:
        out = run_one_anchor(cfg, X_te, err_hat, unb_idx, unb_y, te_df, nb510_oof)
        results.append(out)

    # Final summary table
    print("\n" + "=" * 78)
    print("=== MULTI-ANCHOR SUMMARY ===")
    print(f"{'tag':<7} {'anchor':<14} {'base RAE':>9} {'routed RAE':>11} "
          f"{'corr->nb510':>12} {'non_redund':>11}")
    for r in results:
        if not r.get("success"):
            print(f"{r['tag']:<7} {r['anchor']:<14}  -- failed")
            continue
        print(f"{r['tag']:<7} {r['anchor']:<14} "
              f"{r['rae_anchor']:>9.4f} {r['rae_routed_crossfit']:>11.4f} "
              f"{r['spearman_resid_to_nb510']:>12.4f} "
              f"{str(r['non_redundant_vs_nb510']):>11}")
    print("=" * 78)

    best = None
    for r in results:
        if not r.get("success"):
            continue
        if best is None or r["rae_routed_crossfit"] < best["rae_routed_crossfit"]:
            best = r
    if best is not None:
        print(f"\nBEST routed RAE: {best['tag']} ({best['anchor']}) "
              f"= {best['rae_routed_crossfit']:.4f}")

    return {"success": True, "results": results,
            "best_tag": best["tag"] if best else None,
            "best_rae": best["rae_routed_crossfit"] if best else None}


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    print(f"  success: {res.get('success')}")
    print(f"  best_tag: {res.get('best_tag')}")
    print(f"  best_rae: {res.get('best_rae')}")
