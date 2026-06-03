"""nb1024 -- LassoCV blend on 7 PRE-unblind candidates + stretch grid.

Hypothesis: L1 regularization may select a sparser subset of candidates
than SLSQP's simplex constraint (which spreads weight onto co-linear
predictors), giving more deploy-friendly weights that transfer better
from the 253-unblind cross-fit to the LB.

Pool (7 PRE-unblind candidates):
  chemprop_aux, nb901, nb972, nb914, nb960, nb923, nb120_huber_1_0

Procedure:
  1. Build P_unb (253, 7) and P_te (513, 7).
  2. 5-fold cross-fit LassoCV(positive=True, cv=5) on the 253 unblind:
       - For each outer fold f:
           Fit LassoCV on train folds (inner-CV picks alpha_f).
           Predict held-out fold -> OOF.
       - Pooled cross-fit RAE on the 253.
  3. Deploy: fit LassoCV(positive=True, cv=5) on ALL 253 unblind to
     get the single deploy alpha + weights; apply to P_te (513, 7).
  4. Apply stretch grid {1.00, 1.05, ..., 1.50} around the deploy-blend
     mean on the unblind, pick best stretch s by 253 in-sample RAE
     (sanity-style; full cross-fit of (lasso, s) jointly skipped here
     -- stretch is a 1-param post-hoc decompression).
  5. Final pred = mu_blend + s * (lasso_blend - mu_blend).

Outputs:
  data/processed/te_nb1024.npy
  data/processed/nb1024_summary.json
  submissions/nb1024_lasso_blend.csv
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

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1024"
# Map candidate alias -> (npy stem, csv stem). npy takes precedence.
CANDIDATES = [
    ("chemprop_aux",     "chemprop_aux",        "chemprop_aux"),
    ("nb901",            "nb901",               "nb901_nr_multitask"),
    ("nb972",            "nb972_long_train",    "nb972_long_train_optim"),
    ("nb914",            "nb914",               "nb914_persistence_homology"),
    ("nb960",            "nb960",               "nb960_pseudo_self_train"),
    ("nb923",            "nb923",               "nb923_wl_graph_kernel"),
    ("nb120_huber_1_0",  "nb120_huber_1_0",     "nb120_huber_1_0"),
]
STRETCH_GRID = np.round(np.arange(1.00, 1.501, 0.05), 2).tolist()
N_FOLDS = 5
SEED = 42
NB1014_REF = 0.5994  # multi-seed bag reference (LB-honest)


def load_te(npy_stem: str, csv_stem: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{npy_stem}.npy"
    if npy.exists():
        arr = np.load(npy).astype(np.float64)
        assert len(arr) == len(te_names), \
            f"{npy_stem}: te length {len(arr)} != {len(te_names)}"
        return arr
    sub = pd.read_csv(SUBMISSIONS / f"{csv_stem}.csv")
    # align by Molecule Name in case row order differs
    name_to_p = dict(zip(sub["Molecule Name"].values,
                         sub["pEC50"].values.astype(np.float64)))
    arr = np.array([name_to_p[n] for n in te_names], dtype=np.float64)
    return arr


def best_stretch(blend: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Grid-scan s; return (best_s, best_rae)."""
    mu = float(blend.mean())
    best_s, best_r = 1.0, float("inf")
    for s in STRETCH_GRID:
        stretched = mu + s * (blend - mu)
        r = float(rae(y, stretched))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LassoCV blend on 7 PRE-unblind candidates")
    print("=" * 78)

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values
    n_te = len(te_names)
    preds_513 = np.column_stack([
        load_te(npy, csv, te_names) for _, npy, csv in CANDIDATES
    ])
    aliases = [c[0] for c in CANDIDATES]
    print(f"[load] preds_513 shape = {preds_513.shape}  (cols = {aliases})")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    n_unb = len(y_unb)
    print(f"[load] P_unb shape = {P_unb.shape}, y shape = {y_unb.shape}")

    # ---- Individual in_RAE sanity ----
    print("\n[indiv] in_RAE on 253 unblind:")
    indiv_rae = {}
    for j, a in enumerate(aliases):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[a] = r
        print(f"   {a:20s}: {r:.4f}")

    # =================================================================
    # 5-fold cross-fit LassoCV(positive=True)
    # =================================================================
    print("\n" + "-" * 78)
    print(f"CROSS-FIT LassoCV(positive=True, cv=5)  outer KFold seed={SEED}")
    print("-" * 78)
    outer = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(n_unb, np.nan)
    fold_alphas = []
    fold_weights = []
    fold_val_rae = []
    fold_intercepts = []
    for k, (tr_loc, va_loc) in enumerate(outer.split(np.arange(n_unb))):
        model = LassoCV(positive=True, cv=5, n_alphas=100,
                        max_iter=20000, random_state=SEED)
        model.fit(P_unb[tr_loc], y_unb[tr_loc])
        oof[va_loc] = model.predict(P_unb[va_loc])
        rv = float(rae(y_unb[va_loc], oof[va_loc]))
        fold_alphas.append(float(model.alpha_))
        fold_weights.append(model.coef_.astype(float).tolist())
        fold_intercepts.append(float(model.intercept_))
        fold_val_rae.append(rv)
        nz = int((model.coef_ > 1e-8).sum())
        print(f"   fold {k}: alpha={model.alpha_:.5f}  n_active={nz}/7  "
              f"val_RAE={rv:.4f}")
        print(f"           weights={['%.3f' % w for w in model.coef_]}")
    pooled = float(rae(y_unb, oof))
    print(f"\n[xfit] pooled cross-fit RAE = {pooled:.4f}  "
          f"(vs nb1014 ref {NB1014_REF:.4f}; delta {pooled - NB1014_REF:+.4f})")

    # =================================================================
    # Deploy: refit LassoCV on ALL 253 to pick the single alpha
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (LassoCV refit on ALL 253; apply to 513)")
    print("-" * 78)
    deploy_model = LassoCV(positive=True, cv=5, n_alphas=100,
                           max_iter=20000, random_state=SEED)
    deploy_model.fit(P_unb, y_unb)
    deploy_alpha = float(deploy_model.alpha_)
    deploy_w = deploy_model.coef_.astype(float)
    deploy_b = float(deploy_model.intercept_)
    n_active = int((deploy_w > 1e-8).sum())
    print(f"   deploy alpha     = {deploy_alpha:.5f}")
    print(f"   deploy intercept = {deploy_b:.4f}")
    print(f"   n_active_weights = {n_active}/7")
    for a, w in zip(aliases, deploy_w):
        flag = "  <-- active" if w > 1e-8 else ""
        print(f"     {a:20s}: w={w:.4f}{flag}")

    lasso_blend_unb = deploy_model.predict(P_unb)
    lasso_blend_513 = deploy_model.predict(preds_513)
    in_rae_lasso = float(rae(y_unb, lasso_blend_unb))
    print(f"\n   in-sample RAE (lasso, no stretch) = {in_rae_lasso:.4f}")

    # =================================================================
    # Stretch grid on top of the lasso blend
    # =================================================================
    print("\n" + "-" * 78)
    print(f"STRETCH GRID  s in {STRETCH_GRID[0]}..{STRETCH_GRID[-1]}")
    print("-" * 78)
    best_s, best_rae_stretch = best_stretch(lasso_blend_unb, y_unb)
    mu_blend = float(lasso_blend_unb.mean())
    print(f"   mu (blend mean on 253) = {mu_blend:.4f}")
    print(f"   best stretch s         = {best_s:.2f}")
    print(f"   in-sample RAE (lasso+stretch) = {best_rae_stretch:.4f}  "
          f"(delta {best_rae_stretch - in_rae_lasso:+.4f})")

    # Apply stretch on the 513
    deploy_513 = (mu_blend + best_s * (lasso_blend_513 - mu_blend)).astype(
        np.float32)

    # Cross-fit RAE WITH stretch: re-stretch each OOF fold using its own
    # fold-train mean and a fold-train best s (proper honest estimate).
    print("\n[xfit+stretch] re-running OOF with per-fold stretch...")
    oof_stretched = np.full(n_unb, np.nan)
    fold_s = []
    for k, (tr_loc, va_loc) in enumerate(outer.split(np.arange(n_unb))):
        model = LassoCV(positive=True, cv=5, n_alphas=100,
                        max_iter=20000, random_state=SEED)
        model.fit(P_unb[tr_loc], y_unb[tr_loc])
        tr_blend = model.predict(P_unb[tr_loc])
        va_blend = model.predict(P_unb[va_loc])
        s_f, _ = best_stretch(tr_blend, y_unb[tr_loc])
        mu_tr = float(tr_blend.mean())
        oof_stretched[va_loc] = mu_tr + s_f * (va_blend - mu_tr)
        fold_s.append(s_f)
    pooled_stretch = float(rae(y_unb, oof_stretched))
    print(f"   per-fold s values        = {fold_s}")
    print(f"   pooled cross-fit RAE (lasso+stretch) = {pooled_stretch:.4f}")
    print(f"   delta vs lasso-only       = {pooled_stretch - pooled:+.4f}")

    print(f"\n   te(513) mean/std         = "
          f"{deploy_513.mean():.3f} / {deploy_513.std():.3f}")

    # =================================================================
    # Save
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_lasso_blend.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    beats_nb1014 = bool(pooled_stretch < NB1014_REF - 0.005)
    if beats_nb1014:
        verdict = "BEATS_NB1014"
    elif abs(pooled_stretch - NB1014_REF) <= 0.005:
        verdict = "TIES_NB1014"
    else:
        verdict = "WORSE_THAN_NB1014"
    print(f"\n[verdict] lasso+stretch vs nb1014 ({NB1014_REF:.4f}): "
          f"delta={pooled_stretch - NB1014_REF:+.4f}  -> {verdict}")

    summary = {
        "tag": TAG,
        "candidates": aliases,
        "indiv_in_rae": indiv_rae,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "fold_alphas": fold_alphas,
        "fold_weights": fold_weights,
        "fold_intercepts": fold_intercepts,
        "fold_val_rae": fold_val_rae,
        "fold_s": fold_s,
        "crossfit_rae_lasso": pooled,
        "crossfit_rae_with_stretch": pooled_stretch,
        "deploy_alpha": deploy_alpha,
        "deploy_intercept": deploy_b,
        "deploy_weights": deploy_w.tolist(),
        "deploy_weights_by_name": dict(zip(aliases, deploy_w.tolist())),
        "n_active_weights": n_active,
        "stretch_grid": STRETCH_GRID,
        "deploy_stretch_s": best_s,
        "deploy_mu_blend": mu_blend,
        "in_sample_rae_lasso": in_rae_lasso,
        "in_sample_rae_lasso_stretch": best_rae_stretch,
        "nb1014_ref": NB1014_REF,
        "delta_vs_nb1014": pooled_stretch - NB1014_REF,
        "beats_nb1014": beats_nb1014,
        "verdict": verdict,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pool                       = {aliases}")
    print(f"   crossfit RAE (lasso)       = {pooled:.4f}")
    print(f"   crossfit RAE (lasso+stretch)= {pooled_stretch:.4f}")
    print(f"   deploy alpha               = {deploy_alpha:.5f}")
    print(f"   n_active_weights           = {n_active}/7")
    print(f"   deploy stretch s           = {best_s:.2f}")
    print(f"   nb1014 reference           = {NB1014_REF:.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("crossfit_rae_lasso", "crossfit_rae_with_stretch",
              "deploy_alpha", "n_active_weights", "deploy_stretch_s",
              "delta_vs_nb1014", "beats_nb1014", "verdict",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
