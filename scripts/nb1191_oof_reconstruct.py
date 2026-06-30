"""nb1191_oof_reconstruct -- rebuild nb1191's 253-row OOF as a one-shot artifact.

CONTEXT:
    nb1191 (PRE-unblind pyramid; SLSQP+rank-stretch over 4 anchors) was the
    PRIMARY-1 PRE-clean ceiling at deep-30 RAE 0.4718. The original script
    saved te_nb1191.npy (513-row deploy vector) but never persisted the
    253-row OOF -- nb2880 (James-Stein shrinkage anchored to nb1191) needs
    that OOF as a per-fold prior.

RECIPE (mirrors scripts/nb1191_pre_unblind_pyramid.py exactly):
    Anchors (4):
        0. chemprop_aux  (nb1133_chemprop_aux_pred_oof.npy)             PRE
        1. nb1150        (reconstructed from SLSQP4 over 4 sub-anchors)
        2. nb1158_K32    (nb1158_mean_bag_oof_K32.npy)
        3. nb2112_K28    (nb2103_mean_bag_oof_K28.npy)

    Per-fold (5-fold scaffold CV, kf_seed=1001..1005):
        1. SLSQP simplex fit on fold-train (sum=1, w>=0)
        2. blend = P @ w
        3. mu_tr = blend_tr.mean()
        4. grid-search s in {1.000, 1.025, ..., 1.150} minimizing
           RAE(y_tr, mu_tr + s*(blend_tr - mu_tr))
        5. oof[va] = mu_tr + s*(blend_va - mu_tr)

    Final OOF = mean across 5 kf_seeds (matches nb1191's
    `rae_of_mean_of_seed_oofs` = 0.4697).

NOTE on user request wording:
    The task brief mentioned "nb2240_K20, chemprop_aux, counter_clean - the
    PRE-clean subset". nb1191 in code uses {chemprop_aux, nb1150, nb1158_K32,
    nb2112_K28}. To stay LB-faithful and reproduce nb1191 byte-faithfully
    (so nb2880 JS-shrinkage uses the literal nb1191 prior, not a paraphrase),
    we use the original 4 anchors. If a downstream notebook needs the
    {nb2240_K20, chemprop_aux, counter_clean} variant, that should be a
    NEW notebook (nb1192-style) so we don't conflate two priors under one tag.

OUTPUTS:
    data/processed/nb1191_pred_oof.npy           (253,) float32
    data/processed/nb1191_oof_summary.json
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
from scipy.optimize import minimize

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb1191_oof_reconstruct"
OUT_OOF = DATA_PROCESSED / "nb1191_pred_oof.npy"
OUT_SUMMARY = DATA_PROCESSED / "nb1191_oof_summary.json"

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]

# Mirror nb1191_pre_unblind_pyramid.py ANCHORS exactly
ANCHORS = [
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
    ("nb1150",       "_RECONSTRUCT_nb1150_oof"),
    ("nb1158_K32",   "nb1158_mean_bag_oof_K32.npy"),
    ("nb2112_K28",   "nb2103_mean_bag_oof_K28.npy"),
]

# nb1150 sub-stack: SLSQP4 over 4 base anchors with cached full-pool weights
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS_FULL_POOL = [0.0, 0.2942, 0.0, 0.7058]


def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        p = DATA_PROCESSED / rel
        assert p.exists(), f"missing nb1150 anchor OOF: {p}"
        v = np.load(p).astype(np.float64)
        assert v.shape == (n_unb,), f"{p.name} shape {v.shape}"
        cols.append(v)
    P = np.column_stack(cols)
    w = np.asarray(NB1150_SLSQP4_WEIGHTS_FULL_POOL, dtype=np.float64)
    return P @ w


def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP",
        bounds=bnds,
        constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch_on(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def cv_run_for_seed(P_unb, y_unb, unb_scaffolds, kf_seed):
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = P_unb.shape[0]
    oof_blend = np.full(n_unb, np.nan)
    fold_w, fold_s = [], []
    for tr_loc, va_loc in splits:
        w_f = slsqp_simplex(P_unb[tr_loc], y_unb[tr_loc])
        blend_tr = P_unb[tr_loc] @ w_f
        mu_tr = float(blend_tr.mean())
        s_f, _ = best_stretch_on(
            blend_tr, y_unb[tr_loc], mu_tr, STRETCH_GRID,
        )
        blend_va = P_unb[va_loc] @ w_f
        oof_blend[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        fold_w.append(w_f)
        fold_s.append(s_f)
    pooled = float(rae(y_unb, oof_blend))
    return pooled, oof_blend, fold_w, fold_s


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- rebuild nb1191's 253-row OOF (was never persisted)")
    print("=" * 78)

    te = load_test()
    te_smiles = te["smiles"].values

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    unb_smiles = te_smiles[unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    # ---- assemble anchor matrix ----
    print("\n[anchors]")
    oof_cols, indiv_rae = [], {}
    for disp, oof_rel in ANCHORS:
        if oof_rel == "_RECONSTRUCT_nb1150_oof":
            oof = reconstruct_nb1150_oof(n_unb)
        else:
            p = DATA_PROCESSED / oof_rel
            assert p.exists(), f"missing OOF: {p}"
            oof = np.load(p).astype(np.float64)
        assert oof.shape == (n_unb,), f"{disp} oof {oof.shape}"
        r = float(rae(y_unb, oof))
        indiv_rae[disp] = r
        oof_cols.append(oof)
        print(f"   {disp:14s} oof_RAE={r:.4f}  mean={oof.mean():.3f}  std={oof.std():.3f}")
    P_unb = np.column_stack(oof_cols)
    print(f"[stack] P_unb {P_unb.shape}  K={P_unb.shape[1]}")

    # ---- 5-fold scaffold CV across 5 kf_seeds ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD 5-FOLD CV  kf_seeds={KF_SEEDS}  stretch_grid={STRETCH_GRID}")
    print("-" * 78)
    per_seed = []
    all_oofs = []
    for kf_seed in KF_SEEDS:
        pooled, oof_blend, fold_w, fold_s = cv_run_for_seed(
            P_unb, y_unb, unb_scaffolds, kf_seed,
        )
        per_seed.append({
            "kf_seed": int(kf_seed),
            "pooled_rae": pooled,
            "fold_s": [float(x) for x in fold_s],
            "fold_w_mean": [float(x) for x in np.mean(fold_w, axis=0)],
        })
        all_oofs.append(oof_blend)
        print(f"   seed={kf_seed}  pooled_RAE={pooled:.4f}  "
              f"mean_s={np.mean(fold_s):.3f}  "
              f"w_mean={np.round(np.mean(fold_w, axis=0), 3).tolist()}")

    mean_oof = np.mean(np.column_stack(all_oofs), axis=1)
    mean_oof_rae = float(rae(y_unb, mean_oof))
    pooled_rae_mean_seeds = float(np.mean([r["pooled_rae"] for r in per_seed]))
    pooled_rae_std_seeds = float(np.std([r["pooled_rae"] for r in per_seed]))
    print(f"\n[cv] mean across 5 seeds  pooled_RAE = {pooled_rae_mean_seeds:.4f} "
          f"(+/- {pooled_rae_std_seeds:.4f})")
    print(f"[cv] RAE of mean-of-seed OOFs        = {mean_oof_rae:.4f}")

    # Cross-check against published nb1191 numbers
    expected_pooled = 0.47034   # nb1191_summary.json pooled_rae_mean_seeds
    expected_mean   = 0.46973   # nb1191_summary.json rae_of_mean_of_seed_oofs
    print(f"\n[xref] nb1191_summary.json pooled_rae_mean_seeds = {expected_pooled:.5f}")
    print(f"[xref] reproduced                                  = {pooled_rae_mean_seeds:.5f}")
    print(f"[xref] nb1191_summary.json rae_of_mean_of_seed_oofs = {expected_mean:.5f}")
    print(f"[xref] reproduced                                   = {mean_oof_rae:.5f}")
    reproduced_ok = (abs(pooled_rae_mean_seeds - expected_pooled) < 1e-3
                     and abs(mean_oof_rae - expected_mean) < 1e-3)
    print(f"[xref] match within 1e-3 RAE: {reproduced_ok}")

    # ---- save ----
    np.save(OUT_OOF, mean_oof.astype(np.float32))
    print(f"\n[save] {OUT_OOF}  shape={mean_oof.shape}  dtype=float32")

    summary = {
        "tag": "nb1191_pred_oof",
        "method": "nb1191_OOF_reconstruction_meanofseeds",
        "anchors": [a[0] for a in ANCHORS],
        "anchor_oof_paths": [a[1] for a in ANCHORS],
        "indiv_oof_rae_unb": indiv_rae,
        "kf_seeds": KF_SEEDS,
        "n_folds": N_FOLDS,
        "stretch_grid": STRETCH_GRID,
        "n_unb": n_unb,
        "n_unique_scaffolds": n_unique_scaf,
        "per_seed_results": per_seed,
        "pooled_rae_mean_seeds": pooled_rae_mean_seeds,
        "pooled_rae_std_seeds": pooled_rae_std_seeds,
        "rae_of_mean_of_seed_oofs": mean_oof_rae,
        "mean_oof_mean": float(mean_oof.mean()),
        "mean_oof_std": float(mean_oof.std()),
        "expected_pooled_from_nb1191_summary": expected_pooled,
        "expected_mean_from_nb1191_summary": expected_mean,
        "reproduced_within_1e-3": bool(reproduced_ok),
        "out_oof_path": str(OUT_OOF),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(OUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {OUT_SUMMARY}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   reconstructed nb1191 OOF rows         = {mean_oof.shape[0]}")
    print(f"   pooled_RAE (mean of seeds)            = {pooled_rae_mean_seeds:.4f}")
    print(f"   RAE of mean-of-seed OOFs (mean_rae)   = {mean_oof_rae:.4f}")
    print(f"   reproduces nb1191_summary within 1e-3 = {reproduced_ok}")
    print(f"   wall                                  = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "rae_of_mean_of_seed_oofs",
        "reproduced_within_1e-3",
        "out_oof_path",
    ):
        print(f"  {k}: {res.get(k)}")
