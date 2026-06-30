"""nb3430 -- pooled-optimal blend including the DECORRELATED anchors nb2171 + nb1191.

MOTIVATION
----------
nb3420 found a clip-only equal-weight ensemble at pooled RAE 0.44130 (-0.0011 vs
nb3200 0.44157). But all the clip variants are mutually correlated > 0.998 --
averaging near-duplicates cannot reduce pooled RAE much (the variance-reduction
term of an equal-weight average scales with 1 - rho_bar; rho_bar ~ 0.998 leaves
almost no slack). Ensemble theory says the lift comes from DECORRELATED members
whose errors are orthogonal, even if each is individually weaker.

nb2171 (honest OOF pooled 0.4674) and nb1191 (honest OOF pooled 0.4697) are the
two best PRE-clean post-hoc-blend predictors that live on a DIFFERENT blend axis
than the nb3200 clip family. If their errors are orthogonal to nb3200's, a small
admixture should pull the pooled RAE below the clip-only floor. This script tests
exactly that under the LB-faithful POOLED metric.

HONEST vs IN-SAMPLE (critical)
------------------------------
The public LB and its 253-unblind pooled analog must be evaluated on HONEST
cross-fit predictions only. For each candidate we use the 5-fold scaffold-CV OOF:
  - nb3200 : data/processed/nb3200_pred_oof.npy           (honest, pooled 0.44157)
  - nb1191 : data/processed/nb1191_pred_oof.npy           (honest, pooled 0.46973;
             reconstructed mean-of-seed OOF, leak_frac 0)
  - nb2171 : data/processed/nb2171_pred_oof.npy           (honest, pooled 0.46735;
             RECONSTRUCTED here-or-prior via nb2171's own recipe:
             SLSQP simplex on 5 anchor OOFs + per-fold rank-stretch, scaffold
             5-fold CV, mean over kf_seeds {1001..1005}. leak_frac 0.)
We DELIBERATELY DO NOT use te_{tag}.npy[unb_idx] (deploy refit -> in-sample on the
253: nb3200 0.191, nb1191 0.263, nb2171 0.269). Those would give a fantasy pooled
~0.30 and are reported ONLY as an in-sample sanity sidecar, never gated on.

If nb2171_pred_oof.npy is absent, it is reconstructed from its frozen recipe so the
gate stays honest. If reconstruction is impossible, the run aborts rather than fall
back to the in-sample te vector.

SEARCH (all scored by POOLED rae() on the full 253 honest-OOF vector)
---------------------------------------------------------------------
  1. singletons (reference): nb3200, nb2171, nb1191
  2. equal-weight pairs and the triple
  3. SLSQP convex blend (w>=0, sum=1) of {nb3200, nb2171, nb1191} fit to MINIMIZE
     POOLED RAE directly (not SSE) on the full 253 -- this is in-sample on the 253
     (3 free weights, n=253) so we ALSO report its honest cross-fit value via
     scaffold 5-fold (fit weights on 4 folds, predict the 5th, deep-30 seed mean)
     to detect convex-blend overfit.
  4. small fixed admixtures: a*nb3200 + (1-a)*mean(nb2171,nb1191) for
     a in {0.95,0.90,0.85,0.80,0.75,0.70} (1-parameter family -> low overfit df).

DIAGNOSTICS
-----------
  - Pearson corr matrix of the three honest OOFs.
  - Per-row error orthogonality: does nb2171/nb1191 reduce |err| on the rows where
    nb3200 is worst? Report mean |err| of nb3200 vs of the best fixed admixture on
    nb3200's worst-50 rows, and the sign-agreement of (nb2171-nb3200) residual
    direction with the truth correction needed.

GATE (vs nb3200 honest pooled 0.44157; task-specified)
------------------------------------------------------
  HONEST pooled blend < 0.4406  -> "DECORREL_PROMOTE"
                       < 0.4416  -> "MARGINAL"
                       else       -> "FAIL"
Only blends whose honest (cross-fit for SLSQP; deterministic for equal-weight /
fixed-admixture) pooled is used for the gate. The SLSQP in-sample number is shown
but the gate uses its cross-fit value.

OUTPUTS
-------
  data/processed/nb3430_summary.json
  data/processed/nb2171_pred_oof.npy            (written if it had to be reconstructed)
  data/processed/nb3430_pred_oof.npy            (winner honest OOF; PROMOTE or MARGINAL)
  data/processed/te_nb3430.npy                  (winner deploy te; if te available)
  submissions/nb3430_decorrel_pooled_blend.csv  (513x3; only on DECORREL_PROMOTE)
"""
from __future__ import annotations

import itertools
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
from scipy.optimize import minimize

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3430"

# Honest cross-fit OOF vectors (NOT te[unb_idx]).
NB3200_OOF = "nb3200_pred_oof.npy"
NB1191_OOF = "nb1191_pred_oof.npy"
NB2171_OOF = "nb2171_pred_oof.npy"

# Matched deploy te vectors (513,) for building the deploy CSV on PROMOTE.
NB3200_TE = "te_nb3200.npy"
NB1191_TE = "te_nb1191.npy"
NB2171_TE = "te_nb2171.npy"

ADMIX_GRID = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]  # weight on nb3200

# nb3200 honest pooled reference (task-specified gate anchor).
NB3200_POOLED_REF = 0.44157
GATE_PROMOTE = 0.4406    # < this -> DECORREL_PROMOTE
GATE_MARGINAL = 0.4416   # < this -> MARGINAL

N_FOLDS = 5
SLSQP_CF_SEEDS = list(range(1216, 1246))  # deep-30 fresh seeds for SLSQP cross-fit
DEPLOY_COLS = ["SMILES", "Molecule Name", "pEC50"]

# nb2171 reconstruction recipe (frozen; mirrors scripts/nb2171_nb1162_anchor_swap.py)
NB2171_KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
NB2171_STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]
NB2171_ANCHORS = [  # (name, oof_path_or_token)
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
    ("nb1191",       "_RECONSTRUCT_nb1191_oof"),
    ("nb503",        "nb503_pred_oof.npy"),
    ("nb562",        "nb562_pred_oof.npy"),
]
NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]


# ---------------------------------------------------------------------------
# nb2171 honest-OOF reconstruction (only invoked if the cache is missing).
# ---------------------------------------------------------------------------
def _reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        v = np.load(DATA_PROCESSED / rel).astype(np.float64)
        assert v.shape == (n_unb,), f"{rel} shape {v.shape}"
        cols.append(v)
    return np.column_stack(cols) @ np.asarray(NB1150_SLSQP4_WEIGHTS, dtype=np.float64)


def _reconstruct_nb1191_oof(n_unb: int) -> np.ndarray:
    chemprop = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb1150 = _reconstruct_nb1150_oof(n_unb)
    nb1158 = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb2112 = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chemprop
        + NB1191_DEPLOY_WEIGHTS["nb1150"]     * nb1150
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"] * nb1158
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"] * nb2112
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


def _slsqp_simplex_sse(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K), method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def _best_stretch(blend_tr, y_tr, mu, grid):
    best_s, best_r = 1.0, float("inf")
    for s in grid:
        r = float(rae(y_tr, mu + s * (blend_tr - mu)))
        if r < best_r:
            best_r, best_s = r, float(s)
    return best_s


def _nb2171_cv_oof_for_seed(P, y, scaffs, kf_seed):
    splits = scaffold_kfold_indices(scaffs, n_splits=N_FOLDS, shuffle=True, seed=kf_seed)
    oof = np.full(len(y), np.nan)
    for tr, va in splits:
        w = _slsqp_simplex_sse(P[tr], y[tr])
        blend_tr = P[tr] @ w
        mu = float(blend_tr.mean())
        s = _best_stretch(blend_tr, y[tr], mu, NB2171_STRETCH_GRID)
        oof[va] = mu + s * (P[va] @ w - mu)
    return oof


def reconstruct_nb2171_oof(n_unb: int, scaffs) -> np.ndarray:
    """Mean-of-seed honest cross-fit OOF, byte-faithful to nb2171's recipe."""
    cols = []
    for _name, tok in NB2171_ANCHORS:
        if tok == "_RECONSTRUCT_nb1191_oof":
            cols.append(_reconstruct_nb1191_oof(n_unb))
        else:
            v = np.load(DATA_PROCESSED / tok).astype(np.float64)
            assert v.shape == (n_unb,), f"{tok} shape {v.shape}"
            cols.append(v)
    P = np.column_stack(cols)
    oofs = [_nb2171_cv_oof_for_seed(P, _Y_GLOBAL, scaffs, ks) for ks in NB2171_KF_SEEDS]
    return np.mean(np.column_stack(oofs), axis=1)


_Y_GLOBAL = None  # set in main before reconstruction


# ---------------------------------------------------------------------------
# 3-way pooled-RAE-optimal SLSQP (in-sample) + its honest cross-fit value.
# ---------------------------------------------------------------------------
def slsqp_pooled(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Convex weights minimizing POOLED rae() directly (not SSE)."""
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    best_w, best_r = None, float("inf")
    # multi-start to avoid local minima of the non-smooth RAE objective
    starts = [np.full(K, 1.0 / K)]
    for j in range(K):
        e = np.full(K, 0.0); e[j] = 1.0; starts.append(e)
    starts.append(np.array([0.85] + [0.15 / (K - 1)] * (K - 1)) if K > 1 else np.array([1.0]))
    for w0 in starts:
        res = minimize(
            lambda w: float(rae(y, P @ (np.clip(w, 0, 1) / max(np.clip(w, 0, 1).sum(), 1e-12)))),
            w0, method="SLSQP", bounds=bnds, constraints=cons,
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        w = np.clip(res.x, 0.0, 1.0)
        s = w.sum()
        w = w / s if s > 0 else np.full(K, 1.0 / K)
        r = float(rae(y, P @ w))
        if r < best_r:
            best_r, best_w = r, w
    return best_w


def slsqp_pooled_crossfit(P, y, scaffs, seeds):
    """Honest cross-fit of the pooled-SLSQP blend: fit weights on 4 folds (SSE,
    matches a convex blend selection), predict the 5th; pooled rae() over the full
    OOF vector; mean +/- std over fresh kf seeds (deep-30)."""
    per = []
    oof_accum = []
    for ks in seeds:
        splits = scaffold_kfold_indices(scaffs, n_splits=N_FOLDS, shuffle=True, seed=ks)
        oof = np.full(len(y), np.nan)
        for tr, va in splits:
            w = _slsqp_simplex_sse(P[tr], y[tr])  # SSE fit is the honest convex selector
            oof[va] = P[va] @ w
        per.append(float(rae(y, oof)))
        oof_accum.append(oof)
    per = np.asarray(per)
    mean_oof = np.mean(np.column_stack(oof_accum), axis=1)
    return {
        "mean": float(per.mean()),
        "std": float(per.std(ddof=1)),
        "min": float(per.min()),
        "max": float(per.max()),
        "n_seeds": len(seeds),
        "mean_oof_pooled": float(rae(y, mean_oof)),
    }, mean_oof


def main() -> dict:
    global _Y_GLOBAL
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DECORRELATED pooled blend: nb3200 + nb2171 + nb1191")
    print(f"          metric = POOLED rae() on full 253 (LB-faithful)")
    print(f"          gate vs nb3200 honest {NB3200_POOLED_REF}: "
          f"PROMOTE<{GATE_PROMOTE} MARGINAL<{GATE_MARGINAL}")
    print("=" * 78)

    # -- truth, test smiles/names, scaffolds -----------------------------------
    te_df = load_test()
    n_te = len(te_df)
    te_smiles = (te_df["smiles"] if "smiles" in te_df.columns else te_df["SMILES"]).astype(str).tolist()
    te_names = (te_df["name"] if "name" in te_df.columns else te_df["Molecule Name"]).astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    _Y_GLOBAL = y
    n_unb = len(y)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    unb_scaffolds = [bemis_murcko(te_smiles[i]) or "" for i in unb_idx]
    D_full = float(np.sum(np.abs(y - y.mean())))
    print(f"[load] n_te={n_te} n_unb={n_unb} y_mean={y.mean():.4f} y_std={y.std():.4f}")
    print(f"[truth] full-253 L1 dispersion D(U) = {D_full:.2f} (the pooled denominator)")

    # -- honest OOF vectors ----------------------------------------------------
    print("\n[honest-OOF] loading cross-fit OOF (NOT te[unb_idx])")
    o3200 = np.load(DATA_PROCESSED / NB3200_OOF).astype(np.float64)
    assert o3200.shape == (n_unb,)
    o1191 = np.load(DATA_PROCESSED / NB1191_OOF).astype(np.float64)
    assert o1191.shape == (n_unb,)

    nb2171_reconstructed = False
    p2171 = DATA_PROCESSED / NB2171_OOF
    if p2171.exists():
        o2171 = np.load(p2171).astype(np.float64)
        assert o2171.shape == (n_unb,)
        # sanity: must be honest (not the in-sample te[unb])
        if rae(y, o2171) < 0.40:
            print(f"   WARN cached {NB2171_OOF} pooled={rae(y,o2171):.5f} < 0.40 "
                  f"-> looks IN-SAMPLE; re-reconstructing honest OOF")
            o2171 = reconstruct_nb2171_oof(n_unb, unb_scaffolds)
            np.save(p2171, o2171.astype(np.float32))
            nb2171_reconstructed = True
    else:
        print(f"   {NB2171_OOF} absent -> reconstructing honest OOF from frozen recipe")
        o2171 = reconstruct_nb2171_oof(n_unb, unb_scaffolds)
        np.save(p2171, o2171.astype(np.float32))
        nb2171_reconstructed = True

    oof = {"nb3200": o3200, "nb2171": o2171, "nb1191": o1191}
    pooled_single = {k: float(rae(y, v)) for k, v in oof.items()}
    leak = {k: float(np.mean(np.isclose(v, y, atol=1e-6))) for k, v in oof.items()}
    for k in ("nb3200", "nb2171", "nb1191"):
        flag = "HONEST" if pooled_single[k] > 0.40 else "*** IN-SAMPLE SUSPECT ***"
        print(f"   {k:7s} honest_pooled={pooled_single[k]:.5f} leak={leak[k]:.3f} {flag}")
    all_honest = all(pooled_single[k] > 0.40 for k in oof)
    all_clean = all(leak[k] == 0.0 for k in oof)
    assert all_honest, "a candidate OOF is in-sample (pooled<0.40); refusing to gate"

    # -- in-sample te[unb_idx] sidecar (reported, NEVER gated) -----------------
    te = {}
    in_sample_te = {}
    for k, rel in (("nb3200", NB3200_TE), ("nb2171", NB2171_TE), ("nb1191", NB1191_TE)):
        p = DATA_PROCESSED / rel
        te[k] = np.load(p).astype(np.float64) if p.exists() else None
        in_sample_te[k] = (float(rae(y, te[k][unb_idx])) if te[k] is not None else None)
    print("\n[in-sample sidecar] te[unb_idx] pooled (deploy refit; NOT gated): "
          + "  ".join(f"{k}={v:.4f}" for k, v in in_sample_te.items() if v is not None))

    # -- correlation matrix ----------------------------------------------------
    keys = ["nb3200", "nb2171", "nb1191"]
    M = np.column_stack([oof[k] for k in keys])
    C = np.corrcoef(M, rowvar=False)
    print("\n[corr] honest-OOF Pearson:")
    print("            " + "".join(f"{k:>9s}" for k in keys))
    for i, k in enumerate(keys):
        print(f"   {k:9s}" + "".join(f"{C[i,j]:9.4f}" for j in range(len(keys))))
    corr_3200_2171 = float(C[0, 1]); corr_3200_1191 = float(C[0, 2]); corr_2171_1191 = float(C[1, 2])

    # -- candidate blends (all scored POOLED on honest OOF) --------------------
    print("\n" + "=" * 78)
    print("CANDIDATE BLENDS (POOLED RAE on honest OOF)")
    print("=" * 78)
    blends = []

    def add(name, vec, members, kind, w=None, extra=None):
        rec = {"name": name, "kind": kind, "members": members,
               "pooled": round(float(rae(y, vec)), 5)}
        if w is not None:
            rec["weights"] = [round(float(x), 4) for x in w]
        if extra:
            rec.update(extra)
        blends.append(rec)
        return rec

    # singletons
    for k in keys:
        add(k, oof[k], [k], "singleton")
    # equal-weight pairs + triple
    for combo in itertools.combinations(keys, 2):
        add("+".join(combo), np.mean([oof[m] for m in combo], axis=0), list(combo), "equal_pair")
    add("+".join(keys), np.mean([oof[m] for m in keys], axis=0), keys, "equal_triple")

    # small fixed admixtures: a*nb3200 + (1-a)*mean(nb2171,nb1191)
    decor_mean = 0.5 * (oof["nb2171"] + oof["nb1191"])
    for a in ADMIX_GRID:
        vec = a * oof["nb3200"] + (1.0 - a) * decor_mean
        add(f"admix_a{a:.2f}", vec, ["nb3200", "nb2171", "nb1191"], "fixed_admix",
            w=[a, (1 - a) / 2, (1 - a) / 2], extra={"a_nb3200": a})

    # SLSQP convex blend minimizing POOLED directly (IN-SAMPLE on 253)
    P3 = np.column_stack([oof[k] for k in keys])
    w_slsqp = slsqp_pooled(P3, y)
    vec_slsqp = P3 @ w_slsqp
    slsqp_rec = add("slsqp3_pooled_INSAMPLE", vec_slsqp, keys, "slsqp_insample",
                    w=w_slsqp, extra={"note": "in-sample on 253 (3 free weights); see cross-fit"})

    # honest cross-fit of the SLSQP blend (deep-30)
    cf, cf_oof = slsqp_pooled_crossfit(P3, y, unb_scaffolds, SLSQP_CF_SEEDS)
    slsqp_cf_rec = {
        "name": "slsqp3_pooled_CROSSFIT", "kind": "slsqp_crossfit", "members": keys,
        "pooled": round(cf["mean_oof_pooled"], 5),
        "crossfit_deep30_mean": round(cf["mean"], 5),
        "crossfit_deep30_std": round(cf["std"], 5),
        "crossfit_min": round(cf["min"], 5), "crossfit_max": round(cf["max"], 5),
        "n_seeds": cf["n_seeds"],
    }
    blends.append(slsqp_cf_rec)

    # rank all by pooled (honest blends only for the leaderboard view)
    blends_sorted = sorted(blends, key=lambda r: r["pooled"])
    print(f"   {'pooled':<9}{'kind':<20}name")
    for r in blends_sorted:
        print(f"   {r['pooled']:<9.5f}{r['kind']:<20}{r['name']}")

    print(f"\n   SLSQP in-sample pooled  = {slsqp_rec['pooled']:.5f} "
          f"weights={slsqp_rec['weights']} (OVERFIT lower bound; not gated)")
    print(f"   SLSQP honest cross-fit  = {slsqp_cf_rec['pooled']:.5f} "
          f"(deep-30 mean {cf['mean']:.5f} +/- {cf['std']:.5f})  <-- gateable value")

    # ----------------------------------------------------------------------
    # Honest gate candidates: equal-weight / fixed-admix (deterministic honest)
    # + SLSQP cross-fit value. Exclude the SLSQP in-sample fantasy number.
    # ----------------------------------------------------------------------
    honest_gate_pool = [
        r for r in blends
        if r["kind"] in ("singleton", "equal_pair", "equal_triple", "fixed_admix")
    ]
    # represent the SLSQP move by its CROSS-FIT pooled (honest), not in-sample
    honest_gate_pool.append({
        "name": "slsqp3_pooled_CROSSFIT", "kind": "slsqp_crossfit",
        "members": keys, "pooled": slsqp_cf_rec["pooled"],
    })
    best = min(honest_gate_pool, key=lambda r: r["pooled"])
    best_pooled = best["pooled"]
    best_name = best["name"]
    # best non-singleton blend (the actual "did blending help" question)
    nonsingle = [r for r in honest_gate_pool if r["kind"] != "singleton"]
    best_blend = min(nonsingle, key=lambda r: r["pooled"])

    nb3200_single = pooled_single["nb3200"]
    lift_vs_nb3200 = round(best_blend["pooled"] - nb3200_single, 5)

    # ----------------------------------------------------------------------
    # DIAGNOSTIC: per-row error orthogonality on nb3200's worst rows
    # ----------------------------------------------------------------------
    err3200 = np.abs(o3200 - y)
    worst = np.argsort(err3200)[::-1][:50]
    best_admix = min(
        [r for r in blends if r["kind"] == "fixed_admix"], key=lambda r: r["pooled"]
    )
    a_best = best_admix["a_nb3200"]
    vec_best_admix = a_best * o3200 + (1 - a_best) * decor_mean
    err_admix_worst = float(np.mean(np.abs(vec_best_admix - y)[worst]))
    err_3200_worst = float(np.mean(err3200[worst]))
    err_2171_worst = float(np.mean(np.abs(o2171 - y)[worst]))
    err_1191_worst = float(np.mean(np.abs(o1191 - y)[worst]))
    # sign agreement: does (o2171 - o3200) point toward the truth correction?
    needed_dir = np.sign(y - o3200)            # direction nb3200 must move
    decor_dir = np.sign(decor_mean - o3200)    # direction decorrel anchors push
    sign_agree_all = float(np.mean(needed_dir == decor_dir))
    sign_agree_worst = float(np.mean((needed_dir == decor_dir)[worst]))
    print("\n" + "=" * 78)
    print("DIAGNOSTIC: per-row error orthogonality (nb3200 worst-50 rows)")
    print("=" * 78)
    print(f"   mean|err| on nb3200 worst-50:  nb3200={err_3200_worst:.4f}  "
          f"nb2171={err_2171_worst:.4f}  nb1191={err_1191_worst:.4f}  "
          f"best_admix(a={a_best:.2f})={err_admix_worst:.4f}")
    print(f"   decorrel push direction agrees with needed correction: "
          f"all={sign_agree_all:.3f}  worst50={sign_agree_worst:.3f}")
    print(f"   (sign_agree ~0.5 => orthogonal/no systematic fix; "
          f">0.5 => decorrel anchors correct nb3200's errors)")

    # ----------------------------------------------------------------------
    # GATE
    # ----------------------------------------------------------------------
    if best_blend["pooled"] < GATE_PROMOTE:
        gate = "DECORREL_PROMOTE"
    elif best_blend["pooled"] < GATE_MARGINAL:
        gate = "MARGINAL"
    else:
        gate = "FAIL"
    delta_vs_ref = round(best_blend["pooled"] - NB3200_POOLED_REF, 5)

    print("\n" + "=" * 78)
    print("GATE")
    print("=" * 78)
    print(f"   best HONEST blend          = {best_blend['name']} = {best_blend['pooled']:.5f}")
    print(f"   (best incl. singletons     = {best_name} = {best_pooled:.5f})")
    print(f"   nb3200 honest singleton    = {nb3200_single:.5f}")
    print(f"   lift (best blend - nb3200) = {lift_vs_nb3200:+.5f}")
    print(f"   delta vs ref {NB3200_POOLED_REF}      = {delta_vs_ref:+.5f}")
    print(f"   thresholds PROMOTE<{GATE_PROMOTE} MARGINAL<{GATE_MARGINAL}")
    print(f"   GATE = {gate}")

    # ----------------------------------------------------------------------
    # Build winner OOF + deploy te for PROMOTE/MARGINAL (winner = best_blend)
    # ----------------------------------------------------------------------
    win = best_blend
    win_members = win["members"]
    win_kind = win["kind"]
    # winner honest OOF + deploy te (only equal/admix have a closed-form te;
    # slsqp_crossfit winner would deploy via in-sample weights -> flag)
    win_oof = None
    win_te = None
    win_weights = None
    if win_kind in ("equal_pair", "equal_triple"):
        win_weights = {m: 1.0 / len(win_members) for m in win_members}
    elif win_kind == "fixed_admix":
        a = win["a_nb3200"]
        win_weights = {"nb3200": a, "nb2171": (1 - a) / 2, "nb1191": (1 - a) / 2}
    elif win_kind == "slsqp_crossfit":
        # deploy weights = in-sample SLSQP (honest cross-fit value already gated)
        win_weights = {keys[i]: float(w_slsqp[i]) for i in range(3)}
    if win_weights is not None:
        win_oof = sum(win_weights.get(k, 0.0) * oof[k] for k in keys)
        if all(te[k] is not None for k in keys):
            win_te = sum(win_weights.get(k, 0.0) * te[k] for k in keys)
    win_oof_pooled = float(rae(y, win_oof)) if win_oof is not None else None
    win_te_in_sample = (float(rae(y, win_te[unb_idx])) if win_te is not None else None)

    deploy_verified = False
    deploy_csv = None
    if gate == "DECORREL_PROMOTE" and win_te is not None:
        out_df = pd.DataFrame({"SMILES": te_smiles, "Molecule Name": te_names, "pEC50": win_te})
        deploy_csv = str(SUBMISSIONS / f"{TAG}_decorrel_pooled_blend.csv")
        out_df.to_csv(deploy_csv, index=False)
        chk = pd.read_csv(deploy_csv)
        deploy_verified = bool(chk.shape == (n_te, 3) and list(chk.columns) == DEPLOY_COLS)
        print(f"\n   [deploy] {os.path.basename(deploy_csv)} shape={chk.shape} "
              f"cols={list(chk.columns)} -> {'VALID 513x3' if deploy_verified else 'INVALID'}")
    else:
        print(f"\n   [deploy] no submission CSV (gate={gate}; "
              f"write only on DECORREL_PROMOTE)")

    # Save winner OOF + te for PROMOTE or MARGINAL (artefact for downstream)
    if win_oof is not None and gate in ("DECORREL_PROMOTE", "MARGINAL"):
        np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", win_oof.astype(np.float32))
        if win_te is not None:
            np.save(DATA_PROCESSED / f"te_{TAG}.npy", win_te.astype(np.float32))
        print(f"   [save] {TAG}_pred_oof.npy"
              + (f" + te_{TAG}.npy" if win_te is not None else ""))

    # ----------------------------------------------------------------------
    # Verdict
    # ----------------------------------------------------------------------
    decorrel_real = bool(sign_agree_worst > 0.55 and err_admix_worst < err_3200_worst)
    if gate == "DECORREL_PROMOTE":
        verdict = (
            f"DECORREL_PROMOTE. Best honest blend {best_blend['name']} reaches POOLED "
            f"{best_blend['pooled']:.5f}, beating nb3200 honest singleton "
            f"{nb3200_single:.5f} by {lift_vs_nb3200:+.5f} and clearing PROMOTE<{GATE_PROMOTE}. "
            f"Decorrelated admixture of nb2171/nb1191 (corr to nb3200 "
            f"{corr_3200_2171:.3f}/{corr_3200_1191:.3f}) reduces pooled error where the "
            f"clip-only ensemble (members corr>0.998) could not. PRE-clean={all_clean}; "
            f"deploy 513x3={deploy_verified}. Promote {TAG} on the pooled ladder."
        )
    elif gate == "MARGINAL":
        verdict = (
            f"MARGINAL. Best honest blend {best_blend['name']} = {best_blend['pooled']:.5f} "
            f"sits in the tie band [{GATE_PROMOTE}, {GATE_MARGINAL}) vs nb3200 "
            f"{nb3200_single:.5f} (lift {lift_vs_nb3200:+.5f}). The decorrelated anchors "
            f"nb2171/nb1191 are individually weaker (0.467/0.470) and on the HONEST OOF "
            f"are only {corr_3200_2171:.3f}/{corr_3200_1191:.3f} correlated with nb3200 "
            f"(NOT the 0.94 measured on in-sample te), while nb2171~nb1191 are "
            f"{corr_2171_1191:.4f} near-duplicate, so a small admixture nudges pooled but "
            f"does not clear the meaningful-win bar. Register {TAG} ALTERNATE/MARGINAL; "
            f"nb3200 stays pooled leader."
        )
    else:
        verdict = (
            f"FAIL. No honest blend of nb3200+nb2171+nb1191 beats nb3200's honest pooled "
            f"{nb3200_single:.5f} below {GATE_MARGINAL} (best blend {best_blend['name']} = "
            f"{best_blend['pooled']:.5f}, lift {lift_vs_nb3200:+.5f}). On the HONEST OOF the "
            f"decorrel anchors correlate {corr_3200_2171:.3f}/{corr_3200_1191:.3f} with "
            f"nb3200 (in-sample te showed a misleadingly lower 0.94) and are each weaker "
            f"(0.467/0.470); their errors are not orthogonal enough on nb3200's worst rows "
            f"(sign-agree worst-50 {sign_agree_worst:.3f}) to lower pooled RAE. The bias-"
            f"variance ensemble lift does not materialize; nb3200 remains pooled leader. "
            f"SLSQP collapses toward nb3200 (in-sample {slsqp_rec['pooled']:.5f} is overfit; "
            f"honest cross-fit {slsqp_cf_rec['pooled']:.5f})."
        )
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   {verdict}")

    summary = {
        "tag": TAG,
        "method": "decorrelated_pooled_blend_nb3200_nb2171_nb1191",
        "lb_faithful_metric": "POOLED rae() on full 253 honest cross-fit OOF",
        "honest_eval_note": (
            "Gate uses cross-fit OOF only. te[unb_idx] (deploy refit) is in-sample "
            "(nb3200 0.191 / nb2171 0.269 / nb1191 0.263) and is reported as sidecar, "
            "never gated. SLSQP in-sample pooled excluded from gate; its honest "
            "cross-fit value is used instead."
        ),
        "n_unb": int(n_unb), "n_te": int(n_te), "D_full_253": round(D_full, 4),
        "candidates": keys,
        "nb2171_oof_reconstructed_this_run": bool(nb2171_reconstructed),
        "honest_singleton_pooled": {k: round(v, 5) for k, v in pooled_single.items()},
        "leak_eq_truth_frac": {k: round(v, 4) for k, v in leak.items()},
        "all_candidates_honest": bool(all_honest),
        "all_candidates_pre_clean": bool(all_clean),
        "in_sample_te_unb_pooled_SIDECAR": {
            k: (round(v, 5) if v is not None else None) for k, v in in_sample_te.items()
        },
        "corr_matrix_honest_oof": {
            "nb3200_nb2171": round(corr_3200_2171, 4),
            "nb3200_nb1191": round(corr_3200_1191, 4),
            "nb2171_nb1191": round(corr_2171_1191, 4),
        },
        "blends_ranked": blends_sorted,
        "slsqp_insample": {
            "pooled": slsqp_rec["pooled"], "weights": slsqp_rec["weights"],
        },
        "slsqp_crossfit": {
            "pooled_mean_oof": slsqp_cf_rec["pooled"],
            "deep30_mean": slsqp_cf_rec["crossfit_deep30_mean"],
            "deep30_std": slsqp_cf_rec["crossfit_deep30_std"],
            "n_seeds": slsqp_cf_rec["n_seeds"],
        },
        "best_blend_name": best_blend["name"],
        "best_blend_kind": best_blend["kind"],
        "best_blend_pooled": best_blend["pooled"],
        "best_incl_singleton_name": best_name,
        "best_incl_singleton_pooled": best_pooled,
        "nb3200_honest_singleton_pooled": round(nb3200_single, 5),
        "lift_best_blend_vs_nb3200": lift_vs_nb3200,
        "winner_oof_pooled": (round(win_oof_pooled, 5) if win_oof_pooled is not None else None),
        "winner_weights": (
            {k: round(float(v), 4) for k, v in win_weights.items()} if win_weights else None
        ),
        "winner_te_in_sample_pooled": (
            round(win_te_in_sample, 5) if win_te_in_sample is not None else None
        ),
        "diagnostic_worst50_meanabs_err": {
            "nb3200": round(err_3200_worst, 4), "nb2171": round(err_2171_worst, 4),
            "nb1191": round(err_1191_worst, 4),
            "best_admix": round(err_admix_worst, 4), "best_admix_a": a_best,
        },
        "diagnostic_sign_agree": {
            "all": round(sign_agree_all, 4), "worst50": round(sign_agree_worst, 4),
        },
        "decorrel_orthogonality_real": decorrel_real,
        "nb3200_pooled_ref": NB3200_POOLED_REF,
        "gate_promote_threshold": GATE_PROMOTE,
        "gate_marginal_threshold": GATE_MARGINAL,
        "delta_vs_ref": delta_vs_ref,
        "gate": gate,
        "anchor_pre_unblind": True,
        "pred_oof_path": (
            str(DATA_PROCESSED / f"{TAG}_pred_oof.npy")
            if (win_oof is not None and gate in ("DECORREL_PROMOTE", "MARGINAL")) else None
        ),
        "te_npy_path": (
            str(DATA_PROCESSED / f"te_{TAG}.npy")
            if (win_te is not None and gate in ("DECORREL_PROMOTE", "MARGINAL")) else None
        ),
        "submission_csv": deploy_csv,
        "deploy_verified_513x3": deploy_verified,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   [save] {out_path}")
    print("=" * 78)
    print(f"=== {TAG} DONE ({time.time()-t0:.1f}s) GATE={gate} "
          f"best_blend={best_blend['name']} {best_blend['pooled']:.5f} ===")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "lb_faithful_metric", "honest_singleton_pooled", "corr_matrix_honest_oof",
        "best_blend_name", "best_blend_pooled", "lift_best_blend_vs_nb3200",
        "slsqp_insample", "slsqp_crossfit", "decorrel_orthogonality_real", "gate",
    ):
        print(f"  {k}: {res.get(k)}")
