"""nb1314 -- Bayesian Model Averaging (BMA) over 9 residual-learner OOFs.

Hypothesis:
    nb1281 SLSQP simplex stacking on the same 9 OOFs landed at 0.5513
    (MLE point estimate of simplex weights -- overfits at n=253).
    A proper Bayesian approach with Dirichlet(alpha=1) prior + Gaussian
    likelihood (importance-weighted Monte Carlo on the simplex) should
    regularize the weight posterior toward the prior and gain over the MLE.

Protocol:
  1. Load 9 OOFs (same set as nb1281):
       nb1242, nb1252, nb1190, nb1200, nb1211, nb1183, nb1130, nb1153, nb1172.
  2. Prior:       w ~ Dirichlet(alpha = 1, ..., 1)   (uniform on the simplex)
     Likelihood: y | w  ~  N(P @ w, sigma^2 * I)
                 sigma^2 estimated from MLE residuals (plug-in).
  3. Approximate the posterior by simple Monte Carlo:
       a. Sample N_DRAWS i.i.d. Dirichlet(1) draws.
       b. Importance weight each draw by the Gaussian likelihood
          (in log-space, then subtract max for numerical stability,
          then exponentiate and normalize -> self-normalized importance
          weights on the simplex).
       c. Posterior-mean weights = sum_i (iw_i * w_i)
       d. MAP estimate = arg-max log-likelihood draw.
  4. Evaluate pooled RAE on 253 unblind under:
       - posterior-mean simplex weights
       - MAP draw
       - (also report grid BMA with step 0.10 as a sanity check)
  5. Verdict at 0.003 margin vs nb1290 (0.5390).

Outputs:
  data/processed/nb1314_posterior_mean_oof.npy
  data/processed/nb1314_map_oof.npy
  data/processed/nb1314_summary.json
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

TAG = "nb1314"
SEED = 42
N_DRAWS = 5000
DIRICHLET_ALPHA = 1.0           # uniform prior on simplex
NB1290_REF = 0.5390
MARGIN = 0.003

COMPONENT_FILES = [
    ("nb1242", "nb1242_mean_bag_oof.npy", 0.5431),
    ("nb1252", "nb1252_bob_mean_oof.npy", 0.5446),
    ("nb1190", "nb1190_bob_mean_oof.npy", 0.5499),
    ("nb1200", "nb1200_bob_mean_oof.npy", 0.5495),
    ("nb1211", "nb1211_mean_oof.npy",     0.5451),
    ("nb1183", "nb1183_mean_bag_oof.npy", 0.5513),
    ("nb1130", "nb1130_mean_bag_oof.npy", 0.5673),
    ("nb1153", "nb1153_mean_bag_oof.npy", 0.5640),
    ("nb1172", "nb1172_mean_bag_oof.npy", 0.5659),
]


def _gaussian_log_lik(P: np.ndarray, y: np.ndarray,
                      W: np.ndarray, sigma2: float) -> np.ndarray:
    """Vectorized Gaussian log-likelihood for many weight draws.

    P : (n, K) design matrix of OOF predictions
    y : (n,)   targets
    W : (M, K) M simplex weight draws
    sigma2 : scalar variance plug-in

    Returns (M,) log p(y | w_m, sigma^2) values (additive constants kept,
    they cancel after self-normalization).
    """
    # (n, K) @ (K, M) -> (n, M)
    pred = P @ W.T
    diff = y[:, None] - pred                 # (n, M)
    ssq = np.sum(diff * diff, axis=0)        # (M,)
    n = len(y)
    ll = -0.5 * ssq / sigma2 - 0.5 * n * np.log(2.0 * np.pi * sigma2)
    return ll


def _ml_sigma2(P: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """MLE residual variance under candidate weights w (plug-in)."""
    diff = y - P @ w
    return float(np.mean(diff * diff))


def _simplex_grid(K: int, step: float) -> np.ndarray:
    """Enumerate the K-simplex on an integer grid with `step` resolution.

    Returns an (M, K) array of points whose rows sum to 1.

    For K=9 and step=0.10 (M = C(19,8) ~ 75582) this is fine on CPU.
    """
    n_int = int(round(1.0 / step))
    pts: list[list[float]] = []

    def _rec(remaining: int, depth: int, prefix: list[int]) -> None:
        if depth == K - 1:
            prefix2 = prefix + [remaining]
            pts.append([v / n_int for v in prefix2])
            return
        for v in range(remaining + 1):
            _rec(remaining - v, depth + 1, prefix + [v])

    _rec(n_int, 0, [])
    return np.asarray(pts, dtype=np.float64)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- BMA over 9 OOFs  (Dirichlet(alpha=1) prior, Gaussian lik)")
    print(f"          N_DRAWS = {N_DRAWS}   verdict margin {MARGIN} vs "
          f"nb1290 ({NB1290_REF:.4f})")
    print("=" * 78)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] y_unb shape = {y_unb.shape}")

    preds = {}
    standalone_rae = {}
    for tag, fname, ref in COMPONENT_FILES:
        p = DATA_PROCESSED / fname
        if not p.exists():
            raise FileNotFoundError(f"{p} not found ({tag})")
        v = np.load(p).astype(np.float64)
        if v.shape[0] != n_unb:
            raise ValueError(f"shape mismatch: {tag}={v.shape}")
        preds[tag] = v
        standalone_rae[tag] = float(rae(y_unb, v))
        print(f"   {tag:8s}  {p.name:35s}  RAE={standalone_rae[tag]:.4f}  "
              f"(ref {ref:.4f})")

    tags = [t for t, _, _ in COMPONENT_FILES]
    K = len(tags)
    P = np.column_stack([preds[t] for t in tags])
    print(f"\n[stack] design matrix P shape = {P.shape}  ({K} features, "
          f"{n_unb} rows)")

    # -- plug-in sigma^2 from a sensible centre: uniform-mean predictor.
    w_unif = np.full(K, 1.0 / K)
    sigma2_unif = _ml_sigma2(P, y_unb, w_unif)
    rae_unif = float(rae(y_unb, P @ w_unif))
    print(f"\n[plug-in] sigma^2 under uniform 1/K weights = {sigma2_unif:.6f}")
    print(f"[plug-in] uniform-mean predictor RAE          = {rae_unif:.4f}")

    # ====================================================================
    #   BLOCK 1: Monte-Carlo BMA (Dirichlet(1) draws + Gaussian importance)
    # ====================================================================
    print("\n" + "-" * 78)
    print(f"  BLOCK 1: Monte-Carlo BMA   (N_DRAWS = {N_DRAWS})")
    print("-" * 78)
    rng = np.random.default_rng(SEED)
    alpha_vec = np.full(K, DIRICHLET_ALPHA)
    W = rng.dirichlet(alpha_vec, size=N_DRAWS)        # (M, K), rows on simplex
    assert W.shape == (N_DRAWS, K)
    assert np.allclose(W.sum(axis=1), 1.0)

    log_lik = _gaussian_log_lik(P, y_unb, W, sigma2_unif)
    # Self-normalized importance weights on the simplex (prior is uniform on
    # the simplex when alpha=1, so iw = lik / sum(lik) up to a constant).
    log_lik_shift = log_lik - log_lik.max()
    iw = np.exp(log_lik_shift)
    iw /= iw.sum()

    # Effective sample size for diagnostics.
    ess = float(1.0 / np.sum(iw * iw))
    print(f"   ESS = {ess:.1f}  /  {N_DRAWS}   (frac = {ess / N_DRAWS:.4f})")

    # Posterior-mean weights = iw @ W   -> (K,)
    w_post_mean = iw @ W
    w_post_mean = w_post_mean / w_post_mean.sum()    # renormalize for safety
    p_post_mean = P @ w_post_mean
    rae_post_mean = float(rae(y_unb, p_post_mean))
    print(f"\n   posterior-mean weights:")
    for t, w in zip(tags, w_post_mean):
        print(f"     {t}: {w:.4f}")
    print(f"   posterior-mean RAE = {rae_post_mean:.4f}")

    # MAP draw = arg-max log_lik (under uniform prior).
    map_idx = int(np.argmax(log_lik))
    w_map = W[map_idx]
    p_map = P @ w_map
    rae_map = float(rae(y_unb, p_map))
    print(f"\n   MAP draw index = {map_idx}   log-lik = {log_lik[map_idx]:.4f}")
    print(f"   MAP weights:")
    for t, w in zip(tags, w_map):
        print(f"     {t}: {w:.4f}")
    print(f"   MAP RAE = {rae_map:.4f}")

    # Posterior standard deviation per component (for diagnostics).
    w_post_sq = iw @ (W * W)
    w_post_var = np.clip(w_post_sq - w_post_mean ** 2, 0.0, None)
    w_post_sd = np.sqrt(w_post_var)
    print(f"\n   posterior SD per component:")
    for t, sd in zip(tags, w_post_sd):
        print(f"     {t}: {sd:.4f}")

    # ====================================================================
    #   BLOCK 2: Grid BMA sanity check  (step 0.10 over 9-simplex)
    # ====================================================================
    print("\n" + "-" * 78)
    print("  BLOCK 2: Grid BMA sanity check  (step = 0.10 on 9-simplex)")
    print("-" * 78)
    Wg = _simplex_grid(K, 0.10)
    print(f"   grid size M = {Wg.shape[0]}")
    log_lik_g = _gaussian_log_lik(P, y_unb, Wg, sigma2_unif)
    log_lik_g_shift = log_lik_g - log_lik_g.max()
    iw_g = np.exp(log_lik_g_shift)
    iw_g /= iw_g.sum()
    ess_g = float(1.0 / np.sum(iw_g * iw_g))
    w_post_mean_g = iw_g @ Wg
    w_post_mean_g = w_post_mean_g / w_post_mean_g.sum()
    rae_post_mean_g = float(rae(y_unb, P @ w_post_mean_g))
    map_idx_g = int(np.argmax(log_lik_g))
    w_map_g = Wg[map_idx_g]
    rae_map_g = float(rae(y_unb, P @ w_map_g))
    print(f"   grid ESS = {ess_g:.1f}  /  {Wg.shape[0]}")
    print(f"   grid posterior-mean weights:")
    for t, w in zip(tags, w_post_mean_g):
        print(f"     {t}: {w:.4f}")
    print(f"   grid posterior-mean RAE = {rae_post_mean_g:.4f}")
    print(f"   grid MAP weights:")
    for t, w in zip(tags, w_map_g):
        print(f"     {t}: {w:.4f}")
    print(f"   grid MAP RAE = {rae_map_g:.4f}")

    # ====================================================================
    #   VERDICT
    # ====================================================================
    candidates = {
        "posterior_mean":       rae_post_mean,
        "map":                  rae_map,
        "grid_posterior_mean":  rae_post_mean_g,
        "grid_map":             rae_map_g,
        "uniform_1_over_K":     rae_unif,
    }
    best_tag = min(candidates, key=candidates.get)
    best_rae = candidates[best_tag]

    beats_nb1290 = best_rae < NB1290_REF - MARGIN
    flat_nb1290 = abs(best_rae - NB1290_REF) < MARGIN

    if beats_nb1290:
        verdict = f"BMA_BEATS_NB1290 ({best_tag} @ {best_rae:.4f})"
    elif flat_nb1290:
        verdict = f"BMA_FLAT_VS_NB1290 ({best_tag} @ {best_rae:.4f})"
    else:
        verdict = f"BMA_HURTS_VS_NB1290 ({best_tag} @ {best_rae:.4f})"

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"   nb1290 ref RAE                     = {NB1290_REF:.4f}")
    print(f"   posterior-mean RAE (MC, N={N_DRAWS}) = {rae_post_mean:.4f}")
    print(f"   MAP RAE             (MC)            = {rae_map:.4f}")
    print(f"   grid posterior-mean RAE             = {rae_post_mean_g:.4f}")
    print(f"   grid MAP RAE                        = {rae_map_g:.4f}")
    print(f"   uniform 1/K RAE                     = {rae_unif:.4f}")
    print(f"")
    print(f"   best candidate                      : {best_tag} @ {best_rae:.4f}")
    print(f"   delta vs nb1290 ({NB1290_REF:.4f})       : "
          f"{best_rae - NB1290_REF:+.4f}")
    print(f"   beats_nb1290 (margin {MARGIN})         : {beats_nb1290}")
    print(f"   verdict                             : {verdict}")

    # Persist artifacts.
    np.save(DATA_PROCESSED / f"{TAG}_posterior_mean_oof.npy",
            p_post_mean.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_map_oof.npy",
            p_map.astype(np.float32))
    print(f"\n[save] {TAG}_posterior_mean_oof.npy")
    print(f"[save] {TAG}_map_oof.npy")

    summary = {
        "tag": TAG,
        "n_unb": n_unb,
        "n_draws_mc": N_DRAWS,
        "dirichlet_alpha": DIRICHLET_ALPHA,
        "seed": SEED,
        "component_tags": tags,
        "component_files": [f for _, f, _ in COMPONENT_FILES],
        "standalone_rae": standalone_rae,
        "sigma2_uniform_plugin": sigma2_unif,
        "rae_uniform_1_over_K": rae_unif,
        # MC BMA
        "ess_mc": ess,
        "posterior_mean_weights": {t: float(w)
                                   for t, w in zip(tags, w_post_mean)},
        "posterior_sd_weights": {t: float(s)
                                 for t, s in zip(tags, w_post_sd)},
        "rae_posterior_mean": rae_post_mean,
        "map_index_mc": map_idx,
        "map_log_lik_mc": float(log_lik[map_idx]),
        "map_weights": {t: float(w) for t, w in zip(tags, w_map)},
        "rae_map": rae_map,
        # Grid BMA sanity
        "grid_step": 0.10,
        "grid_size": int(Wg.shape[0]),
        "ess_grid": ess_g,
        "grid_posterior_mean_weights": {t: float(w)
                                        for t, w in zip(tags, w_post_mean_g)},
        "rae_grid_posterior_mean": rae_post_mean_g,
        "grid_map_weights": {t: float(w) for t, w in zip(tags, w_map_g)},
        "rae_grid_map": rae_map_g,
        # Verdict
        "candidate_rae_table": candidates,
        "best_candidate_tag": best_tag,
        "best_candidate_rae": best_rae,
        "nb1290_ref": NB1290_REF,
        "delta_best_vs_nb1290": best_rae - NB1290_REF,
        "beats_nb1290": bool(beats_nb1290),
        "flat_vs_nb1290": bool(flat_nb1290),
        "margin": MARGIN,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("standalone_rae",
              "sigma2_uniform_plugin", "rae_uniform_1_over_K",
              "ess_mc",
              "posterior_mean_weights", "posterior_sd_weights",
              "rae_posterior_mean",
              "map_weights", "rae_map",
              "ess_grid",
              "grid_posterior_mean_weights", "rae_grid_posterior_mean",
              "grid_map_weights", "rae_grid_map",
              "best_candidate_tag", "best_candidate_rae",
              "delta_best_vs_nb1290", "beats_nb1290", "verdict"):
        print(f"  {k}: {res.get(k)}")
