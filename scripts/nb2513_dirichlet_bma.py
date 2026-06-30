"""nb2513 -- Bayesian Model Averaging via Dirichlet posterior over anchor weights.

NEW PARADIGM (vs SLSQP/GA point estimates):
    Full Bayesian posterior over simplex weights via Dirichlet sampling.
    Posterior mean -> blend weights (BMA), then 5-fold scaffold CV on 253.

PROTOCOL:
    Anchors (PRE-clean only):
        - nb2240_K20      (oof: nb2240_mean_bag_oof_K20.npy , te: te_nb2240_K20.npy)
        - chemprop_aux    (oof: nb1133_chemprop_aux_pred_oof.npy , te: te_chemprop_aux.npy)
        - nb1191          SKIP (no pred_oof artefact available, per instruction)

    Prior:        Dirichlet(alpha = [1,...,1])  uniform on simplex
    Likelihood:   N(y_unb | sum_k w_k * pred_k_oof, sigma^2)
                  sigma estimated from residual MAD on best single anchor
    Posterior:    Importance-weight 10000 Dirichlet samples by Gaussian likelihood
                  (Dirichlet is not conjugate for this likelihood, so we do
                  importance sampling from the prior, then take weighted mean)
    Final:        posterior mean weights -> simplex blend

CV:
    5-fold scaffold CV. Inside each fold we refit the Dirichlet posterior on
    fold-train, evaluate on fold-val. Deploy refit on full 253 -> te (513,).

GATE:
    mean_rae < 0.4570 -> PROMOTE
    < 0.4601           -> MARGINAL_BEAT
    else               -> FAIL

Outputs:
    data/processed/nb2513_summary.json
    data/processed/nb2513_pred_oof.npy   (253,)
    data/processed/te_nb2513.npy         (513,)
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

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2513"

# ---- protocol ----
N_FOLDS = 5
KF_SEED = 1001
N_DIRICHLET_SAMPLES = 10000
PRIOR_ALPHA = 1.0           # uniform Dirichlet(1,...,1)
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4601
RNG_SEED = 1729

# ---- anchor candidates ----
# (tag, oof_path, te_path)
ANCHOR_CANDIDATES = [
    ("nb2240_K20",   DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
                      DATA_PROCESSED / "te_nb2240_K20.npy"),
    ("chemprop_aux", DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
                      DATA_PROCESSED / "te_chemprop_aux.npy"),
    ("nb1191",       DATA_PROCESSED / "nb1191_pred_oof.npy",
                      DATA_PROCESSED / "te_nb1191.npy"),
]


def _save_summary(payload: dict):
    out = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[save] {out}")


def _load_anchors():
    """Load only anchors whose oof + te files exist."""
    used = []
    oofs = []
    tes = []
    for tag, oof_p, te_p in ANCHOR_CANDIDATES:
        if oof_p.exists() and te_p.exists():
            oofs.append(np.load(oof_p).astype(np.float64))
            tes.append(np.load(te_p).astype(np.float64))
            used.append(tag)
            print(f"[anchor] LOADED  {tag:14s}  oof={oof_p.name}  te={te_p.name}")
        else:
            print(f"[anchor] SKIP    {tag:14s}  oof_exists={oof_p.exists()} te_exists={te_p.exists()}")
    if len(used) < 2:
        raise RuntimeError(f"need >=2 anchors with both oof+te; got {used}")
    return used, np.stack(oofs, axis=1), np.stack(tes, axis=1)


def _dirichlet_bma_weights(P_train: np.ndarray, y_train: np.ndarray,
                            sigma: float, alpha: float,
                            n_samples: int, rng: np.random.Generator):
    """Importance-sample Dirichlet(alpha) prior, reweight by Gaussian likelihood.

    Returns posterior-mean weights (K,) on simplex.

    P_train : (n_train, K)  anchor predictions on the training fold
    y_train : (n_train,)    truth on training fold
    sigma   : scalar likelihood scale
    alpha   : Dirichlet concentration (uniform if =1)
    """
    K = P_train.shape[1]
    n = P_train.shape[0]

    # 1) prior samples on simplex
    W = rng.dirichlet([alpha] * K, size=n_samples)        # (S, K)

    # 2) blended predictions  (S, n)
    Y_pred = W @ P_train.T                                 # (S, n)

    # 3) log-likelihood:  log N(y | Y_pred, sigma^2)
    #    drop constants -> use SSE
    sse = ((Y_pred - y_train[None, :]) ** 2).sum(axis=1)   # (S,)
    log_lik = -0.5 * sse / (sigma ** 2)

    # 4) self-normalised importance weights
    log_lik -= log_lik.max()      # numerical stability
    iw = np.exp(log_lik)
    iw_sum = iw.sum()
    if iw_sum <= 0 or not np.isfinite(iw_sum):
        # degenerate: fall back to prior mean (uniform on simplex)
        return np.full(K, 1.0 / K), 0.0
    iw /= iw_sum

    # 5) posterior mean weights
    w_post = (iw[:, None] * W).sum(axis=0)
    # renormalise to be safe
    w_post = w_post / w_post.sum()

    # effective sample size (diagnostic)
    ess = 1.0 / (iw ** 2).sum()
    return w_post, float(ess)


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Dirichlet BMA over PRE-clean anchors")
    print("=" * 78)

    rng = np.random.default_rng(RNG_SEED)

    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    n_unb = len(y_unb)

    used, OOF, TE = _load_anchors()
    K = len(used)
    n_test = TE.shape[0]
    assert OOF.shape == (n_unb, K), f"OOF shape {OOF.shape} expected ({n_unb},{K})"
    assert TE.shape[0] == n_test
    print(f"[load] K={K} anchors  OOF={OOF.shape}  TE={TE.shape}  y_unb={y_unb.shape}")

    # Per-anchor sanity
    indiv_rae = {tag: float(rae(y_unb, OOF[:, k])) for k, tag in enumerate(used)}
    for tag, r in indiv_rae.items():
        print(f"[indiv] {tag:14s} OOF RAE = {r:.4f}")
    best_idx = int(np.argmin(list(indiv_rae.values())))
    best_tag = used[best_idx]
    print(f"[indiv] best single anchor = {best_tag} ({indiv_rae[best_tag]:.4f})")

    # sigma from residual MAD on best anchor (robust)
    resid_best = y_unb - OOF[:, best_idx]
    mad = float(np.median(np.abs(resid_best - np.median(resid_best))))
    sigma = max(1.4826 * mad, 1e-3)   # MAD -> std under normality
    print(f"[likelihood] sigma (1.4826 * MAD on {best_tag}) = {sigma:.4f}")

    # ---- scaffolds for CV ----
    te = load_test()
    te_smiles = te["smiles"].astype(str).tolist() if "smiles" in te.columns else te["SMILES"].astype(str).tolist()
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}")

    splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                    shuffle=True, seed=KF_SEED)

    print("\n" + "-" * 78)
    print(f"5-FOLD SCAFFOLD CV  Dirichlet({PRIOR_ALPHA},...) prior, {N_DIRICHLET_SAMPLES} samples")
    print("-" * 78)

    pred_oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold = []
    per_fold_weights = []
    for f_idx, (tr_loc, va_loc) in enumerate(splits):
        ts = time.time()
        w_post, ess = _dirichlet_bma_weights(
            P_train=OOF[tr_loc], y_train=y_unb[tr_loc],
            sigma=sigma, alpha=PRIOR_ALPHA,
            n_samples=N_DIRICHLET_SAMPLES,
            rng=np.random.default_rng(RNG_SEED + f_idx),
        )
        pred_va = OOF[va_loc] @ w_post
        pred_oof[va_loc] = pred_va
        fold_rae = float(rae(y_unb[va_loc], pred_va))
        per_fold.append({
            "fold": f_idx,
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "rae": fold_rae,
            "ess": ess,
            "weights": {tag: float(w_post[k]) for k, tag in enumerate(used)},
            "wall_sec": round(time.time() - ts, 2),
        })
        per_fold_weights.append(w_post)
        wstr = " ".join(f"{tag}={w_post[k]:.3f}" for k, tag in enumerate(used))
        print(f"   fold={f_idx}  n_tr={len(tr_loc):3d} n_va={len(va_loc):3d}  "
              f"RAE={fold_rae:.4f}  ESS={ess:.0f}  {wstr}")

    rae_pooled = float(rae(y_unb, pred_oof))
    fold_raes = np.array([r["rae"] for r in per_fold])
    mean_rae = float(fold_raes.mean())
    std_rae = float(fold_raes.std())
    print(f"\n[cv] pooled OOF RAE = {rae_pooled:.4f}")
    print(f"[cv] mean fold-RAE  = {mean_rae:.4f} +/- {std_rae:.4f}")

    # ---- gate ----
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"[gate] promote<{GATE_PROMOTE}  marginal<{GATE_MARGINAL}  -> {verdict}")

    # ---- deploy: refit Dirichlet BMA on FULL 253 ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit Dirichlet BMA on all 253; predict 513)")
    print("-" * 78)
    w_deploy, ess_deploy = _dirichlet_bma_weights(
        P_train=OOF, y_train=y_unb,
        sigma=sigma, alpha=PRIOR_ALPHA,
        n_samples=N_DIRICHLET_SAMPLES,
        rng=np.random.default_rng(RNG_SEED + 999),
    )
    te_deploy = (TE @ w_deploy).astype(np.float32)
    te_deploy = np.clip(te_deploy, 3.0, 9.0).astype(np.float32)
    te_unb_in = float(rae(y_unb, te_deploy[unb_idx].astype(np.float64)))
    wstr_deploy = " ".join(f"{tag}={w_deploy[k]:.3f}" for k, tag in enumerate(used))
    print(f"   deploy weights:  {wstr_deploy}  ESS={ess_deploy:.0f}")
    print(f"   te_deploy mean={te_deploy.mean():.3f}  std={te_deploy.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")

    # ---- save ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof.astype(np.float32))
    np.save(te_path, te_deploy)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    summary = {
        "tag": TAG,
        "method": "dirichlet_bma_importance_sample_posterior_mean",
        "anchors_used": used,
        "anchors_skipped": [t for t, _, _ in ANCHOR_CANDIDATES if t not in used],
        "K": K,
        "n_unb": n_unb,
        "n_te": n_test,
        "n_unique_scaffolds": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "rng_seed": RNG_SEED,
        "prior_alpha": PRIOR_ALPHA,
        "n_dirichlet_samples": N_DIRICHLET_SAMPLES,
        "likelihood_sigma": sigma,
        "indiv_oof_rae_unb": indiv_rae,
        "best_single_anchor": best_tag,
        "per_fold": per_fold,
        "deploy_weights": {tag: float(w_deploy[k]) for k, tag in enumerate(used)},
        "deploy_ess": ess_deploy,
        "rae_pooled_oof": rae_pooled,
        "mean_fold_rae": mean_rae,
        "std_fold_rae": std_rae,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "te_unb_in_sample_rae": te_unb_in,
        "deploy_te_mean": float(te_deploy.mean()),
        "deploy_te_std": float(te_deploy.std()),
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    _save_summary(summary)

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchors_used        = {used}")
    print(f"   mean fold-RAE       = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   pooled OOF RAE      = {rae_pooled:.4f}")
    print(f"   deploy weights      = {wstr_deploy}")
    print(f"   verdict             = {verdict}")
    print(f"   wall                = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("verdict", "mean_fold_rae", "std_fold_rae", "rae_pooled_oof",
              "te_unb_in_sample_rae", "deploy_weights"):
        print(f"  {k}: {res.get(k)}")
