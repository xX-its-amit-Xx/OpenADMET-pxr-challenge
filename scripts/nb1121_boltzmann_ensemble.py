"""nb1121 -- Boltzmann ensemble across hyperparameter uncertainty.

nb1014 produced a point-estimate optimum (w0=0.7616, s=1.248) for the
chemprop_aux + nb972_long_train blend with rank-stretch. SLSQP + grid
search collapses the full posterior of plausible (w0, s) configurations
onto a single point; the posterior shape itself encodes uncertainty
that can be integrated over.

Hypothesis: a Boltzmann-weighted ensemble across 20 (w0, w1, s)
samples drawn from a prior centered on the nb1014 optimum may give a
tighter / more honest LB estimate than the point-estimate SLSQP, by
implicitly marginalizing over the (w0, s) likelihood surface instead
of picking the argmax.

Procedure:
  1. Draw 20 (w0, s) samples:
       w0 ~ Normal(0.76, 0.05)   clamped to [0.50, 0.95]
       s  ~ Normal(1.25, 0.10)   clamped to [1.00, 1.60]
       w1 = 1 - w0
  2. For each sample compute pred_i = mu + s * ((w0*p_cp + w1*p_nb972) - mu)
     where mu is the in-fold blend mean.
  3. Score in_RAE on the 253 unblind in-sample (i.e. on the same
     fold as the prediction).
  4. Weight each sample by exp(-beta * in_RAE_i)  with beta = 10.
  5. Boltzmann-weighted average across 20 samples => ensemble pred.
  6. Repeat under 5 KFold seeds for honest pooled cross-fit RAE.

Outputs:
  data/processed/te_nb1121.npy
  data/processed/nb1121_summary.json
  submissions/nb1121_boltzmann_ensemble.csv
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
import pandas as pd
from sklearn.model_selection import KFold

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1121"
CANDIDATES = ["chemprop_aux", "nb972_long_train"]
N_SAMPLES = 20
N_FOLDS = 5
SEEDS = [0, 1, 7, 42, 137]
BETA = 10.0

# Prior centered on nb1014 deploy point.
W0_MEAN, W0_STD = 0.76, 0.05
S_MEAN, S_STD = 1.25, 0.10
W0_LO, W0_HI = 0.50, 0.95
S_LO, S_HI = 1.00, 1.60

NB1014_REF_MEAN_POOLED = 0.5930
NB1070_REF_MEAN_POOLED = 0.5814


def load_te(name: str, te_names: np.ndarray) -> np.ndarray:
    npy = DATA_PROCESSED / f"te_{name}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float64)
    sub = pd.read_csv(SUBMISSIONS / f"{name}.csv")
    assert (sub["Molecule Name"].values == te_names).all(), (
        f"{name}: submission row order does not match test order")
    return sub["pEC50"].values.astype(np.float64)


def sample_hyperparams(rng: np.random.Generator,
                       n: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw n (w0, s) samples from clamped Normal priors."""
    w0 = np.clip(rng.normal(W0_MEAN, W0_STD, size=n), W0_LO, W0_HI)
    s = np.clip(rng.normal(S_MEAN, S_STD, size=n), S_LO, S_HI)
    return w0, s


def blend_then_stretch(p_cp: np.ndarray, p_nb972: np.ndarray,
                       w0: float, s: float, mu: float) -> np.ndarray:
    blend = w0 * p_cp + (1.0 - w0) * p_nb972
    return mu + s * (blend - mu)


def boltzmann_predict(P_tr: np.ndarray, y_tr: np.ndarray,
                      P_va: np.ndarray,
                      w0_samples: np.ndarray,
                      s_samples: np.ndarray,
                      beta: float) -> tuple[np.ndarray, np.ndarray, float]:
    """For each sample, fit mu on train fold, score in_RAE on train,
    apply to val fold; return Boltzmann-weighted val prediction.

    Returns (pred_va, weights, mu_train_avg).
    """
    n_samp = len(w0_samples)
    in_raes = np.zeros(n_samp)
    val_preds = np.zeros((n_samp, P_va.shape[0]))
    mus = np.zeros(n_samp)
    for i in range(n_samp):
        w0_i, s_i = float(w0_samples[i]), float(s_samples[i])
        blend_tr = w0_i * P_tr[:, 0] + (1.0 - w0_i) * P_tr[:, 1]
        mu_tr = float(blend_tr.mean())
        pred_tr = mu_tr + s_i * (blend_tr - mu_tr)
        in_raes[i] = float(rae(y_tr, pred_tr))
        val_preds[i] = blend_then_stretch(
            P_va[:, 0], P_va[:, 1], w0_i, s_i, mu_tr)
        mus[i] = mu_tr
    # Boltzmann weights: exp(-beta * in_RAE), with numerical-stable shift.
    logits = -beta * in_raes
    logits -= logits.max()
    w = np.exp(logits)
    w /= w.sum()
    ens_val = (w[:, None] * val_preds).sum(axis=0)
    return ens_val, w, float((w * mus).sum())


def run_one_seed(P_unb: np.ndarray, y_unb: np.ndarray,
                 seed: int) -> dict:
    """5-fold pooled cross-fit using the Boltzmann ensemble per fold."""
    n_unb = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n_unb, np.nan)
    folds = []
    rng = np.random.default_rng(seed * 1000 + 7)
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        w0_samp, s_samp = sample_hyperparams(rng, N_SAMPLES)
        ens_va, w_b, mu_avg = boltzmann_predict(
            P_unb[tr_loc], y_unb[tr_loc], P_unb[va_loc],
            w0_samp, s_samp, BETA)
        oof[va_loc] = ens_va
        rae_va = float(rae(y_unb[va_loc], ens_va))
        eff_n = float(1.0 / (w_b ** 2).sum())
        folds.append({
            "fold": k, "n_va": int(len(va_loc)),
            "val_rae": rae_va, "w_eff_n": eff_n,
            "w_max": float(w_b.max()),
            "w0_eff": float((w_b * w0_samp).sum()),
            "s_eff": float((w_b * s_samp).sum()),
            "mu_eff": mu_avg,
        })
    pooled = float(rae(y_unb, oof))
    return {"seed": seed, "folds": folds, "pooled_rae": pooled, "oof": oof}


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Boltzmann ensemble over {N_SAMPLES} (w0, s) samples "
          f"(beta={BETA})")
    print("=" * 78)

    # ---- Load 513 test ----
    te = load_test()
    te_names = te["name"].values
    preds_513 = np.column_stack([load_te(c, te_names) for c in CANDIDATES])
    print(f"[load] preds_513 shape = {preds_513.shape}")

    # ---- Load 253 unblind ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    P_unb = preds_513[unb_idx]
    print(f"[load] P_unb shape = {P_unb.shape}, y shape = {y_unb.shape}")

    print("\n[indiv] in_RAE on 253 unblind:")
    indiv_rae = {}
    for j, c in enumerate(CANDIDATES):
        r = float(rae(y_unb, P_unb[:, j]))
        indiv_rae[c] = r
        print(f"   {c:30s}: {r:.4f}")

    # =================================================================
    # Honest pooled cross-fit RAE under N_SEEDS KFold seeds
    # =================================================================
    print("\n" + "-" * 78)
    print(f"BOLTZMANN CROSS-FIT  (N_FOLDS={N_FOLDS}, N_SAMPLES={N_SAMPLES})")
    print(f"   prior w0 ~ N({W0_MEAN},{W0_STD})  clamp [{W0_LO},{W0_HI}]")
    print(f"   prior s  ~ N({S_MEAN},{S_STD})   clamp [{S_LO},{S_HI}]")
    print(f"   beta     = {BETA}")
    print("-" * 78)

    per_seed_rae = []
    seed_results = []
    all_w0_eff = []
    all_s_eff = []
    for seed in SEEDS:
        res = run_one_seed(P_unb, y_unb, seed)
        per_seed_rae.append(res["pooled_rae"])
        seed_results.append({
            "seed": seed,
            "pooled_rae": res["pooled_rae"],
            "folds": res["folds"],
        })
        for f in res["folds"]:
            all_w0_eff.append(f["w0_eff"])
            all_s_eff.append(f["s_eff"])
        fold_vals = [round(f["val_rae"], 3) for f in res["folds"]]
        eff_ns = [round(f["w_eff_n"], 1) for f in res["folds"]]
        print(f"   seed {seed:>3d}: pooled_RAE = {res['pooled_rae']:.4f}  "
              f"folds {fold_vals}  eff_n {eff_ns}")

    mean_rae = float(np.mean(per_seed_rae))
    std_rae = float(np.std(per_seed_rae))
    mean_w0_eff = float(np.mean(all_w0_eff))
    mean_s_eff = float(np.mean(all_s_eff))
    print(f"\n[bag] mean pooled CV RAE     = {mean_rae:.4f}  "
          f"(std {std_rae:.4f})")
    print(f"[ref] nb1014 mean pooled RAE = {NB1014_REF_MEAN_POOLED:.4f}")
    print(f"[ref] nb1070 mean pooled RAE = {NB1070_REF_MEAN_POOLED:.4f}")
    print(f"[bag] mean w0_eff (Boltzmann) = {mean_w0_eff:.4f}")
    print(f"[bag] mean s_eff  (Boltzmann) = {mean_s_eff:.4f}")

    # =================================================================
    # Deploy: draw N_SAMPLES, score on 253 in-sample, apply to 513
    # =================================================================
    print("\n" + "-" * 78)
    print("DEPLOY  (Boltzmann ensemble on 513 using 253-in-sample scoring)")
    print("-" * 78)
    rng_dep = np.random.default_rng(20260604)
    w0_dep, s_dep = sample_hyperparams(rng_dep, N_SAMPLES)
    ens_unb, w_dep, mu_dep_avg = boltzmann_predict(
        P_unb, y_unb, P_unb, w0_dep, s_dep, BETA)
    in_rae_dep = float(rae(y_unb, ens_unb))

    # Apply same weights to the 513.
    val_preds_513 = np.zeros((N_SAMPLES, preds_513.shape[0]))
    for i in range(N_SAMPLES):
        w0_i, s_i = float(w0_dep[i]), float(s_dep[i])
        blend_tr = w0_i * P_unb[:, 0] + (1.0 - w0_i) * P_unb[:, 1]
        mu_tr = float(blend_tr.mean())
        val_preds_513[i] = blend_then_stretch(
            preds_513[:, 0], preds_513[:, 1], w0_i, s_i, mu_tr)
    deploy_513 = (w_dep[:, None] * val_preds_513).sum(axis=0).astype(np.float32)

    eff_n_dep = float(1.0 / (w_dep ** 2).sum())
    w0_eff_dep = float((w_dep * w0_dep).sum())
    s_eff_dep = float((w_dep * s_dep).sum())
    print(f"   N_SAMPLES               = {N_SAMPLES}")
    print(f"   eff sample size         = {eff_n_dep:.2f} / {N_SAMPLES}")
    print(f"   max weight              = {w_dep.max():.3f}")
    print(f"   eff w0 (deploy)         = {w0_eff_dep:.4f}")
    print(f"   eff s  (deploy)         = {s_eff_dep:.4f}")
    print(f"   in-sample RAE (253)     = {in_rae_dep:.4f}  "
          "(overfit lower bound)")
    print(f"   te(513) mean/std        = "
          f"{deploy_513.mean():.3f} / {deploy_513.std():.3f}")

    # =================================================================
    # Save
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy_513)
    plain = SUBMISSIONS / f"{TAG}_boltzmann_ensemble.csv"
    pd.DataFrame({
        "SMILES": te["smiles"].values,
        "Molecule Name": te_names,
        "pEC50": deploy_513,
    }).to_csv(plain, index=False)
    print(f"\n[save] te_{TAG}.npy")
    print(f"[save] {plain}")

    delta_vs_nb1014 = mean_rae - NB1014_REF_MEAN_POOLED
    delta_vs_nb1070 = mean_rae - NB1070_REF_MEAN_POOLED
    beats_nb1070 = delta_vs_nb1070 < -0.005
    if beats_nb1070:
        verdict = "BEATS_NB1070"
    elif abs(delta_vs_nb1070) <= 0.005:
        verdict = "TIES_NB1070"
    elif delta_vs_nb1014 < -0.005:
        verdict = "BEATS_NB1014_ONLY"
    else:
        verdict = "WORSE_THAN_BOTH"
    print(f"\n[verdict] delta vs nb1014 = {delta_vs_nb1014:+.4f}")
    print(f"[verdict] delta vs nb1070 = {delta_vs_nb1070:+.4f}  -> {verdict}")

    summary = {
        "tag": TAG,
        "candidates": CANDIDATES,
        "indiv_in_rae": indiv_rae,
        "n_samples": N_SAMPLES,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "beta": BETA,
        "prior_w0_mean": W0_MEAN, "prior_w0_std": W0_STD,
        "prior_w0_clamp": [W0_LO, W0_HI],
        "prior_s_mean": S_MEAN, "prior_s_std": S_STD,
        "prior_s_clamp": [S_LO, S_HI],
        "per_seed_pooled_rae": per_seed_rae,
        "mean_pooled_rae": mean_rae,
        "std_pooled_rae": std_rae,
        "nb1014_ref_mean_pooled": NB1014_REF_MEAN_POOLED,
        "nb1070_ref_mean_pooled": NB1070_REF_MEAN_POOLED,
        "delta_vs_nb1014": delta_vs_nb1014,
        "delta_vs_nb1070": delta_vs_nb1070,
        "verdict": verdict,
        "mean_w0_eff": mean_w0_eff,
        "mean_s_eff": mean_s_eff,
        "deploy_w0_eff": w0_eff_dep,
        "deploy_s_eff": s_eff_dep,
        "deploy_eff_n": eff_n_dep,
        "deploy_w_max": float(w_dep.max()),
        "in_sample_rae_overfit_bound": in_rae_dep,
        "deploy_te_mean": float(deploy_513.mean()),
        "deploy_te_std": float(deploy_513.std()),
        "seed_results": seed_results,
        "plain_submission": str(plain),
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pool                       = {CANDIDATES}")
    print(f"   N samples / beta           = {N_SAMPLES} / {BETA}")
    print(f"   per-seed pooled RAE        = "
          f"{[f'{r:.4f}' for r in per_seed_rae]}")
    print(f"   mean (honest bagged CV)    = {mean_rae:.4f}  "
          f"(std {std_rae:.4f})")
    print(f"   nb1014 ref                 = {NB1014_REF_MEAN_POOLED:.4f}")
    print(f"   nb1070 ref                 = {NB1070_REF_MEAN_POOLED:.4f}")
    print(f"   delta vs nb1070            = {delta_vs_nb1070:+.4f}")
    print(f"   verdict                    = {verdict}")
    print(f"   deploy (w0_eff, s_eff)     = ({w0_eff_dep:.3f}, {s_eff_dep:.3f})")
    print(f"   wall                       = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("per_seed_pooled_rae", "mean_pooled_rae", "std_pooled_rae",
              "delta_vs_nb1014", "delta_vs_nb1070", "verdict",
              "mean_w0_eff", "mean_s_eff",
              "deploy_eff_n", "in_sample_rae_overfit_bound",
              "plain_submission"):
        print(f"  {k}: {res.get(k)}")
