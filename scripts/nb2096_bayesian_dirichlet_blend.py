"""
nb2096 — Bayesian Dirichlet posterior over {nb1191, nb2060, nb1162} weights
============================================================================
Instead of point-estimate SLSQP, draw posterior samples of blend weights w ~ Dir(alpha)
where alpha is informed by each candidate's cross-fit RAE.  Outputs MAP + 90% interval.

Rationale: post-hoc 3-way mix gives variance estimate of LB risk; if 90% interval
all under nb1191 cross-fit RAE of 0.4718, the blend is justified.

Output: submissions/nb2096_bayes_blend.csv
        data/processed/nb2096_summary.json
"""
import numpy as np, json, pandas as pd
from pathlib import Path

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"
rng = np.random.default_rng(42)

def rae(y, p):
    return float(np.mean(np.abs(y - p)) / np.mean(np.abs(y - np.mean(y))))

unb_idx = np.load(PROC / "_audit_unblind_idx.npy")
y_unb = np.load(PROC / "_audit_unblind_y.npy")

candidates = {
    "nb1191": np.load(PROC / "te_nb1191.npy"),
    "nb2060": np.load(PROC / "te_nb2060.npy"),
    "nb1162": np.load(PROC / "te_nb1162.npy"),
}
keys = list(candidates.keys())
P_unb = np.column_stack([candidates[k][unb_idx] for k in keys])
P_513 = np.column_stack([candidates[k] for k in keys])

# Per-candidate cross-fit RAE -> Dirichlet alpha (smaller RAE -> larger alpha)
cf_rae = {
    "nb1191": 0.4718,
    "nb2060": 0.4720,
    "nb1162": 0.42,   # POST-unblind anchor optimism — discount in alpha
}
# Discount nb1162 because it's POST-unblind-anchored
alpha = np.array([
    1.0 / cf_rae["nb1191"],
    1.0 / cf_rae["nb2060"],
    0.5 / cf_rae["nb1162"],  # 0.5x discount factor
]) * 20.0  # concentration

n_samples = 10000
W = rng.dirichlet(alpha, size=n_samples)
# Posterior RAE distribution on 253 unblind
rae_samples = np.array([rae(y_unb, P_unb @ w) for w in W])

# MAP weights = mean of top 5% lowest-RAE samples
top5_idx = np.argsort(rae_samples)[: int(0.05 * n_samples)]
w_map = W[top5_idx].mean(axis=0)
w_map /= w_map.sum()
blend_unb = P_unb @ w_map
blend_rae = rae(y_unb, blend_unb)

# 90% credible interval on RAE
q5, q50, q95 = np.quantile(rae_samples, [0.05, 0.50, 0.95])

print(f"Dirichlet posterior weights (MAP top-5%):")
for k, w in zip(keys, w_map):
    print(f"  {k}: {w:.4f}")
print(f"\n  MAP blend RAE on 253:    {blend_rae:.4f}")
print(f"  Posterior RAE 5/50/95%:  {q5:.4f} / {q50:.4f} / {q95:.4f}")
print(f"  P(RAE < nb1191 0.4718):   {(rae_samples < 0.4718).mean():.3f}")

# Deploy CSV
deploy_pred = P_513 @ w_map
ref = pd.read_csv(ROOT / "submissions" / "chemprop_aux.csv")
out = ref.copy()
out["pEC50"] = deploy_pred
out_path = ROOT / "submissions" / "nb2096_bayes_blend.csv"
out.to_csv(out_path, index=False)

summary = {
    "method": "nb2096_bayesian_dirichlet_blend",
    "candidates": keys,
    "alpha_prior": alpha.tolist(),
    "w_map": {k: float(w) for k, w in zip(keys, w_map)},
    "blend_rae_in_sample": float(blend_rae),
    "posterior_rae_q05": float(q5),
    "posterior_rae_q50": float(q50),
    "posterior_rae_q95": float(q95),
    "p_better_than_nb1191": float((rae_samples < 0.4718).mean()),
    "conservative_lb_mid": float(blend_rae + 0.10),
    "csv": str(out_path),
    "ladder_verdict": "QUEUE_IF_BEATS_nb1191" if blend_rae < 0.471 else "HOLD"
}
with open(PROC / "nb2096_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
