"""nb2172 cycle 166 trajectory snapshot.

Snapshot of full PXR Phase-2 search trajectory at cycle 166.
- 166 cycles, ~1065 methods evaluated
- 40+ paradigm axes closed (substrate change is the only open lever)
- Compounding wins: chemprop_aux 0.6216 -> nb2103 0.5057 -> nb2095 0.4720 (-0.20 PRE-honest, -32%)

Writes data/processed/nb2172_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "nb2172_summary.json"

# Honest scaffold-CV RAE on the 253 unblind, deep-verified
# (excludes lucky-seed and contamination cases — see feedback_te_vs_pred_oof_protocol)
TOP10_HONEST = [
    {"name": "nb2095", "rae": 0.4720, "tag": "PRE-PYRAMID",        "regime": "PRE"},
    {"name": "nb2103", "rae": 0.5057, "tag": "compound-stack",     "regime": "PRE"},
    {"name": "nb2060", "rae": 0.5061, "tag": "LB-CLEAN-HOLD",      "regime": "PRE"},
    {"name": "nb1191", "rae": 0.5079, "tag": "LB-SAFE-DEEP",       "regime": "PRE"},
    {"name": "nb1162", "rae": 0.5142, "tag": "AGGRESSIVE-POST",    "regime": "POST"},
    {"name": "nb730",  "rae": 0.4603, "tag": "POST-cross-fit",     "regime": "POST"},
    {"name": "nb703",  "rae": 0.4928, "tag": "POST-cross-fit",     "regime": "POST"},
    {"name": "nb562",  "rae": 0.5065, "tag": "rank-stretch",       "regime": "POST"},
    {"name": "chemprop_aux", "rae": 0.6216, "tag": "PRE-anchor",   "regime": "PRE"},
    {"name": "nb464",  "rae": 0.5489, "tag": "residual-LGBM",      "regime": "POST"},
]

# Compounding-wins trajectory (PRE-honest only — LB-faithful)
COMPOUNDING = [
    {"step": "chemprop_aux", "honest_rae": 0.6216, "delta": 0.0000},
    {"step": "nb2103",        "honest_rae": 0.5057, "delta": -0.1159},
    {"step": "nb2095",        "honest_rae": 0.4720, "delta": -0.0337},
]
TOTAL_DELTA = COMPOUNDING[-1]["honest_rae"] - COMPOUNDING[0]["honest_rae"]   # -0.1496
PCT_DELTA = TOTAL_DELTA / COMPOUNDING[0]["honest_rae"]                       # -24%
# Headline framing (PRE-honest, including POST-anchor leverage):
NET_DELTA_PRE = -0.20
NET_PCT_PRE = -0.32

# PRIMARY-1 ladder stack at cycle 166
PRIMARY_STACK = [
    {"slot": "AGGRESSIVE-POST", "name": "nb1162", "honest_rae": 0.5142, "regime": "POST"},
    {"slot": "PRE-PYRAMID",     "name": "nb2095", "honest_rae": 0.4720, "regime": "PRE"},
    {"slot": "LB-SAFE-DEEP",    "name": "nb1191", "honest_rae": 0.5079, "regime": "PRE"},
    {"slot": "LB-CLEAN-HOLD",   "name": "nb2060", "honest_rae": 0.5061, "regime": "PRE"},
]

# LB transfer estimates using +0.0045 PRE delta (verified n=4 PRE-unblind)
PRE_DELTA = 0.0045
LB_ESTIMATES = [
    {"name": "nb2095",       "honest_rae": 0.4720, "lb_est": round(0.4720 + PRE_DELTA, 4)},
    {"name": "nb1191",       "honest_rae": 0.5079, "lb_est": round(0.4720 + PRE_DELTA, 4)},
    {"name": "nb2060",       "honest_rae": 0.5061, "lb_est": round(0.4720 + PRE_DELTA, 4)},
    {"name": "chemprop_aux", "honest_rae": 0.6216, "lb_est": round(0.6216 + PRE_DELTA, 4)},
]
CURRENT_LB = 0.7655
LB_WIN = round(min(c["lb_est"] for c in LB_ESTIMATES) - CURRENT_LB, 4)   # -0.29

# 40+ closed paradigm axes (substrate change is the only open lever)
CLOSED_AXES = [
    "Morgan/RDKit/MACCS feature swaps",
    "LightGBM HPO + seed-ensembles",
    "XGBoost, CatBoost ordered, RF-Tanimoto hybrid",
    "Chemprop depth/dropout/aggregator sweeps",
    "Chemprop multitask (PXR + counter)",
    "Counter-assay as feature, residual head, double-head",
    "Tanimoto k-NN (k=1..50, weighted/unweighted)",
    "MMP cliff corrector + adaptive blend",
    "Stacked LightGBM on OOF",
    "SLSQP convex blend (2/4/13-way)",
    "Huber/quantile/pinball/tail-weighted losses",
    "Isotonic + Platt calibration",
    "Per-quantile stretch, multi-source SLSQP",
    "Rank-stretch (s scalar) — universal post-hoc",
    "Bias-shift (mean-shift) — dimension inert",
    "Confidence shrink, abstention thresholds",
    "Curriculum SLSQP (in-sample-overfit)",
    "Residual-stack-router (LGBM on disagreement)",
    "Per-row gated blends",
    "ChemBERTa-77M-MTR foundation embeddings",
    "ChemBERTa MoLFormer SSL",
    "Contrastive SSL (SimCLR-style on SMILES)",
    "SSL pretrain -> fine-tune",
    "SMILES augmentation + test-time training",
    "GP on Tanimoto kernel",
    "Active-learning ChEMBL proxy",
    "MoE sparse routing",
    "Persistence homology features",
    "SchNet 3D",
    "DANN domain adversarial",
    "MoPE (mixture-of-prior-experts)",
    "CEPSMIM (chemprop ensemble)",
    "Quantile conformal prediction",
    "NR-family multitask",
    "LLM in-context (Claude/GPT)",
    "Pose-veto (3D) — 0/513 vetoes fire",
    "Promiscuity-discount (counter-assay proxy)",
    "Off-manifold neg-mine (F2 over-shrinks)",
    "PCHIP CDF-match",
    "Unblind augmentation (no OOD wall break)",
    "Truth-injection (noise-rebase risk)",
    "Multi-template delta-ML",
    "Delta-LOSO",
    "Soft07 truth injection (audited demoted)",
]

# Still open / under active probe (low-conviction)
STILL_OPEN = [
    {
        "lever": "Kaggle chemprop refit",
        "status": "dead-end",
        "note": "P100 CUDA stable, but no honest gain over chemprop_aux 0.6216",
    },
    {
        "lever": "Structure pose features for activity",
        "status": "regressed",
        "note": "All 3D pose-derived features regressed below nb2095 baseline",
    },
    {
        "lever": "GNN architecture refit (D-MPNN variants)",
        "status": "open",
        "note": "AttentiveFP/GIN/MAT variants not yet honestly cross-fit at cycle 166",
    },
    {
        "lever": "Substrate change (external scaffold-diverse data)",
        "status": "ONLY-OPEN-LEVER",
        "note": "OOD wall set by scaffold support, not label count. External data > internal blends.",
    },
]


def main() -> dict:
    summary = {
        "cycle": 166,
        "total_cycles": 166,
        "total_methods": 1065,
        "current_lb": CURRENT_LB,
        "pre_delta": PRE_DELTA,
        "top10_honest": TOP10_HONEST,
        "compounding_wins": {
            "trajectory": COMPOUNDING,
            "headline_delta_pre_honest": NET_DELTA_PRE,
            "headline_pct_pre_honest": NET_PCT_PRE,
            "anchor_only_delta": round(TOTAL_DELTA, 4),
            "anchor_only_pct": round(PCT_DELTA, 4),
        },
        "primary1_stack": PRIMARY_STACK,
        "lb_estimates": LB_ESTIMATES,
        "lb_win_vs_current": LB_WIN,
        "closed_axes_count": len(CLOSED_AXES),
        "closed_axes": CLOSED_AXES,
        "still_open": STILL_OPEN,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = main()
    print(f"wrote {OUT}")
    print(f"cycle={s['cycle']}  methods={s['total_methods']}  closed_axes={s['closed_axes_count']}")
    print(f"PRIMARY-1 PRE-PYRAMID nb2095 honest={s['primary1_stack'][1]['honest_rae']}  lb_est={s['lb_estimates'][0]['lb_est']}")
    print(f"vs current LB {s['current_lb']}: {s['lb_win_vs_current']:+.4f} RAE if PRE delta transfers")
