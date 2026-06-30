"""nb2300_trajectory.py — Cycle 179 trajectory snapshot.

Comprehensive summary of the PXR Challenge activity-track campaign:
  - 179 cycles, ~1175 distinct methods executed
  - Compounding PRE-honest scaffold-CV wins from chemprop_aux 0.6216 down to nb2240 0.4601
  - 5-way PRIMARY-1 stack (nb2240/nb1162/nb2171/nb2095/nb1191)
  - 50+ closed exploration axes, 2 actively open
  - Phase-2 deadline 2026-07-01 (~23 days remaining)

Writes data/processed/nb2300_trajectory_summary.json with the full structured
snapshot for downstream ladder / readme / cron consumers.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "data" / "processed" / "nb2300_trajectory_summary.json"


def build_compounding_chain() -> list[dict]:
    """Ordered list of compounding PRE-honest scaffold-CV wins."""
    return [
        {
            "rank": 1,
            "candidate": "chemprop_aux",
            "pre_honest_rae": 0.6216,
            "delta_from_prev": None,
            "note": "First LB-faithful anchor (replaced cross-fit-overfit nb703/562/730)",
        },
        {
            "rank": 2,
            "candidate": "nb2103",
            "pre_honest_rae": 0.5057,
            "delta_from_prev": -0.1159,
            "note": "SHAP-K fine grid; first sub-0.51 PRE-honest",
        },
        {
            "rank": 3,
            "candidate": "nb2095",
            "pre_honest_rae": 0.4720,
            "delta_from_prev": -0.0337,
            "note": "PRE pyramid stack; breaks 0.48",
        },
        {
            "rank": 4,
            "candidate": "nb2171",
            "pre_honest_rae": 0.4682,
            "delta_from_prev": -0.0038,
            "note": "nb1162 anchor swap; tighter family weighting",
        },
        {
            "rank": 5,
            "candidate": "nb2240",
            "pre_honest_rae": 0.4601,
            "delta_from_prev": -0.0081,
            "note": "nb2171 + K=20 RFE; current PRIMARY-1",
        },
    ]


def build_lb_transfer_projection() -> dict:
    """LB transfer using verified +0.0045 PRE-delta (from PRE-unblind regime)."""
    pre_delta = 0.0045
    primary_pre = 0.4601
    projected_lb = round(primary_pre + pre_delta, 4)
    current_best_lb = 0.7655
    win_vs_current = round(current_best_lb - projected_lb, 4)
    return {
        "pre_delta_calibration": pre_delta,
        "primary_candidate": "nb2240",
        "primary_pre_honest_rae": primary_pre,
        "projected_lb_rae": projected_lb,
        "current_best_lb_rae": current_best_lb,
        "expected_win_vs_current": win_vs_current,
        "expected_win_pct": round(win_vs_current / current_best_lb * 100, 1),
        "regime": "PRE-unblind (trained on 4139 only); LB delta validated n=4",
    }


def build_primary_stack() -> list[dict]:
    """5-way PRIMARY-1 rotation stack."""
    return [
        {"slot": "PRIMARY-1", "candidate": "nb2240", "pre_honest_rae": 0.4601,
         "role": "K=20 RFE anchor, deploy default"},
        {"slot": "PRIMARY-2", "candidate": "nb1162", "pre_honest_rae": 0.4654,
         "role": "Deep-30 multi-seed anchor; backbone of nb2171"},
        {"slot": "PRIMARY-3", "candidate": "nb2171", "pre_honest_rae": 0.4682,
         "role": "Family-weighted anchor swap"},
        {"slot": "PRIMARY-4", "candidate": "nb2095", "pre_honest_rae": 0.4720,
         "role": "PRE pyramid stack"},
        {"slot": "PRIMARY-5", "candidate": "nb1191", "pre_honest_rae": 0.4757,
         "role": "Independent deep-30; rotation diversity"},
    ]


def build_closed_axes() -> dict:
    """50+ exhausted exploration axes grouped by category."""
    return {
        "model_paradigms": [
            "LightGBM (Huber / MSE / quantile / monotone / DART / Tweedie)",
            "XGBoost (MSE / Huber / pseudo-Huber)",
            "CatBoost (RMSE / ordered / categorical)",
            "Random Forest (sklearn / cuML / extra-trees)",
            "Sklearn Stack (ridge / lasso / kernel-ridge / SVR)",
            "Neural Nets (MLP / dropout-MC / tanh-head)",
            "GNN (Chemprop / SchNet / SE3-EGNN / multitask)",
            "ChemBERTa (anchor / counter / family)",
            "MolFormer (extract / embed)",
            "Gaussian Process (Tanimoto / Matern)",
            "Bayesian Dirichlet blend",
            "Persistent Homology (TDA)",
            "Symbolic regression (PySR residual)",
        ],
        "calibration_methods": [
            "Rank-stretch (scalar / per-quantile / Murcko-gated)",
            "Isotonic regression (per-fold / pooled)",
            "Quantile / conformal stacking",
            "Conformal shrinkage",
            "Confidence-aware shrinkage",
            "Bias-shift / mean re-anchoring",
            "Heteroscedastic NLL",
            "PCHIP CDF-match",
            "Pinball / tail-weighted loss",
            "Tanh-target compression",
            "Soft07 truth injection (cross-fit guarded)",
        ],
        "feature_substrates": [
            "Morgan FP (radius 1-4, bits 1024/2048/4096)",
            "RDKit descriptors (217-D)",
            "Combined Morgan + RDKit (2265-D)",
            "Mordred substrate (1600+ descriptors, full)",
            "MACCS keys (167-bit)",
            "ChemBERTa-77M-MTR embeddings",
            "MolFormer embeddings",
            "SHAP-K top selection (K=20..50 grid)",
            "LASSO selection",
            "Mutual Information selection",
            "Boruta wrapper selection",
            "Permutation Importance selection",
            "Spectral graph features",
            "Persistent Homology features",
            "3D conformer / pharmacophore features",
            "Similarity-matrix as features",
            "MMP transform features",
            "Cliff-transformation features",
            "Counter-assay pEC50 as feature",
            "Single-conc log2FC as feature",
        ],
        "blend_methods": [
            "SLSQP convex combination (2/3/4/5/13-way)",
            "Plain average / weighted average",
            "Median / trimmed-median",
            "Best-of-bag (BoB) selection",
            "Trimmed adversarial blend",
            "Meta-pyramid stack",
            "Per-fold pyramid",
            "Bag-of-bags",
            "Tri-vote / diverse-vote",
            "Dirichlet posterior weighting",
            "Meta-LGBM stacker",
            "Sklearn stacking regressor",
            "Multi-K stack (RFE feature counts)",
            "Decorrelated greedy",
            "Family ablation greedy",
        ],
        "data_augmentation": [
            "SMILES TTA (test-time augmentation)",
            "Unblind label augmentation (253)",
            "Single-conc auxiliary task",
            "Counter-assay multitask head",
            "PubChem PXR external pull",
            "ChEMBL PXR active kNN",
            "Tox21 broad analogy transfer",
            "Large-ADME analogy transfer",
            "NR-family transfer / CLIP",
            "Cross-target profile features",
            "MMP transform model",
            "Pseudo self-training (nb2011)",
        ],
        "structural_priors": [
            "Boltz-2 pose veto (0/513 fired)",
            "Pocket-aware predictor",
            "PDB fragment-residue DB",
            "Pose IFP + GBSA",
            "3D atom features (biotype)",
            "Multi-template delta",
        ],
    }


def build_actively_open() -> list[dict]:
    """Live axes still being explored."""
    return [
        {
            "candidate": "nb2263",
            "name": "lucky-aware RFE",
            "status": "running",
            "hypothesis": "Bootstrap-stabilized RFE eliminates lucky-feature subsets; "
                          "if nb2240 K=20 was a lucky local optimum, this finds the robust K.",
            "expected_pre_honest": "0.4570-0.4620",
        },
        {
            "candidate": "nb2291",
            "name": "Mordred substrate (full 1600-D)",
            "status": "running",
            "hypothesis": "Mordred 1600+ descriptors contain physchem signal absent from "
                          "RDKit-217 + Morgan; first orthogonal substrate not yet exhausted.",
            "expected_pre_honest": "0.4550-0.4650",
        },
    ]


def build_summary() -> dict:
    today = date.today().isoformat()
    chain = build_compounding_chain()
    lb_projection = build_lb_transfer_projection()
    return {
        "snapshot_date": today,
        "cycle": 179,
        "total_methods": 1175,
        "phase_2_deadline": "2026-07-01",
        "days_remaining": (date(2026, 7, 1) - date.today()).days,
        "compounding_wins": chain,
        "compounding_net_delta_rae": round(
            chain[-1]["pre_honest_rae"] - chain[0]["pre_honest_rae"], 4
        ),
        "compounding_net_delta_pct": round(
            (chain[-1]["pre_honest_rae"] - chain[0]["pre_honest_rae"])
            / chain[0]["pre_honest_rae"] * 100,
            1,
        ),
        "primary_stack": build_primary_stack(),
        "lb_transfer_projection": lb_projection,
        "closed_axes": build_closed_axes(),
        "actively_open": build_actively_open(),
        "closed_axes_count": sum(
            len(v) for v in build_closed_axes().values()
        ),
        "regime_note": (
            "All compounding wins are PRE-unblind scaffold-CV honest cross-fit. "
            "POST-unblind cross-fit candidates (nb703/562/730 etc.) deprecated due "
            "to unreliable LB transfer (+0.10 shift observed on 253-unblind)."
        ),
    }


def main() -> None:
    summary = build_summary()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {OUT_PATH}")
    print(f"  cycle={summary['cycle']} methods={summary['total_methods']}")
    print(f"  net PRE delta={summary['compounding_net_delta_rae']} "
          f"({summary['compounding_net_delta_pct']}%)")
    print(f"  PRIMARY-1={summary['primary_stack'][0]['candidate']} "
          f"PRE={summary['primary_stack'][0]['pre_honest_rae']}")
    print(f"  projected LB={summary['lb_transfer_projection']['projected_lb_rae']} "
          f"(vs current {summary['lb_transfer_projection']['current_best_lb_rae']})")


if __name__ == "__main__":
    main()
