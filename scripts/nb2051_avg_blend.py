"""nb2051 -- 50/50 arithmetic-mean blend of nb1162 + nb1191.

Protocol:
1. Load submissions/nb1162_deploy_nb1153.csv and submissions/nb1191_deploy_pre_pyramid.csv.
2. Verify SMILES + Molecule Name alignment (must be 513 matching rows).
3. Arithmetic mean on pEC50 -> submissions/nb2051_avg_1162_1191.csv.
4. Compute Pearson r between the two prediction vectors on full 513.
5. Diversity gate: Pearson <= 0.92 to deploy. If > 0.92, SKIP (duplicate).
6. In-sample compare on 253 unblind (te[unb_idx] RAE) vs each individual.
7. Write data/processed/nb2051_summary.json with decision and stats.

Caveat (feedback_lb_two_regime_calibration / te_vs_pred_oof_protocol):
in_RAE on te[unb_idx] is in-sample for any POST-unblind anchor and may be
optimistic. Use it only as a diagnostic; LB-faithful number must come from
cross-fit OOFs of the inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
SUB_DIR = ROOT / "submissions"
PROC_DIR = ROOT / "data" / "processed"

TAG = "nb2051"
IN1_PATH = SUB_DIR / "nb1162_deploy_nb1153.csv"
IN2_PATH = SUB_DIR / "nb1191_deploy_pre_pyramid.csv"
OUT_CSV = SUB_DIR / "nb2051_avg_1162_1191.csv"
SUMMARY = PROC_DIR / "nb2051_summary.json"

DIVERSITY_GATE = 0.92  # Pearson r upper bound; > gate => duplicate, SKIP deploy.


def rae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true))))


def main() -> dict:
    d1 = pd.read_csv(IN1_PATH)
    d2 = pd.read_csv(IN2_PATH)

    # ---- Alignment checks ----
    if len(d1) != 513 or len(d2) != 513:
        raise ValueError(f"row count mismatch: nb1162={len(d1)}, nb1191={len(d2)}, need 513 each")
    if not (d1["SMILES"].values == d2["SMILES"].values).all():
        raise ValueError("SMILES columns do not align between nb1162 and nb1191")
    if not (d1["Molecule Name"].values == d2["Molecule Name"].values).all():
        raise ValueError("Molecule Name columns do not align between nb1162 and nb1191")

    p1 = d1["pEC50"].to_numpy(dtype=np.float64)
    p2 = d2["pEC50"].to_numpy(dtype=np.float64)

    # ---- Arithmetic mean ----
    p_blend = 0.5 * p1 + 0.5 * p2

    # ---- Pearson diversity ----
    r, _ = pearsonr(p1, p2)
    pearson_r = float(r)

    # ---- In-sample evaluation on 253 unblind ----
    unb_idx = np.load(PROC_DIR / "_audit_unblind_idx.npy")
    y_unb = np.load(PROC_DIR / "_audit_unblind_y.npy").astype(np.float64)

    in_rae_nb1162 = rae(y_unb, p1[unb_idx])
    in_rae_nb1191 = rae(y_unb, p2[unb_idx])
    in_rae_blend = rae(y_unb, p_blend[unb_idx])
    delta_vs_best_input = float(in_rae_blend - min(in_rae_nb1162, in_rae_nb1191))

    # ---- Diversity gate ----
    pass_gate = pearson_r <= DIVERSITY_GATE
    decision = "DEPLOY_PRIMARY_1_BLEND" if pass_gate else "SKIP_DUPLICATE"

    # ---- Write CSV only if gate passes ----
    if pass_gate:
        out_df = pd.DataFrame({
            "SMILES": d1["SMILES"].values,
            "Molecule Name": d1["Molecule Name"].values,
            "pEC50": p_blend,
        })
        out_df.to_csv(OUT_CSV, index=False)
        csv_written = str(OUT_CSV.resolve())
    else:
        csv_written = None

    summary = {
        "tag": TAG,
        "inputs": {
            "nb1162_csv": str(IN1_PATH.resolve()),
            "nb1191_csv": str(IN2_PATH.resolve()),
        },
        "n_rows": int(len(d1)),
        "alignment_ok": True,
        "blend_weights": {"nb1162": 0.5, "nb1191": 0.5},
        "pred_stats_513": {
            "nb1162": {"mean": float(p1.mean()), "std": float(p1.std()),
                       "min": float(p1.min()), "max": float(p1.max())},
            "nb1191": {"mean": float(p2.mean()), "std": float(p2.std()),
                       "min": float(p2.min()), "max": float(p2.max())},
            "blend":  {"mean": float(p_blend.mean()), "std": float(p_blend.std()),
                       "min": float(p_blend.min()), "max": float(p_blend.max())},
        },
        "pearson_r_513": pearson_r,
        "diversity_gate": DIVERSITY_GATE,
        "pass_gate": bool(pass_gate),
        "in_sample_253_unblind": {
            "in_rae_nb1162": float(in_rae_nb1162),
            "in_rae_nb1191": float(in_rae_nb1191),
            "in_rae_blend":  float(in_rae_blend),
            "delta_blend_vs_best_input": delta_vs_best_input,
        },
        "decision": decision,
        "ladder_action": "ADD_PRIMARY_1_BLEND" if pass_gate else "NONE",
        "csv_written": csv_written,
        "caveat": (
            "in_rae on te[unb_idx] is in-sample for POST-unblind anchors; use as "
            "diagnostic only. LB-faithful blend RAE requires cross-fit OOFs."
        ),
    }

    SUMMARY.write_text(json.dumps(summary, indent=2))

    # Console report
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
