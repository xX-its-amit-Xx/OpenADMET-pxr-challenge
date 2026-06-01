"""Classify te_*.npy files as LEAK / SOFT_INJECT / NORMAL_BIAS.

For each model in {nb432, nb472, nb481, nb492, nb502, nb503, nb562, nb700,
nb701, nb702, nb703}:
  1. Load te_<nb>*.npy and project onto the 253-row unblind subset.
  2. Compute RAE(te[unb_idx], truth).
  3. Load nb_pred_oof.npy when available and compute RAE(oof, truth).
  4. Apply heuristics:
     - LEAK if te_rae < 0.30 (a 4139-train model cannot honestly achieve <0.30).
     - LEAK if Pearson(te[unb_idx], truth) > 0.95.
     - LEAK if any te[unb_idx] cell equals truth exactly for >5 cells.
     - SOFT_INJECT if te[unb_idx] ~ 0.7*truth + 0.3*deploy
        diagnosed via |fit_w_truth - 0.7| < 0.15 and the (truth, oof) regression
        captures >99% of variance (i.e. te[unb] is a near-perfect linear blend).
     - NORMAL_BIAS otherwise (small in-sample-vs-OOF gap, ~0.05-0.15).
Outputs data/processed/te_files_classification.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent / "data" / "processed"


def rae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = float(np.sum(np.abs(y_true - y_pred)))
    den = float(np.sum(np.abs(y_true - y_true.mean())))
    return num / den if den > 0 else float("nan")


def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)) + 1e-12)
    return float(np.sum(a * b) / denom)


TE_FILES = {
    "nb432": "te_nb432.npy",
    "nb472": "te_nb472_stretched.npy",
    "nb481": "te_nb481_stretched.npy",
    "nb492": "te_nb492_stretched.npy",
    "nb502": "te_nb502_stretched.npy",
    "nb503": "te_nb503_clean.npy",
    "nb562": "te_nb562_clean.npy",
    "nb700": None,
    "nb701": None,
    "nb702": "te_nb702.npy",
    "nb703": "te_nb703_clean.npy",
}

OOF_FILES = {
    "nb432": None,
    "nb472": "nb472_pred_oof.npy",
    "nb481": "nb481_pred_oof.npy",
    "nb492": "nb492_pred_oof.npy",
    "nb502": "nb502_pred_oof.npy",
    "nb503": "nb503_pred_oof.npy",
    "nb562": "nb562_pred_oof.npy",
    "nb700": "nb700_pred_oof.npy",
    "nb701": "nb701_pred_oof.npy",
    "nb702": "nb702_pred_oof.npy",
    "nb703": "nb703_pred_oof.npy",
}


def classify(te_rae, oof_rae, p_truth, exact_eq, fit_w_truth, sub_blend_r2):
    # LEAK criteria
    if te_rae is not None and te_rae < 0.30:
        return "LEAK", "te_rae < 0.30 (impossibly tight for 4139-train model)"
    if p_truth is not None and p_truth > 0.95:
        return "LEAK", f"Pearson(te,truth)={p_truth:.4f} > 0.95"
    if exact_eq is not None and exact_eq > 5:
        return "LEAK", f"{exact_eq} exact truth matches (>5)"
    # SOFT_INJECT — needs both truth-weight ~0.7 and a near-perfect blend fit
    if (
        fit_w_truth is not None
        and abs(fit_w_truth - 0.7) < 0.15
        and sub_blend_r2 is not None
        and sub_blend_r2 > 0.99
    ):
        return "SOFT_INJECT", f"te ~ 0.7*truth + 0.3*deploy (w_truth={fit_w_truth:.3f}, R2={sub_blend_r2:.4f})"
    # NORMAL_BIAS — small in-sample gap vs OOF
    if te_rae is not None and oof_rae is not None and 0.0 <= oof_rae - te_rae <= 0.20:
        return "NORMAL_BIAS", f"oof_rae - te_rae = {oof_rae - te_rae:+.4f} (small in-sample bias)"
    if te_rae is not None and oof_rae is None:
        return "UNKNOWN", "no pred_oof available"
    return "NORMAL_BIAS", "no leak signature; gap within tolerance"


def main():
    unb_idx = np.load(ROOT / "_audit_unblind_idx.npy")
    truth = np.load(ROOT / "_audit_unblind_y.npy")

    rows = []
    for nb in TE_FILES:
        te_name = TE_FILES[nb]
        oof_name = OOF_FILES.get(nb)
        te_path = ROOT / te_name if te_name else None
        oof_path = ROOT / oof_name if oof_name else None

        te_rae = oof_rae = p_truth = None
        exact_eq = None
        fit_w_truth = None
        sub_blend_r2 = None
        mae_te_oof = None

        sub = None
        if te_path is not None and te_path.exists():
            te = np.load(te_path)
            if te.ndim == 1 and te.shape[0] >= int(unb_idx.max()) + 1:
                sub = te[unb_idx].astype(float)
                te_rae = rae(truth, sub)
                p_truth = pearson(truth, sub)
                exact_eq = int(np.sum(sub == truth))

        oof = None
        if oof_path is not None and oof_path.exists():
            oof_raw = np.load(oof_path)
            if oof_raw.shape == (253,):
                oof = oof_raw.astype(float)
                oof_rae = rae(truth, oof)

        if sub is not None and oof is not None:
            # Regress te[unb] on (truth, oof, 1) and read the fit weights.
            X = np.hstack([truth.reshape(-1, 1), oof.reshape(-1, 1), np.ones((253, 1))])
            coef, *_ = np.linalg.lstsq(X, sub, rcond=None)
            fit_w_truth = float(coef[0])
            pred_blend = X @ coef
            ss_res = float(np.sum((sub - pred_blend) ** 2))
            ss_tot = float(np.sum((sub - sub.mean()) ** 2))
            sub_blend_r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
            mae_te_oof = float(np.mean(np.abs(sub - oof)))

        klass, reason = classify(te_rae, oof_rae, p_truth, exact_eq, fit_w_truth, sub_blend_r2)
        rows.append(
            dict(
                nb=nb,
                te_file=te_name or "",
                oof_file=oof_name or "",
                te_present=te_path is not None and te_path.exists(),
                oof_present=oof_path is not None and oof_path.exists(),
                te_rae=te_rae,
                oof_rae=oof_rae,
                pearson_te_truth=p_truth,
                exact_eq_truth=exact_eq,
                fit_w_truth=fit_w_truth,
                blend_r2=sub_blend_r2,
                mae_te_oof=mae_te_oof,
                classification=klass,
                reason=reason,
            )
        )

    df = pd.DataFrame(rows)
    out = ROOT / "te_files_classification.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
