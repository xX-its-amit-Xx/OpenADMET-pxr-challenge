"""nb493a -- ANCHOR / RESIDUAL CORRELATION AUDIT.

Goal: decide whether the new alt-anchor routers (nb490 chemprop_aux, nb491
nb420, nb492 nb464) actually carry diversifying signal vs the original
nb432-anchored routers (nb472, nb481, nb482), which we already know are
~0.95+ collinear on both the prediction OOF and the residual OOF.

Two views on the 253 unblind rows:

  (A) PRED OOF correlation -- final cross-fit predictions.
  (B) RESIDUAL OOF correlation -- the actual LGBM residual signal each router
      adds on top of its anchor.  This is the direct collinearity check
      because the residual is what SLSQP actually weights when blending.

For nb482 the residual_oof file is not saved on disk; reconstruct from
  resid_oof = (pred_oof - nb432[unb_idx]) / alpha[unb_idx]
using te_nb432.npy and te_nb482_alpha.npy.

Outputs a 6x6 Pearson matrix for each view and lists the pairs that fall
below the 0.7 collinearity threshold (target: alt-anchor routers achieve
< 0.7 corr against nb472/481/482).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from pxr.paths import DATA_PROCESSED, DATA_RAW


ROUTERS = ["nb472", "nb481", "nb482", "nb490", "nb491", "nb492"]


def load_unblind_idx_and_y():
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"].tolist())}
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    idx = np.array([name_to_idx[n] for n in unb["Molecule Name"]], dtype=int)
    y = unb["pEC50"].astype(float).values.astype(np.float32)
    return idx, y, te_df


def load_pred_oofs():
    out = {}
    for tag in ROUTERS:
        p = DATA_PROCESSED / f"{tag}_pred_oof.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        out[tag] = np.load(p).astype(np.float64).ravel()
    return out


def load_resid_oofs(unb_idx: np.ndarray) -> dict[str, np.ndarray]:
    """Return resid_oof per router on the 253 unblind rows.

    nb482 lacks an on-disk resid_oof; reconstruct from pred_oof and the
    nb432 anchor + nb482 alpha gate.
    """
    out: dict[str, np.ndarray] = {}
    for tag in ["nb472", "nb481", "nb490", "nb491", "nb492"]:
        p = DATA_PROCESSED / f"{tag}_resid_oof.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        out[tag] = np.load(p).astype(np.float64).ravel()

    # ---- nb482 reconstruction ----
    pred = np.load(DATA_PROCESSED / "nb482_pred_oof.npy").astype(np.float64).ravel()
    nb432 = np.load(DATA_PROCESSED / "te_nb432.npy").astype(np.float64).ravel()
    alpha = np.load(DATA_PROCESSED / "te_nb482_alpha.npy").astype(np.float64).ravel()
    anchor_u = nb432[unb_idx]
    alpha_u = alpha[unb_idx]
    # Guard against zero gate (sigmoid output ~ (0,1) so safe in practice).
    resid = (pred - anchor_u) / np.clip(alpha_u, 1e-6, None)
    out["nb482"] = resid
    return out


def pearson_matrix(d: dict[str, np.ndarray]) -> pd.DataFrame:
    tags = list(d.keys())
    M = np.stack([d[t] for t in tags], axis=0)
    C = np.corrcoef(M)
    return pd.DataFrame(C, index=tags, columns=tags)


def format_table(df: pd.DataFrame) -> str:
    return df.round(3).to_string()


def list_pairs_below(df: pd.DataFrame, thresh: float) -> list[tuple[str, str, float]]:
    tags = list(df.index)
    out = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            r = float(df.iat[i, j])
            if r < thresh:
                out.append((tags[i], tags[j], r))
    return out


def anchor_class(tag: str) -> str:
    return "nb432" if tag in {"nb472", "nb481", "nb482"} else "alt"


def cross_anchor_summary(df: pd.DataFrame) -> dict:
    tags = list(df.index)
    cross_vals = []
    within_alt = []
    within_n432 = []
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            ca = anchor_class(tags[i])
            cb = anchor_class(tags[j])
            r = float(df.iat[i, j])
            if ca != cb:
                cross_vals.append(r)
            elif ca == "alt":
                within_alt.append(r)
            else:
                within_n432.append(r)
    return {
        "cross_anchor_mean": float(np.mean(cross_vals)) if cross_vals else float("nan"),
        "cross_anchor_max":  float(np.max(cross_vals))  if cross_vals else float("nan"),
        "cross_anchor_min":  float(np.min(cross_vals))  if cross_vals else float("nan"),
        "within_nb432_mean": float(np.mean(within_n432)) if within_n432 else float("nan"),
        "within_alt_mean":   float(np.mean(within_alt))  if within_alt  else float("nan"),
    }


def main() -> dict:
    print("=" * 78)
    print("nb493a -- ANCHOR / RESIDUAL CORRELATION AUDIT")
    print("=" * 78)

    unb_idx, unb_y, te_df = load_unblind_idx_and_y()
    print(f"n_unb = {len(unb_idx)}  (expected 253)")

    pred = load_pred_oofs()
    resid = load_resid_oofs(unb_idx)

    # ---- Shape sanity ----
    for tag in ROUTERS:
        if pred[tag].shape[0] != len(unb_idx):
            raise ValueError(f"{tag}_pred_oof shape {pred[tag].shape} != 253")
        if resid[tag].shape[0] != len(unb_idx):
            raise ValueError(f"{tag}_resid_oof shape {resid[tag].shape} != 253")

    Cp = pearson_matrix(pred)
    Cr = pearson_matrix(resid)

    print("\n--- PRED OOF Pearson (253 rows) ---")
    print(format_table(Cp))
    print("\n--- RESIDUAL OOF Pearson (253 rows) ---")
    print(format_table(Cr))

    print("\nRESIDUAL pairs with |r| < 0.7  (good = decorrelated):")
    lt07 = list_pairs_below(Cr, 0.7)
    if not lt07:
        print("  (none)")
    else:
        for a, b, r in sorted(lt07, key=lambda x: x[2]):
            print(f"  {a:6s} vs {b:6s}   r = {r:+.3f}")

    print("\nRESIDUAL pairs with |r| < 0.9:")
    lt09 = list_pairs_below(Cr, 0.9)
    if not lt09:
        print("  (none)")
    else:
        for a, b, r in sorted(lt09, key=lambda x: x[2]):
            print(f"  {a:6s} vs {b:6s}   r = {r:+.3f}")

    summ = cross_anchor_summary(Cr)
    print("\nResidual cross-anchor summary:")
    for k, v in summ.items():
        print(f"  {k:22s} = {v:+.3f}")

    verdict = (
        "DIVERSIFYING" if summ["cross_anchor_mean"] < 0.7
        else "WEAK DIVERSIFICATION" if summ["cross_anchor_mean"] < 0.9
        else "STILL COLLINEAR -- anchor strategy did NOT help"
    )
    print(f"\nVerdict: {verdict}")

    # Save matrices
    out_dir = DATA_PROCESSED
    Cp.to_csv(out_dir / "nb493a_pred_pearson.csv")
    Cr.to_csv(out_dir / "nb493a_resid_pearson.csv")
    print(f"\nSaved {out_dir / 'nb493a_pred_pearson.csv'}")
    print(f"Saved {out_dir / 'nb493a_resid_pearson.csv'}")

    return {
        "success": True,
        "n_pairs_resid_lt_07": len(lt07),
        "n_pairs_resid_lt_09": len(lt09),
        "verdict": verdict,
        "summary": summ,
        "pred_pearson_csv": str(out_dir / "nb493a_pred_pearson.csv"),
        "resid_pearson_csv": str(out_dir / "nb493a_resid_pearson.csv"),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== RESULT ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
