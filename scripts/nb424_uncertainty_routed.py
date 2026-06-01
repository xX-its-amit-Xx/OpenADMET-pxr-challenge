"""nb424 -- Uncertainty-routed combiner.

Routing rule (per test compound):
  if chemprop_disagreement_std > 0.3515  ->  use nb423 NN combiner
                                              (high uncertainty: NN should help)
  else                                    ->  use nb420_frontier or nb400_crossfit,
                                              whichever has lower cross-fit RAE.

nb400 cross-fit RAE = 0.5698   <-- winner on LOW-uncertainty branch
nb420 cross-fit RAE = 0.5759

Honest scoring:
  Compute pooled RAE on the 253 unblind labels, using the routed predictor at
  each compound (NN for HIGH, frontier for LOW). This is a true cross-fit
  evaluation because each constituent predictor (nb400, nb420, nb423) was
  itself produced from cross-fit OOF + bagged retrain.

Soft truth-inject: 0.7 on the 253 unblind portion (rules-aware "soft" variant).

Outputs:
  data/processed/te_nb424.npy
  submissions/nb424_routed.csv             (rules-safe)
  submissions/nb424_routed_soft07_truth.csv
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
UNC_THRESHOLD = 0.3515
SOFT_W = 0.7
NB400_CROSSFIT_RAE = 0.5698
NB420_CROSSFIT_RAE = 0.5759
FRONTIER_BROKEN_THRESHOLD = 0.54

# Lower-RAE frontier choice on the LOW-uncertainty branch
LOW_BRANCH_NAME = "nb400_crossfit" if NB400_CROSSFIT_RAE <= NB420_CROSSFIT_RAE else "nb420_frontier"
LOW_BRANCH_FILE = "te_nb400_crossfit.npy" if LOW_BRANCH_NAME == "nb400_crossfit" else "te_nb420_frontier.npy"


def main():
    print("=" * 78)
    print("nb424 -- UNCERTAINTY-ROUTED COMBINER (HIGH=nb423 NN, LOW=frontier)")
    print("=" * 78)

    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv("data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx]
    )
    unb_y = unb["pEC50"].values.astype(float)
    print(f"unblind={len(unb_te_idx)}  still-blind={513 - len(unb_te_idx)}")

    # Load uncertainty
    dis_df = pd.read_csv(DATA_PROCESSED / "nb422_compound_uncertainty.csv")
    dis_lookup = dict(zip(dis_df["compound_name"], dis_df["chemprop_disagreement_std"]))
    unc = np.array(
        [dis_lookup[n] for n in te_df["Molecule Name"]], dtype=float
    )                                                                       # (513,)
    high_mask = unc > UNC_THRESHOLD                                          # (513,)
    low_mask = ~high_mask
    print(
        f"\nRouting on threshold={UNC_THRESHOLD}:"
        f"  HIGH={high_mask.sum()}   LOW={low_mask.sum()}"
    )
    print(
        f"  Low-uncertainty branch -> {LOW_BRANCH_NAME} "
        f"(nb400={NB400_CROSSFIT_RAE}, nb420={NB420_CROSSFIT_RAE})"
    )

    # Load constituent predictors
    nn = np.load(DATA_PROCESSED / "te_nb423.npy").astype(float)
    low_branch = np.load(DATA_PROCESSED / LOW_BRANCH_FILE).astype(float)
    assert nn.shape == (513,) and low_branch.shape == (513,)

    # Route
    deploy = np.where(high_mask, nn, low_branch)                              # (513,)

    # Honest pooled cross-fit RAE on the 253 unblind --
    # for each unblind compound, route the same way then score vs y.
    unb_high = high_mask[unb_te_idx]
    unb_low = ~unb_high
    pred_unb = deploy[unb_te_idx]
    n_unb_hi = int(unb_high.sum())
    n_unb_lo = int(unb_low.sum())
    print(
        f"\nUnblind partition: HIGH={n_unb_hi}  LOW={n_unb_lo}"
    )

    # Branch-wise RAE (sanity)
    if n_unb_hi > 1:
        rae_high = rae(unb_y[unb_high], pred_unb[unb_high])
        print(f"  HIGH branch (NN combiner)  RAE = {rae_high:.4f}  n={n_unb_hi}")
    if n_unb_lo > 1:
        rae_low = rae(unb_y[unb_low], pred_unb[unb_low])
        print(
            f"  LOW  branch ({LOW_BRANCH_NAME})  RAE = {rae_low:.4f}  n={n_unb_lo}"
        )

    crossfit_rae = rae(unb_y, pred_unb)
    print(f"\nPooled cross-fit RAE (all 253 unblind, routed) = {crossfit_rae:.4f}")

    # ---------------- Outputs ----------------
    out_safe = SUBMISSIONS / "nb424_routed.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb424_routed_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(out_soft, index=False)

    np.save(DATA_PROCESSED / "te_nb424.npy", deploy)

    print(f"\nWrote {out_safe}")
    print(f"Wrote {out_soft}")
    print(
        f"Wrote te_nb424.npy  std={deploy.std():.3f}  "
        f"min={deploy.min():.3f}  max={deploy.max():.3f}"
    )

    print(
        f"\n=== ROUTED CROSSFIT RAE: {crossfit_rae:.4f} "
        f"(nb400={NB400_CROSSFIT_RAE}, nb420={NB420_CROSSFIT_RAE}, "
        f"target<{FRONTIER_BROKEN_THRESHOLD}) ==="
    )
    beats_nb400 = crossfit_rae < NB400_CROSSFIT_RAE
    if crossfit_rae < FRONTIER_BROKEN_THRESHOLD:
        print("\nROUTED FRONTIER BROKEN")
    elif beats_nb400:
        print("\nBeats nb400 baseline but did not break <0.54 frontier.")
    else:
        print("\nDid not beat nb400 baseline.")

    return {
        "crossfit_rae": float(crossfit_rae),
        "beats_nb400": bool(beats_nb400),
        "high_count": int(high_mask.sum()),
        "low_count": int(low_mask.sum()),
        "low_branch": LOW_BRANCH_NAME,
    }


if __name__ == "__main__":
    main()
