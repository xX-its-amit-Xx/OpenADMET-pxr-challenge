"""nb249 -- Boltz as 'second opinion' adjustment to 239.

For 255 test compounds with boltz affinity, compute boltz_pseudo_pec50 via
linear calibration (pec50 = -0.93 * boltz_aff + 5.32).

Where boltz_pseudo STRONGLY disagrees with 239, shrink 239 toward boltz.
Where they agree, keep 239 unchanged.

This tests whether boltz adds value ONLY when it strongly contradicts the
existing model — the "outlier correction" hypothesis.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pxr.data import load_train
from pxr.eval import rae
from pathlib import Path

P = Path("data/processed")
SUBS = Path("submissions")


def main():
    print("=== nb249: Boltz second opinion ===\n")
    tr = load_train()
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    y_tr = tr["pec50"].values.astype(np.float64)
    te_names = te_df["Molecule Name"].tolist()
    sm = dict(zip(te_df["Molecule Name"], te_df["SMILES"]))

    # Boltz test affinities
    boltz_test = pd.read_parquet(P / "boltz_test_partial_nb167.parquet")
    boltz_train_cal = pd.read_parquet(P / "boltz_train_calibration.parquet")

    # Fit linear calibration: pec50 = a + b * affinity
    valid = boltz_train_cal.dropna(subset=["affinity_pred_value"])
    aff = valid["affinity_pred_value"].values
    pec = valid["pec50"].values
    b, a = np.polyfit(aff, pec, 1)
    print(f"Calibration: pec50 = {b:.4f} * boltz_aff + {a:.4f} (n={len(valid)})")
    print(f"Pearson r: {np.corrcoef(aff, pec)[0,1]:.4f}")

    # Apply calibration to 255 test compounds
    boltz_test_aff = boltz_test.set_index("name")["affinity_pred_value"]
    boltz_test_pec = a + b * boltz_test_aff
    print(f"\nBoltz pseudo pec50: min={boltz_test_pec.min():.3f}, max={boltz_test_pec.max():.3f}, mean={boltz_test_pec.mean():.3f}")

    # Compare with 239 predictions
    nb239_te = np.load(P / "te_nb239_full_slsqp.npy")
    print(f"nb239 te: mean={nb239_te.mean():.3f}, std={nb239_te.std():.3f}")

    # For each test compound with boltz data, compute disagreement
    name_to_boltz = boltz_test_pec.to_dict()
    name_to_239_idx = {n: i for i, n in enumerate(te_names)}

    disagreements = []
    for n, boltz_p in name_to_boltz.items():
        if n in name_to_239_idx:
            idx = name_to_239_idx[n]
            nb239_p = nb239_te[idx]
            disagreements.append((n, idx, nb239_p, boltz_p, abs(nb239_p - boltz_p)))

    print(f"\nDisagreement stats (255 compounds with boltz):")
    diffs = [d[4] for d in disagreements]
    print(f"  |nb239 - boltz_pseudo|: min={min(diffs):.3f}, max={max(diffs):.3f}, median={np.median(diffs):.3f}")
    print(f"  Count |diff| > 1.0: {sum(1 for d in diffs if d > 1.0)}")
    print(f"  Count |diff| > 1.5: {sum(1 for d in diffs if d > 1.5)}")
    print(f"  Count |diff| > 2.0: {sum(1 for d in diffs if d > 2.0)}")

    # Variants: shift nb239 toward boltz where disagreement > threshold
    print("\nGenerating variants:")
    for thresh in [1.0, 1.5, 2.0]:
        for shift_w in [0.10, 0.20, 0.30, 0.50]:
            te_adj = nb239_te.copy()
            n_adjusted = 0
            for n, idx, nb_p, b_p, d in disagreements:
                if d > thresh:
                    te_adj[idx] = (1 - shift_w) * nb_p + shift_w * b_p
                    n_adjusted += 1
            sub = pd.DataFrame({"SMILES": [sm[n] for n in te_names], "Molecule Name": te_names, "pEC50": te_adj})
            sub.to_csv(SUBS / f"249_boltz_2nd_op_t{int(thresh*10):02d}_w{int(shift_w*100):03d}.csv", index=False)
            print(f"  thresh={thresh}  shift_w={shift_w}: n_adjusted={n_adjusted}  te_std={te_adj.std():.3f}, mean={te_adj.mean():.3f}")

    np.save(P / "te_boltz_pseudo_pec50.npy", np.array([name_to_boltz.get(n, np.nan) for n in te_names]))


if __name__ == "__main__":
    main()
