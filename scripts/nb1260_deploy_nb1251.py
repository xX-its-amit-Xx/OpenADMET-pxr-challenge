"""nb1260 -- DEPLOY artifact for nb1251 (best-fixed-w 0.55/0.45 blend of
nb1242 ChEMBL-kNN+MACCS residual bag and nb1211 BoB-of-BoBs blend) on 513.

PRECEDENT
---------
nb1251 grid sweep over w_nb1242 in {0.30..0.70 step 0.05} on 253 unblind
showed best fixed-w blend at w_nb1242=0.55, w_nb1211=0.45 with pooled RAE
0.5394 -- a -0.0037 improvement over nb1242 alone (0.5431) and -0.0057
over nb1211 alone (0.5451).

PROTOCOL
--------
1. Load te_nb1250.npy   (513,) -- nb1242 deploy (MACCS+ChEMBL residual bag).
2. Load te_nb1220.npy   (513,) -- nb1211 deploy (BoB-of-BoBs).
3. te_nb1260 = 0.55 * te_nb1250 + 0.45 * te_nb1220.
4. Save data/processed/te_nb1260.npy and
   submissions/nb1260_deploy_nb1251.csv (513 rows).

NOTE
----
This is a pure post-hoc blend on already-trained deploy artifacts.  No new
fit -- both inputs were trained on ALL 253 unblind rows (POST-unblind tier).
The honest cross-fit RAE for the recipe is the nb1251 best_fixed_w number
(0.5394).  in_RAE on te[unb_idx] is in-sample optimistic; use 0.5394 as
LB anchor.

Outputs:
  data/processed/te_nb1260.npy             (513,) float32
  submissions/nb1260_deploy_nb1251.csv     (513 rows, SMILES+Molecule Name+pEC50)
  data/processed/nb1260_summary.json
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

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1260"

W_NB1242 = 0.55
W_NB1211 = 0.45
NB1251_HONEST_CROSSFIT = 0.5394

SUBMISSIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "submissions")
)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)


def _save_submission_csv(te_pred, te_smiles, te_names, csv_path, label):
    assert te_pred.shape[0] == 513, (
        f"{label}: te_pred shape {te_pred.shape}, expected (513,)"
    )
    assert np.all(np.isfinite(te_pred)), f"{label}: te_pred has NaN/Inf"
    sub = pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred.astype(np.float64),
    })
    assert len(sub) == 513
    assert list(sub.columns) == ["SMILES", "Molecule Name", "pEC50"]
    assert sub.isna().sum().sum() == 0, f"{label}: CSV has NaN"
    sub.to_csv(csv_path, index=False)
    return {"csv_path": csv_path, "n_rows": int(len(sub)),
            "columns": list(sub.columns)}


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY nb1251 (best-fixed-w {W_NB1242}*nb1250 + "
          f"{W_NB1211}*nb1220) on 513 test")
    print(f"          honest cross-fit anchor = {NB1251_HONEST_CROSSFIT:.4f}")
    print("=" * 78)

    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_test = len(te_smiles)
    assert n_test == 513, f"expected 513 test rows, got {n_test}"

    te_1250_path = DATA_PROCESSED / "te_nb1250.npy"
    te_1220_path = DATA_PROCESSED / "te_nb1220.npy"
    if not te_1250_path.exists():
        raise FileNotFoundError(f"{te_1250_path} missing")
    if not te_1220_path.exists():
        raise FileNotFoundError(f"{te_1220_path} missing")

    te_1250 = np.load(te_1250_path).astype(np.float64)
    te_1220 = np.load(te_1220_path).astype(np.float64)
    assert te_1250.shape == (n_test,), f"te_1250 shape {te_1250.shape}"
    assert te_1220.shape == (n_test,), f"te_1220 shape {te_1220.shape}"
    assert np.all(np.isfinite(te_1250))
    assert np.all(np.isfinite(te_1220))
    print(f"[load] te_nb1250  shape={te_1250.shape}  "
          f"mean={te_1250.mean():.4f}  std={te_1250.std():.4f}")
    print(f"[load] te_nb1220  shape={te_1220.shape}  "
          f"mean={te_1220.mean():.4f}  std={te_1220.std():.4f}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    rae_1250_in = float(rae(y_unb, te_1250[unb_idx]))
    rae_1220_in = float(rae(y_unb, te_1220[unb_idx]))
    print(f"[diag] in_RAE(te_nb1250[unb]) = {rae_1250_in:.4f}")
    print(f"[diag] in_RAE(te_nb1220[unb]) = {rae_1220_in:.4f}")

    # Blend.
    te_nb1260 = W_NB1242 * te_1250 + W_NB1211 * te_1220
    in_rae_blend = float(rae(y_unb, te_nb1260[unb_idx]))
    pred_corr = float(np.corrcoef(te_1250, te_1220)[0, 1])

    print("\n" + "=" * 78)
    print("BLEND")
    print("=" * 78)
    print(f"   w_nb1242 = {W_NB1242:.4f}")
    print(f"   w_nb1211 = {W_NB1211:.4f}")
    print(f"   pred corr (te_nb1250, te_nb1220) = {pred_corr:.4f}")
    print(f"   te_nb1260  mean={te_nb1260.mean():.4f}  std={te_nb1260.std():.4f}  "
          f"min={te_nb1260.min():.4f}  max={te_nb1260.max():.4f}")
    print(f"   in_RAE(te_nb1260[unb]) = {in_rae_blend:.4f}  "
          f"(honest cross-fit anchor = {NB1251_HONEST_CROSSFIT:.4f})")

    # Save te.
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_nb1260.astype(np.float32))
    print(f"\n[save] {te_path}")

    csv_path = os.path.join(SUBMISSIONS_DIR, f"{TAG}_deploy_nb1251.csv")
    csv_info = _save_submission_csv(
        te_nb1260, te_smiles, te_names, csv_path, "nb1260"
    )
    print(f"[save] {csv_path}  rows={csv_info['n_rows']}  "
          f"cols={csv_info['columns']}")

    summary = {
        "tag": TAG,
        "recipe": f"{W_NB1242}*te_nb1250 + {W_NB1211}*te_nb1220",
        "components": ["te_nb1250 (nb1242 deploy)", "te_nb1220 (nb1211 deploy)"],
        "w_nb1242": W_NB1242,
        "w_nb1211": W_NB1211,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "te_nb1250_stats": {
            "mean": float(te_1250.mean()), "std": float(te_1250.std()),
            "min": float(te_1250.min()),   "max": float(te_1250.max()),
        },
        "te_nb1220_stats": {
            "mean": float(te_1220.mean()), "std": float(te_1220.std()),
            "min": float(te_1220.min()),   "max": float(te_1220.max()),
        },
        "te_nb1260_stats": {
            "mean": float(te_nb1260.mean()), "std": float(te_nb1260.std()),
            "min": float(te_nb1260.min()),   "max": float(te_nb1260.max()),
        },
        "in_rae_nb1250_unb": rae_1250_in,
        "in_rae_nb1220_unb": rae_1220_in,
        "in_rae_nb1260_unb": in_rae_blend,
        "pred_corr_nb1250_nb1220": pred_corr,
        "nb1251_honest_crossfit_anchor": NB1251_HONEST_CROSSFIT,
        "te_path": str(te_path),
        "csv_path": csv_path,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "POST-unblind deploy blend; both components were trained on ALL "
            "253 unblind rows, so in_RAE on te[unb_idx] is in-sample optimistic. "
            "LB-faithful anchor is nb1251 best_fixed_w cross-fit RAE 0.5394."
        ),
    }
    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {summary_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== STRUCTURED SUMMARY ====")
    print(f"  te_mean: {res['te_nb1260_stats']['mean']:.4f}")
    print(f"  te_std:  {res['te_nb1260_stats']['std']:.4f}")
    print(f"  te_min:  {res['te_nb1260_stats']['min']:.4f}")
    print(f"  te_max:  {res['te_nb1260_stats']['max']:.4f}")
    print(f"  in_rae_unb: {res['in_rae_nb1260_unb']:.4f}")
    print(f"  honest_anchor: {res['nb1251_honest_crossfit_anchor']:.4f}")
    print(f"  te_path: {res['te_path']}")
    print(f"  csv_path: {res['csv_path']}")
