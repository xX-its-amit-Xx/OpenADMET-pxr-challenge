"""nb1321 -- MEGA-deploy ensemble: naive mean of TOP-5 deploys at the
0.539-0.545 honest band.

HYPOTHESIS
----------
Multiple deploys converge to ~0.5390-0.5407 honest. Naive mean across 5 of
them may have lower variance than any one (variance-reduction at deploy
level), even if honest cross-fit is identical.

COMPONENTS
----------
  te_nb1260.npy  -- nb1251 best-w 0.55/0.45 blend (0.5394 honest)
  te_nb1261.npy  -- nb1252+nb1211 0.5/0.5 blend  (0.5407 honest)
  te_nb1320.npy  -- nb1290 direct 2-way 0.35*nb1190 + 0.65*nb1242 (0.5390)
                    Built INLINE here if not present on disk.
  te_nb1250.npy  -- nb1242 ChEMBL-feat residual bag deploy (0.5431)
  te_nb1220.npy  -- nb1211 BoB-of-BoBs blend deploy (0.5451)

PROTOCOL
--------
1. Load 4 fixed deploys (nb1260, nb1261, nb1250, nb1220).
2. Build te_nb1320 inline as 0.35*te_nb1190 + 0.65*te_nb1250 if not on disk.
3. te_nb1321 = naive mean across the 5 deploys (row-wise mean over 5).
4. Save data/processed/te_nb1321.npy and
   submissions/nb1321_mega_deploy_ensemble.csv.
5. Validate 513 rows, no NaN. Compute pairwise pred Pearson among the 5 te_X.

NOTE
----
This is a pure post-hoc deploy-level ensemble.  Honest cross-fit is NOT
determined by 5-way OOF blending; it's approximately the mean of the 5
honest anchors (~0.5410 expected).  in_RAE on te[unb_idx] is in-sample
optimistic since every component is POST-unblind.

Outputs:
  data/processed/te_nb1321.npy              (513,) float32
  data/processed/te_nb1320.npy              (513,) float32 if built inline
  submissions/nb1321_mega_deploy_ensemble.csv  (513 rows)
  data/processed/nb1321_summary.json
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

TAG = "nb1321"

# Honest cross-fit anchors per component (from prior notebook summaries).
ANCHORS = {
    "te_nb1260": 0.5394,  # nb1251 best-w 0.55/0.45 (nb1242, nb1211)
    "te_nb1261": 0.5407,  # nb1252 + nb1211 0.5/0.5
    "te_nb1320": 0.5390,  # nb1290 direct 0.35*nb1190 + 0.65*nb1242
    "te_nb1250": 0.5431,  # nb1242 ChEMBL-feat residual bag
    "te_nb1220": 0.5451,  # nb1211 BoB-of-BoBs blend
}

# nb1290 winner: w_nb1190 = 0.35, w_nb1242 = 0.65 (best_fixed_w @ 0.5390).
W_NB1320_NB1190 = 0.35
W_NB1320_NB1242 = 0.65

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


def _build_te_nb1320_inline() -> tuple[np.ndarray, dict]:
    """Build te_nb1320 = 0.35*te_nb1190 + 0.65*te_nb1250 if not on disk."""
    te_path = DATA_PROCESSED / "te_nb1320.npy"
    if te_path.exists():
        arr = np.load(te_path).astype(np.float64)
        return arr, {"built": False, "path": str(te_path)}
    print("[inline] te_nb1320 missing -- building 0.35*te_nb1190 + 0.65*te_nb1250")
    te_1190_path = DATA_PROCESSED / "te_nb1190.npy"
    te_1250_path = DATA_PROCESSED / "te_nb1250.npy"
    if not te_1190_path.exists():
        raise FileNotFoundError(f"{te_1190_path} missing")
    if not te_1250_path.exists():
        raise FileNotFoundError(f"{te_1250_path} missing")
    te_1190 = np.load(te_1190_path).astype(np.float64)
    te_1250 = np.load(te_1250_path).astype(np.float64)
    assert te_1190.shape == (513,), f"te_nb1190 shape {te_1190.shape}"
    assert te_1250.shape == (513,), f"te_nb1250 shape {te_1250.shape}"
    assert np.all(np.isfinite(te_1190))
    assert np.all(np.isfinite(te_1250))
    te_1320 = W_NB1320_NB1190 * te_1190 + W_NB1320_NB1242 * te_1250
    np.save(te_path, te_1320.astype(np.float32))
    print(f"[inline] saved {te_path}")
    return te_1320, {
        "built": True, "path": str(te_path),
        "recipe": f"{W_NB1320_NB1190}*te_nb1190 + {W_NB1320_NB1242}*te_nb1250",
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- MEGA-deploy ensemble: naive mean of TOP-5 deploys")
    print(f"          (nb1260, nb1261, nb1320, nb1250, nb1220) at 0.539-0.545 band")
    print("=" * 78)

    te = load_test()
    te_smiles = te["smiles"].values
    te_names = te["name"].values
    n_test = len(te_smiles)
    assert n_test == 513, f"expected 513 test rows, got {n_test}"

    # Step A -- build te_nb1320 inline if needed.
    te_nb1320, nb1320_info = _build_te_nb1320_inline()

    # Step B -- load remaining 4 deploys.
    component_files = [
        "te_nb1260", "te_nb1261", "te_nb1250", "te_nb1220",
    ]
    components: dict[str, np.ndarray] = {}
    for name in component_files:
        p = DATA_PROCESSED / f"{name}.npy"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing")
        arr = np.load(p).astype(np.float64)
        assert arr.shape == (n_test,), f"{name} shape {arr.shape}"
        assert np.all(np.isfinite(arr)), f"{name} has NaN/Inf"
        components[name] = arr
        print(f"[load] {name}  mean={arr.mean():.4f}  std={arr.std():.4f}  "
              f"anchor={ANCHORS[name]:.4f}")

    components["te_nb1320"] = te_nb1320
    print(f"[load] te_nb1320  mean={te_nb1320.mean():.4f}  "
          f"std={te_nb1320.std():.4f}  anchor={ANCHORS['te_nb1320']:.4f}")

    # Maintain canonical ordering matching the protocol spec.
    order = ["te_nb1260", "te_nb1261", "te_nb1320", "te_nb1250", "te_nb1220"]
    P_te = np.column_stack([components[k] for k in order])
    assert P_te.shape == (n_test, 5)

    # ---- in_RAE diagnostics per component ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    in_rae_per_comp: dict[str, float] = {}
    print("\n[diag] in_RAE per component on 253 unblind:")
    for k in order:
        v = float(rae(y_unb, components[k][unb_idx]))
        in_rae_per_comp[k] = v
        print(f"   {k}: in_RAE = {v:.4f}   (honest anchor = {ANCHORS[k]:.4f})")

    # ---- Pairwise pred Pearson on full 513 ----
    print("\n[diag] pairwise pred Pearson on 513 deploys:")
    corr_mat = np.corrcoef(P_te.T)  # (5, 5)
    pairwise = {}
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if j <= i:
                continue
            c = float(corr_mat[i, j])
            pairwise[f"{a}__{b}"] = c
            print(f"   {a}  vs  {b}:  rho = {c:.4f}")

    # ---- Mega-deploy ensemble: naive row-wise mean ----
    te_nb1321 = P_te.mean(axis=1)
    assert te_nb1321.shape == (n_test,)
    assert np.all(np.isfinite(te_nb1321)), "te_nb1321 has NaN/Inf"

    in_rae_blend = float(rae(y_unb, te_nb1321[unb_idx]))

    # ---- Stats ----
    te_stats = {
        "mean": float(te_nb1321.mean()),
        "std":  float(te_nb1321.std()),
        "min":  float(te_nb1321.min()),
        "max":  float(te_nb1321.max()),
    }
    print("\n" + "=" * 78)
    print("MEGA-DEPLOY ENSEMBLE")
    print("=" * 78)
    print(f"   recipe: row-wise mean over 5 deploys (uniform 1/5 weight each)")
    print(f"   te_nb1321 mean={te_stats['mean']:.4f}  std={te_stats['std']:.4f}  "
          f"min={te_stats['min']:.4f}  max={te_stats['max']:.4f}")
    print(f"   in_RAE(te_nb1321[unb]) = {in_rae_blend:.4f}")

    # Expected LB anchor = mean of 5 honest cross-fit anchors.
    expected_lb_anchor = float(np.mean(list(ANCHORS.values())))
    print(f"   expected LB anchor = mean(5 anchors) = {expected_lb_anchor:.4f}")

    # ---- Save artifacts ----
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_nb1321.astype(np.float32))
    print(f"\n[save] {te_path}")

    csv_path = os.path.join(SUBMISSIONS_DIR, f"{TAG}_mega_deploy_ensemble.csv")
    csv_info = _save_submission_csv(
        te_nb1321, te_smiles, te_names, csv_path, "nb1321"
    )
    print(f"[save] {csv_path}  rows={csv_info['n_rows']}  "
          f"cols={csv_info['columns']}")

    summary = {
        "tag": TAG,
        "recipe": "row-wise mean of 5 deploys (uniform 1/5 weight)",
        "components": order,
        "component_anchors": {k: ANCHORS[k] for k in order},
        "component_in_rae_unb": in_rae_per_comp,
        "nb1320_inline": nb1320_info,
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "te_nb1321_stats": te_stats,
        "in_rae_nb1321_unb": in_rae_blend,
        "pairwise_pred_pearson": pairwise,
        "expected_lb_anchor": expected_lb_anchor,
        "te_path": str(te_path),
        "csv_path": csv_path,
        "wall_sec": round(time.time() - t0, 2),
        "note": (
            "POST-unblind deploy-level ensemble; each component was trained on ALL "
            "253 unblind rows, so in_RAE on te[unb_idx] is in-sample optimistic. "
            "Honest cross-fit RAE for a naive uniform 5-way mean is NOT determined "
            "by 5-way OOF blending; it's approximately the mean of the 5 honest "
            "anchors (~0.5415) modulo variance-reduction gains."
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
    print(f"  te_mean: {res['te_nb1321_stats']['mean']:.4f}")
    print(f"  te_std:  {res['te_nb1321_stats']['std']:.4f}")
    print(f"  te_min:  {res['te_nb1321_stats']['min']:.4f}")
    print(f"  te_max:  {res['te_nb1321_stats']['max']:.4f}")
    print(f"  in_rae_unb: {res['in_rae_nb1321_unb']:.4f}")
    print(f"  expected_lb_anchor: {res['expected_lb_anchor']:.4f}")
    print(f"  te_path: {res['te_path']}")
    print(f"  csv_path: {res['csv_path']}")
    print("  pairwise_pred_pearson:")
    for k, v in res["pairwise_pred_pearson"].items():
        print(f"    {k}: {v:.4f}")
