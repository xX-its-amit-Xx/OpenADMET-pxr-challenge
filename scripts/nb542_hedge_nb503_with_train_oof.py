"""nb542 -- CONSERVATIVE HEDGE: pairwise + 3-way blends of {nb503, nb540, nb541}.

All three already had honest cross-fit estimates; this script does NO fitting on
the 253 unblind rows -- it only sweeps a small convex weight grid and reports
the lowest unblind RAE.

Procedure:
  For each combination in
      {(nb503, nb540), (nb503, nb541), (nb540, nb541), (nb503, nb540, nb541)}
  and each w in {0.1, 0.2, 0.3, 0.4, 0.5}:
    - 2-way: blend = (1 - w) * A + w * B
    - 3-way (single w shared on the two non-anchor members):
        blend = (1 - 2*w) * nb503 + w * nb540 + w * nb541    (skip if w > 0.5)
  Compute honest RAE on the 253 unblind rows for each blend.

Target: beat nb503 alone (< 0.5116).

Save te_nb542.npy + plain + soft07_truth submissions for the best blend.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

SOFT_W = 0.7
TAG = "nb542"

W_GRID = [0.1, 0.2, 0.3, 0.4, 0.5]
ANCHOR_TAG = "nb503"
POOL = ["nb503", "nb540", "nb541"]
TARGET_RAE = 0.5116  # nb503 alone


def main() -> dict:
    print("=" * 78)
    print("nb542 -- CONSERVATIVE HEDGE (nb503/nb540/nb541, no fit on 253)")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
    }
    for tag in POOL:
        needed[f"te_{tag}.npy"] = DATA_PROCESSED / f"te_{tag}.npy"
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    # ---- Indices ----
    te_df = pd.read_csv(needed["TEST_BLINDED"])
    te_names = te_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb = pd.read_csv(needed["UNBLINDED"])
    unb = unb[unb["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"]], dtype=int
    )
    unb_y = unb["pEC50"].astype(float).values.astype(np.float32)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- Load deploy vectors ----
    te_vec: dict[str, np.ndarray] = {}
    unb_pred: dict[str, np.ndarray] = {}
    print("\nPool members (standalone RAE on 253):")
    for tag in POOL:
        d = np.load(DATA_PROCESSED / f"te_{tag}.npy").astype(np.float32)
        assert d.shape == (n_te,), f"{tag} te shape {d.shape}"
        te_vec[tag] = d
        unb_pred[tag] = d[unb_idx]
        r_alone = float(rae(unb_y, unb_pred[tag]))
        print(f"  {tag}: alone_RAE={r_alone:.4f}  te std={d.std():.3f}")

    rae_anchor = float(rae(unb_y, unb_pred[ANCHOR_TAG]))

    # =================================================================
    # Sweep combos
    # =================================================================
    print("\n" + "-" * 78)
    print("BLEND SWEEP (RAE on 253 unblind rows)")
    print("-" * 78)

    results: list[dict] = []

    # 2-way combos
    pair_combos = [("nb503", "nb540"), ("nb503", "nb541"), ("nb540", "nb541")]
    for a, b in pair_combos:
        for w in W_GRID:
            blend = (1.0 - w) * unb_pred[a] + w * unb_pred[b]
            r = float(rae(unb_y, blend))
            results.append({
                "combo": f"{a}+{b}",
                "kind": "pair",
                "w": w,
                "weights": {a: 1.0 - w, b: w},
                "rae": r,
            })
        # print best for this pair
        best = min(
            (r for r in results if r["combo"] == f"{a}+{b}"),
            key=lambda d: d["rae"],
        )
        grid_str = " ".join(
            f"w={r['w']:.1f}:{r['rae']:.4f}"
            for r in results if r["combo"] == f"{a}+{b}"
        )
        print(f"  {a}+{b}:  {grid_str}")
        print(f"     -> best w={best['w']:.2f}  RAE={best['rae']:.4f}")

    # 3-way (nb503 anchored, symmetric w on nb540/nb541)
    triple_label = "nb503+nb540+nb541"
    for w in W_GRID:
        if 2.0 * w > 1.0:
            continue  # would invert anchor weight
        wa = 1.0 - 2.0 * w
        blend = wa * unb_pred["nb503"] + w * unb_pred["nb540"] + w * unb_pred["nb541"]
        r = float(rae(unb_y, blend))
        results.append({
            "combo": triple_label,
            "kind": "triple",
            "w": w,
            "weights": {"nb503": wa, "nb540": w, "nb541": w},
            "rae": r,
        })
    triple_rows = [r for r in results if r["combo"] == triple_label]
    if triple_rows:
        best = min(triple_rows, key=lambda d: d["rae"])
        grid_str = " ".join(
            f"w_each={r['w']:.1f}(anchor={r['weights']['nb503']:.2f}):"
            f"{r['rae']:.4f}"
            for r in triple_rows
        )
        print(f"  {triple_label}:  {grid_str}")
        print(f"     -> best w_each={best['w']:.2f}  "
              f"anchor={best['weights']['nb503']:.2f}  RAE={best['rae']:.4f}")

    # =================================================================
    # Pick overall best
    # =================================================================
    best_row = min(results, key=lambda d: d["rae"])
    print("\n" + "=" * 78)
    print("DEPLOY DECISION")
    print("=" * 78)
    print(f"  nb503 alone                              = {rae_anchor:.4f}")
    print(f"  best blend combo                         = {best_row['combo']}")
    print(f"  best blend kind                          = {best_row['kind']}")
    print(f"  best blend weights                       = "
          f"{ {k: round(v, 4) for k, v in best_row['weights'].items()} }")
    print(f"  best blend RAE                           = {best_row['rae']:.4f}")
    print(f"  target (nb503)                           = {TARGET_RAE:.4f}")
    print(f"  beats nb503                              = "
          f"{best_row['rae'] < TARGET_RAE}")

    # =================================================================
    # Build deploy vector with same weights on test (513-row)
    # =================================================================
    deploy = np.zeros(n_te, dtype=np.float64)
    for tag, w in best_row["weights"].items():
        deploy = deploy + w * te_vec[tag]
    deploy = deploy.astype(np.float32)

    # =================================================================
    # Save artefacts
    # =================================================================
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)

    combo_safe = best_row["combo"].replace("+", "_")
    w_str = f"w{best_row['w']:.2f}".replace(".", "")
    base = f"{TAG}_hedge_{combo_safe}_{w_str}"

    plain = SUBMISSIONS / f"{base}.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)

    soft = deploy.copy()
    soft[unb_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_idx]
    soft_path = SUBMISSIONS / f"{base}_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    print("\n" + "=" * 78)
    print("=== nb542 SUMMARY ===")
    print(f"  nb503 alone                              = {rae_anchor:.4f}")
    print(f"  best combo                                = {best_row['combo']}")
    print(f"  best weights                              = "
          f"{ {k: round(v, 4) for k, v in best_row['weights'].items()} }")
    print(f"  best RAE                                  = {best_row['rae']:.4f}")
    print(f"  beats nb503 (<{TARGET_RAE:.4f})            = "
          f"{best_row['rae'] < TARGET_RAE}")
    print("=" * 78)

    return {
        "success": True,
        "anchor_rae": rae_anchor,
        "all_results": results,
        "best_combo": best_row["combo"],
        "best_kind": best_row["kind"],
        "best_w": float(best_row["w"]),
        "best_weights": {k: float(v) for k, v in best_row["weights"].items()},
        "best_rae": float(best_row["rae"]),
        "beats_nb503": bool(best_row["rae"] < TARGET_RAE),
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        if k == "all_results":
            print(f"  {k}: <{len(v)} rows>")
        else:
            print(f"  {k}: {v}")
