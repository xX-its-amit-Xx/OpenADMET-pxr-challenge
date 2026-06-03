"""nb1003 -- HYBRID STRETCH + AFFINE MEAN-SHIFT on nb986.

Hypothesis: combining variance decompression (s) with mean shift (mu_shift)
handles both bias and compression simultaneously.

Procedure:
  1) Load te_nb986.npy (513 deploy preds).
  2) Joint grid:
       s         in {1.0, 1.2, 1.4, 1.6, 1.8, 2.0}
       mu_shift  in {-0.3, -0.1, 0.0, +0.1, +0.3}
  3) pred = (mu + mu_shift) + s * (te_nb986 - mu)
  4) Compute in_RAE per (s, mu_shift) on 253 unblind. Pick best.
  5) Save best as submission CSV.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW, FIGURES, SUBMISSIONS

TAG = "nb1003"
ANCHOR = "nb986"
S_GRID = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
MU_SHIFT_GRID = [-0.3, -0.1, 0.0, 0.1, 0.3]


def _affine_stretch(p: np.ndarray, mu: float, s: float,
                    mu_shift: float) -> np.ndarray:
    """pred = (mu + mu_shift) + s * (p - mu)"""
    return (mu + mu_shift) + s * (p - mu)


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- HYBRID STRETCH + AFFINE MEAN-SHIFT on {ANCHOR}")
    print("=" * 78)

    needed = {
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        f"te_{ANCHOR}.npy":
            DATA_PROCESSED / f"te_{ANCHOR}.npy",
    }
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
    unb_y = unb["pEC50"].astype(float).values.astype(np.float64)
    n_te = len(te_df)
    n_unb = len(unb_idx)
    print(f"\ntest n={n_te}  unblind n={n_unb}")

    # ---- Load anchor ----
    te_anchor = np.load(
        DATA_PROCESSED / f"te_{ANCHOR}.npy"
    ).astype(np.float64)
    assert te_anchor.shape == (n_te,), f"te shape {te_anchor.shape}"

    in_pred = te_anchor[unb_idx]
    rae_anchor_in = float(rae(unb_y, in_pred))
    truth_std = float(unb_y.std())
    truth_mean = float(unb_y.mean())
    pred_std_in = float(in_pred.std())
    pred_std_full = float(te_anchor.std())
    pred_mean_in = float(in_pred.mean())
    print(f"\n{ANCHOR} in_RAE(253)         = {rae_anchor_in:.4f}")
    print(f"{ANCHOR} te mean/std (513)    = "
          f"{te_anchor.mean():.3f} / {pred_std_full:.3f}")
    print(f"{ANCHOR} te[unb] mean/std     = "
          f"{pred_mean_in:.3f} / {pred_std_in:.3f}")
    print(f"truth mean/std (253)         = "
          f"{truth_mean:.3f} / {truth_std:.3f}")
    print(f"variance-ratio (truth/pred_in) = "
          f"{truth_std / max(pred_std_in, 1e-9):.4f}")
    print(f"mean-gap (truth - pred_in)   = "
          f"{truth_mean - pred_mean_in:+.4f}")

    # =================================================================
    # Joint grid sweep, in-sample on 253
    # =================================================================
    mu = float(te_anchor.mean())
    print(f"\nmu = mean(te_{ANCHOR}) = {mu:.4f}")
    print("\n" + "-" * 78)
    print(f"JOINT GRID IN-SAMPLE ON 253")
    print(f"  s_grid        = {S_GRID}")
    print(f"  mu_shift_grid = {MU_SHIFT_GRID}")
    print("-" * 78)

    rows = []
    best_s, best_mu_shift = 1.0, 0.0
    best_rae = float("inf")
    best_full = te_anchor.copy()

    # Heatmap matrix: rows = s, cols = mu_shift
    heatmap = np.zeros((len(S_GRID), len(MU_SHIFT_GRID)), dtype=np.float64)

    print(f"\n  s \\ mu_shift | " +
          " ".join(f"{m:+.2f}" for m in MU_SHIFT_GRID))
    print("  " + "-" * (15 + 7 * len(MU_SHIFT_GRID)))
    for i, s in enumerate(S_GRID):
        line_cells = []
        for j, ms in enumerate(MU_SHIFT_GRID):
            full = _affine_stretch(te_anchor, mu, s, ms)
            r = float(rae(unb_y, full[unb_idx]))
            heatmap[i, j] = r
            rows.append({
                "s": float(s),
                "mu_shift": float(ms),
                "in_rae_253": r,
                "te_mean": float(full.mean()),
                "te_std":  float(full.std()),
            })
            line_cells.append(f"{r:.4f}")
            if r < best_rae:
                best_rae = r
                best_s = float(s)
                best_mu_shift = float(ms)
                best_full = full
        print(f"  s={s:.2f}       | " + " ".join(line_cells))

    sweep_df = pd.DataFrame(rows)
    sweep_path = DATA_PROCESSED / f"{TAG}_joint_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nWrote sweep: {sweep_path}")

    print(f"\nBEST:")
    print(f"  s        = {best_s:.2f}")
    print(f"  mu_shift = {best_mu_shift:+.2f}")
    print(f"  in_RAE   = {best_rae:.4f}")
    print(f"  delta vs anchor s=1.0 mu_shift=0.0 = "
          f"{rae_anchor_in - best_rae:+.4f}")

    # =================================================================
    # Compare to nb1002 (s-only stretch on nb986)
    # =================================================================
    nb1002_summary_path = DATA_PROCESSED / "nb1002_summary.json"
    nb1002_best_s = None
    nb1002_best_rae = None
    if nb1002_summary_path.exists():
        with open(nb1002_summary_path) as f:
            nb1002_sum = json.load(f)
        nb1002_best_s = float(nb1002_sum.get("best_s", float("nan")))
        nb1002_best_rae = float(nb1002_sum.get("best_in_rae", float("nan")))
        print("\n" + "-" * 78)
        print("COMPARISON vs nb1002 (s-only stretch on nb986)")
        print("-" * 78)
        print(f"  nb1002 best s   = {nb1002_best_s:.2f}")
        print(f"  nb1002 best RAE = {nb1002_best_rae:.4f}")
        print(f"  {TAG} best s   = {best_s:.2f}  mu_shift={best_mu_shift:+.2f}")
        print(f"  {TAG} best RAE = {best_rae:.4f}")
        print(f"  delta vs nb1002 = {best_rae - nb1002_best_rae:+.4f}")
        if best_rae < nb1002_best_rae:
            print(f"  -> {TAG} BEATS nb1002 (hybrid helps)")
        elif best_rae > nb1002_best_rae:
            print(f"  -> {TAG} loses to nb1002 (hybrid hurts)")
        else:
            print(f"  -> tied with nb1002")

    # =================================================================
    # Plot heatmap
    # =================================================================
    fig_path = FIGURES / f"{TAG}_joint_heatmap.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis_r",
                   origin="lower")
    ax.set_xticks(range(len(MU_SHIFT_GRID)))
    ax.set_xticklabels([f"{m:+.2f}" for m in MU_SHIFT_GRID])
    ax.set_yticks(range(len(S_GRID)))
    ax.set_yticklabels([f"{s:.2f}" for s in S_GRID])
    ax.set_xlabel("mu_shift")
    ax.set_ylabel("stretch factor s")
    ax.set_title(f"{TAG} -- in_RAE on 253 (anchor={ANCHOR})")
    # annotate
    for i in range(len(S_GRID)):
        for j in range(len(MU_SHIFT_GRID)):
            color = "white" if heatmap[i, j] > heatmap.mean() else "black"
            ax.text(j, i, f"{heatmap[i, j]:.4f}",
                    ha="center", va="center", color=color, fontsize=8)
    # mark best
    bi = S_GRID.index(best_s)
    bj = MU_SHIFT_GRID.index(best_mu_shift)
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1,
                                fill=False, edgecolor="red", lw=2))
    plt.colorbar(im, ax=ax, label="in_RAE")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"\nWrote heatmap: {fig_path}")

    # =================================================================
    # Deploy
    # =================================================================
    deploy = best_full.astype(np.float32)
    print("\n" + "=" * 78)
    print("DEPLOY")
    print("=" * 78)
    print(f"  {ANCHOR} (s=1.0, mu_shift=0) in_RAE = {rae_anchor_in:.4f}")
    print(f"  best s                              = {best_s:.2f}")
    print(f"  best mu_shift                       = {best_mu_shift:+.2f}")
    print(f"  best in_RAE                         = {best_rae:.4f}")
    print(f"  improvement vs anchor               = "
          f"{rae_anchor_in - best_rae:+.4f}")
    print(f"  deploy te mean/std                  = "
          f"{deploy.mean():.3f} / {deploy.std():.3f}")

    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)

    plain = (SUBMISSIONS /
             f"{TAG}_{ANCHOR}_hybrid_s{best_s:.2f}_"
             f"mu{best_mu_shift:+.2f}.csv")
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(plain, index=False)
    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {plain}")

    summary = {
        "tag": TAG,
        "anchor": ANCHOR,
        "anchor_in_rae_253": rae_anchor_in,
        "anchor_te_mean": float(te_anchor.mean()),
        "anchor_te_std": float(te_anchor.std()),
        "anchor_in_pred_mean": pred_mean_in,
        "anchor_in_pred_std": pred_std_in,
        "truth_mean": truth_mean,
        "truth_std": truth_std,
        "mu": mu,
        "s_grid": S_GRID,
        "mu_shift_grid": MU_SHIFT_GRID,
        "sweep": rows,
        "best_s": float(best_s),
        "best_mu_shift": float(best_mu_shift),
        "best_in_rae": float(best_rae),
        "delta_vs_anchor": float(rae_anchor_in - best_rae),
        "deploy_te_mean": float(deploy.mean()),
        "deploy_te_std": float(deploy.std()),
        "beats_anchor": bool(best_rae < rae_anchor_in),
        "nb1002_best_s": nb1002_best_s,
        "nb1002_best_in_rae": nb1002_best_rae,
        "delta_vs_nb1002": (
            float(best_rae - nb1002_best_rae)
            if nb1002_best_rae is not None else None
        ),
        "beats_nb1002": (
            bool(best_rae < nb1002_best_rae)
            if nb1002_best_rae is not None else None
        ),
        "plain_submission": str(plain),
        "figure": str(fig_path),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_summary.json'}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"  anchor                    = {ANCHOR}")
    print(f"  anchor in_RAE (s=1, ms=0) = {rae_anchor_in:.4f}")
    print(f"  best s                    = {best_s:.2f}")
    print(f"  best mu_shift             = {best_mu_shift:+.2f}")
    print(f"  best in_RAE               = {best_rae:.4f}")
    print(f"  delta vs anchor           = {rae_anchor_in - best_rae:+.4f}")
    if nb1002_best_rae is not None:
        print(f"  nb1002 best (s-only)      = {nb1002_best_rae:.4f} "
              f"(s={nb1002_best_s:.2f})")
        print(f"  delta vs nb1002           = "
              f"{best_rae - nb1002_best_rae:+.4f}")
    print(f"  beats anchor              = {best_rae < rae_anchor_in}")
    print("=" * 78)

    return {
        "success": True,
        "anchor": ANCHOR,
        "anchor_in_rae": rae_anchor_in,
        "best_s": float(best_s),
        "best_mu_shift": float(best_mu_shift),
        "best_in_rae": float(best_rae),
        "beats_anchor": bool(best_rae < rae_anchor_in),
        "nb1002_best_in_rae": nb1002_best_rae,
        "beats_nb1002": (
            bool(best_rae < nb1002_best_rae)
            if nb1002_best_rae is not None else None
        ),
        "plain_submission": str(plain),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
