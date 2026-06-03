"""nb1002 -- RANK STRETCH on nb986 (asym blend w_chem=0.80 + 0.20*nb972).

nb986 has slightly different mu/std than nb982 (different blend weights),
so the stretch optimum may differ from nb1000's s=1.25.

Procedure:
  1) Load te_nb986.npy (513 deploy preds, in_RAE=0.6105 at s=1.0).
  2) Grid s in {1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0}.
  3) Compute stretched = mu + s*(te - mu); in_RAE on 253. Pick best.
  4) Compare to nb1000 stretch on nb982.
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

TAG = "nb1002"
ANCHOR = "nb986"
STRETCH_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]


def _stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def main() -> dict:
    print("=" * 78)
    print(f"{TAG} -- RANK STRETCH on {ANCHOR} (asym blend)")
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
    pred_std_in = float(in_pred.std())
    pred_std_full = float(te_anchor.std())
    print(f"\n{ANCHOR} in_RAE(253)         = {rae_anchor_in:.4f}")
    print(f"{ANCHOR} te mean/std (513)    = "
          f"{te_anchor.mean():.3f} / {pred_std_full:.3f}")
    print(f"{ANCHOR} te[unb] mean/std     = "
          f"{in_pred.mean():.3f} / {pred_std_in:.3f}")
    print(f"truth mean/std (253)         = "
          f"{unb_y.mean():.3f} / {truth_std:.3f}")
    print(f"variance-ratio (truth/pred_in) = "
          f"{truth_std / max(pred_std_in, 1e-9):.4f}")

    # ---- Compare anchors to nb982 ----
    nb982_path = DATA_PROCESSED / "te_nb982.npy"
    if nb982_path.exists():
        te_982 = np.load(nb982_path).astype(np.float64)
        print(f"\nnb982 te mean/std (513)      = "
              f"{te_982.mean():.3f} / {te_982.std():.3f}")
        print(f"{ANCHOR} vs nb982 diff(mean)  = "
              f"{te_anchor.mean() - te_982.mean():+.4f}")
        print(f"{ANCHOR} vs nb982 diff(std)   = "
              f"{te_anchor.std() - te_982.std():+.4f}")

    # =================================================================
    # Grid sweep, in-sample on 253
    # =================================================================
    mu = float(te_anchor.mean())
    print(f"\nmu = mean(te_{ANCHOR}) = {mu:.4f}")
    print("\n" + "-" * 78)
    print(f"GRID STRETCH IN-SAMPLE ON 253  grid={STRETCH_GRID}")
    print("-" * 78)

    rows = []
    best_s, best_rae, best_full = 1.0, float("inf"), te_anchor.copy()
    for s in STRETCH_GRID:
        full = _stretch(te_anchor, mu, s)
        r = float(rae(unb_y, full[unb_idx]))
        rows.append({"s": float(s), "in_rae_253": r,
                     "te_mean": float(full.mean()),
                     "te_std":  float(full.std())})
        flag = ""
        if r < best_rae:
            best_rae = r
            best_s = float(s)
            best_full = full
            flag = "  <- best"
        print(f"  s={s:.2f}  in_RAE={r:.4f}  "
              f"te_std={full.std():.3f}{flag}")

    sweep_df = pd.DataFrame(rows)
    sweep_path = DATA_PROCESSED / f"{TAG}_stretch_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nWrote sweep: {sweep_path}")

    # =================================================================
    # Compare to nb1000 (stretch on nb982)
    # =================================================================
    print("\n" + "-" * 78)
    print("COMPARISON vs nb1000 (stretch on nb982)")
    print("-" * 78)
    nb1000_summary_path = DATA_PROCESSED / "nb1000_summary.json"
    nb1000_best_s = None
    nb1000_best_rae = None
    if nb1000_summary_path.exists():
        with open(nb1000_summary_path) as f:
            nb1000_sum = json.load(f)
        nb1000_best_s = float(nb1000_sum.get("best_s", float("nan")))
        nb1000_best_rae = float(nb1000_sum.get("best_in_rae", float("nan")))
        print(f"  nb1000 anchor   = nb982")
        print(f"  nb1000 best s   = {nb1000_best_s:.2f}")
        print(f"  nb1000 best RAE = {nb1000_best_rae:.4f}")
        print(f"  {TAG} anchor   = {ANCHOR}")
        print(f"  {TAG} best s   = {best_s:.2f}")
        print(f"  {TAG} best RAE = {best_rae:.4f}")
        print(f"  delta vs nb1000 = {best_rae - nb1000_best_rae:+.4f}")
        if best_rae < nb1000_best_rae:
            print(f"  -> {TAG} BEATS nb1000")
        elif best_rae > nb1000_best_rae:
            print(f"  -> {TAG} loses to nb1000")
        else:
            print(f"  -> tied with nb1000")
    else:
        print(f"  nb1000_summary.json missing -- skip compare")

    # =================================================================
    # Plot
    # =================================================================
    fig_path = FIGURES / f"{TAG}_stretch_curve.png"
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["s"], sweep_df["in_rae_253"],
             marker="o", lw=1.8, color="#1f77b4", label=f"{ANCHOR} stretch")
    plt.axvline(best_s, color="#d62728", ls="--", lw=1,
                label=f"best s={best_s:.2f}  RAE={best_rae:.4f}")
    plt.axhline(rae_anchor_in, color="#7f7f7f", ls=":", lw=1,
                label=f"{ANCHOR} s=1.0  RAE={rae_anchor_in:.4f}")
    if nb1000_best_rae is not None:
        plt.axhline(nb1000_best_rae, color="#2ca02c", ls=":", lw=1,
                    label=f"nb1000 best (nb982) RAE={nb1000_best_rae:.4f}")
    plt.xlabel("stretch factor s")
    plt.ylabel("in-sample RAE on 253 unblind")
    plt.title(f"{TAG} -- stretch sweep on te_{ANCHOR}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f"\nWrote plot: {fig_path}")

    # =================================================================
    # Deploy
    # =================================================================
    deploy = best_full.astype(np.float32)
    print("\n" + "=" * 78)
    print("DEPLOY")
    print("=" * 78)
    print(f"  {ANCHOR} (s=1.0) in_RAE         = {rae_anchor_in:.4f}")
    print(f"  best s                          = {best_s:.2f}")
    print(f"  best in_RAE                     = {best_rae:.4f}")
    print(f"  improvement vs s=1.0            = "
          f"{rae_anchor_in - best_rae:+.4f}")
    print(f"  deploy te mean/std              = "
          f"{deploy.mean():.3f} / {deploy.std():.3f}")

    np.save(DATA_PROCESSED / f"te_{TAG}.npy", deploy)

    plain = SUBMISSIONS / f"{TAG}_{ANCHOR}_stretch_s{best_s:.2f}.csv"
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
        "anchor_in_pred_std": pred_std_in,
        "truth_mean": float(unb_y.mean()),
        "truth_std": truth_std,
        "mu": mu,
        "stretch_grid": STRETCH_GRID,
        "sweep": rows,
        "best_s": float(best_s),
        "best_in_rae": float(best_rae),
        "delta_vs_anchor": float(rae_anchor_in - best_rae),
        "deploy_te_mean": float(deploy.mean()),
        "deploy_te_std": float(deploy.std()),
        "beats_anchor": bool(best_rae < rae_anchor_in),
        "nb1000_best_s": nb1000_best_s,
        "nb1000_best_in_rae": nb1000_best_rae,
        "delta_vs_nb1000": (
            float(best_rae - nb1000_best_rae)
            if nb1000_best_rae is not None else None
        ),
        "beats_nb1000": (
            bool(best_rae < nb1000_best_rae)
            if nb1000_best_rae is not None else None
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
    print(f"  anchor in_RAE (s=1.0)     = {rae_anchor_in:.4f}")
    print(f"  best s                    = {best_s:.2f}")
    print(f"  best in_RAE               = {best_rae:.4f}")
    print(f"  delta vs anchor           = {rae_anchor_in - best_rae:+.4f}")
    if nb1000_best_rae is not None:
        print(f"  nb1000 best (nb982)       = "
              f"{nb1000_best_rae:.4f} (s={nb1000_best_s:.2f})")
        print(f"  delta vs nb1000           = "
              f"{best_rae - nb1000_best_rae:+.4f}")
    print(f"  beats anchor              = {best_rae < rae_anchor_in}")
    print("=" * 78)

    return {
        "success": True,
        "anchor": ANCHOR,
        "anchor_in_rae": rae_anchor_in,
        "best_s": float(best_s),
        "best_in_rae": float(best_rae),
        "beats_anchor": bool(best_rae < rae_anchor_in),
        "nb1000_best_in_rae": nb1000_best_rae,
        "beats_nb1000": (
            bool(best_rae < nb1000_best_rae)
            if nb1000_best_rae is not None else None
        ),
        "plain_submission": str(plain),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
