"""nb1101 — LOW-ACTIVITY diagnostic: is the weak-compound error fixable BIAS or irreducible VARIANCE?

User hypothesis + our F2 post-mortem: we do well on strong binders, over-predict weak/inactive compounds.
Research verdict: low tail = removable left-censoring BIAS (over-prediction) + irreducible aleatoric noise.
This quantifies it on the 253 with nb3200, and visualizes:
  - error & BIAS (yhat - y) stratified by TRUE pEC50  -> is low-end systematically over-predicted?
  - error stratified by PREDICTED pEC50               -> can we trust 'predicted-low' for a redirect/calibration?
  - RAE decomposition by bin                          -> how much RAE lives in the low tail (the headroom)
  - oracle: RAE if the low tail were perfectly fixed  -> upper bound on the lever
  - novelty overlay (top-1 train Tanimoto)            -> F2 novel-scaffold angle
Figures -> data/processed/figures/nb1101_*.png
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.eval import rae
from src.pxr.chem import morgan_fp_batch
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "data/processed"; FIG = f"{P}/figures"; os.makedirs(FIG, exist_ok=True)


def fpf(sm): return (morgan_fp_batch(sm).astype(np.float32) > 0).astype(np.float32)


def main():
    unb = np.load(f"{P}/_audit_unblind_idx.npy"); y = np.load(f"{P}/_audit_unblind_y.npy")
    anchor = np.load(f"{P}/nb3200_pred_oof.npy"); err = anchor - y          # +ve = over-prediction
    te = load_test(); tr = load_train().dropna(subset=["pec50"])
    smte = te["smiles"].to_numpy()[unb].tolist()
    # novelty: top-1 train Tanimoto
    Fte = fpf(smte); Ftr = fpf(tr["smiles"].tolist())
    inter = Fte @ Ftr.T; sim = inter / np.clip(Fte.sum(1)[:, None] + Ftr.sum(1)[None, :] - inter, 1, None)
    top1 = sim.max(1)

    print(f"nb3200 RAE {rae(y, anchor):.4f} | n={len(y)} | y range {y.min():.2f}-{y.max():.2f} median {np.median(y):.2f}")
    # stratify by TRUE pEC50 (quartiles)
    qs = np.quantile(y, [0, .25, .5, .75, 1.0])
    print(f"\n=== ERROR by TRUE pEC50 quartile (bias = mean(yhat - y); +ve = OVER-predict) ===")
    print(f"{'bin':16s} {'n':>4s} {'mean_y':>7s} {'bias':>7s} {'MAE':>6s} {'%of_RAE_num':>11s}")
    tot_num = np.sum(np.abs(err))
    rows = []
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        m = (y >= lo) & (y <= hi) if i == 3 else (y >= lo) & (y < hi)
        bias = np.mean(err[m]); mae = np.mean(np.abs(err[m])); frac = np.sum(np.abs(err[m])) / tot_num
        rows.append((f"[{lo:.2f},{hi:.2f}]", m.sum(), y[m].mean(), bias, mae, frac))
        print(f"{rows[-1][0]:16s} {m.sum():>4d} {y[m].mean():>7.2f} {bias:>+7.2f} {mae:>6.2f} {frac:>10.1%}")

    # stratify by PREDICTED pEC50 (for redirect/calibration feasibility)
    print(f"\n=== ERROR by PREDICTED pEC50 quartile (can we trust 'predicted-low'?) ===")
    qp = np.quantile(anchor, [0, .25, .5, .75, 1.0])
    for i in range(4):
        lo, hi = qp[i], qp[i + 1]
        m = (anchor >= lo) & (anchor <= hi) if i == 3 else (anchor >= lo) & (anchor < hi)
        print(f"pred[{lo:.2f},{hi:.2f}] n={m.sum():>3d} bias={np.mean(err[m]):+.2f} MAE={np.mean(np.abs(err[m])):.2f} "
              f"true_mean={y[m].mean():.2f}")

    # ORACLE: RAE if the bottom-quartile true compounds were predicted perfectly
    low = y <= qs[1]
    fixed = anchor.copy(); fixed[low] = y[low]
    print(f"\nORACLE RAE if bottom-quartile TRUE compounds perfect: {rae(y, fixed):.4f} (vs {rae(y, anchor):.4f}, "
          f"headroom {rae(y, fixed)-rae(y, anchor):+.4f})")
    # ORACLE: if we just removed the BIAS on low-predicted (shift them down by their mean bias)
    lowp = anchor <= qp[1]
    deb = anchor.copy(); deb[lowp] = anchor[lowp] - np.mean(err[lowp])
    print(f"ORACLE RAE if predicted-low de-biased (shift by mean bias): {rae(y, deb):.4f} "
          f"(headroom {rae(y, deb)-rae(y, anchor):+.4f}) [in-sample, optimistic]")
    # novelty interaction
    nov = top1 < 0.4
    print(f"\nnovel (top1<0.4) n={nov.sum()}: bias={np.mean(err[nov]):+.2f} MAE={np.mean(np.abs(err[nov])):.2f} | "
          f"familiar bias={np.mean(err[~nov]):+.2f} MAE={np.mean(np.abs(err[~nov])):.2f}")
    nov_low = nov & low
    print(f"novel & low-activity n={nov_low.sum()}: bias={np.mean(err[nov_low]) if nov_low.sum() else 0:+.2f} "
          f"MAE={np.mean(np.abs(err[nov_low])) if nov_low.sum() else 0:.2f}")

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].scatter(y, anchor, c=top1, cmap="viridis", s=28, alpha=0.8)
    lims = [y.min() - 0.3, y.max() + 0.3]; ax[0].plot(lims, lims, "k--", lw=1)
    ax[0].set_xlabel("true pEC50"); ax[0].set_ylabel("nb3200 predicted"); ax[0].set_title("Pred vs true (color=train sim)")
    cb = fig.colorbar(ax[0].collections[0], ax=ax[0]); cb.set_label("top-1 train Tanimoto")
    # binned bias
    cents = [(qs[i] + qs[i + 1]) / 2 for i in range(4)]
    biases = [r[3] for r in rows]
    ax[1].bar(range(4), biases, color=["#d62728" if b > 0 else "#2ca02c" for b in biases])
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xticks(range(4))
    ax[1].set_xticklabels([f"Q{i+1}\n{r[2]:.1f}" for i, r in enumerate(rows)])
    ax[1].set_ylabel("bias = mean(pred - true)"); ax[1].set_title("Over-prediction by true-pEC50 quartile")
    # error vs novelty
    ax[2].scatter(top1, np.abs(err), c=(y <= qs[1]), cmap="coolwarm", s=28, alpha=0.8)
    ax[2].set_xlabel("top-1 train Tanimoto"); ax[2].set_ylabel("|error|")
    ax[2].set_title("|error| vs novelty (red=low-activity)")
    plt.tight_layout(); plt.savefig(f"{FIG}/nb1101_lowactivity_diagnostic.png", dpi=110); plt.close()
    print(f"\nwrote {FIG}/nb1101_lowactivity_diagnostic.png")
    json.dump({"rae": float(rae(y, anchor)), "bins": [[r[0], int(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in rows],
               "oracle_lowtrue_perfect": float(rae(y, fixed)), "novel_low_n": int(nov_low.sum())},
              open(f"{P}/nb1101_lowactivity.json", "w"), indent=2)


if __name__ == "__main__":
    main()
