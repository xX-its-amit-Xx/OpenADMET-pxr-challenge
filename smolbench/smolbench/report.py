"""Figures + text report for a BenchmarkResult (matplotlib, no interactivity)."""
from __future__ import annotations
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_C = {"pos": "#3a7ca5", "neg": "#d1495b", "acc": "#edae49", "grid": "#cccccc"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def make_figures(result, out_dir, y_true=None, target_name="target"):
    """Write heatmap, per-model boxplot, and (if y_true for the test set) pred-vs-truth."""
    os.makedirs(out_dir, exist_ok=True)
    df = result.results
    paths = []

    # 1. featurizer x model heatmap of best RAE
    piv = df.pivot_table(index="featurizer", columns="model", values="RAE", aggfunc="min")
    fig, ax = plt.subplots(figsize=(1.1 * piv.shape[1] + 2, 0.6 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v > np.nanmedian(piv.values) else "black")
    ax.set_title(f"Scaffold-CV RAE by featurizer x model (lower=better)")
    fig.colorbar(im, ax=ax, label="RAE"); fig.tight_layout()
    p = os.path.join(out_dir, "heatmap_featurizer_model.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 2. per-model best RAE bar
    fig, ax = plt.subplots(figsize=(7, 4))
    s = df.groupby("model")["RAE"].min().sort_values()
    ax.barh(s.index, s.values, color=_C["pos"]); ax.invert_yaxis()
    ax.set_xlabel("best scaffold-CV RAE"); ax.set_title("Best RAE achievable per model family")
    fig.tight_layout(); p = os.path.join(out_dir, "model_ranking.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 3. holdout pred vs truth (if provided)
    if y_true is not None and result.test_predictions is not None:
        from .metrics import all_metrics
        p_ = result.test_predictions; y_ = np.asarray(y_true, float)
        m = all_metrics(y_, p_)
        fig, ax = plt.subplots(figsize=(5.4, 5.2))
        ax.scatter(y_, p_, s=26, alpha=0.6, c=_C["pos"], edgecolor="none")
        lims = [min(y_.min(), p_.min()) - 0.3, max(y_.max(), p_.max()) + 0.3]
        ax.plot(lims, lims, "k--", alpha=0.5); ax.fill_between(lims, [l - 1 for l in lims], [l + 1 for l in lims], color="gray", alpha=0.08)
        ax.set_xlim(lims); ax.set_ylim(lims); ax.set_xlabel(f"measured {target_name}"); ax.set_ylabel("predicted")
        ax.set_title(f"Test set: RAE {m['RAE']:.3f} | MAE {m['MAE']:.3f} | R2 {m['R2']:.2f} | ρ {m['Spearman']:.2f}")
        fig.tight_layout(); p = os.path.join(out_dir, "test_pred_vs_truth.png"); fig.savefig(p, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def text_report(result):
    lines = ["=" * 66, "smolbench benchmark report", "=" * 66,
             f"train molecules: {result.meta['n_train']} | CV: {result.meta['cv']} "
             f"({result.meta['n_folds']} folds) | configs: {result.meta['n_configs']} "
             f"| runtime {result.meta['runtime_s']}s", ""]
    lines.append("TOP 8 CONFIGURATIONS (by scaffold-CV RAE):")
    lines.append(result.top(8)[["featurizer", "prep", "model", "RAE", "MAE", "R2", "Spearman"]].to_string(index=False))
    lines += ["", "INSIGHTS:"] + [f"  - {s}" for s in result.insights]
    return "\n".join(lines)
