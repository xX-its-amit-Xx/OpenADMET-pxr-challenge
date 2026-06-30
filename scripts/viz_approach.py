"""viz_approach.py — standardized visualization + documentation for ONE PXR modeling approach.

Produces a panel of: (1) pred-vs-actual, (2) residual dist + residual-vs-pred, (3) feature-correlation heatmap,
(4) feature importance (quick LGBM), (5) before/after data-prep label dist, (6) ensemble-member correlation,
(7) coverage (Tanimoto-to-test). Plots ONLY what the supplied artifacts allow; skips the rest gracefully.
Writes C:/pxr_work/viz/<name>/{panel.png, metrics.json, *.png}. Used live by workers AND retroactively by the viz agent.

Usage:
  python scripts/viz_approach.py --name aimnet2 [--oof oof.npy] [--feats feats.npz|feats.npy[:key]]
     [--members a.npy,b.npy,...] [--member-names lgbm,xgb,...] [--submission submissions/nb1177.csv]
     [--prep-before col.npy --prep-after col.npy] [--title "AIMNet2 QM scalars"]
OOF/feature arrays must align to load_train() order (4139). Run with no plot-able artifact -> writes a stub note.
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_ROOT = "C:/pxr_work/viz"


def _load_arr(spec):
    """'path.npy' or 'path.npz:key' -> ndarray."""
    if spec is None: return None
    if ":" in spec and spec.split(":")[0].endswith(".npz"):
        p, k = spec.rsplit(":", 1); return np.load(p)[k]
    a = np.load(spec, allow_pickle=True)
    return a[a.files[0]] if hasattr(a, "files") else a


def rae(yt, yp):
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    d = np.abs(yt - np.median(yt)).sum()
    return float(np.abs(yt - yp).sum() / d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--oof"); ap.add_argument("--feats"); ap.add_argument("--members")
    ap.add_argument("--member-names"); ap.add_argument("--submission")
    ap.add_argument("--prep-before"); ap.add_argument("--prep-after"); ap.add_argument("--title")
    ap.add_argument("--truth", help="override y array (e.g. 253-unblind truth) to align OOF/feature panels")
    a = ap.parse_args()
    outdir = os.path.join(OUT_ROOT, a.name); os.makedirs(outdir, exist_ok=True)
    title = a.title or a.name
    metrics = {"name": a.name}

    from src.pxr.data import load_train
    if a.truth:
        y = _load_arr(a.truth).astype(float)
        metrics["truth_source"] = os.path.basename(a.truth); metrics["n_eval"] = int(len(y))
    else:
        y = load_train().dropna(subset=["pec50"]).reset_index(drop=True)["pec50"].to_numpy()

    panels = []   # (fn, plotter)
    oof = _load_arr(a.oof) if a.oof else None
    if oof is not None and len(oof) == len(y):
        metrics["oof_rae"] = round(rae(y, oof), 4)
        metrics["oof_r"] = round(float(np.corrcoef(y, oof)[0, 1]), 4)
        res = y - oof

        def p_predact(ax):
            ax.scatter(y, oof, s=6, alpha=0.25, edgecolors="none")
            lo, hi = min(y.min(), oof.min()), max(y.max(), oof.max())
            ax.plot([lo, hi], [lo, hi], "r--", lw=1)
            ax.set_xlabel("true pEC50"); ax.set_ylabel("OOF pred")
            ax.set_title(f"pred vs actual (RAE {metrics['oof_rae']}, r {metrics['oof_r']})")

        def p_resid(ax):
            ax.scatter(oof, res, s=6, alpha=0.25, edgecolors="none")
            ax.axhline(0, color="r", ls="--", lw=1)
            ax.set_xlabel("OOF pred"); ax.set_ylabel("residual (true-pred)")
            ax.set_title(f"residuals (MAE {np.abs(res).mean():.3f}, bias {res.mean():+.3f})")

        def p_resdist(ax):
            ax.hist(res, bins=50, color="#4477aa"); ax.axvline(0, color="r", ls="--")
            ax.set_xlabel("residual"); ax.set_title("residual distribution")
        panels += [("pred_vs_actual.png", p_predact), ("residuals.png", p_resid), ("resid_dist.png", p_resdist)]

    feats = _load_arr(a.feats) if a.feats else None
    if feats is not None and feats.ndim == 2 and len(feats) == len(y):
        feats = np.nan_to_num(feats.astype(float))
        nf = feats.shape[1]
        # correlation of each feature with y; show top-correlated subset heatmap
        with np.errstate(invalid="ignore"):
            cors = np.array([np.corrcoef(feats[:, j], y)[0, 1] if feats[:, j].std() > 0 else 0 for j in range(nf)])
        cors = np.nan_to_num(cors)
        top = np.argsort(-np.abs(cors))[:min(20, nf)]
        metrics["n_features"] = int(nf)
        metrics["max_abs_feat_corr_with_y"] = round(float(np.abs(cors).max()), 4)

        def p_featcorr(ax):
            sub = feats[:, top]
            cm = np.corrcoef(np.column_stack([sub, y]).T)
            im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"top-{len(top)} feature corr (+y, last row/col)")
            ax.set_xticks([]); ax.set_yticks([]); plt.colorbar(im, ax=ax, fraction=0.046)

        def p_imp(ax):
            try:
                from lightgbm import LGBMRegressor
                m = LGBMRegressor(n_estimators=200, num_leaves=31, n_jobs=4, verbose=-1).fit(feats, y)
                imp = m.feature_importances_; o = np.argsort(-imp)[:20]
                ax.barh(range(len(o)), imp[o][::-1], color="#66aa55")
                ax.set_yticks(range(len(o))); ax.set_yticklabels([f"f{j}" for j in o[::-1]], fontsize=6)
                ax.set_title("LGBM feature importance (top 20)")
            except Exception as e:
                ax.text(0.1, 0.5, f"imp n/a: {str(e)[:40]}"); ax.set_axis_off()
        panels += [("feature_corr.png", p_featcorr), ("feature_importance.png", p_imp)]

    if a.members:
        mpaths = a.members.split(","); names = (a.member_names or "").split(",")
        marr = []
        for i, mp in enumerate(mpaths):
            try:
                v = _load_arr(mp)
                if len(v) == len(y): marr.append((names[i] if i < len(names) and names[i] else f"m{i}", v))
            except Exception: pass
        if len(marr) >= 2:
            mat = np.corrcoef(np.array([v for _, v in marr]))
            metrics["member_corr_min"] = round(float(mat[np.triu_indices(len(marr), 1)].min()), 4)

            def p_membercorr(ax):
                im = ax.imshow(mat, cmap="viridis", vmin=mat.min(), vmax=1)
                ax.set_xticks(range(len(marr))); ax.set_xticklabels([n for n, _ in marr], rotation=90, fontsize=6)
                ax.set_yticks(range(len(marr))); ax.set_yticklabels([n for n, _ in marr], fontsize=6)
                ax.set_title("ensemble member OOF correlation"); plt.colorbar(im, ax=ax, fraction=0.046)
            panels.append(("member_corr.png", p_membercorr))

    pb = _load_arr(a.prep_before) if a.prep_before else None
    pa = _load_arr(a.prep_after) if a.prep_after else None
    if pb is not None or pa is not None:
        def p_prep(ax):
            if pb is not None: ax.hist(pb, bins=40, alpha=0.5, label="before", color="#cc6677")
            if pa is not None: ax.hist(pa, bins=40, alpha=0.5, label="after", color="#4477aa")
            ax.legend(); ax.set_title("data-prep: before vs after")
        panels.append(("data_prep.png", p_prep))

    # coverage: Tanimoto of test->train (context for the approach); cheap, always available
    try:
        from src.pxr.data import load_test
        from src.pxr.chem import morgan_fp_batch
        te = load_test().reset_index(drop=True)
        trsm = load_train().dropna(subset=["pec50"]).reset_index(drop=True)["smiles"].tolist()
        ft = morgan_fp_batch(te["smiles"].tolist()).astype(bool)
        fr = morgan_fp_batch(trsm[:2000]).astype(bool)  # cap for speed
        def tani_max(q):
            inter = (fr & q).sum(1); uni = (fr | q).sum(1); s = inter / np.maximum(uni, 1)
            return s.max()
        cov = np.array([tani_max(ft[i]) for i in range(len(ft))])
        metrics["test_to_train_tanimoto_median"] = round(float(np.median(cov)), 3)

        def p_cov(ax):
            ax.hist(cov, bins=40, color="#aa7744"); ax.axvline(0.4, color="r", ls="--", label="0.4 coverage line")
            ax.set_xlabel("max Tanimoto test->train"); ax.legend()
            ax.set_title(f"coverage (median {np.median(cov):.2f})")
        panels.append(("coverage.png", p_cov))
    except Exception:
        pass

    # combined panel
    if panels:
        n = len(panels); cols = 3; rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.atleast_1d(axes).ravel()
        for i, (fn, fnc) in enumerate(panels):
            fnc(axes[i])
            f2, a2 = plt.subplots(figsize=(5, 4)); fnc(a2); f2.tight_layout()
            f2.savefig(os.path.join(outdir, fn), dpi=90); plt.close(f2)
        for j in range(len(panels), len(axes)): axes[j].set_axis_off()
        fig.suptitle(f"PXR approach: {title}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(outdir, "panel.png"), dpi=100); plt.close(fig)
        metrics["panels"] = [fn for fn, _ in panels]
    else:
        metrics["note"] = "no plot-able artifacts supplied"

    json.dump(metrics, open(os.path.join(outdir, "metrics.json"), "w"), indent=2)
    print(f"viz[{a.name}] -> {outdir}  ({len(panels)} panels)  metrics={json.dumps(metrics)[:200]}")


if __name__ == "__main__":
    main()
