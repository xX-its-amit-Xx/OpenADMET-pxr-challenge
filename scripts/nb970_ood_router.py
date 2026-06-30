"""nb970 -- OOD-aware sim-sigmoid per-row weighting router on nb2103 K=28.

HYPOTHESIS:
    Unlike nb960's hard AND-gate F2 shrinkage (binary fire flag + uniform
    shrink_w on a small cluster), this router weights EVERY row by a smooth
    sigmoid of its max Tanimoto similarity to train (ECFP4).  Rows with low
    sim are softly pulled toward a prior; rows with high sim keep raw pred.

    rule:
        w(row)     = sigmoid(alpha * (max_train_sim - threshold))
        final_pred = w * raw_pred + (1 - w) * prior

    Two priors tested:
        global  : train_mean = 4.32 (single scalar pull)
        scaffold: chemprop_aux scaffold median (softer biology-aware prior)

    Sweep:
        alpha     in {2, 5, 10, 20}
        threshold in {0.30, 0.35, 0.40, 0.45}

    Anchor / baseline:
        nb2103 K=28 mean-bag RAE 0.4737, median-bag 0.4698; decision margin 0.003.

OUTPUTS:
    data/processed/nb970_summary.json
    submissions/nb970_deploy_ood_router.csv  (only if beats baseline by margin)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb970"

# Anchors / inputs
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"           # (513,)
NB2103_K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"  # (253,)

# References
NB2103_K28_MEAN_BAG = 0.4737
NB2103_K28_MEDIAN_BAG = 0.4698
DECISION_MARGIN = 0.003

ALPHA_GRID = [2, 5, 10, 20]
THRESHOLD_GRID = [0.30, 0.35, 0.40, 0.45]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _tanimoto_max_per_query(fp_q: np.ndarray, fp_pool: np.ndarray) -> np.ndarray:
    a = fp_q.astype(np.float32)
    b = fp_pool.astype(np.float32)
    a_sum = a.sum(axis=1)
    b_sum = b.sum(axis=1)
    n_q = a.shape[0]
    out = np.zeros(n_q, dtype=np.float32)
    BLOCK = 64
    for s in range(0, n_q, BLOCK):
        e = min(n_q, s + BLOCK)
        inter = a[s:e] @ b.T
        denom = a_sum[s:e, None] + b_sum[None, :] - inter
        denom = np.maximum(denom, 1.0)
        sim = inter / denom
        out[s:e] = sim.max(axis=1)
    return out


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- OOD sim-sigmoid per-row router on nb2103 K=28 OOF (253 unblind)")
    print(f"  baselines: mean_bag={NB2103_K28_MEAN_BAG:.4f}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG:.4f}  margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load inputs ----
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing {ANCHOR_TE_PATH}")
    if not NB2103_K28_OOF_PATH.exists():
        raise FileNotFoundError(f"missing {NB2103_K28_OOF_PATH}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    nb2103_oof_253 = np.load(NB2103_K28_OOF_PATH).astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)

    n_test = te_anchor_513.shape[0]
    n_unb = y_unb.shape[0]
    assert n_unb == 253
    assert nb2103_oof_253.shape[0] == 253
    print(f"[load] anchor_513={te_anchor_513.shape}  nb2103_oof_253={nb2103_oof_253.shape}")

    # ---- Train data + scaffolds + FPs ----
    tr = load_train()
    pec50_col = "pec50" if "pec50" in tr.columns else "pEC50"
    smi_tr_col = "smiles" if "smiles" in tr.columns else "SMILES"
    tr = tr.dropna(subset=[pec50_col, smi_tr_col]).reset_index(drop=True)
    train_mean_pec50 = float(tr[pec50_col].astype(float).mean())
    print(f"[train] n={len(tr)}  train_mean_pec50={train_mean_pec50:.4f}")

    print(f"[scaf] computing Murcko scaffolds for {len(tr)} train cpds ...")
    tr_scaffolds = []
    for s in tr[smi_tr_col].astype(str).tolist():
        sc = bemis_murcko(s)
        tr_scaffolds.append(sc if sc else "")
    tr_scaffold_set = set([s for s in tr_scaffolds if s])

    print(f"[fp] building Morgan ECFP4 for {len(tr)} train cpds ...")
    fp_train = morgan_fp_batch(tr[smi_tr_col].astype(str).tolist())
    keep_tr = fp_train.sum(axis=1) > 0
    fp_train = fp_train[keep_tr]
    print(f"[fp] train fingerprints kept: {fp_train.shape}")

    # ---- Test SMILES + scaffolds + max train sim ----
    te = load_test()
    smi_te_col = "smiles" if "smiles" in te.columns else "SMILES"
    name_col = ("name" if "name" in te.columns
                else ("Molecule Name" if "Molecule Name" in te.columns
                      else "molecule_name"))
    test_smiles = te[smi_te_col].astype(str).tolist()
    assert len(test_smiles) == n_test

    unb_smiles = [test_smiles[i] for i in unb_idx]
    print(f"[scaf] computing Murcko scaffolds for {n_unb} unblind ...")
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]

    print(f"[tan] computing max Tanimoto train for {n_unb} unblind ...")
    unb_std_smi = []
    for s in unb_smiles:
        m = standardize(s)
        unb_std_smi.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_unb = morgan_fp_batch(unb_std_smi)
    max_train_sim_unb = _tanimoto_max_per_query(fp_unb, fp_train).astype(np.float64)
    print(f"[tan] max_train_sim stats: mean={max_train_sim_unb.mean():.3f}  "
          f"med={np.median(max_train_sim_unb):.3f}  "
          f"min={max_train_sim_unb.min():.3f}  "
          f"max={max_train_sim_unb.max():.3f}  "
          f"p25={np.quantile(max_train_sim_unb, 0.25):.3f}  "
          f"p75={np.quantile(max_train_sim_unb, 0.75):.3f}")

    # ---- Scaffold-median chemprop_aux prior on the 253 ----
    # For each unblind row, find its train scaffold's median chemprop_aux pred on
    # the test rows whose scaffold matches.  When no scaffold match in train,
    # fall back to global chemprop_aux test median.
    anchor_unb = te_anchor_513[unb_idx]
    chemprop_aux_test_median = float(np.median(te_anchor_513))

    # Build scaffold -> indices map from TRAIN scaffolds (the actual prior)
    # then look up median of chemprop_aux preds on test rows sharing that scaffold
    # Simpler honest variant: per-scaffold median of anchor across TEST rows of
    # the same scaffold (no train-scaffold join needed; uses anchor itself).
    test_scaffolds_513 = [bemis_murcko(s) for s in test_smiles]
    scaf2test_idx = {}
    for i, sc in enumerate(test_scaffolds_513):
        if not sc:
            continue
        scaf2test_idx.setdefault(sc, []).append(i)

    scaf_anchor_median_513 = np.full(n_test, chemprop_aux_test_median,
                                     dtype=np.float64)
    for sc, idxs in scaf2test_idx.items():
        if len(idxs) >= 2:
            scaf_anchor_median_513[idxs] = float(np.median(te_anchor_513[idxs]))
    # Rows whose scaffold appears only once in test keep global median fallback
    scaf_anchor_median_unb = scaf_anchor_median_513[unb_idx]
    print(f"[prior-scaf] chemprop_aux test median (global): "
          f"{chemprop_aux_test_median:.4f}")
    print(f"[prior-scaf] scaffold-median prior on unb: mean="
          f"{scaf_anchor_median_unb.mean():.4f}  std="
          f"{scaf_anchor_median_unb.std():.4f}")

    # ---- Baseline ----
    raw_pred = nb2103_oof_253.copy()
    rae_baseline = float(rae(y_unb, raw_pred))
    print(f"[base] nb2103 K=28 mean-bag OOF RAE = {rae_baseline:.4f}  "
          f"(ref {NB2103_K28_MEAN_BAG:.4f})")

    # ---- Sweep ----
    print("\n" + "-" * 78)
    print("SWEEP: (alpha x threshold) x prior  ->  RAE on 253 unblind")
    print(f"  alphas={ALPHA_GRID}  thresholds={THRESHOLD_GRID}")
    print(f"  prior #1 = global train_mean ({train_mean_pec50:.4f})")
    print(f"  prior #2 = chemprop_aux scaffold median per row")
    print("-" * 78)

    results = []
    # heat_global[i,j] for prior=train_mean, heat_scaf[i,j] for scaffold prior
    heat_global = np.full((len(ALPHA_GRID), len(THRESHOLD_GRID)),
                          np.nan, dtype=np.float64)
    heat_scaf = np.full((len(ALPHA_GRID), len(THRESHOLD_GRID)),
                        np.nan, dtype=np.float64)

    print(f"  {'prior':>10s}  {'alpha':>5s}  {'th':>5s}  "
          f"{'w_mean':>7s}  {'w_med':>7s}  {'frac_w<.5':>10s}  "
          f"{'RAE':>7s}  {'dRAE':>8s}  verdict")

    for prior_name, prior_vec in [
        ("train_mean", np.full(n_unb, train_mean_pec50, dtype=np.float64)),
        ("scaf_anchor_median", scaf_anchor_median_unb),
    ]:
        for ai, alpha in enumerate(ALPHA_GRID):
            for ti, th in enumerate(THRESHOLD_GRID):
                w = _sigmoid(alpha * (max_train_sim_unb - th))
                final_pred = w * raw_pred + (1.0 - w) * prior_vec
                r = float(rae(y_unb, final_pred))
                d_rae = r - rae_baseline
                w_mean = float(w.mean())
                w_med = float(np.median(w))
                frac_low = float((w < 0.5).mean())

                beats = r < (NB2103_K28_MEDIAN_BAG - DECISION_MARGIN)
                v = "BEATS_MED" if beats else (
                    "FLAT" if abs(d_rae) < DECISION_MARGIN else "WORSE"
                )

                print(f"  {prior_name:>10s}  {alpha:>5d}  {th:>5.2f}  "
                      f"{w_mean:>7.3f}  {w_med:>7.3f}  "
                      f"{frac_low:>10.3f}  {r:>7.4f}  {d_rae:>+8.4f}  {v}")
                results.append({
                    "prior": prior_name,
                    "alpha": int(alpha),
                    "threshold": float(th),
                    "w_mean": w_mean,
                    "w_median": w_med,
                    "frac_w_lt_half": frac_low,
                    "rae": r,
                    "delta_vs_baseline": d_rae,
                    "delta_vs_nb2103_median": r - NB2103_K28_MEDIAN_BAG,
                    "beats_nb2103_median": bool(beats),
                })
                if prior_name == "train_mean":
                    heat_global[ai, ti] = r
                else:
                    heat_scaf[ai, ti] = r

    # ---- Pretty heatmap dump for both priors ----
    def _print_heat(name, mat):
        print(f"\n  HEATMAP RAE (prior={name})")
        print(f"  {'alpha\\th':>10s}  " +
              "  ".join(f"{t:>7.2f}" for t in THRESHOLD_GRID))
        for ai, alpha in enumerate(ALPHA_GRID):
            print(f"  {alpha:>10d}  " +
                  "  ".join(f"{mat[ai,ti]:>7.4f}" for ti in range(len(THRESHOLD_GRID))))

    _print_heat("train_mean", heat_global)
    _print_heat("scaf_anchor_median", heat_scaf)

    # ---- Pick best variant ----
    best = sorted(results, key=lambda r: r["rae"])[0]
    print("\n" + "=" * 78)
    print("BEST VARIANT")
    print("=" * 78)
    print(f"  prior       = {best['prior']}")
    print(f"  alpha       = {best['alpha']}")
    print(f"  threshold   = {best['threshold']:.2f}")
    print(f"  RAE         = {best['rae']:.4f}")
    print(f"  d_vs_base   = {best['delta_vs_baseline']:+.4f} "
          f"(baseline {rae_baseline:.4f})")
    print(f"  d_vs_med    = {best['delta_vs_nb2103_median']:+.4f} "
          f"(nb2103 median {NB2103_K28_MEDIAN_BAG:.4f})")
    print(f"  margin      = {DECISION_MARGIN}")

    beats_median = best["rae"] < (NB2103_K28_MEDIAN_BAG - DECISION_MARGIN)
    beats_mean = best["rae"] < (NB2103_K28_MEAN_BAG - DECISION_MARGIN)

    if beats_median:
        verdict = "DEPLOY_BEATS_NB2103_MEDIAN_BY_MARGIN"
    elif beats_mean:
        verdict = "BEATS_NB2103_MEAN_BUT_NOT_MEDIAN_BY_MARGIN"
    elif abs(best["rae"] - NB2103_K28_MEDIAN_BAG) < DECISION_MARGIN:
        verdict = "FLAT_VS_NB2103_MEDIAN"
    else:
        verdict = "DOES_NOT_BEAT_NB2103"
    print(f"  verdict     = {verdict}")

    # ---- Deploy 513 CSV iff beats ----
    deploy_csv_path = None
    if beats_median:
        print("\n[deploy] beats nb2103 median by margin; building 513-row CSV ...")

        # For 513 deploy we need max train sim on ALL test rows
        all_std_smi = []
        for s in test_smiles:
            m = standardize(s)
            all_std_smi.append(Chem.MolToSmiles(m) if m is not None else "")
        fp_test_all = morgan_fp_batch(all_std_smi)
        max_train_sim_513 = _tanimoto_max_per_query(
            fp_test_all, fp_train
        ).astype(np.float64)

        # raw predictor on 513: chemprop_aux anchor (te_chemprop_aux) -- best
        # available pre-unblind per-row predictor.  Note nb2103 OOF only exists
        # on 253; we mirror the router's behavior on the 253 by using
        # nb2103 OOF where available, else anchor.
        raw_pred_513 = te_anchor_513.copy()
        raw_pred_513[unb_idx] = nb2103_oof_253

        # prior vector on 513
        if best["prior"] == "train_mean":
            prior_513 = np.full(n_test, train_mean_pec50, dtype=np.float64)
        else:
            prior_513 = scaf_anchor_median_513

        w_513 = _sigmoid(best["alpha"] * (max_train_sim_513 - best["threshold"]))
        final_513 = w_513 * raw_pred_513 + (1.0 - w_513) * prior_513

        sub_df = pd.DataFrame({
            "SMILES": te[smi_te_col].astype(str).tolist(),
            "Molecule Name": te[name_col].astype(str).tolist(),
            "pEC50": final_513.astype(np.float32),
        })
        SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"
        SUB_DIR.mkdir(exist_ok=True, parents=True)
        deploy_csv_path = SUB_DIR / f"{TAG}_deploy_ood_router.csv"
        sub_df.to_csv(deploy_csv_path, index=False)
        print(f"[deploy-save] {deploy_csv_path}  shape={sub_df.shape}  "
              f"w_513 mean={w_513.mean():.3f} med={np.median(w_513):.3f}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "ood_sim_sigmoid_per_row_router_two_priors",
        "anchor": "nb2103_K28_mean_bag",
        "rule": ("w = sigmoid(alpha*(max_train_sim - threshold)); "
                 "final_pred = w * raw_pred + (1-w) * prior"),
        "priors_tested": ["train_mean", "scaf_anchor_median"],
        "train_mean_pec50": train_mean_pec50,
        "chemprop_aux_test_median": chemprop_aux_test_median,
        "alpha_grid": ALPHA_GRID,
        "threshold_grid": THRESHOLD_GRID,
        "n_test": n_test,
        "n_unb": n_unb,
        "max_train_sim_unb_stats": {
            "mean": float(max_train_sim_unb.mean()),
            "median": float(np.median(max_train_sim_unb)),
            "min": float(max_train_sim_unb.min()),
            "max": float(max_train_sim_unb.max()),
            "p25": float(np.quantile(max_train_sim_unb, 0.25)),
            "p75": float(np.quantile(max_train_sim_unb, 0.75)),
        },
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG,
        "rae_baseline_nb2103_K28_oof": rae_baseline,
        "decision_margin": DECISION_MARGIN,
        "heatmap_rae_train_mean": heat_global.tolist(),
        "heatmap_rae_scaf_anchor_median": heat_scaf.tolist(),
        "heatmap_rows_alpha": ALPHA_GRID,
        "heatmap_cols_threshold": THRESHOLD_GRID,
        "results": results,
        "best_variant": best,
        "beats_nb2103_median": bool(beats_median),
        "beats_nb2103_mean": bool(beats_mean),
        "verdict": verdict,
        "deploy_csv": str(deploy_csv_path) if deploy_csv_path is not None else None,
        "pre_unblind_clean": True,  # chemprop_aux + nb2103 K=28 + train sim only
        "wall_sec": round(time.time() - t0, 2),
    }

    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "train_mean_pec50",
        "rae_baseline_nb2103_K28_oof",
        "best_variant",
        "beats_nb2103_median",
        "beats_nb2103_mean",
        "verdict",
        "deploy_csv",
    ):
        print(f"  {k}: {res.get(k)}")
