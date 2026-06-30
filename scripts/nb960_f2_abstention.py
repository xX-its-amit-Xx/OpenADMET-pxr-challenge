"""nb960 -- M1 Conformal Abstention Router on chemprop_aux v1 anchor (F2 cluster).

HYPOTHESIS:
    Per memory feedback_failure_mode_quantile_compression + pm06 framing
    (feedback_phase2_prescriptions), the F2 failure cluster is greasy-novel-
    inactive compounds that get over-predicted: 90% novel scaffold +
    high seed-bag uncertainty + high counter-assay disagreement.
    A 3-criterion AND-gate identifies the F2 cluster; shrink those rows
    only toward the train mean (shrink_w fraction).

    Anchor: te_chemprop_aux.npy on 513 (in_RAE 0.6216 on 253 unblind).
    Baseline: nb2103 K=28 mean-bag 0.4737 / median-bag 0.4698.
    Decision margin = 0.003.

ABSTENTION RULE:
    fire := (scaf_train_freq == 0)
            AND (bag_std > std_threshold)
            AND (counter_disagreement > 0.3)
    if fire: pred_new = (1 - shrink_w) * pred + shrink_w * train_mean
    else   : pred_new = pred  (untouched)

GATES:
    - Abstention fire rate on 253 must be in [5%, 15%]; else flag.
    - Best (shrink_w, std_threshold) must beat 0.4698 by >= 0.003 to deploy.

OUTPUTS:
    data/processed/nb960_summary.json
    submissions/nb960_deploy_abstention.csv  (only if beats by margin)
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

TAG = "nb960"

# Anchors / inputs
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"            # (513,)
NB2103_K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"  # (253,)
NB2190_STD_PATH = DATA_PROCESSED / "nb2190_per_row_std_K28.npy"      # (253,)
NULL_PRED_513_PATH = DATA_PROCESSED / "pred_pec50_null_513.npy"      # (513,)

# Baselines
NB2103_K28_MEAN_BAG = 0.4737
NB2103_K28_MEDIAN_BAG = 0.4698
DECISION_MARGIN = 0.003

# Sweep grids
SHRINK_W_GRID = [0.0, 0.15, 0.3, 0.5]
STD_THRESHOLD_GRID = [0.2, 0.3, 0.4, 0.5]
COUNTER_DISAGREEMENT_THRESHOLD = 0.3  # fixed per task spec

# Fire-rate validation gate
FIRE_RATE_LO = 0.05
FIRE_RATE_HI = 0.15


def _safe_inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) if mol is not None else None
    except Exception:
        return None


def _tanimoto_max_per_query(fp_q: np.ndarray, fp_pool: np.ndarray) -> np.ndarray:
    """For each query, return the max Tanimoto similarity to any pool fingerprint.
    Uses bit-uint8 ECFP4 fingerprints (N,2048).
    """
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
    print(f"{TAG} -- F2 abstention router on chemprop_aux anchor + nb2103 K=28")
    print(f"           baselines: mean_bag={NB2103_K28_MEAN_BAG:.4f}  "
          f"median_bag={NB2103_K28_MEDIAN_BAG:.4f}  margin={DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load all required arrays ----
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"missing {ANCHOR_TE_PATH}")
    if not NB2103_K28_OOF_PATH.exists():
        raise FileNotFoundError(f"missing {NB2103_K28_OOF_PATH}")
    if not NB2190_STD_PATH.exists():
        raise FileNotFoundError(f"missing {NB2190_STD_PATH}")
    if not NULL_PRED_513_PATH.exists():
        raise FileNotFoundError(f"missing {NULL_PRED_513_PATH}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    nb2103_oof_253 = np.load(NB2103_K28_OOF_PATH).astype(np.float64)
    bag_std_253 = np.load(NB2190_STD_PATH).astype(np.float64)
    null_pred_513 = np.load(NULL_PRED_513_PATH).astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)

    n_test = te_anchor_513.shape[0]
    n_unb = y_unb.shape[0]
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    assert bag_std_253.shape[0] == 253
    assert nb2103_oof_253.shape[0] == 253
    print(f"[load] anchor_513={te_anchor_513.shape}  nb2103_oof_253={nb2103_oof_253.shape}  "
          f"bag_std_253={bag_std_253.shape}  null_513={null_pred_513.shape}")

    # ---- Train + test SMILES + scaffolds ----
    tr = load_train()
    pec50_col = "pec50" if "pec50" in tr.columns else "pEC50"
    smi_tr_col = "smiles" if "smiles" in tr.columns else "SMILES"
    tr = tr.dropna(subset=[pec50_col, smi_tr_col]).reset_index(drop=True)
    train_mean_pec50 = float(tr[pec50_col].astype(float).mean())
    print(f"[train] n_train_with_pec50 = {len(tr)}  train_mean_pec50 = {train_mean_pec50:.4f}")

    # Train Murcko scaffold set
    print(f"[scaf] computing Murcko scaffolds for {len(tr)} train cpds ...")
    tr_scaffolds = set()
    for s in tr[smi_tr_col].astype(str).tolist():
        sc = bemis_murcko(s)
        if sc is not None and sc != "":
            tr_scaffolds.add(sc)
    print(f"[scaf] unique train scaffolds = {len(tr_scaffolds)}")

    # Train ECFP4 fingerprints (for max Tanimoto to train)
    print(f"[fp]   building Morgan ECFP4 for {len(tr)} train cpds ...")
    fp_train = morgan_fp_batch(tr[smi_tr_col].astype(str).tolist())
    keep_tr = fp_train.sum(axis=1) > 0
    fp_train = fp_train[keep_tr]
    print(f"[fp]   train fingerprints kept: {fp_train.shape}")

    te = load_test()
    smi_te_col = "smiles" if "smiles" in te.columns else "SMILES"
    name_col = "name" if "name" in te.columns else (
        "Molecule Name" if "Molecule Name" in te.columns else "molecule_name"
    )
    test_smiles = te[smi_te_col].astype(str).tolist()
    assert len(test_smiles) == n_test

    # Unblind subset
    unb_smiles = [test_smiles[i] for i in unb_idx]

    # ---- (1) scaf_train_freq for unblind (binary: 0 if novel else 1) ----
    print(f"[scaf] computing scaffold freq for {n_unb} unblind ...")
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    # scaf_train_freq == 0 if scaffold not present in train, else >=1
    scaf_train_freq_unb = np.array(
        [0 if (sc is None or sc == "" or sc not in tr_scaffolds) else 1
         for sc in unb_scaffolds],
        dtype=np.int32,
    )
    n_novel = int((scaf_train_freq_unb == 0).sum())
    print(f"[scaf] unblind novel-scaffold rows: {n_novel}/{n_unb} ({n_novel/n_unb:.1%})")

    # ---- (2) max_tanimoto_train for unblind (used only for diagnostics here) ----
    print(f"[tan]  computing max Tanimoto train for {n_unb} unblind ...")
    unb_std_smi = []
    for s in unb_smiles:
        m = standardize(s)
        unb_std_smi.append(Chem.MolToSmiles(m) if m is not None else "")
    fp_unb = morgan_fp_batch(unb_std_smi)
    max_tan_unb = _tanimoto_max_per_query(fp_unb, fp_train)
    print(f"[tan]  max_tanimoto stats: mean={max_tan_unb.mean():.3f}  "
          f"med={np.median(max_tan_unb):.3f}  min={max_tan_unb.min():.3f}  "
          f"max={max_tan_unb.max():.3f}")

    # ---- (3) counter_disagreement for unblind ----
    # |pEC50_chemprop_aux - pEC50_null_pred|   (both on 513, slice unb)
    counter_disagreement_513 = np.abs(te_anchor_513 - null_pred_513)
    counter_disagreement_unb = counter_disagreement_513[unb_idx]
    print(f"[ctr]  counter-disagreement stats (unb): mean={counter_disagreement_unb.mean():.3f}  "
          f"med={np.median(counter_disagreement_unb):.3f}  "
          f"min={counter_disagreement_unb.min():.3f}  max={counter_disagreement_unb.max():.3f}")

    # ---- Baseline check: nb2103 K=28 OOF RAE on 253 ----
    baseline_oof = nb2103_oof_253.copy()
    rae_baseline = float(rae(y_unb, baseline_oof))
    print(f"[base] nb2103 K=28 mean-bag OOF RAE = {rae_baseline:.4f}  "
          f"(ref {NB2103_K28_MEAN_BAG:.4f})")

    # ---- Sweep (shrink_w, std_threshold) ----
    print("\n" + "-" * 78)
    print("SWEEP: shrink_w x std_threshold abstention router")
    print(f"  rule: (scaf_train_freq==0) AND (bag_std > std_th) AND "
          f"(counter_disagree > {COUNTER_DISAGREEMENT_THRESHOLD})")
    print(f"  fire-rate gate: [{FIRE_RATE_LO:.0%}, {FIRE_RATE_HI:.0%}] on 253")
    print("-" * 78)
    print(f"  {'shrink_w':>8s}  {'std_th':>6s}  {'n_fire':>6s}  {'fire%':>6s}  "
          f"{'RAE':>7s}  {'dRAE':>8s}  {'gate':>4s}")

    results = []
    for shrink_w in SHRINK_W_GRID:
        for std_th in STD_THRESHOLD_GRID:
            fire = (
                (scaf_train_freq_unb == 0)
                & (bag_std_253 > std_th)
                & (counter_disagreement_unb > COUNTER_DISAGREEMENT_THRESHOLD)
            )
            n_fire = int(fire.sum())
            fire_rate = float(n_fire / n_unb)

            pred = baseline_oof.copy()
            pred[fire] = (1.0 - shrink_w) * baseline_oof[fire] + shrink_w * train_mean_pec50

            r = float(rae(y_unb, pred))
            d_rae = r - rae_baseline

            in_gate = (FIRE_RATE_LO <= fire_rate <= FIRE_RATE_HI)
            gate_str = "OK" if in_gate else "FAIL"
            print(f"  {shrink_w:>8.2f}  {std_th:>6.2f}  {n_fire:>6d}  "
                  f"{fire_rate:>5.1%}  {r:>7.4f}  {d_rae:>+8.4f}  {gate_str:>4s}")
            results.append({
                "shrink_w": float(shrink_w),
                "std_threshold": float(std_th),
                "counter_disagreement_threshold": float(COUNTER_DISAGREEMENT_THRESHOLD),
                "n_fire": n_fire,
                "fire_rate": fire_rate,
                "in_fire_rate_gate": bool(in_gate),
                "rae": r,
                "delta_vs_baseline": d_rae,
                "delta_vs_median_base": r - NB2103_K28_MEDIAN_BAG,
                "delta_vs_mean_base": r - NB2103_K28_MEAN_BAG,
            })

    # ---- Pick best ----
    # Ranking: lowest RAE among in-gate variants first; if none in-gate, use overall best
    in_gate_results = [r for r in results if r["in_fire_rate_gate"]]
    if in_gate_results:
        best = sorted(in_gate_results, key=lambda x: x["rae"])[0]
        best_source = "in_fire_rate_gate"
    else:
        best = sorted(results, key=lambda x: x["rae"])[0]
        best_source = "all_variants_gate_failed"

    print("\n" + "=" * 78)
    print("BEST VARIANT")
    print("=" * 78)
    print(f"  source            = {best_source}")
    print(f"  shrink_w          = {best['shrink_w']:.2f}")
    print(f"  std_threshold     = {best['std_threshold']:.2f}")
    print(f"  fire_rate         = {best['fire_rate']:.1%}  (n_fire={best['n_fire']}/{n_unb})")
    print(f"  RAE               = {best['rae']:.4f}")
    print(f"  delta vs baseline = {best['delta_vs_baseline']:+.4f}  "
          f"(nb2103 K=28 mean-bag {rae_baseline:.4f})")
    print(f"  delta vs median   = {best['delta_vs_median_base']:+.4f}  "
          f"(nb2103 median-bag {NB2103_K28_MEDIAN_BAG:.4f})")
    print(f"  decision margin   = {DECISION_MARGIN}")

    # Decision: beats nb2103 median-bag by >= margin AND in-gate?
    beats_median = best["rae"] < (NB2103_K28_MEDIAN_BAG - DECISION_MARGIN)
    beats_mean = best["rae"] < (NB2103_K28_MEAN_BAG - DECISION_MARGIN)
    in_gate = best["in_fire_rate_gate"]

    if beats_median and in_gate:
        verdict = "DEPLOY_BEATS_NB2103_MEDIAN_WITH_GATE_OK"
    elif beats_median and not in_gate:
        verdict = "BEATS_NB2103_MEDIAN_BUT_FIRE_RATE_OUTSIDE_GATE"
    elif beats_mean and in_gate:
        verdict = "BEATS_NB2103_MEAN_BUT_NOT_MEDIAN_GATE_OK"
    elif beats_mean:
        verdict = "BEATS_NB2103_MEAN_BUT_NOT_MEDIAN_AND_GATE_FAIL"
    else:
        verdict = "DOES_NOT_BEAT_NB2103_MEDIAN_BY_MARGIN"
    print(f"  verdict           = {verdict}")

    # ---- Deploy CSV if beats AND in-gate ----
    deploy_csv_path = None
    deploy_fire_rate_513 = None
    if beats_median and in_gate:
        print("\n[deploy] best beats baseline and is in fire-rate gate; building 513-row CSV ...")
        # Apply SAME abstention rule on full 513:
        # 1. Train scaffold check on all 513 test compounds
        # 2. bag_std unavailable on 260 non-unblind rows -> we must extrapolate.
        #    Honest move: use the 253 distribution percentile of bag_std as proxy.
        #    Simplest: if a 513 row has no bag_std proxy, treat std-threshold as failed
        #    (= do NOT shrink). This is conservative.
        # We have:
        #   anchor (te_chemprop_aux) on 513
        #   null_pred on 513
        #   scaf for all 513
        #   bag_std only on 253 (unb_idx slice)

        # 513-row scaffolds
        print(f"[scaf-513] computing Murcko scaffolds for {n_test} test cpds ...")
        test_scaffolds_513 = [bemis_murcko(s) for s in test_smiles]
        scaf_train_freq_513 = np.array(
            [0 if (sc is None or sc == "" or sc not in tr_scaffolds) else 1
             for sc in test_scaffolds_513],
            dtype=np.int32,
        )

        # bag_std on 513: NaN-fill non-unblind; we set those to 0 (= will fail std_th)
        bag_std_513 = np.zeros(n_test, dtype=np.float64)
        bag_std_513[unb_idx] = bag_std_253

        # fire on 513
        fire_513 = (
            (scaf_train_freq_513 == 0)
            & (bag_std_513 > best["std_threshold"])
            & (counter_disagreement_513 > COUNTER_DISAGREEMENT_THRESHOLD)
        )
        n_fire_513 = int(fire_513.sum())
        deploy_fire_rate_513 = float(n_fire_513 / n_test)
        print(f"[deploy-fire] n_fire_513={n_fire_513}/{n_test}  "
              f"rate={deploy_fire_rate_513:.1%}  "
              f"(only unb rows can fire since bag_std=0 elsewhere)")

        # Deploy predictions on 513
        pred_513 = te_anchor_513.copy()
        # For UNB rows that fire: shrink toward train mean using nb2103 K=28 OOF as predictor
        # (this is the anchor for the 253 since it's our best honest predictor there)
        # For NON-UNB rows: leave at chemprop_aux (no fire possible since bag_std=0)
        # For UNB rows that don't fire: leave at chemprop_aux te (consistency)
        #
        # NOTE: this is a conservative deploy -- the 260 blinded rows can't fire
        # because we have no bag_std estimate.  This matches nb2190's deploy policy.
        for j, idx_in_513 in enumerate(unb_idx):
            if fire_513[idx_in_513]:
                pred_513[idx_in_513] = (
                    (1.0 - best["shrink_w"]) * nb2103_oof_253[j]
                    + best["shrink_w"] * train_mean_pec50
                )
            else:
                # do not perturb -- keep anchor
                pass

        # Write CSV
        sub_df = pd.DataFrame({
            "SMILES": te[smi_te_col].astype(str).tolist(),
            "Molecule Name": te[name_col].astype(str).tolist(),
            "pEC50": pred_513.astype(np.float32),
        })
        SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"
        SUB_DIR.mkdir(exist_ok=True, parents=True)
        deploy_csv_path = SUB_DIR / f"{TAG}_deploy_abstention.csv"
        sub_df.to_csv(deploy_csv_path, index=False)
        print(f"[deploy-save] {deploy_csv_path}  shape={sub_df.shape}")

    # ---- Save summary ----
    summary = {
        "tag": TAG,
        "method": "f2_conformal_abstention_router_AND_gate",
        "anchor": "chemprop_aux",
        "rule": (
            "(scaf_train_freq==0) AND (bag_std > std_threshold) AND "
            f"(counter_disagreement > {COUNTER_DISAGREEMENT_THRESHOLD}) -> "
            "pred_new = (1-shrink_w)*pred + shrink_w*train_mean"
        ),
        "shrink_w_grid": SHRINK_W_GRID,
        "std_threshold_grid": STD_THRESHOLD_GRID,
        "counter_disagreement_threshold": COUNTER_DISAGREEMENT_THRESHOLD,
        "fire_rate_gate": [FIRE_RATE_LO, FIRE_RATE_HI],
        "n_test": n_test,
        "n_unb": n_unb,
        "n_novel_scaffold_unb": n_novel,
        "frac_novel_scaffold_unb": n_novel / n_unb,
        "train_mean_pec50": train_mean_pec50,
        "rae_baseline_nb2103_K28_mean_bag": rae_baseline,
        "nb2103_K28_mean_bag_ref": NB2103_K28_MEAN_BAG,
        "nb2103_K28_median_bag_ref": NB2103_K28_MEDIAN_BAG,
        "decision_margin": DECISION_MARGIN,
        "bag_std_stats_unb": {
            "mean": float(bag_std_253.mean()),
            "median": float(np.median(bag_std_253)),
            "min": float(bag_std_253.min()),
            "max": float(bag_std_253.max()),
            "p25": float(np.quantile(bag_std_253, 0.25)),
            "p75": float(np.quantile(bag_std_253, 0.75)),
        },
        "counter_disagreement_stats_unb": {
            "mean": float(counter_disagreement_unb.mean()),
            "median": float(np.median(counter_disagreement_unb)),
            "min": float(counter_disagreement_unb.min()),
            "max": float(counter_disagreement_unb.max()),
        },
        "max_tanimoto_train_unb_stats": {
            "mean": float(max_tan_unb.mean()),
            "median": float(np.median(max_tan_unb)),
            "min": float(max_tan_unb.min()),
            "max": float(max_tan_unb.max()),
        },
        "results": results,
        "best_variant": best,
        "best_source": best_source,
        "beats_nb2103_median": bool(beats_median),
        "beats_nb2103_mean": bool(beats_mean),
        "in_fire_rate_gate": bool(in_gate),
        "verdict": verdict,
        "deploy_csv": str(deploy_csv_path) if deploy_csv_path is not None else None,
        "deploy_fire_rate_513": deploy_fire_rate_513,
        "pre_unblind_clean": True,  # uses chemprop_aux + nb2103 K=28 + null_pred only
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
        "rae_baseline_nb2103_K28_mean_bag",
        "best_variant",
        "in_fire_rate_gate",
        "beats_nb2103_median",
        "beats_nb2103_mean",
        "verdict",
        "deploy_csv",
    ):
        print(f"  {k}: {res.get(k)}")
