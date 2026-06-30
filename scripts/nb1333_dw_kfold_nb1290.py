"""nb1333 -- Distance-weighted KFold for nb1290 (LB-faithful cross-fit estimate).

Random KFold on the 253 unblind compounds gives a moderate-similarity holdout
per fold (training and held-out folds drawn iid, so Tanimoto support is shared).
The LB on the 513 test set sees a different mixture, with novel-scaffold tails
the 5-fold random split does NOT reproduce.

Distance-weighted KFold replaces random splits with Tanimoto-distance clusters.
Each held-out fold is now a tight chemical cluster; the training side has the
other clusters; inter-fold similarity is low. This pushes the cross-fit RAE
upward toward an LB-faithful estimate.

PROTOCOL
  1. Load 253 unblind SMILES via _audit_unblind_idx.npy -> TEST_BLINDED.csv.
  2. Compute Morgan-2048 ECFP4 and pairwise Tanimoto.
  3. AgglomerativeClustering(n_clusters=5, linkage='average',
     metric='precomputed') on (1 - Tanimoto).
  4. Use cluster labels as 5 holdout-folds.
  5. For each cluster fold: fit 2-way SLSQP weights on (nb1190, nb1242) using
     OTHER clusters; predict the held-out cluster.
  6. Pool per-cluster RAE; report overall pooled DW-KFold RAE.
  7. Compare to random-KFold nb1290 (rae_slsqp_cross_fit, rae_best_fixed_w).

PURPOSE
  NOT to beat nb1290 -- to get a more pessimistic, LB-faithful number.
  Expected DW-KFold RAE >= random-KFold RAE.

OUTPUTS
  data/processed/nb1333_dw_oof.npy        (253,) float32 DW-cross-fit blend
  data/processed/nb1333_summary.json
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
from scipy.optimize import minimize
from sklearn.cluster import AgglomerativeClustering

from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, DATA_RAW

TAG = "nb1333"

# Reference numbers (from nb1290_summary.json on 253 unblind).
NB1290_SLSQP_REF = 0.5475
NB1290_MEAN_REF = 0.5397
NB1290_BESTW_REF = 0.5390  # best random-KFold blend RAE (w[nb1190]=0.35)

# Conservative reference for LB calibration.
LB_REF_RANDOM = NB1290_BESTW_REF

N_CLUSTERS = 5


def _slsqp_blend_weights(P_tr: np.ndarray, y_tr: np.ndarray) -> np.ndarray:
    K = P_tr.shape[1]
    w0 = np.full(K, 1.0 / K)

    def _loss(w: np.ndarray) -> float:
        pred = P_tr @ w
        diff = y_tr - pred
        return float(np.mean(diff * diff))

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        _loss, w0, method="SLSQP",
        bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
    s = w.sum()
    if s <= 0:
        return np.full(K, 1.0 / K)
    return w / s


def _morgan_fp_matrix(smiles: list[str], radius: int = 2,
                      n_bits: int = 2048) -> np.ndarray:
    """Compute Morgan/ECFP4 bit-vectors as (N, n_bits) uint8 matrix."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    arr = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    n_fail = 0
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_fail += 1
            continue
        try:
            gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fp = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
            arr[i] = fp
        except Exception:
            try:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius, nBits=n_bits)
                from rdkit.DataStructs import ConvertToNumpyArray
                tmp = np.zeros(n_bits, dtype=np.int8)
                ConvertToNumpyArray(fp, tmp)
                arr[i] = tmp.astype(np.uint8)
            except Exception:
                n_fail += 1
    if n_fail > 0:
        print(f"[fp] {n_fail} SMILES failed Morgan FP")
    return arr


def _pairwise_tanimoto(fps: np.ndarray) -> np.ndarray:
    """Tanimoto similarity matrix from binary FP matrix.

    sim[i,j] = |a AND b| / |a OR b|
    """
    fps_f = fps.astype(np.float32)
    inter = fps_f @ fps_f.T  # (N, N) int counts in float
    sums = fps_f.sum(axis=1)  # (N,)
    union = sums[:, None] + sums[None, :] - inter
    sim = np.divide(inter, union,
                    out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float64)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Distance-weighted KFold cross-fit for nb1290 components")
    print(f"          n_clusters = {N_CLUSTERS}, linkage='average', metric='precomputed'")
    print(f"          random-KFold reference: best_fixed_w = {LB_REF_RANDOM:.4f}")
    print("=" * 78)

    # ---- Load anchors ----
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy").astype(int)
    n_unb = len(y_unb)
    assert n_unb == 253, n_unb

    p1 = np.load(DATA_PROCESSED / "nb1190_bob_mean_oof.npy").astype(np.float64)
    p2 = np.load(DATA_PROCESSED / "nb1242_mean_bag_oof.npy").astype(np.float64)
    assert p1.shape[0] == n_unb and p2.shape[0] == n_unb

    # ---- Load SMILES for 253 unblind ----
    test_csv = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    assert test_csv.shape[0] == 513
    smi_513 = test_csv["SMILES"].astype(str).tolist()
    smi_253 = [smi_513[i] for i in unb_idx.tolist()]
    print(f"[load] 253 unblind SMILES via TEST_BLINDED.csv[unb_idx]")

    # ---- Morgan FP + pairwise Tanimoto ----
    print(f"\n[fp]  computing Morgan-2048 ECFP4 fingerprints ...")
    fps = _morgan_fp_matrix(smi_253, radius=2, n_bits=2048)
    print(f"[fp]  fp matrix shape = {fps.shape}, sparsity = "
          f"{fps.mean():.4f} on-bits/bit")

    print(f"[sim] computing pairwise Tanimoto on (253, 253) ...")
    T = _pairwise_tanimoto(fps)
    # Off-diagonal stats
    iu = np.triu_indices(n_unb, k=1)
    tan_off = T[iu]
    print(f"[sim] off-diagonal Tanimoto: "
          f"mean={tan_off.mean():.4f}  median={np.median(tan_off):.4f}  "
          f"p90={np.percentile(tan_off, 90):.4f}  max={tan_off.max():.4f}")

    D = 1.0 - T
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)
    D = np.clip(D, 0.0, 1.0)

    # ---- AgglomerativeClustering on precomputed distance ----
    print(f"\n[cluster] AgglomerativeClustering(n_clusters={N_CLUSTERS}, "
          f"linkage='average', metric='precomputed')")
    agg = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        linkage="average",
        metric="precomputed",
    )
    labels = agg.fit_predict(D).astype(int)
    sizes = {int(c): int((labels == c).sum()) for c in range(N_CLUSTERS)}
    print(f"[cluster] cluster sizes: {sizes}")

    # Inter-cluster Tanimoto: for every pair (i, j) with i, j in different
    # clusters, take T[i, j]; report mean.
    inter_pairs_mask = labels[:, None] != labels[None, :]
    intra_pairs_mask = (labels[:, None] == labels[None, :]) & (
        np.arange(n_unb)[:, None] != np.arange(n_unb)[None, :])
    inter_tan_mean = float(T[inter_pairs_mask].mean()) if inter_pairs_mask.any() else float("nan")
    intra_tan_mean = float(T[intra_pairs_mask].mean()) if intra_pairs_mask.any() else float("nan")
    print(f"[cluster] mean Tanimoto INTER-cluster  = {inter_tan_mean:.4f}")
    print(f"[cluster] mean Tanimoto INTRA-cluster  = {intra_tan_mean:.4f}")
    print(f"[cluster] separation (intra - inter)   = "
          f"{intra_tan_mean - inter_tan_mean:+.4f}")

    # ---- DW-KFold cross-fit: holdout each cluster, train on others ----
    print("\n" + "-" * 78)
    print("  BLOCK: DW-KFold SLSQP cross-fit (per-cluster holdout)")
    print("-" * 78)
    P = np.column_stack([p1, p2])
    dw_oof = np.full(n_unb, np.nan, dtype=np.float64)
    cluster_records: list[dict] = []
    for c in range(N_CLUSTERS):
        va_mask = labels == c
        tr_mask = ~va_mask
        n_tr = int(tr_mask.sum())
        n_va = int(va_mask.sum())
        if n_tr < 2 or n_va < 1:
            print(f"   cluster {c}: SKIP (n_tr={n_tr}, n_va={n_va})")
            continue
        w = _slsqp_blend_weights(P[tr_mask], y_unb[tr_mask])
        pred_va = P[va_mask] @ w
        dw_oof[va_mask] = pred_va
        # Per-cluster RAE (with the cluster's OWN mean as denominator -- that
        # is what rae() does internally; numerically stable when n_va>=2.
        # We report both this and the pooled RAE later.
        if n_va >= 2:
            cluster_rae = float(rae(y_unb[va_mask], pred_va))
        else:
            cluster_rae = float("nan")
        # Also report intra-cluster top-1 similarity to training side.
        sub = T[np.ix_(va_mask, tr_mask)]
        top1_to_train = sub.max(axis=1)
        rec = {
            "cluster": int(c),
            "n_tr": n_tr,
            "n_va": n_va,
            "weights": [float(x) for x in w],
            "cluster_rae": cluster_rae,
            "mean_top1_sim_to_train": float(top1_to_train.mean()),
            "median_top1_sim_to_train": float(np.median(top1_to_train)),
            "y_mean": float(y_unb[va_mask].mean()),
            "y_std": float(y_unb[va_mask].std()),
        }
        cluster_records.append(rec)
        print(f"   cluster {c}: n_tr={n_tr:3d}  n_va={n_va:3d}  "
              f"w=[{w[0]:.3f}, {w[1]:.3f}]  RAE={cluster_rae:.4f}  "
              f"top1_sim_to_train mean={top1_to_train.mean():.3f}")

    # ---- Pooled DW-KFold RAE ----
    finite_mask = np.isfinite(dw_oof)
    n_predicted = int(finite_mask.sum())
    rae_dw_pooled = float(rae(y_unb[finite_mask], dw_oof[finite_mask]))
    print(f"\n[pool] n_predicted = {n_predicted} / {n_unb}")
    print(f"[pool] pooled DW-KFold SLSQP cross-fit RAE = {rae_dw_pooled:.4f}")

    # Also pool the fixed-w=0.35*p1+0.65*p2 (random-KFold best) over DW folds:
    fixed_w_oof = 0.35 * p1 + 0.65 * p2
    rae_fixed_dw = float(rae(y_unb, fixed_w_oof))  # same on full 253
    # And a per-cluster fixed-w pool using OOF (constant w, so identical to
    # full-pool) -- track for completeness:
    print(f"[pool] pooled fixed-w (w[nb1190]=0.35) RAE  = {rae_fixed_dw:.4f}")

    # ---- Compare to random-KFold ----
    delta_vs_random_best = rae_dw_pooled - NB1290_BESTW_REF
    delta_vs_random_slsqp = rae_dw_pooled - NB1290_SLSQP_REF
    print(f"\n[delta] vs random-KFold best_fixed_w ({NB1290_BESTW_REF:.4f}) = "
          f"{delta_vs_random_best:+.4f}")
    print(f"[delta] vs random-KFold SLSQP cross-fit ({NB1290_SLSQP_REF:.4f}) = "
          f"{delta_vs_random_slsqp:+.4f}")

    # ---- LB estimate adjustment ----
    # Per MEMORY.md "te vs pred_oof evaluation protocol":
    #   LB estimate = 0.51 * pred_oof + 0.49 * te[unb_idx] (in-sample optimism).
    # When we don't have te[unb_idx], we just upgrade pred_oof using DW shift.
    # Conservative ladder: replace random pred_oof with DW pred_oof.
    lb_est_random = NB1290_BESTW_REF  # treat random-KFold as the prior anchor
    lb_est_dw = rae_dw_pooled
    lb_shift = lb_est_dw - lb_est_random
    print(f"\n[lb-est] random-KFold pred_oof anchor       = {lb_est_random:.4f}")
    print(f"[lb-est] DW-KFold pred_oof anchor (this run) = {lb_est_dw:.4f}")
    print(f"[lb-est] DW shift to apply to LB band        = {lb_shift:+.4f}")

    verdict_parts = []
    if rae_dw_pooled > NB1290_BESTW_REF + 0.003:
        verdict_parts.append("DW_PESSIMISTIC")
    elif rae_dw_pooled < NB1290_BESTW_REF - 0.003:
        verdict_parts.append("DW_UNEXPECTEDLY_OPTIMISTIC")
    else:
        verdict_parts.append("DW_FLAT_VS_RANDOM")
    verdict = "  ".join(verdict_parts) + f"  (DW={rae_dw_pooled:.4f}, random={NB1290_BESTW_REF:.4f})"
    print(f"\n[verdict] {verdict}")

    # ---- Persist ----
    np.save(DATA_PROCESSED / f"{TAG}_dw_oof.npy",
            dw_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_dw_oof.npy'}")

    summary = {
        "tag": TAG,
        "n_unb": int(n_unb),
        "n_clusters": int(N_CLUSTERS),
        "components": ["nb1190", "nb1242"],
        "cluster_sizes": sizes,
        "mean_inter_cluster_tanimoto": inter_tan_mean,
        "mean_intra_cluster_tanimoto": intra_tan_mean,
        "tanimoto_offdiag_mean": float(tan_off.mean()),
        "tanimoto_offdiag_median": float(np.median(tan_off)),
        "tanimoto_offdiag_p90": float(np.percentile(tan_off, 90)),
        "cluster_records": cluster_records,
        "rae_dw_kfold_slsqp_pooled": rae_dw_pooled,
        "rae_fixed_w_full_pool": rae_fixed_dw,
        "n_predicted": n_predicted,
        "random_kfold_ref_bestw": NB1290_BESTW_REF,
        "random_kfold_ref_slsqp": NB1290_SLSQP_REF,
        "random_kfold_ref_mean": NB1290_MEAN_REF,
        "delta_vs_random_best_fixed_w": delta_vs_random_best,
        "delta_vs_random_slsqp_cross_fit": delta_vs_random_slsqp,
        "lb_est_random_anchor": lb_est_random,
        "lb_est_dw_anchor": lb_est_dw,
        "lb_est_dw_shift": lb_shift,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("cluster_sizes", "mean_inter_cluster_tanimoto",
              "mean_intra_cluster_tanimoto",
              "rae_dw_kfold_slsqp_pooled", "rae_fixed_w_full_pool",
              "delta_vs_random_best_fixed_w",
              "delta_vs_random_slsqp_cross_fit",
              "lb_est_dw_shift", "verdict"):
        print(f"  {k}: {res.get(k)}")
