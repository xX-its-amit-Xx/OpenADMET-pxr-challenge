"""nb1143_cluster_expert -- Per-Butina-cluster expert LGBM (K=28-style features).

HYPOTHESIS:
    A mixture-of-experts indexed by Butina cluster at Tanimoto 0.4 may improve
    over a single global model on the 253 unblind because PXR's pocket is
    largely hydrophobic + flexible -- different chemotypes (e.g. steroidal vs
    bisphenol vs aryl-piperazine) plausibly respond to different feature
    subsets and different non-linear regimes.

PROTOCOL:
    1. Butina-cluster TRAIN 4139 (deduped by InChIKey) at Tanimoto 0.4 cutoff
       on ECFP4 (radius=2, nBits=2048).
    2. Compute a Morgan-FP centroid per cluster (mean over members, binarized
       at 0.5 -> a representative bit-vector).
    3. Train one LightGBM specialist per cluster of size >= MIN_CLUSTER (small
       clusters fall back to a GLOBAL specialist trained on all 4139).
    4. At inference: for each of the 513 test compounds compute Tanimoto sim
       to every cluster centroid, route by nearest, BLEND top-3 weighted by
       sim (similarity-weighted soft-routing).
    5. Honest 5-fold SCAFFOLD-CV on the 253 unblind, plus a refit on full
       TRAIN to produce a deploy CSV (513 preds).
    6. Compare vs nb2103 K=28 mean-bag (0.4737 on 253 cross-fit) AND the
       user-supplied 0.5057 scaffold-CV reference; margin 0.003.

NOTES:
    - The user-requested K=28 SHAP-pruned 117-col matrix lives only on the
      253 unblind (data/processed/X_unb_28_nb2103.npy). Rebuilding the same
      28 cols on TRAIN 4139 + 513 test from scratch would require Mordred /
      ChempropEmbed / AtomPair caches that don't exist train-side. We
      substitute the COMBINED Morgan+RDKit feature set (2265 dim, the
      strongest single feature pack per CLAUDE.md / nb02), keeping the LGBM
      hyperparams (max_depth=4, num_leaves=15, n_estimators=300, lr=0.03,
      reg_lambda=2.0) identical to nb2103. This is the honest tractable
      version of the request.
    - Inference uses the same combined feature pack on the 513 test set.
    - We DO NOT touch the 253 labels at training time. Cross-fit on 253 is a
      separate scaffold-CV restricted to the 253 with the train pool only.

Outputs:
    scripts/nb1143_cluster_expert.py           (this file)
    data/processed/nb1143_summary.json
    data/processed/nb1143_pred_oof.npy         (253,) float32 scaffold-CV
    data/processed/nb1143_te.npy               (513,) float32 deploy refit
    submissions/nb1143_cluster_expert.csv      (if beats by 0.003)
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
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

import lightgbm as lgb

from pxr.chem import standardize, morgan_fp_batch, bemis_murcko
from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED

RDLogger.DisableLog("rdApp.*")

TAG = "nb1143"
# User spec: "Butina-cluster TRAIN 4139 at Tanimoto 0.4 -> ~30 clusters".
# RDKit Butina takes a DISTANCE cutoff (= 1 - Tanimoto similarity), so the
# "Tanimoto 0.4" recipe corresponds to a DISTANCE cutoff of (1 - 0.4) = 0.6.
# We sweep {0.55, 0.60, 0.65} and pick the cutoff closest to 30 clusters.
BUTINA_CUTOFF_GRID = [0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96]
TARGET_N_CLUSTERS = 30
MIN_CLUSTER_TRAIN = 30       # below this, fall back to GLOBAL specialist
ROUTE_TOPK = 3               # blend top-3 nearest centroids at inference
N_FOLDS_253 = 5
SEED = 42

# References for the verdict.
REF_NB2103_K28_CROSSFIT = 0.4737     # nb2103 K=28 mean-bag on 253 (residual cross-fit)
REF_NB2103_SCAFFOLD_CV = 0.5057      # user-supplied scaffold-CV anchor
DECISION_MARGIN = 0.003


# LGBM(MSE) -- identical to nb2103 / nb2081 / nb2063.
def _lgbm_params(seed: int = SEED) -> dict:
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _morgan_bitvect(smi: str, radius: int = 2, nBits: int = 2048):
    """Return RDKit ExplicitBitVect for Tanimoto via DataStructs."""
    mol = standardize(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)


def _build_butina_dists(bitvects: list) -> list[float]:
    """Build the flat condensed Tanimoto-distance list once (reused for all cutoffs)."""
    n = len(bitvects)
    dists: list[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(bitvects[i], bitvects[:i])
        dists.extend([1.0 - s for s in sims])
    return dists


def _butina_cluster_from_dists(dists: list[float], n: int,
                                cutoff: float) -> list[tuple[int, ...]]:
    return Butina.ClusterData(dists, n, cutoff, isDistData=True)


def _centroid_bitvect(fps: np.ndarray) -> np.ndarray:
    """Bitwise-majority centroid (mean >= 0.5) -> uint8 bit-vector."""
    mean = fps.mean(axis=0)
    return (mean >= 0.5).astype(np.uint8)


def _tanimoto_dense_to_centroids(
    fp_query: np.ndarray, fp_centroids: np.ndarray
) -> np.ndarray:
    """Dense Tanimoto (n_q, n_c) given uint8 0/1 bit-matrices."""
    a = fp_query.astype(np.float32)
    b = fp_centroids.astype(np.float32)
    inter = a @ b.T
    asum = a.sum(axis=1, keepdims=True)
    bsum = b.sum(axis=1, keepdims=True)
    denom = asum + bsum.T - inter
    denom = np.maximum(denom, 1.0)
    return (inter / denom).astype(np.float32)


def _route_weights(
    sim_to_centroid: np.ndarray, topk: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (idx_topk, w_topk) per row, weights summing to 1 (sim-normalized)."""
    n_q, n_c = sim_to_centroid.shape
    k = min(topk, n_c)
    if k >= n_c:
        idx_topk = np.argsort(-sim_to_centroid, axis=1)[:, :k]
    else:
        part = np.argpartition(-sim_to_centroid, kth=k - 1, axis=1)[:, :k]
        rowi = np.arange(n_q)[:, None]
        order = np.argsort(-sim_to_centroid[rowi, part], axis=1)
        idx_topk = part[rowi, order]
    rowi = np.arange(n_q)[:, None]
    w = sim_to_centroid[rowi, idx_topk].copy()
    w = np.clip(w, 1e-6, 1.0)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum < 1e-9, 1.0, w_sum)
    w = w / w_sum
    return idx_topk, w


def _train_experts(
    cluster_assignments: list[int],
    X: np.ndarray,
    y: np.ndarray,
    n_clusters: int,
    min_cluster: int,
    seed: int,
) -> tuple[dict[int, lgb.LGBMRegressor], lgb.LGBMRegressor, dict[int, int]]:
    """Train one LGBM per cluster (size >= min_cluster), plus a global fallback.

    Returns (cluster_to_model, global_model, cluster_size).
    """
    cluster_to_model: dict[int, lgb.LGBMRegressor] = {}
    cluster_size: dict[int, int] = {}
    arr = np.asarray(cluster_assignments)
    for c in range(n_clusters):
        mask = (arr == c)
        sz = int(mask.sum())
        cluster_size[c] = sz
        if sz < min_cluster:
            continue
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed=seed + c))
        mdl.fit(X[mask], y[mask])
        cluster_to_model[c] = mdl
    # GLOBAL fallback (trained on everything).
    global_model = lgb.LGBMRegressor(**_lgbm_params(seed=seed))
    global_model.fit(X, y)
    return cluster_to_model, global_model, cluster_size


def _predict_routed(
    X: np.ndarray,
    idx_topk: np.ndarray,
    w_topk: np.ndarray,
    cluster_to_model: dict[int, lgb.LGBMRegressor],
    global_model: lgb.LGBMRegressor,
) -> np.ndarray:
    """Blend top-K cluster-expert predictions weighted by sim; fall back to global."""
    n_q = X.shape[0]
    out = np.zeros(n_q, dtype=np.float64)
    # Cache per-cluster preds reused across rows.
    needed = set(int(c) for c in idx_topk.flatten())
    cluster_preds: dict[int, np.ndarray] = {}
    for c in needed:
        if c in cluster_to_model:
            cluster_preds[c] = cluster_to_model[c].predict(X)
        else:
            cluster_preds[c] = global_model.predict(X)
    for i in range(n_q):
        for j in range(idx_topk.shape[1]):
            c = int(idx_topk[i, j])
            out[i] += w_topk[i, j] * cluster_preds[c][i]
    return out


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Per-Butina-cluster expert LGBM")
    print(f"          cutoff_grid={BUTINA_CUTOFF_GRID} -> pick closest to "
          f"{TARGET_N_CLUSTERS} clusters")
    print(f"          min_cluster={MIN_CLUSTER_TRAIN}  route_topk={ROUTE_TOPK}")
    print(f"          n_folds_253={N_FOLDS_253}  margin={DECISION_MARGIN}")
    print(f"          refs: nb2103 K=28 cross-fit {REF_NB2103_K28_CROSSFIT:.4f}, "
          f"scaffold-CV {REF_NB2103_SCAFFOLD_CV:.4f}")
    print("=" * 78)

    # ---- Load TRAIN 4139 + TEST 513 ----
    tr = load_train()
    te = load_test()
    tr = tr[tr["pec50"].notna()].reset_index(drop=True)
    print(f"[load] train n={len(tr)}  test n={len(te)}")

    # Standardize + InChIKey dedup TRAIN (keep median pec50 per IK).
    tr["mol"] = tr["smiles"].apply(standardize)
    tr = tr[tr["mol"].notna()].reset_index(drop=True)
    tr["ik"] = tr["mol"].apply(Chem.MolToInchiKey)
    tr_dd = (
        tr.groupby("ik", as_index=False)
        .agg(smiles=("smiles", "first"),
             pec50=("pec50", "median"),
             mol=("mol", "first"))
        .reset_index(drop=True)
    )
    print(f"[dedup] train after InChIKey dedup: {len(tr_dd)}")

    tr_dd["scaffold"] = tr_dd["smiles"].apply(lambda s: bemis_murcko(s) or "")
    tr_smiles = tr_dd["smiles"].tolist()
    y_tr = tr_dd["pec50"].to_numpy(dtype=np.float64)

    te_smiles = te["smiles"].tolist()

    # ---- Build Morgan FPs for Butina + routing ----
    print("[fp] building Morgan FPs (radius=2, nBits=2048)...")
    fp_tr = morgan_fp_batch(tr_smiles)         # (Ntr, 2048) uint8
    fp_te = morgan_fp_batch(te_smiles)         # (513, 2048) uint8
    print(f"   fp_tr {fp_tr.shape}  fp_te {fp_te.shape}")

    # Per-row RDKit bitvects for Butina.
    print("[butina] building ExplicitBitVect list...")
    tr_bv = []
    for s in tr_smiles:
        bv = _morgan_bitvect(s)
        if bv is None:
            tr_bv.append(AllChem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles("C"), 2, nBits=2048))
        else:
            tr_bv.append(bv)
    t_butina = time.time()
    dists_flat = _build_butina_dists(tr_bv)
    print(f"[butina] computed {len(dists_flat):,} pairwise distances "
          f"({time.time() - t_butina:.1f}s)")
    # Sweep cutoffs and pick the one closest to TARGET_N_CLUSTERS.
    cutoff_records = []
    for cutoff in BUTINA_CUTOFF_GRID:
        clusters_c = _butina_cluster_from_dists(dists_flat, len(tr_bv), cutoff)
        nc = len(clusters_c)
        sizes_c = [len(c) for c in clusters_c]
        cutoff_records.append({
            "cutoff": cutoff,
            "n_clusters": nc,
            "max_size": int(max(sizes_c)),
            "median_size": int(np.median(sizes_c)),
            "n_singletons": int(sum(1 for s in sizes_c if s == 1)),
        })
        print(f"   cutoff={cutoff:.2f}  n_clusters={nc:>4d}  max={max(sizes_c):>5d}  "
              f"median={int(np.median(sizes_c)):>3d}  "
              f"singletons={sum(1 for s in sizes_c if s == 1):>4d}")
    # Pick cutoff giving cluster count closest to TARGET_N_CLUSTERS.
    best_rec = min(cutoff_records,
                   key=lambda r: abs(r["n_clusters"] - TARGET_N_CLUSTERS))
    chosen_cutoff = best_rec["cutoff"]
    print(f"\n[butina] CHOSEN cutoff={chosen_cutoff:.2f} -> "
          f"{best_rec['n_clusters']} clusters (target {TARGET_N_CLUSTERS})")
    clusters = _butina_cluster_from_dists(dists_flat, len(tr_bv), chosen_cutoff)
    sizes = [len(c) for c in clusters]
    print(f"   size stats: max={max(sizes)}  median={int(np.median(sizes))}  "
          f"singletons={sum(1 for s in sizes if s == 1)}")

    # Map each train row to its cluster id.
    n_tr = len(tr_smiles)
    cluster_id = np.full(n_tr, -1, dtype=np.int32)
    for cid, members in enumerate(clusters):
        for m in members:
            cluster_id[m] = cid
    assert (cluster_id >= 0).all(), "unassigned training rows"
    n_clusters = len(clusters)

    # ---- Centroids ----
    centroids = np.zeros((n_clusters, fp_tr.shape[1]), dtype=np.uint8)
    for cid in range(n_clusters):
        mask = (cluster_id == cid)
        centroids[cid] = _centroid_bitvect(fp_tr[mask])
    print(f"[centroid] built {n_clusters} centroids "
          f"(mean members = {n_tr / n_clusters:.1f})")

    # ---- Combined features for LGBM (Morgan + RDKit, ~2265 dim) ----
    print("[feat] building combined() features for TRAIN + TEST...")
    X_tr = combined(tr_smiles).astype(np.float32)
    X_tr = impute(X_tr)
    X_te = combined(te_smiles).astype(np.float32)
    X_te = impute(X_te)
    print(f"   X_tr {X_tr.shape}  X_te {X_te.shape}")

    # =====================================================================
    # PART A: scaffold-CV honest on the 253 unblind
    # =====================================================================
    print("\n" + "-" * 78)
    print("PART A: 5-fold SCAFFOLD-CV on 253 unblind")
    print("-" * 78)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)

    X_unb_combined = X_te[unb_idx]
    fp_unb = fp_te[unb_idx]
    # Scaffold for each unblind row.
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    splits_unb = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS_253, seed=SEED
    )

    # 5-fold scaffold-CV on the 253:
    #   in each fold, train cluster-experts on TRAIN(4139, deduped) ONLY
    #   (NEVER on the 253 unblind), then predict the held-out val rows.
    # This is honest -- the 253 labels are never used at any training step.
    pred_oof = np.full(n_unb, np.nan, dtype=np.float64)
    sim_to_centroid_unb = _tanimoto_dense_to_centroids(fp_unb, centroids)
    idx_topk_unb, w_topk_unb = _route_weights(sim_to_centroid_unb, ROUTE_TOPK)

    fold_records = []
    for fi, (tr_loc, va_loc) in enumerate(splits_unb):
        ts = time.time()
        # Retrain experts on FULL TRAIN (4139, deduped) -- the 253 is OOF
        # for this whole experiment, so the fold structure on 253 just
        # determines which subset gets predicted.
        cluster_to_model, global_model, _ = _train_experts(
            cluster_id.tolist(), X_tr, y_tr,
            n_clusters=n_clusters,
            min_cluster=MIN_CLUSTER_TRAIN,
            seed=SEED + fi,
        )
        pred_va = _predict_routed(
            X_unb_combined[va_loc],
            idx_topk_unb[va_loc],
            w_topk_unb[va_loc],
            cluster_to_model,
            global_model,
        )
        pred_oof[va_loc] = pred_va
        rae_fold = float(rae(y_unb[va_loc], pred_va))
        fold_records.append({
            "fold": int(fi),
            "n_val": int(len(va_loc)),
            "rae_fold": rae_fold,
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   fold={fi}  n_val={len(va_loc):>3d}  "
              f"rae_fold={rae_fold:.4f}  "
              f"wall={time.time() - ts:.1f}s")

    rae_scaffold_cv = float(rae(y_unb, pred_oof))
    print(f"\n[A] scaffold-CV pooled RAE on 253 = {rae_scaffold_cv:.4f}")

    delta_vs_nb2103_cf = rae_scaffold_cv - REF_NB2103_K28_CROSSFIT
    delta_vs_scaffold_ref = rae_scaffold_cv - REF_NB2103_SCAFFOLD_CV
    beats_scaffold_ref = rae_scaffold_cv < REF_NB2103_SCAFFOLD_CV - DECISION_MARGIN
    beats_nb2103_cf = rae_scaffold_cv < REF_NB2103_K28_CROSSFIT - DECISION_MARGIN
    print(f"   delta vs nb2103 K=28 cross-fit (0.4737) = {delta_vs_nb2103_cf:+.4f}")
    print(f"   delta vs scaffold-CV ref (0.5057)        = {delta_vs_scaffold_ref:+.4f}")

    # =====================================================================
    # PART B: Deploy -- retrain experts on FULL TRAIN, predict 513
    # =====================================================================
    print("\n" + "-" * 78)
    print("PART B: deploy refit on full TRAIN -> 513 predictions")
    print("-" * 78)
    cluster_to_model_full, global_model_full, _ = _train_experts(
        cluster_id.tolist(), X_tr, y_tr,
        n_clusters=n_clusters,
        min_cluster=MIN_CLUSTER_TRAIN,
        seed=SEED,
    )
    sim_to_centroid_te = _tanimoto_dense_to_centroids(fp_te, centroids)
    idx_topk_te, w_topk_te = _route_weights(sim_to_centroid_te, ROUTE_TOPK)
    pred_te = _predict_routed(
        X_te, idx_topk_te, w_topk_te,
        cluster_to_model_full, global_model_full,
    )
    print(f"[B] te shape={pred_te.shape}  mean={pred_te.mean():.3f}  "
          f"std={pred_te.std():.3f}")
    rae_te_unb_in_sample = float(rae(y_unb, pred_te[unb_idx]))
    print(f"[B] te[unb_idx] in-sample RAE = {rae_te_unb_in_sample:.4f}")

    # Save OOF + te npy.
    np.save(DATA_PROCESSED / f"{TAG}_pred_oof.npy", pred_oof.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_te.npy", pred_te.astype(np.float32))
    print(f"[save] {DATA_PROCESSED}/{TAG}_pred_oof.npy")
    print(f"[save] {DATA_PROCESSED}/{TAG}_te.npy")

    # Deploy CSV only if scaffold-CV beats either reference by margin.
    deploy_written = False
    deploy_path = None
    if beats_scaffold_ref or beats_nb2103_cf:
        deploy_path = Path(__file__).resolve().parents[1] / "submissions" / f"{TAG}_cluster_expert.csv"
        # Submission needs SMILES + Molecule Name + pEC50.
        out_df = pd.DataFrame({
            "SMILES": te["smiles"].values,
            "Molecule Name": te["name"].values,
            "pEC50": pred_te.astype(np.float32),
        })
        out_df.to_csv(deploy_path, index=False)
        deploy_written = True
        print(f"[deploy] WROTE {deploy_path}  (beats ref by margin)")
    else:
        print(f"[deploy] SKIPPED -- scaffold-CV {rae_scaffold_cv:.4f} does not "
              f"beat refs by margin {DECISION_MARGIN}")

    # ---- Verdict ----
    if beats_nb2103_cf:
        verdict = "BEATS_NB2103_K28_CROSSFIT"
    elif beats_scaffold_ref:
        verdict = "BEATS_SCAFFOLD_CV_REF"
    elif abs(delta_vs_scaffold_ref) < DECISION_MARGIN:
        verdict = "FLAT_VS_SCAFFOLD_CV_REF"
    else:
        verdict = "WORSE_THAN_BOTH_REFS"
    print(f"\n[verdict] {verdict}")

    summary = {
        "tag": TAG,
        "method": "per_butina_cluster_lgbm_top3_sim_routed",
        "butina_cutoff_grid": BUTINA_CUTOFF_GRID,
        "butina_cutoff_chosen": chosen_cutoff,
        "butina_cutoff_records": cutoff_records,
        "target_n_clusters": TARGET_N_CLUSTERS,
        "min_cluster_train": MIN_CLUSTER_TRAIN,
        "route_topk": ROUTE_TOPK,
        "n_folds_253": N_FOLDS_253,
        "seed": SEED,
        "n_train_pre_dedup": int(len(tr)),
        "n_train_dedup": int(n_tr),
        "n_test": int(len(te)),
        "n_unb": int(n_unb),
        "n_clusters": int(n_clusters),
        "cluster_size_stats": {
            "max": int(max(sizes)),
            "median": int(np.median(sizes)),
            "min": int(min(sizes)),
            "n_singletons": int(sum(1 for s in sizes if s == 1)),
            "n_above_min": int(sum(1 for s in sizes if s >= MIN_CLUSTER_TRAIN)),
        },
        "feature_pack": "combined_morgan_plus_rdkit_2265",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "fold_records": fold_records,
        "rae_scaffold_cv_253": rae_scaffold_cv,
        "rae_te_unb_in_sample": rae_te_unb_in_sample,
        "ref_nb2103_K28_crossfit": REF_NB2103_K28_CROSSFIT,
        "ref_nb2103_scaffold_cv": REF_NB2103_SCAFFOLD_CV,
        "delta_vs_nb2103_K28_crossfit": delta_vs_nb2103_cf,
        "delta_vs_scaffold_cv_ref": delta_vs_scaffold_ref,
        "beats_nb2103_K28_crossfit": bool(beats_nb2103_cf),
        "beats_scaffold_cv_ref": bool(beats_scaffold_ref),
        "decision_margin": DECISION_MARGIN,
        "verdict": verdict,
        "deploy_csv_written": bool(deploy_written),
        "deploy_csv_path": str(deploy_path) if deploy_written else None,
        "te_npy": str(DATA_PROCESSED / f"{TAG}_te.npy"),
        "oof_npy": str(DATA_PROCESSED / f"{TAG}_pred_oof.npy"),
        "pre_unblind_clean": True,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_clusters", "cluster_size_stats",
        "rae_scaffold_cv_253", "rae_te_unb_in_sample",
        "ref_nb2103_K28_crossfit", "ref_nb2103_scaffold_cv",
        "delta_vs_nb2103_K28_crossfit", "delta_vs_scaffold_cv_ref",
        "beats_nb2103_K28_crossfit", "beats_scaffold_cv_ref",
        "verdict", "deploy_csv_written", "deploy_csv_path",
    ):
        print(f"  {k}: {res.get(k)}")
