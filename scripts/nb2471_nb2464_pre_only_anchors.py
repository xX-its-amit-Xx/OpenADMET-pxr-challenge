"""nb2471 -- Stress test nb2464 with PRE-UNBLIND-ONLY anchors.

CONTEXT:
    nb2464 scored 0.4428 deep-30 with anchor pool {nb2240, nb730, nb562,
    chemprop_aux}. Per `feedback_anchor_contamination_chain.md`:
      * nb730: te bit-identical to pred_oof (coarse-lambda coincidence),
      * nb562: te[unb]=0.4172 vs pred_oof=0.5065 (+0.089 in-sample leak),
    so the 0.4428 may be leak-amplified. Re-run with anchor pool reduced to
    {nb2240, chemprop_aux} -- the only verified-clean PRE-unblind anchors.

PROTOCOL (mirror of nb2464, anchor pool shrunk):
    - Anchor pool: nb2240_mean_bag_oof_K20.npy + nb1133_chemprop_aux_pred_oof.npy
      (chemprop_aux trained on 4139 PRE-unblind only; nb2240 is a residual on
       chemprop_aux -> tower-clean).
    - Lens: MACCS+Morgan+physchem -> StdScale -> PCA-32 (identical to nb2464).
    - Mapper: resolution=10, gain=0.3, AgglomerativeClustering(n_clusters=2)
      per cube, min_cluster_n=10 (identical).
    - 30-seed scaffold CV (kf_seeds=1001..1030).
    - Per-cluster SLSQP simplex over 2 anchors.
    - Deploy: refit per-cluster SLSQP on all 253 -> te_nb2471.npy on 513.

GATE:
    - 30-seed mean < 0.4570 -> "PRE_CLEAN_HOLDS" (genuine, contamination-free)
    - 30-seed mean > 0.4570 -> "CONTAMINATION_DRIVEN" (0.4428 was leak-amplified)
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
from rdkit import RDLogger
from scipy.optimize import minimize
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

from pxr.chem import morgan_fp_batch, bemis_murcko, compute_physchem
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

try:
    import kmapper as km
    HAVE_KMAPPER = True
except Exception as e:
    HAVE_KMAPPER = False
    KMAPPER_IMPORT_ERR = str(e)

TAG = "nb2471"

# ---- mapper hyperparams (identical to nb2464) ----
PCA_DIM = 32
RESOLUTION = 10
GAIN = 0.3
MIN_CLUSTER_N = 10

# ---- CV setup: 30-seed bag ----
N_FOLDS = 5
KF_SEEDS = list(range(1001, 1031))  # 30 seeds

GATE_PRE_HOLDS = 0.4570

# ---- PRE-clean anchors only ----
NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"
CHEMPROP_AUX_OOF_PATH = DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy"
NB2240_TE_PATH = DATA_PROCESSED / "te_nb2240_K20.npy"
CHEMPROP_AUX_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"


def _slsqp_convex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Simplex-constrained least-squares: w >= 0, sum w = 1; min ||Pw - y||^2."""
    k = P.shape[1]
    w0 = np.full(k, 1.0 / k)

    def obj(w):
        r = P @ w - y
        return float(r @ r)

    def grad(w):
        r = P @ w - y
        return 2.0 * (P.T @ r)

    cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
             "jac": lambda w: np.ones_like(w)}]
    bnds = [(0.0, 1.0)] * k
    res = minimize(obj, w0, jac=grad, method="SLSQP", bounds=bnds,
                   constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9, "disp": False})
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        w = np.full(k, 1.0 / k)
    return w


def _build_lens(te_smiles):
    """MACCS-167 (if cached) + Morgan-2048 + 9 physchem -> StdScale -> PCA-32."""
    n_test = len(te_smiles)
    X_morgan = morgan_fp_batch(te_smiles, radius=2, n_bits=2048).astype(np.float32)
    maccs_path = DATA_PROCESSED / "te_maccs.npy"
    if maccs_path.exists():
        X_maccs = np.load(maccs_path).astype(np.float32)
        if X_maccs.shape[0] != n_test:
            X_maccs = None
    else:
        X_maccs = None
    phys_rows = []
    for smi in te_smiles:
        d = compute_physchem(smi)
        if d is None:
            phys_rows.append([np.nan] * 9)
        else:
            phys_rows.append([
                d.get("mw", np.nan), d.get("logp", np.nan), d.get("tpsa", np.nan),
                d.get("hbd", np.nan), d.get("hba", np.nan),
                d.get("rotbonds", np.nan), d.get("fsp3", np.nan),
                d.get("rings", np.nan), d.get("charge", np.nan),
            ])
    X_phys = np.asarray(phys_rows, dtype=np.float32)
    col_med = np.nanmedian(X_phys, axis=0)
    for j in range(X_phys.shape[1]):
        m = np.isnan(X_phys[:, j])
        if m.any():
            X_phys[m, j] = col_med[j]
    parts = [X_morgan, X_phys]
    if X_maccs is not None:
        parts.insert(0, X_maccs)
    X = np.concatenate(parts, axis=1).astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0)
    col_std = X.std(axis=0)
    keep_cols = col_std > 1e-12
    if keep_cols.sum() < X.shape[1]:
        X = X[:, keep_cols]
    scaler = StandardScaler(with_mean=False)
    X_std = scaler.fit_transform(X).astype(np.float64)
    X_std = np.where(np.isfinite(X_std), X_std, 0.0)
    pca = PCA(n_components=min(PCA_DIM, X_std.shape[1], X_std.shape[0] - 1),
              random_state=0)
    X_pca = pca.fit_transform(X_std).astype(np.float64)
    return X_pca, int(X.shape[1])


def _mapper_clusters(X_lens: np.ndarray, n_total: int):
    """KeplerMapper l2norm -> AgglomerativeClustering n_clusters=2 per cube."""
    mapper = km.KeplerMapper(verbose=0)
    proj = mapper.fit_transform(X_lens, projection="l2norm")
    graph = mapper.map(
        proj,
        X_lens,
        cover=km.Cover(n_cubes=RESOLUTION, perc_overlap=GAIN),
        clusterer=AgglomerativeClustering(n_clusters=2),
    )
    clusters = []
    for node_id, member_ids in graph["nodes"].items():
        ids = np.asarray(member_ids, dtype=int)
        ids = ids[(ids >= 0) & (ids < n_total)]
        if len(ids) >= MIN_CLUSTER_N:
            clusters.append(ids)
    return clusters


def _per_cluster_blend_predict(
    P_train, y_train, clusters_train,
    P_query, M_query_full, fallback,
):
    """Fit per-cluster SLSQP on TRAIN; predict on QUERY by soft-avg over membership."""
    C = len(clusters_train)
    K = P_train.shape[1]
    if C == 0:
        return fallback.copy(), np.zeros((0, K))
    W_per_cluster = np.zeros((C, K), dtype=np.float64)
    valid_mask = np.zeros(C, dtype=bool)
    for c_idx, tr_loc in enumerate(clusters_train):
        if len(tr_loc) < MIN_CLUSTER_N:
            continue
        w = _slsqp_convex(P_train[tr_loc], y_train[tr_loc])
        W_per_cluster[c_idx] = w
        valid_mask[c_idx] = True
    n_q = P_query.shape[0]
    pred_clusters = np.zeros((n_q, C), dtype=np.float64)
    for c_idx in range(C):
        if not valid_mask[c_idx]:
            continue
        pred_clusters[:, c_idx] = P_query @ W_per_cluster[c_idx]
    M_valid = M_query_full[:, valid_mask]
    pred_valid = pred_clusters[:, valid_mask]
    mem_sum = M_valid.sum(axis=1)
    weighted = (M_valid * pred_valid).sum(axis=1)
    out = np.where(mem_sum > 0, weighted / np.where(mem_sum > 0, mem_sum, 1.0),
                   fallback)
    return out, W_per_cluster


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PRE-clean anchor stress test of nb2464 cluster blend")
    print("=" * 78)
    if not HAVE_KMAPPER:
        out = {"tag": TAG, "status": "INSTALL_FAILED",
               "error": KMAPPER_IMPORT_ERR}
        with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
            json.dump(out, f, indent=2)
        print("INSTALL_FAILED")
        return out

    # ---- Load data ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test} n_unb={n_unb}")

    # ---- Anchors: PRE-clean ONLY ----
    anchor_oof = {
        "nb2240": np.load(NB2240_OOF_PATH).astype(np.float64),
        "chemprop_aux": np.load(CHEMPROP_AUX_OOF_PATH).astype(np.float64),
    }
    anchor_te = {
        "nb2240": np.load(NB2240_TE_PATH).astype(np.float64),
        "chemprop_aux": np.load(CHEMPROP_AUX_TE_PATH).astype(np.float64),
    }
    anchor_provenance = {
        "nb2240": "PRE-clean (residual stack on chemprop_aux, K=20 mean bag)",
        "chemprop_aux": "PRE-clean (trained on 4139 PRE-unblind only)",
    }
    anchors_excluded = {
        "nb730": "DEPRECATED: te bit-identical to pred_oof (coarse-lambda coincidence)",
        "nb562": "DEPRECATED: te[unb]=0.4172 vs pred_oof=0.5065 (+0.089 in-sample leak)",
        "nb1191": "EXCLUDED: only te_nb1191[unb_idx] (deploy-refit) available; not an honest OOF",
    }
    anchor_names = list(anchor_oof.keys())
    K = len(anchor_names)
    P_unb = np.column_stack([anchor_oof[k] for k in anchor_names])
    P_te = np.column_stack([anchor_te[k] for k in anchor_names])
    rae_anchors = {k: float(rae(y_unb, anchor_oof[k])) for k in anchor_names}
    for k in anchor_names:
        print(f"   anchor {k:14s}  unb-RAE = {rae_anchors[k]:.4f}  [{anchor_provenance[k]}]")
    for k, why in anchors_excluded.items():
        print(f"   excluded {k:12s}  {why}")

    # ---- Lens (identical to nb2464) ----
    print("\n" + "-" * 78)
    print(f"LENS: MACCS+Morgan+physchem -> StdScale -> PCA-{PCA_DIM}  (l2norm projection)")
    print("-" * 78)
    X_pca_te, raw_dim = _build_lens(te_smiles)
    X_pca_unb = X_pca_te[unb_idx]
    print(f"   raw_dim={raw_dim}  PCA dim={X_pca_te.shape[1]}")

    # ---- Mapper diagnostics on full unb / te ----
    print("\n" + "-" * 78)
    print(f"MAPPER  resolution={RESOLUTION}  gain={GAIN}  min_cluster_n={MIN_CLUSTER_N}")
    print("-" * 78)
    clusters_unb_full = _mapper_clusters(X_pca_unb, n_unb)
    clusters_te_full = _mapper_clusters(X_pca_te, n_test)
    print(f"   clusters_unb (full) = {len(clusters_unb_full)}")
    print(f"   clusters_te  (full) = {len(clusters_te_full)}")

    # ---- Scaffolds ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"\n[scaffold] unique={n_unique_scaf}")

    # ---- Outer 5-fold scaffold CV, 30-seed bag ----
    print("\n" + "-" * 78)
    print(f"SCAFFOLD CV  folds={N_FOLDS}  kf_seeds={KF_SEEDS[0]}..{KF_SEEDS[-1]} (n={len(KF_SEEDS)})")
    print("-" * 78)
    bag_oof = np.zeros(n_unb, dtype=np.float64)
    per_seed_summary = []
    for s_idx, kf_seed in enumerate(KF_SEEDS):
        splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                        shuffle=True, seed=kf_seed)
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        for f_idx, (tr_loc, va_loc) in enumerate(splits):
            X_tr = X_pca_unb[tr_loc]
            X_va = X_pca_unb[va_loc]
            X_comb = np.vstack([X_tr, X_va])
            clusters_comb = _mapper_clusters(X_comb, len(X_comb))
            n_tr = len(tr_loc)
            tr_clusters_local = []
            va_membership_per_cluster = []
            for ids in clusters_comb:
                tr_ids = ids[ids < n_tr]
                va_ids = ids[ids >= n_tr] - n_tr
                if len(tr_ids) >= MIN_CLUSTER_N:
                    tr_clusters_local.append(tr_ids)
                    va_membership_per_cluster.append(va_ids)
            n_va = len(va_loc)
            C_valid = len(tr_clusters_local)
            M_va = np.zeros((n_va, C_valid), dtype=np.float64)
            for c_idx, va_ids in enumerate(va_membership_per_cluster):
                M_va[va_ids, c_idx] = 1.0
            P_tr = P_unb[tr_loc]
            y_tr = y_unb[tr_loc]
            P_va = P_unb[va_loc]
            va_fallback = anchor_oof["nb2240"][va_loc]
            pred_va, _ = _per_cluster_blend_predict(
                P_tr, y_tr, tr_clusters_local,
                P_va, M_va, va_fallback,
            )
            oof[va_loc] = pred_va
        if np.isnan(oof).any():
            n_na = int(np.isnan(oof).sum())
            oof[np.isnan(oof)] = anchor_oof["nb2240"][np.isnan(oof)]
        rae_seed = float(rae(y_unb, oof))
        per_seed_summary.append({"kf_seed": int(kf_seed), "pooled_rae": rae_seed})
        bag_oof += oof
        print(f"   [{s_idx+1:2d}/{len(KF_SEEDS)}] kf_seed={kf_seed}  pooled_RAE={rae_seed:.4f}")
    bag_oof /= len(KF_SEEDS)
    rae_standalone = float(rae(y_unb, bag_oof))
    pooled_mean = float(np.mean([r["pooled_rae"] for r in per_seed_summary]))
    pooled_std = float(np.std([r["pooled_rae"] for r in per_seed_summary]))
    print(f"\n[standalone 30-seed]  mean = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"[standalone bag]     RAE = {rae_standalone:.4f}")

    verdict = "PRE_CLEAN_HOLDS" if pooled_mean < GATE_PRE_HOLDS else "CONTAMINATION_DRIVEN"
    print(f"[gate]  target < {GATE_PRE_HOLDS}  ->  {verdict}")

    # ---- Deploy ----
    print("\n" + "-" * 78)
    print("DEPLOY (refit per-cluster SLSQP on all 253; predict 513)")
    print("-" * 78)
    X_comb_full = np.vstack([X_pca_unb, X_pca_te])
    clusters_comb_full = _mapper_clusters(X_comb_full, len(X_comb_full))
    n_u = n_unb
    deploy_tr_clusters = []
    deploy_te_membership = []
    for ids in clusters_comb_full:
        tr_ids = ids[ids < n_u]
        te_ids = ids[ids >= n_u] - n_u
        if len(tr_ids) >= MIN_CLUSTER_N:
            deploy_tr_clusters.append(tr_ids)
            deploy_te_membership.append(te_ids)
    C_dep = len(deploy_tr_clusters)
    M_te = np.zeros((n_test, C_dep), dtype=np.float64)
    for c_idx, te_ids in enumerate(deploy_te_membership):
        M_te[te_ids, c_idx] = 1.0
    te_fallback = anchor_te["nb2240"]
    te_pred, W_dep = _per_cluster_blend_predict(
        P_unb, y_unb, deploy_tr_clusters,
        P_te, M_te, te_fallback,
    )
    te_pred = np.clip(te_pred, 3.0, 9.0).astype(np.float32)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    coverage_te = float((M_te.sum(axis=1) > 0).mean())
    print(f"   deploy clusters = {C_dep}")
    print(f"   te_deploy mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}")
    print(f"   te coverage (any cluster) = {coverage_te:.3f}")

    # ---- Save ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, bag_oof.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_pre_only_mapper_blend.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": te_pred,
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # Compare to nb2464 ceiling claim of 0.4428
    nb2464_claim = 0.4428
    delta_vs_nb2464 = pooled_mean - nb2464_claim
    summary = {
        "tag": TAG,
        "method": "tda_mapper_cluster_conditional_PRE_ONLY_2anchor_slsqp_blend",
        "intent": "Stress test nb2464 0.4428 with leak-free PRE-only anchors",
        "anchor_pool": anchor_names,
        "anchor_provenance": anchor_provenance,
        "anchors_excluded": anchors_excluded,
        "lens": {
            "raw_dim": int(raw_dim),
            "pca_dim": int(X_pca_te.shape[1]),
            "projection": "l2norm",
            "resolution": RESOLUTION,
            "gain": GAIN,
            "min_cluster_n": MIN_CLUSTER_N,
            "n_clusters_unb_full": int(len(clusters_unb_full)),
            "n_clusters_te_full": int(len(clusters_te_full)),
            "n_clusters_deploy": int(C_dep),
        },
        "anchor_in_rae": rae_anchors,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "n_folds": int(N_FOLDS),
        "n_kf_seeds": len(KF_SEEDS),
        "kf_seed_first": int(KF_SEEDS[0]),
        "kf_seed_last": int(KF_SEEDS[-1]),
        "per_seed_pooled": per_seed_summary,
        "pooled_rae_mean_seeds": pooled_mean,
        "pooled_rae_std_seeds": pooled_std,
        "rae_standalone_bag": rae_standalone,
        "gate_pre_holds_threshold": GATE_PRE_HOLDS,
        "verdict": verdict,
        "comparison_nb2464_claim": {
            "nb2464_pooled_mean_claim": nb2464_claim,
            "nb2471_pooled_mean": pooled_mean,
            "delta_pre_minus_nb2464": delta_vs_nb2464,
            "interpretation": (
                "POST anchors (nb730/nb562) drove the 0.4428"
                if delta_vs_nb2464 > 0.01
                else "PRE anchors alone reproduce the gain"
            ),
        },
        "te_unb_in_sample_rae": te_unb_in,
        "te_coverage_any_cluster": coverage_te,
        "deploy_te_mean": float(te_pred.mean()),
        "deploy_te_std": float(te_pred.std()),
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled_RAE  mean (30 seeds) = {pooled_mean:.4f} +/- {pooled_std:.4f}")
    print(f"   standalone bag_RAE          = {rae_standalone:.4f}")
    print(f"   gate target {GATE_PRE_HOLDS:.4f}        -> {verdict}")
    print(f"   delta vs nb2464 (0.4428)    = {delta_vs_nb2464:+.4f}")
    print(f"   te[unb_idx] in-sample RAE   = {te_unb_in:.4f}")
    print(f"   wall                        = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "pooled_rae_mean_seeds",
        "pooled_rae_std_seeds",
        "rae_standalone_bag",
        "verdict",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
