"""nb2210 -- Murcko-aware per-cluster rank-stretch on nb2171.

Hypothesis: the global rank-stretch s in nb2171 (mean s ~ 1.009 across folds)
is a one-parameter scalar that compresses the entire OOF blend uniformly. But
the variance-compression failure mode is not uniform across the chemical
manifold -- novel-scaffold rows (rare-scaffold quantile compression, see
feedback memos) need MORE stretch than common-scaffold rows. A per-cluster s
can extract the localised stretch capacity without overfitting like per-row
isotonic.

Pipeline:
  1. Compute Bemis-Murcko scaffolds for the 4139 train rows + 253 unblind rows.
     Cluster Murcko scaffolds by ECFP4 Tanimoto >= 0.5 using a single-linkage
     greedy clusterer. Project each of the 253 unblind rows to its cluster id.
  2. Reconstruct the nb2171 SLSQP+per-fold-stretch mean OOF per row across the
     same 5 kf_seeds {1001..1005}. This is the "anchor OOF" the cluster-s is
     fit on.
  3. Outer scaffold-5-fold CV (kf_seed=2210). Inside each outer training fold
     fit s_c per cluster c via 4-fold INNER CV on residuals (mu_c + s_c * (p - mu_c))
     where mu_c = mean(p_train_c), s_c chosen on the standard grid. Singleton
     clusters or clusters with < MIN_C_TRAIN training rows fall back to the
     global s (refit on the full outer training fold).
  4. Apply per-cluster s to outer-validation rows; pool predictions; compute
     scaffold-CV RAE on all 253.
  5. Compare vs nb2171 baseline 0.4682. Gate = improvement >= 0.003.
  6. If passes: deep-30 verify (canonical 5 + 25 extra seeds).

Outputs:
    scripts/nb2210_murcko_stretch.py
    data/processed/nb2210_summary.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import defaultdict

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2210"
GATE_DELTA = 0.003          # require improvement >= 0.003 vs nb2171 baseline
NB2171_BASELINE = 0.4682    # reference RAE from nb2171 deep-30 mean OOF

N_OUTER_FOLDS = 5
OUTER_KF_SEED = 2210
N_INNER_FOLDS = 4
INNER_KF_SEED = 1234
DEEP_EXTRA_SEEDS = list(range(2211, 2211 + 25))

KF_SEEDS_NB2171 = [1001, 1002, 1003, 1004, 1005]
STRETCH_GRID = [1.000, 1.025, 1.050, 1.075, 1.100, 1.125, 1.150]

TANIMOTO_THRESHOLD = 0.50
MIN_C_TRAIN = 5             # clusters with fewer training rows fall back to global s

# nb2171 5-anchor stack
ANCHORS = [
    ("nb2103_K28",   "nb2103_mean_bag_oof_K28.npy"),
    ("chemprop_aux", "nb1133_chemprop_aux_pred_oof.npy"),
    ("nb1191",       "_RECONSTRUCT_nb1191_oof"),
    ("nb503",        "nb503_pred_oof.npy"),
    ("nb562",        "nb562_pred_oof.npy"),
]
NB1191_DEPLOY_WEIGHTS = {
    "chemprop_aux": 0.0,
    "nb1150":       0.641721304028517,
    "nb1158_K32":   0.23970131778546713,
    "nb2112_K28":   0.11857737818601592,
}
NB1191_DEPLOY_S = 1.031
NB1150_SLSQP4_OOFS = [
    "nb1133_chemprop_aux_pred_oof.npy",
    "nb503_pred_oof.npy",
    "nb1133_nb1014_pred_oof.npy",
    "nb2103_mean_bag_oof_K28.npy",
]
NB1150_SLSQP4_WEIGHTS = [0.0, 0.2942, 0.0, 0.7058]


# ---------------------------------------------------------------------------
# Anchor reconstruction (identical to nb2171)
# ---------------------------------------------------------------------------

def reconstruct_nb1150_oof(n_unb: int) -> np.ndarray:
    cols = []
    for rel in NB1150_SLSQP4_OOFS:
        v = np.load(DATA_PROCESSED / rel).astype(np.float64)
        assert v.shape == (n_unb,)
        cols.append(v)
    return np.column_stack(cols) @ np.asarray(NB1150_SLSQP4_WEIGHTS)


def reconstruct_nb1191_oof(n_unb: int) -> np.ndarray:
    chem = np.load(DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy").astype(np.float64)
    nb1150 = reconstruct_nb1150_oof(n_unb)
    nb1158 = np.load(DATA_PROCESSED / "nb1158_mean_bag_oof_K32.npy").astype(np.float64)
    nb2112 = np.load(DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy").astype(np.float64)
    blend = (
        NB1191_DEPLOY_WEIGHTS["chemprop_aux"] * chem
        + NB1191_DEPLOY_WEIGHTS["nb1150"]     * nb1150
        + NB1191_DEPLOY_WEIGHTS["nb1158_K32"] * nb1158
        + NB1191_DEPLOY_WEIGHTS["nb2112_K28"] * nb2112
    )
    mu = float(blend.mean())
    return mu + NB1191_DEPLOY_S * (blend - mu)


def load_anchor_matrix(n_unb: int) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for disp, rel in ANCHORS:
        if rel == "_RECONSTRUCT_nb1191_oof":
            v = reconstruct_nb1191_oof(n_unb)
        else:
            v = np.load(DATA_PROCESSED / rel).astype(np.float64)
        assert v.shape == (n_unb,)
        cols.append(v)
        names.append(disp)
    return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# nb2171 reconstruction: SLSQP convex blend + per-fold stretch, seed-averaged
# ---------------------------------------------------------------------------

def slsqp_simplex(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[1]
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0)] * K
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.full(K, 1.0 / K),
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 1000},
    )
    w = np.clip(res.x, 0.0, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def best_stretch(blend_tr: np.ndarray, y_tr: np.ndarray, mu: float) -> float:
    best_s, best_r = 1.0, float("inf")
    for s in STRETCH_GRID:
        pred = mu + s * (blend_tr - mu)
        r = float(rae(y_tr, pred))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s


def nb2171_mean_oof(
    P_unb: np.ndarray, y_unb: np.ndarray, unb_scaffolds: list[str],
    seeds: list[int] = KF_SEEDS_NB2171,
) -> np.ndarray:
    """Reconstruct the nb2171 anchor: mean across kf_seeds of the SLSQP +
    per-fold-stretch OOFs."""
    n = P_unb.shape[0]
    all_oofs = []
    for sd in seeds:
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_OUTER_FOLDS, shuffle=True, seed=sd,
        )
        oof = np.full(n, np.nan)
        for tr, va in splits:
            w = slsqp_simplex(P_unb[tr], y_unb[tr])
            b_tr = P_unb[tr] @ w
            mu_tr = float(b_tr.mean())
            s_f = best_stretch(b_tr, y_unb[tr], mu_tr)
            b_va = P_unb[va] @ w
            oof[va] = mu_tr + s_f * (b_va - mu_tr)
        assert not np.isnan(oof).any()
        all_oofs.append(oof)
    return np.mean(np.column_stack(all_oofs), axis=1)


# ---------------------------------------------------------------------------
# Murcko-scaffold Tanimoto clustering (single-linkage greedy)
# ---------------------------------------------------------------------------

def tanimoto_matrix_uint8(fps: np.ndarray) -> np.ndarray:
    """Symmetric (N, N) Tanimoto matrix from uint8 bit-vectors."""
    bits = fps.astype(np.int32)
    counts = bits.sum(axis=1)
    inter = bits @ bits.T
    union = counts[:, None] + counts[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(union > 0, inter / union, 0.0)
    return T


def cluster_scaffolds_single_linkage(
    unique_scaffolds: list[str], thresh: float,
) -> dict[str, int]:
    """Single-linkage greedy: scan all pairs, if Tanimoto >= thresh, union."""
    fps = morgan_fp_batch(unique_scaffolds).astype(np.uint8)
    T = tanimoto_matrix_uint8(fps)
    n = len(unique_scaffolds)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    iu, ju = np.triu_indices(n, k=1)
    mask = T[iu, ju] >= thresh
    for i, j in zip(iu[mask], ju[mask]):
        union(int(i), int(j))

    # canonical cluster id = root index
    root_to_id: dict[int, int] = {}
    out: dict[str, int] = {}
    for idx, scaf in enumerate(unique_scaffolds):
        r = find(idx)
        if r not in root_to_id:
            root_to_id[r] = len(root_to_id)
        out[scaf] = root_to_id[r]
    return out


# ---------------------------------------------------------------------------
# Per-cluster stretch fitting
# ---------------------------------------------------------------------------

def fit_per_cluster_s(
    p_tr: np.ndarray, y_tr: np.ndarray, clust_tr: np.ndarray,
    n_inner_folds: int = N_INNER_FOLDS, seed: int = INNER_KF_SEED,
) -> tuple[dict[int, float], float]:
    """For each cluster c with >= MIN_C_TRAIN training rows, choose s_c that
    minimises inner-CV RAE on rows in cluster c, where the local mean mu_c
    is the cluster's training mean. For tiny clusters fall back to global s.
    Returns (cluster_id -> s, global_s)."""
    rng = np.random.default_rng(seed)

    # global s fit on this outer training fold
    mu_g = float(p_tr.mean())
    s_global = best_stretch(p_tr, y_tr, mu_g)

    cluster_ids = np.unique(clust_tr)
    out: dict[int, float] = {}

    for c in cluster_ids:
        mask = clust_tr == c
        n_c = int(mask.sum())
        if n_c < MIN_C_TRAIN:
            continue

        idx_c = np.flatnonzero(mask)
        # inner k-fold over this cluster's rows
        perm = rng.permutation(idx_c)
        folds = np.array_split(perm, min(n_inner_folds, n_c))

        # per-grid-value pooled RAE on inner-OOF cluster rows
        oof_by_s: dict[float, np.ndarray] = {
            s: np.full(n_c, np.nan) for s in STRETCH_GRID
        }
        local_pos = {idx: pos for pos, idx in enumerate(idx_c)}
        for va_idx in folds:
            if len(va_idx) == 0:
                continue
            tr_idx = np.setdiff1d(idx_c, va_idx, assume_unique=False)
            if len(tr_idx) == 0:
                continue
            mu_inner = float(p_tr[tr_idx].mean())
            for s in STRETCH_GRID:
                pred = mu_inner + s * (p_tr[va_idx] - mu_inner)
                for k, vi in enumerate(va_idx):
                    oof_by_s[s][local_pos[vi]] = pred[k]

        best_s, best_r = s_global, float("inf")
        for s in STRETCH_GRID:
            oof = oof_by_s[s]
            valid = ~np.isnan(oof)
            if valid.sum() < 2:
                continue
            r = float(rae(y_tr[idx_c[valid]], oof[valid]))
            if r < best_r:
                best_r = r
                best_s = float(s)
        out[int(c)] = best_s

    return out, s_global


# ---------------------------------------------------------------------------
# Outer CV with per-cluster stretch
# ---------------------------------------------------------------------------

def outer_cv_per_cluster(
    p_anchor: np.ndarray, y_unb: np.ndarray,
    unb_scaffolds: list[str], clust_id_per_row: np.ndarray,
    outer_seed: int = OUTER_KF_SEED,
) -> tuple[float, np.ndarray, list[dict]]:
    n = len(y_unb)
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_OUTER_FOLDS, shuffle=True, seed=outer_seed,
    )
    oof = np.full(n, np.nan)
    fold_diag = []
    for fi, (tr, va) in enumerate(splits):
        s_map, s_global = fit_per_cluster_s(
            p_anchor[tr], y_unb[tr], clust_id_per_row[tr],
        )
        mu_tr = float(p_anchor[tr].mean())
        # apply per-row: use cluster-s if cluster has one, else global s
        s_row = np.array(
            [s_map.get(int(c), s_global) for c in clust_id_per_row[va]],
            dtype=np.float64,
        )
        oof[va] = mu_tr + s_row * (p_anchor[va] - mu_tr)
        fold_diag.append({
            "fold": int(fi),
            "n_tr": int(len(tr)),
            "n_va": int(len(va)),
            "n_clusters_fit": int(len(s_map)),
            "s_global": float(s_global),
            "s_per_cluster_used_va_unique": int(len(np.unique(s_row))),
        })
    assert not np.isnan(oof).any()
    return float(rae(y_unb, oof)), oof, fold_diag


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Murcko-cluster per-cluster rank-stretch on nb2171 anchor")
    print("=" * 78)

    # ----- load test + train + unblind -----
    te = load_test()
    tr = load_train()
    te_smiles = te["smiles"].values
    tr_smiles = tr["smiles"].values
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    n_tr = len(tr_smiles)
    print(f"[load] n_test={len(te_smiles)}  n_train={n_tr}  n_unb={n_unb}")

    # ----- Murcko scaffolds for the union (4139 + 253) -----
    unb_smiles = te_smiles[unb_idx]
    unb_scaf = [bemis_murcko(s) or "" for s in unb_smiles]
    tr_scaf = [bemis_murcko(s) or "" for s in tr_smiles]
    union_scaffolds = sorted({s for s in (tr_scaf + unb_scaf) if s})
    print(f"[scaffold] unique Murcko in 4139+253 union: {len(union_scaffolds)}")

    # ----- cluster Murcko scaffolds by Tanimoto >= 0.5 -----
    print(f"[cluster] single-linkage Tanimoto >= {TANIMOTO_THRESHOLD}")
    t_clust = time.time()
    scaf_to_cluster = cluster_scaffolds_single_linkage(
        union_scaffolds, thresh=TANIMOTO_THRESHOLD,
    )
    n_clusters = len(set(scaf_to_cluster.values()))
    print(
        f"[cluster] {len(union_scaffolds)} scaffolds -> {n_clusters} clusters "
        f"({time.time() - t_clust:.1f}s)"
    )

    # project the 253 unblind rows to cluster ids (NA scaffolds -> -1)
    clust_id_per_row = np.array(
        [scaf_to_cluster.get(s, -1) for s in unb_scaf], dtype=np.int64,
    )
    # cluster id distribution on the 253
    unique_c_unb, counts_c_unb = np.unique(clust_id_per_row, return_counts=True)
    n_clusters_present = int((unique_c_unb >= 0).sum())
    print(
        f"[cluster] clusters present on 253 unblind = {n_clusters_present}  "
        f"(NA={int((clust_id_per_row < 0).sum())})"
    )

    # ----- nb2171 anchor: deep-30 mean OOF per row -----
    print("\n[anchor] reconstructing nb2171 SLSQP+stretch OOF (5 kf_seeds)")
    P_unb, anchor_names = load_anchor_matrix(n_unb)
    p_anchor = nb2171_mean_oof(P_unb, y_unb, unb_scaf)
    anchor_rae = float(rae(y_unb, p_anchor))
    print(
        f"[anchor] anchor stack {anchor_names}  shape={P_unb.shape}  "
        f"RAE(mean OOF) = {anchor_rae:.4f}"
    )

    # ----- outer CV with per-cluster stretch -----
    print("\n" + "-" * 78)
    print(f"OUTER SCAFFOLD CV  outer_seed={OUTER_KF_SEED}  inner_folds={N_INNER_FOLDS}")
    print("-" * 78)
    pooled_rae, oof_2210, fold_diag = outer_cv_per_cluster(
        p_anchor, y_unb, unb_scaf, clust_id_per_row,
    )
    print(f"[main] outer pooled RAE = {pooled_rae:.4f}  (vs anchor {anchor_rae:.4f})")
    for fd in fold_diag:
        print(
            f"   fold {fd['fold']}: n_tr={fd['n_tr']}  n_va={fd['n_va']}  "
            f"clusters_fit={fd['n_clusters_fit']}  s_global={fd['s_global']:.3f}  "
            f"unique_s_va={fd['s_per_cluster_used_va_unique']}"
        )

    # ----- gate eval -----
    delta_vs_nb2171 = NB2171_BASELINE - pooled_rae
    gate_pass = delta_vs_nb2171 >= GATE_DELTA
    print("\n" + "-" * 78)
    print("GATE EVALUATION")
    print("-" * 78)
    print(
        f"   nb2171 baseline RAE  = {NB2171_BASELINE:.4f}\n"
        f"   nb2210 RAE           = {pooled_rae:.4f}\n"
        f"   delta (lower better) = {delta_vs_nb2171:+.4f}\n"
        f"   gate threshold       = >= {GATE_DELTA}\n"
        f"   gate                 = {'PASS' if gate_pass else 'FAIL'}"
    )

    # ----- deep-30 verify only on pass -----
    deep30 = None
    if gate_pass:
        print("\n" + "-" * 78)
        print(f"DEEP-30 VERIFY  seeds={DEEP_EXTRA_SEEDS[:3]}..{DEEP_EXTRA_SEEDS[-1]}")
        print("-" * 78)
        all_seeds = [OUTER_KF_SEED] + DEEP_EXTRA_SEEDS
        per = []
        for sd in all_seeds:
            r, _, _ = outer_cv_per_cluster(
                p_anchor, y_unb, unb_scaf, clust_id_per_row, outer_seed=sd,
            )
            per.append({"seed": int(sd), "rae": float(r)})
        raes = np.asarray([p["rae"] for p in per])
        deep30 = {
            "n_seeds": int(len(per)),
            "per_seed": per,
            "mean_rae": float(raes.mean()),
            "std_rae":  float(raes.std()),
            "min_rae":  float(raes.min()),
            "max_rae":  float(raes.max()),
        }
        print(
            f"   n_seeds={deep30['n_seeds']}  mean={deep30['mean_rae']:.4f}  "
            f"std={deep30['std_rae']:.4f}  "
            f"range=[{deep30['min_rae']:.4f}, {deep30['max_rae']:.4f}]"
        )

    # cluster-coverage diagnostics on unblind
    coverage = {
        "n_clusters_present_unb": int(n_clusters_present),
        "n_rows_in_clusters_ge_MIN_C_TRAIN": int(
            sum(
                int(c) >= 0 and int((clust_id_per_row == c).sum()) >= MIN_C_TRAIN
                for c in np.unique(clust_id_per_row)
                for _ in range(int((clust_id_per_row == c).sum()))
            )
            if False else (
                # vectorised version
                sum(
                    int((clust_id_per_row == c).sum())
                    for c in unique_c_unb
                    if int(c) >= 0 and int((clust_id_per_row == c).sum()) >= MIN_C_TRAIN
                )
            )
        ),
        "max_cluster_size_unb": int(counts_c_unb.max()) if len(counts_c_unb) else 0,
        "median_cluster_size_unb": float(np.median(counts_c_unb)) if len(counts_c_unb) else 0.0,
    }

    summary = {
        "tag": TAG,
        "method": "murcko_cluster_per_cluster_rank_stretch_on_nb2171",
        "anchor": "nb2171_mean_of_5_kf_seeds_SLSQP_perfoldstretch",
        "anchor_oof_rae_unb": anchor_rae,
        "nb2171_baseline_rae": NB2171_BASELINE,
        "n_unb": n_unb,
        "n_train": n_tr,
        "n_unique_murcko_union": len(union_scaffolds),
        "tanimoto_threshold": TANIMOTO_THRESHOLD,
        "n_clusters_total": int(n_clusters),
        "cluster_coverage_unblind": coverage,
        "min_c_train_for_local_s": MIN_C_TRAIN,
        "inner_folds": N_INNER_FOLDS,
        "outer_folds": N_OUTER_FOLDS,
        "outer_kf_seed": OUTER_KF_SEED,
        "stretch_grid": STRETCH_GRID,
        "outer_pooled_rae": pooled_rae,
        "fold_diag": fold_diag,
        "gate_delta_required": GATE_DELTA,
        "delta_vs_nb2171": delta_vs_nb2171,
        "gate_pass": bool(gate_pass),
        "deep_30_verify": deep30,
        "wall_sec": round(time.time() - t0, 2),
    }
    json_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {json_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   anchor RAE (nb2171 reconstructed) = {anchor_rae:.4f}")
    print(f"   nb2210 outer pooled RAE           = {pooled_rae:.4f}")
    print(f"   delta vs nb2171 baseline          = {delta_vs_nb2171:+.4f}")
    print(f"   gate (>= {GATE_DELTA} improvement)        = {gate_pass}")
    if deep30:
        print(f"   deep-30 mean RAE                  = {deep30['mean_rae']:.4f}")
    print(f"   wall                              = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== RESULT ====")
    for k in (
        "anchor_oof_rae_unb",
        "outer_pooled_rae",
        "delta_vs_nb2171",
        "gate_pass",
        "n_clusters_total",
        "deep_30_verify",
    ):
        print(f"  {k}: {res.get(k)}")
