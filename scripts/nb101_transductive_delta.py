"""nb101 — Transductive Delta-ML via Test-Test Similarity Graph.

Extends nb76's best model (OOF RAE=0.4164) by adding self-consistency
constraints across test-test molecular pairs. For each test-test pair
with Tanimoto >= SIM_THRESH, the Delta-ML model predicts an expected
D pEC50. We solve:

  minimize: Σ_{(i,j)∈train-test} w_ij * (pred_j - (y_i + D_ij))²
          + Σ_{(i,j)∈test-test}  w_ij * (pred_i - pred_j - D_ij)²
          + λ * Σ_i (pred_i - base_i)²

where base_i is the initial Delta-ML prediction for test compound i.
This forces predictions to be mutually consistent across the test graph.

Analogy: molecular replacement in crystallography — use symmetry (here:
structural similarity) to refine individual measurements.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats, optimize, sparse
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED = 42
N_FOLDS = 5
SIM_THRESH_PAIRS = 0.35    # threshold for building delta pairs
SIM_THRESH_GRAPH = 0.40    # threshold for test-test graph edges
LAMBDA_REG = 2.0           # regularization toward base Delta-ML prediction
LGBM_PARAMS = dict(
    n_estimators=1000, num_leaves=64, learning_rate=0.05,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)
LGBM_DELTA = dict(
    n_estimators=500, num_leaves=64, learning_rate=0.05,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED, verbose=-1, n_jobs=4
)

props_list = ["mw", "logp", "tpsa", "hbd", "hba", "rotbonds", "rings"]


def compute_physchem_array(smiles_list, compute_physchem_fn):
    rows = []
    for smi in smiles_list:
        try:
            p = compute_physchem_fn(smi)
            rows.append([p.get(k, 0) or 0 for k in props_list])
        except Exception:
            rows.append([0.0] * len(props_list))
    return np.array(rows, dtype=np.float32)


def make_delta_feats(fp_anc, fp_com, fp_diff, sim, anc_pec50, phys_diff):
    """Compress fingerprints to 64-dim blocks for delta model."""
    B, D = fp_com.shape
    B32 = max(1, D // 32)
    fp_com_64  = fp_com.reshape(B, 32, B32).mean(-1)
    fp_diff_64 = fp_diff.reshape(B, 32, B32).mean(-1)
    return np.hstack([fp_com_64, fp_diff_64, sim[:, None],
                      anc_pec50[:, None], phys_diff]).astype(np.float32)


def tanimoto_fast(fps_a, fps_b):
    a = fps_a.astype(np.float32); b = fps_b.astype(np.float32)
    dot = a @ b.T
    sa  = a.sum(1)[:, None]; sb = b.sum(1)[None, :]
    return dot / np.maximum(sa + sb - dot, 1e-6)


def full_metrics(y_true, y_pred, label=""):
    yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float)
    msk = np.isfinite(yt) & np.isfinite(yp); yt, yp = yt[msk], yp[msk]
    mae_v = float(np.mean(np.abs(yt - yp)))
    rae_v = mae_v / float(np.mean(np.abs(yt - yt.mean()))) if yt.std() > 0 else np.nan
    r2    = 1 - np.sum((yt-yp)**2) / np.sum((yt-yt.mean())**2) if yt.std() > 0 else np.nan
    pr, _ = stats.pearsonr(yt, yp); sp, _ = stats.spearmanr(yt, yp)
    if label:
        print(f"  [{label}] RAE={rae_v:.4f}  MAE={mae_v:.4f}  R2={r2:.4f}  "
              f"r={pr:.4f}  rho={sp:.4f}")
    return dict(RAE=rae_v, MAE=mae_v, R2=r2, Pearson=pr, Spearman=sp)


def graph_smooth_predictions(base_preds_te, fps_te, fps_tr, y_tr,
                              delta_model, phys_arr_tr, phys_arr_te,
                              sim_thresh=SIM_THRESH_GRAPH,
                              lambda_reg=LAMBDA_REG):
    """
    Solve the graph-consistency optimization for test predictions.

    minimize Σ_{test-train} w_ij*(p_i - (y_j + D_ji))²
           + Σ_{test-test}  w_ij*(p_i - p_j - D_ij)²
           + λ * Σ_i (p_i - base_i)²

    Rearranges to a linear system: A @ p = b.
    """
    N_te = len(base_preds_te)
    N_tr = len(y_tr)

    print(f"  Computing test-train similarity matrix ({N_te}x{N_tr})...")
    sim_te_tr = tanimoto_fast(fps_te, fps_tr)  # (N_te, N_tr)

    print(f"  Computing test-test similarity matrix ({N_te}x{N_te})...")
    sim_te_te = tanimoto_fast(fps_te, fps_te)  # (N_te, N_te)
    np.fill_diagonal(sim_te_te, 0)

    # Get delta model predictions for test-train edges
    print("  Computing train-test delta predictions...")
    best_tr_idx = sim_te_tr.argmax(1)
    best_sims   = sim_te_tr.max(1)

    fp_refs = fps_tr[best_tr_idx]
    fp_com  = np.minimum(fps_te, fp_refs)
    fp_dif  = np.abs(fps_te - fp_refs).astype(np.float32)
    F_te    = make_delta_feats(fp_refs, fp_com, fp_dif, best_sims,
                               y_tr[best_tr_idx],
                               phys_arr_te - phys_arr_tr[best_tr_idx])
    delta_te_tr = delta_model.predict(F_te)  # predicted D from nearest train neighbor

    # Compute expected D for test-test edges
    print(f"  Building test-test graph (threshold={sim_thresh})...")
    i_idx, j_idx = np.where(sim_te_te >= sim_thresh)
    mask_upper = i_idx < j_idx
    i_idx, j_idx = i_idx[mask_upper], j_idx[mask_upper]
    n_edges = len(i_idx)
    print(f"  Test-test edges: {n_edges}")

    if n_edges == 0:
        print("  No test-test edges — returning base predictions")
        return base_preds_te.copy()

    # Delta predictions for test-test pairs
    fp_i = fps_te[i_idx]; fp_j = fps_te[j_idx]
    fp_com_tt = np.minimum(fp_i, fp_j)
    fp_dif_tt = np.abs(fp_i - fp_j).astype(np.float32)
    sims_tt   = sim_te_te[i_idx, j_idx]
    F_tt_ij = make_delta_feats(fp_i, fp_com_tt, fp_dif_tt, sims_tt,
                               base_preds_te[i_idx],
                               phys_arr_te[i_idx] - phys_arr_te[j_idx])
    F_tt_ji = make_delta_feats(fp_j, fp_com_tt, fp_dif_tt, sims_tt,
                               base_preds_te[j_idx],
                               phys_arr_te[j_idx] - phys_arr_te[i_idx])
    delta_ij = delta_model.predict(F_tt_ij)   # D(i->j): pred_j - pred_i
    delta_ji = delta_model.predict(F_tt_ji)   # D(j->i) — should be ≈ -delta_ij
    # Average for symmetry
    delta_ij_sym = 0.5 * (delta_ij - delta_ji)

    # Build linear system: A @ p = b
    # For each test-test edge (i,j): w_ij*(p_i - p_j - D_ij)² + w_ji*(p_j - p_i + D_ij)²
    # Gradient w.r.t. p_i: 2*w_ij*(p_i - p_j - D_ij) = 0
    # Assembles as: A[i,i] += w; A[i,j] -= w; b[i] += w*D_ij
    A = np.diag(np.full(N_te, lambda_reg, dtype=np.float64))
    b = lambda_reg * base_preds_te.astype(np.float64)

    # Train-test anchor terms: w_ij*(p_i - (y_j + delta_ji))²  [using nearest neighbor]
    anchor_w = best_sims ** 2  # weight by similarity
    anchor_target = y_tr[best_tr_idx] + delta_te_tr
    for i in range(N_te):
        w = anchor_w[i]
        A[i, i] += w
        b[i]    += w * anchor_target[i]

    # Test-test consistency terms
    edge_w = sims_tt ** 2
    for k in range(n_edges):
        i_, j_, w = int(i_idx[k]), int(j_idx[k]), edge_w[k]
        d = delta_ij_sym[k]
        # (p_i - p_j - d)² contributes:
        A[i_, i_] += w;  A[j_, j_] += w
        A[i_, j_] -= w;  A[j_, i_] -= w
        b[i_] += w * d;  b[j_] -= w * d

    print("  Solving linear system...")
    p_smooth = np.linalg.solve(A, b)
    return p_smooth


def main():
    print("=== nb101: Transductive Delta-ML with Test-Test Graph ===")
    from pxr.chem import compute_physchem
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    print("Computing Morgan fingerprints...")
    fps_tr = morgan_fp_batch(tr["smiles"].tolist()).astype(np.float32)
    fps_te = morgan_fp_batch(te["smiles"].tolist()).astype(np.float32)

    print("Computing physchem arrays...")
    phys_arr_tr = compute_physchem_array(tr["smiles"].tolist(), compute_physchem)
    phys_arr_te = compute_physchem_array(te["smiles"].tolist(), compute_physchem)

    print("Computing combined features for direct model...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # ── Build Delta model ─────────────────────────────────────────────────────
    print("\nBuilding training pairs for Delta model...")
    sim_tr_tr = tanimoto_fast(fps_tr, fps_tr)
    np.fill_diagonal(sim_tr_tr, 0)
    i_idx, j_idx = np.where(sim_tr_tr >= SIM_THRESH_PAIRS)
    mask_up = i_idx < j_idx
    i_idx, j_idx = i_idx[mask_up], j_idx[mask_up]
    print(f"Training pairs (Tan>={SIM_THRESH_PAIRS}): {len(i_idx):,}")

    MAX_PAIRS = 400_000
    if len(i_idx) > MAX_PAIRS:
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(i_idx), MAX_PAIRS, replace=False)
        i_idx, j_idx = i_idx[sel], j_idx[sel]

    y_delta_ij = (y_tr[j_idx] - y_tr[i_idx]).astype(np.float32)
    y_delta_ji = -y_delta_ij

    fps_i = fps_tr[i_idx]; fps_j = fps_tr[j_idx]
    fp_com = np.minimum(fps_i, fps_j)
    fp_dif = np.abs(fps_i - fps_j).astype(np.float32)
    sims_ij = sim_tr_tr[i_idx, j_idx]

    F_ij = make_delta_feats(fps_i, fp_com, fp_dif, sims_ij,
                            y_tr[i_idx], phys_arr_tr[j_idx]-phys_arr_tr[i_idx])
    F_ji = make_delta_feats(fps_j, fp_com, fp_dif, sims_ij,
                            y_tr[j_idx], phys_arr_tr[i_idx]-phys_arr_tr[j_idx])
    F_all = np.vstack([F_ij, F_ji])
    y_all = np.concatenate([y_delta_ij, y_delta_ji])

    print(f"Delta dataset: {F_all.shape}, D range [{y_all.min():.2f}, {y_all.max():.2f}]")
    delta_model = lgb.LGBMRegressor(**LGBM_DELTA)
    delta_model.fit(F_all, y_all, callbacks=[lgb.log_evaluation(-1)])
    print("Delta model trained.")

    # ── Base Delta-ML OOF (same as nb76) ──────────────────────────────────────
    print("\n=== Base Delta-ML OOF ===")
    oof_delta_base = np.full(len(y_tr), np.nan)
    oof_direct     = np.full(len(y_tr), np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_dir = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        oof_direct[va_idx] = m_dir.predict(X_tr[va_idx])

        fps_va = fps_tr[va_idx]; fps_fold_tr = fps_tr[tr_idx]
        sim_vt = tanimoto_fast(fps_va, fps_fold_tr)
        best_t = sim_vt.argmax(1); best_g = tr_idx[best_t]
        best_s = sim_vt.max(1)

        fp_refs = fps_tr[best_g]
        F_va = make_delta_feats(fp_refs, np.minimum(fps_va, fp_refs),
                                np.abs(fps_va-fp_refs).astype(np.float32),
                                best_s, y_tr[best_g],
                                phys_arr_tr[va_idx]-phys_arr_tr[best_g])
        oof_delta_base[va_idx] = y_tr[best_g] + delta_model.predict(F_va)
        r_d = rae(y_tr[va_idx], oof_delta_base[va_idx])
        print(f"  fold {fold+1}  delta_base RAE={r_d:.4f}", flush=True)

    full_metrics(y_tr, oof_delta_base, "delta_ml_base")

    # ── Final test predictions: base Delta-ML ─────────────────────────────────
    print("\nComputing base test predictions (Delta-ML)...")
    m_dir_final = lgb.train(LGBM_PARAMS, lgb.Dataset(X_tr, label=y_tr),
                            callbacks=[lgb.log_evaluation(-1)])
    te_direct = m_dir_final.predict(X_te)

    sim_te_tr_full = tanimoto_fast(fps_te, fps_tr)
    best_tr_idx    = sim_te_tr_full.argmax(1)
    best_sims_te   = sim_te_tr_full.max(1)
    print(f"Test->train: mean sim={best_sims_te.mean():.3f}  max={best_sims_te.max():.3f}")

    fp_refs_te = fps_tr[best_tr_idx]
    F_te_base  = make_delta_feats(fp_refs_te,
                                  np.minimum(fps_te, fp_refs_te),
                                  np.abs(fps_te-fp_refs_te).astype(np.float32),
                                  best_sims_te, y_tr[best_tr_idx],
                                  phys_arr_te - phys_arr_tr[best_tr_idx])
    te_delta_base = y_tr[best_tr_idx] + delta_model.predict(F_te_base)
    print(f"Base test: min={te_delta_base.min():.2f}  med={np.median(te_delta_base):.2f}  "
          f"max={te_delta_base.max():.2f}")

    # ── Graph smoothing on test predictions ───────────────────────────────────
    print("\n=== Graph Smoothing (transductive) ===")
    te_smooth = graph_smooth_predictions(
        te_delta_base, fps_te, fps_tr, y_tr,
        delta_model, phys_arr_tr, phys_arr_te,
        sim_thresh=SIM_THRESH_GRAPH,
        lambda_reg=LAMBDA_REG
    )
    te_smooth = np.clip(te_smooth, y_tr.min() - 0.5, y_tr.max() + 0.5)
    print(f"Smoothed test: min={te_smooth.min():.2f}  med={np.median(te_smooth):.2f}  "
          f"max={te_smooth.max():.2f}")

    # Compare base vs smoothed on a held-out proxy (we can't do true OOF for test)
    delta_change = np.abs(te_smooth - te_delta_base)
    print(f"Mean prediction change from smoothing: {delta_change.mean():.4f} log-units")

    # Save both base (=nb76) and smoothed versions
    np.save(DATA_PROCESSED / "oof_nb101_delta_base.npy", oof_delta_base)
    np.save(DATA_PROCESSED / "te_nb101_delta_base.npy", te_delta_base)
    np.save(DATA_PROCESSED / "te_nb101_delta_smooth.npy", te_smooth)

    sub_base = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_delta_base})
    sub_base.to_csv(SUBMISSIONS / "101_delta_ml_base.csv", index=False)

    sub_smooth = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_smooth})
    sub_smooth.to_csv(SUBMISSIONS / "101_transductive_delta.csv", index=False)

    print(f"\nSaved 101_delta_ml_base.csv and 101_transductive_delta.csv")

    # Also save OOF for ensemble
    full_metrics(y_tr, oof_delta_base, "nb101_delta_oof_summary")


if __name__ == "__main__":
    main()
