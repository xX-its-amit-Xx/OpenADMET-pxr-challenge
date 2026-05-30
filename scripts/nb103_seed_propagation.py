"""nb103 — Seed-Anchored SAR Propagation Network.

The PXR test set was constructed by taking 63 training hits (pEC50>=6,
selectivity>=1.5) and expanding analogs via Tanimoto>0.4. We know exactly
which 61 training compounds are the seeds and their pEC50 values.

Strategy:
  1. Identify seed compounds (61 training hits)
  2. For each test compound, find its nearest seed (anchor)
  3. Build a Matched Molecular Pair (MMP) transform database from all
     training compound pairs within each seed's scaffold neighborhood
  4. Use Free-Wilson / Delta-ML anchored to the seed's exact pEC50
  5. For test compounds near multiple seeds, weighted blend

This transforms "predict 513 unknowns" into "propagate 61 known potent
compound activities through a structured analog graph."
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, morgan_fp_batch, compute_physchem
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

SEED_PXRSEED = 42
N_FOLDS = 5
SEED_PEC50_THRESH = 6.0    # pEC50 >= 6 for seed compounds
SEED_SEL_THRESH   = 1.5    # selectivity >= 1.5 for seed compounds
SEED_SIM_THRESH   = 0.35   # min Tanimoto for anchor assignment
NEIGHBOR_SIM_THRESH = 0.40 # neighborhood for Free-Wilson analysis

LGBM_DELTA = dict(
    n_estimators=500, num_leaves=64, learning_rate=0.05,
    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED_PXRSEED, verbose=-1, n_jobs=4
)
LGBM_GLOBAL = dict(
    n_estimators=1000, num_leaves=64, learning_rate=0.05,
    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=SEED_PXRSEED, verbose=-1, n_jobs=4
)

props_list = ["mw", "logp", "tpsa", "hbd", "hba", "rotbonds", "rings"]


def tanimoto_fast(fps_a, fps_b):
    a = fps_a.astype(np.float32); b = fps_b.astype(np.float32)
    dot = a @ b.T
    sa = a.sum(1)[:, None]; sb = b.sum(1)[None, :]
    return dot / np.maximum(sa + sb - dot, 1e-6)


def compute_physchem_array(smiles_list):
    rows = []
    for smi in smiles_list:
        try:
            p = compute_physchem(smi)
            rows.append([p.get(k, 0) or 0 for k in props_list])
        except Exception:
            rows.append([0.0] * len(props_list))
    return np.array(rows, dtype=np.float32)


def make_delta_feats(fp_anc, fp_com, fp_diff, sim, anc_pec50, phys_diff):
    B, D = fp_com.shape
    B32 = max(1, D // 32)
    fp_com_64  = fp_com.reshape(B, 32, B32).mean(-1)
    fp_diff_64 = fp_diff.reshape(B, 32, B32).mean(-1)
    return np.hstack([fp_com_64, fp_diff_64, sim[:, None],
                      anc_pec50[:, None], phys_diff]).astype(np.float32)


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


def train_seed_delta_model(fps_seeds, y_seeds, fps_train, y_train,
                            phys_seeds, phys_train, sim_thresh=NEIGHBOR_SIM_THRESH):
    """
    Build a delta model trained specifically on seed-anchored pairs.
    For each training compound that is similar to a seed, compute (seed->compound) delta.
    This teaches the model how activity changes as we move away from hits.
    """
    # Find seed-neighbor pairs: training compounds within sim_thresh of any seed
    sim_matrix = tanimoto_fast(fps_seeds, fps_train)  # (n_seeds, n_train)

    i_seed_list, j_train_list, sim_list = [], [], []
    for i_seed in range(len(fps_seeds)):
        for j_train in range(len(fps_train)):
            s = sim_matrix[i_seed, j_train]
            if s >= sim_thresh:
                i_seed_list.append(i_seed)
                j_train_list.append(j_train)
                sim_list.append(s)

    if len(i_seed_list) == 0:
        print(f"  WARNING: No seed-neighbor pairs found at threshold={sim_thresh}")
        return None

    i_seed_arr = np.array(i_seed_list)
    j_train_arr = np.array(j_train_list)
    sim_arr = np.array(sim_list, dtype=np.float32)

    # Delta: y_train[j] - y_seeds[i]  (how much activity changes from seed to neighbor)
    y_delta = y_train[j_train_arr] - y_seeds[i_seed_arr]

    fp_i = fps_seeds[i_seed_arr]
    fp_j = fps_train[j_train_arr]
    fp_com = np.minimum(fp_i, fp_j)
    fp_dif = np.abs(fp_i - fp_j).astype(np.float32)
    phys_d = phys_train[j_train_arr] - phys_seeds[i_seed_arr]

    F_pairs = make_delta_feats(fp_i, fp_com, fp_dif, sim_arr,
                                y_seeds[i_seed_arr], phys_d)

    # Also add reverse pairs for symmetry
    F_rev = make_delta_feats(fp_j, fp_com, fp_dif, sim_arr,
                              y_train[j_train_arr], -phys_d)

    F_all = np.vstack([F_pairs, F_rev])
    y_all = np.concatenate([y_delta, -y_delta])

    print(f"  Seed-delta training: {len(F_pairs)} pairs, "
          f"D range [{y_delta.min():.2f}, {y_delta.max():.2f}]")

    model = lgb.LGBMRegressor(**LGBM_DELTA)
    model.fit(F_all, y_all.astype(np.float32), callbacks=[lgb.log_evaluation(-1)])
    return model


def predict_from_seeds(fps_query, phys_query, fps_seeds, y_seeds, phys_seeds,
                        seed_delta_model, fps_all_train=None, y_all_train=None,
                        phys_all_train=None, global_delta_model=None,
                        top_k_seeds=3):
    """
    For each query compound:
      1. Find its top-K nearest seeds
      2. Predict delta from each seed using seed_delta_model
      3. pEC50_pred = seed_pec50 + delta_pred (weighted by similarity)
    Falls back to global delta model for low-similarity compounds.
    """
    sim_to_seeds = tanimoto_fast(fps_query, fps_seeds)  # (N_query, N_seeds)
    N = len(fps_query)
    preds = np.full(N, np.nan)
    confidences = np.zeros(N, dtype=np.float32)

    for i in range(N):
        sim_row = sim_to_seeds[i]
        top_k_idx = np.argsort(sim_row)[::-1][:top_k_seeds]
        top_sims  = sim_row[top_k_idx]

        if top_sims[0] < SEED_SIM_THRESH:
            # No nearby seed — use global delta model
            if global_delta_model is not None and fps_all_train is not None:
                sim_tr = tanimoto_fast(fps_query[i:i+1], fps_all_train)[0]
                best_t = sim_tr.argmax(); best_s = sim_tr.max()
                fp_ref = fps_all_train[best_t:best_t+1]
                fp_com = np.minimum(fps_query[i:i+1], fp_ref)
                fp_dif = np.abs(fps_query[i:i+1] - fp_ref).astype(np.float32)
                F_g = make_delta_feats(fp_ref, fp_com, fp_dif,
                                       np.array([best_s], dtype=np.float32),
                                       y_all_train[best_t:best_t+1],
                                       (phys_query[i:i+1] - phys_all_train[best_t:best_t+1]))
                preds[i] = y_all_train[best_t] + global_delta_model.predict(F_g)[0]
                confidences[i] = best_s * 0.5  # lower confidence for global
            continue

        # Weighted average from top-K seeds
        weighted_pred = 0.0
        total_weight  = 0.0
        for rank, (k, s) in enumerate(zip(top_k_idx, top_sims)):
            if s < SEED_SIM_THRESH:
                break
            w = s ** 2  # similarity-squared weighting
            fp_seed = fps_seeds[k:k+1]
            fp_q    = fps_query[i:i+1]
            fp_com  = np.minimum(fp_q, fp_seed)
            fp_dif  = np.abs(fp_q - fp_seed).astype(np.float32)
            F_pred  = make_delta_feats(fp_seed, fp_com, fp_dif,
                                        np.array([s], dtype=np.float32),
                                        y_seeds[k:k+1],
                                        phys_query[i:i+1] - phys_seeds[k:k+1])
            delta   = seed_delta_model.predict(F_pred)[0]
            weighted_pred += w * (y_seeds[k] + delta)
            total_weight  += w

        preds[i]      = weighted_pred / total_weight if total_weight > 0 else np.nan
        confidences[i] = top_sims[0]

    return preds, confidences


def main():
    print("=== nb103: Seed-Anchored SAR Propagation ===")

    # Load data
    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED_PXRSEED)

    # ── Identify seed compounds ──────────────────────────────────────────────
    counter_map  = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names    = raw_train["Molecule Name"].values
    pec50_null   = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    selectivity  = y_tr - np.where(np.isnan(pec50_null), 0.0, pec50_null)

    # Seeds = high pEC50 AND selective (exactly the challenge's test set construction)
    seed_mask = (y_tr >= SEED_PXRSEED_THRESH) & (selectivity >= SEED_SEL_THRESH)
    # Also include high-pEC50 with missing counter (assume selective)
    seed_mask |= (y_tr >= SEED_PXRSEED_THRESH) & np.isnan(pec50_null)

    seed_idx   = np.where(seed_mask)[0]
    y_seeds    = y_tr[seed_idx]
    print(f"Identified {len(seed_idx)} seed compounds (pEC50>={SEED_PXRSEED_THRESH}, "
          f"selectivity>={SEED_SEL_THRESH})")
    print(f"  Seed pEC50: mean={y_seeds.mean():.3f}  min={y_seeds.min():.3f}  "
          f"max={y_seeds.max():.3f}")

    # ── Compute fingerprints ─────────────────────────────────────────────────
    print("\nComputing Morgan fingerprints...")
    fps_tr = morgan_fp_batch(tr["smiles"].tolist()).astype(np.float32)
    fps_te = morgan_fp_batch(te["smiles"].tolist()).astype(np.float32)
    fps_seeds = fps_tr[seed_idx]

    print("Computing physchem arrays...")
    phys_tr    = compute_physchem_array(tr["smiles"].tolist())
    phys_te    = compute_physchem_array(te["smiles"].tolist())
    phys_seeds = phys_tr[seed_idx]

    # ── Check test-seed similarity ────────────────────────────────────────────
    print("\nTest-seed similarity analysis:")
    sim_te_seeds = tanimoto_fast(fps_te, fps_seeds)
    best_seed_sim = sim_te_seeds.max(axis=1)
    print(f"  Test->nearest seed: mean={best_seed_sim.mean():.3f}  "
          f"median={np.median(best_seed_sim):.3f}  "
          f"min={best_seed_sim.min():.3f}  max={best_seed_sim.max():.3f}")
    print(f"  Test compounds with seed sim >= 0.4: {(best_seed_sim >= 0.4).sum()} / {len(fps_te)}")
    print(f"  Test compounds with seed sim >= 0.35: {(best_seed_sim >= 0.35).sum()} / {len(fps_te)}")

    # ── Train seed-delta model ────────────────────────────────────────────────
    print("\nTraining seed-anchored delta model...")
    seed_delta_model = train_seed_delta_model(
        fps_seeds, y_seeds, fps_tr, y_tr,
        phys_seeds, phys_tr, sim_thresh=NEIGHBOR_SIM_THRESH
    )

    # ── Train global delta model (fallback) ─────────────────────────────────
    print("\nTraining global delta model (fallback for low-seed-sim compounds)...")
    sim_tr_tr = tanimoto_fast(fps_tr, fps_tr)
    np.fill_diagonal(sim_tr_tr, 0)
    i_idx, j_idx = np.where(sim_tr_tr >= 0.35)
    mask_up = i_idx < j_idx
    i_idx, j_idx = i_idx[mask_up], j_idx[mask_up]
    MAX_P = 300_000
    if len(i_idx) > MAX_P:
        rng = np.random.default_rng(SEED_PXRSEED)
        sel = rng.choice(len(i_idx), MAX_P, replace=False)
        i_idx, j_idx = i_idx[sel], j_idx[sel]

    fps_i = fps_tr[i_idx]; fps_j = fps_tr[j_idx]
    fp_com = np.minimum(fps_i, fps_j)
    fp_dif = np.abs(fps_i - fps_j).astype(np.float32)
    sims_ij = sim_tr_tr[i_idx, j_idx]
    y_d = (y_tr[j_idx] - y_tr[i_idx]).astype(np.float32)
    F_ij = make_delta_feats(fps_i, fp_com, fp_dif, sims_ij,
                             y_tr[i_idx], phys_tr[j_idx]-phys_tr[i_idx])
    F_ji = make_delta_feats(fps_j, fp_com, fp_dif, sims_ij,
                             y_tr[j_idx], phys_tr[i_idx]-phys_tr[j_idx])
    global_delta_model = lgb.LGBMRegressor(**LGBM_DELTA)
    global_delta_model.fit(np.vstack([F_ij, F_ji]),
                            np.concatenate([y_d, -y_d]),
                            callbacks=[lgb.log_evaluation(-1)])
    print("Global delta model trained.")

    # ── Combined features for standard LGBM (backup) ─────────────────────────
    print("\nComputing combined features (backup model)...")
    X_tr = impute(combined(tr["smiles"].tolist()))
    X_te = impute(combined(te["smiles"].tolist()))

    # ── Scaffold CV ──────────────────────────────────────────────────────────
    print("\n=== Scaffold 5-fold CV ===")
    oof_seed  = np.full(len(y_tr), np.nan)
    oof_delta = np.full(len(y_tr), np.nan)  # global delta for comparison

    for fold, (tr_idx_cv, va_idx_cv) in enumerate(splits):
        # Within-fold seeds: seeds that are in the training portion of this fold
        fold_seed_mask = seed_mask[tr_idx_cv]
        fold_seed_global_idx = tr_idx_cv[fold_seed_mask]

        if fold_seed_global_idx.sum() == 0 if isinstance(fold_seed_mask.sum(), np.integer) else len(fold_seed_global_idx) == 0:
            # No seeds in this fold — fall back to global delta
            sim_vt = tanimoto_fast(fps_tr[va_idx_cv], fps_tr[tr_idx_cv])
            best_t = sim_vt.argmax(1); best_g = tr_idx_cv[best_t]
            best_s = sim_vt.max(1)
            fp_refs = fps_tr[best_g]
            fps_va = fps_tr[va_idx_cv]
            F_va = make_delta_feats(fp_refs, np.minimum(fps_va, fp_refs),
                                     np.abs(fps_va-fp_refs).astype(np.float32),
                                     best_s, y_tr[best_g],
                                     phys_tr[va_idx_cv]-phys_tr[best_g])
            oof_delta[va_idx_cv] = y_tr[best_g] + global_delta_model.predict(F_va)
            oof_seed[va_idx_cv] = oof_delta[va_idx_cv]
            print(f"  fold {fold+1}  no fold-seeds, using global delta", flush=True)
            continue

        fps_fold_seeds = fps_tr[fold_seed_global_idx]
        y_fold_seeds   = y_tr[fold_seed_global_idx]
        phys_fold_seeds = phys_tr[fold_seed_global_idx]

        # Retrain seed-delta model on fold-train seeds only (to avoid leakage)
        fold_seed_delta = train_seed_delta_model(
            fps_fold_seeds, y_fold_seeds,
            fps_tr[tr_idx_cv], y_tr[tr_idx_cv],
            phys_fold_seeds, phys_tr[tr_idx_cv],
            sim_thresh=NEIGHBOR_SIM_THRESH
        )

        fps_va_cv = fps_tr[va_idx_cv]
        phys_va_cv = phys_tr[va_idx_cv]

        if fold_seed_delta is not None:
            preds_seed, conf_seed = predict_from_seeds(
                fps_va_cv, phys_va_cv,
                fps_fold_seeds, y_fold_seeds, phys_fold_seeds,
                fold_seed_delta,
                fps_all_train=fps_tr[tr_idx_cv], y_all_train=y_tr[tr_idx_cv],
                phys_all_train=phys_tr[tr_idx_cv], global_delta_model=global_delta_model,
                top_k_seeds=3
            )
            oof_seed[va_idx_cv] = preds_seed

        # Global delta fallback for comparison
        sim_vt = tanimoto_fast(fps_va_cv, fps_tr[tr_idx_cv])
        best_t_g = sim_vt.argmax(1); best_g = tr_idx_cv[best_t_g]
        best_s_g = sim_vt.max(1)
        fp_refs_g = fps_tr[best_g]
        F_va_g = make_delta_feats(fp_refs_g, np.minimum(fps_va_cv, fp_refs_g),
                                   np.abs(fps_va_cv-fp_refs_g).astype(np.float32),
                                   best_s_g, y_tr[best_g],
                                   phys_va_cv - phys_tr[best_g])
        oof_delta[va_idx_cv] = y_tr[best_g] + global_delta_model.predict(F_va_g)

        # Replace NaN in seed predictions with global delta
        nan_mask_seed = ~np.isfinite(oof_seed[va_idx_cv])
        oof_seed[va_idx_cv[nan_mask_seed]] = oof_delta[va_idx_cv[nan_mask_seed]]

        r_seed  = rae(y_tr[va_idx_cv], oof_seed[va_idx_cv])
        r_delta = rae(y_tr[va_idx_cv], oof_delta[va_idx_cv])
        print(f"  fold {fold+1}  seed_prop RAE={r_seed:.4f}  global_delta RAE={r_delta:.4f}  "
              f"fold_seeds={len(fold_seed_global_idx)}", flush=True)

    full_metrics(y_tr, oof_seed,  "nb103_seed_propagation")
    full_metrics(y_tr, oof_delta, "nb103_global_delta_baseline")

    # ── Test predictions ─────────────────────────────────────────────────────
    print("\n=== Final test predictions ===")
    # Retrain seed-delta on ALL seeds
    seed_delta_final = train_seed_delta_model(
        fps_seeds, y_seeds, fps_tr, y_tr,
        phys_seeds, phys_tr, sim_thresh=NEIGHBOR_SIM_THRESH
    )

    te_preds_seed, te_conf = predict_from_seeds(
        fps_te, phys_te,
        fps_seeds, y_seeds, phys_seeds,
        seed_delta_final,
        fps_all_train=fps_tr, y_all_train=y_tr,
        phys_all_train=phys_tr, global_delta_model=global_delta_model,
        top_k_seeds=3
    )

    # Fill any remaining NaN with global delta
    nan_te = ~np.isfinite(te_preds_seed)
    if nan_te.any():
        sim_te_tr = tanimoto_fast(fps_te[nan_te], fps_tr)
        best_t_nan = sim_te_tr.argmax(1)
        best_s_nan = sim_te_tr.max(1)
        fp_refs_nan = fps_tr[best_t_nan]
        fps_nan = fps_te[nan_te]
        F_nan = make_delta_feats(fp_refs_nan, np.minimum(fps_nan, fp_refs_nan),
                                  np.abs(fps_nan-fp_refs_nan).astype(np.float32),
                                  best_s_nan, y_tr[best_t_nan],
                                  phys_te[nan_te] - phys_tr[best_t_nan])
        te_preds_seed[nan_te] = y_tr[best_t_nan] + global_delta_model.predict(F_nan)
        print(f"  Filled {nan_te.sum()} NaN test preds with global delta")

    te_preds_seed = np.clip(te_preds_seed, y_tr.min()-0.5, y_tr.max()+0.5)

    print(f"  Seed-confidence: mean={te_conf.mean():.3f}  "
          f"high-conf(>=0.4): {(te_conf >= 0.4).sum()} / {len(fps_te)}")
    print(f"  Test: min={te_preds_seed.min():.2f}  "
          f"med={np.median(te_preds_seed):.2f}  max={te_preds_seed.max():.2f}")

    np.save(DATA_PROCESSED / "oof_nb103_seed_propagation.npy", oof_seed)
    np.save(DATA_PROCESSED / "te_nb103_seed_propagation.npy", te_preds_seed)

    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": te_preds_seed})
    out = SUBMISSIONS / "103_seed_propagation.csv"
    sub.to_csv(out, index=False)
    print(f"\nSaved: {out}")


# Fix variable name typo
SEED_PXRSEED_THRESH = SEED_PEC50_THRESH


if __name__ == "__main__":
    main()
