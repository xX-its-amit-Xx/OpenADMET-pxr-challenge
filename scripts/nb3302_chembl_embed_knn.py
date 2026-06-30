"""nb3302 -- ChEMBL/ChemBERTa-pretrained embedding kNN correction on the
novel-scaffold tail.

NEW PARADIGM (per phase2_prescriptions + feedback_unblind_augmentation memory):
    route FOUNDATION EMBEDDINGS at the novel-scaffold tail, NOT the train
    manifold. The OOD wall (90.5% novel-scaffold test rows, scaf_train_freq==0)
    is set by scaffold support, and post-hoc/SLSQP on the chemprop_aux anchor
    family has co-converged at ~0.4720 (post-hoc) / ~0.4422-0.4426 (clip
    winner). This is a SUBSTRATE-CHANGE attack: use a pretrained-embedding kNN
    (ChemBERTa-768, a manifold the trees never see) to give the novel-scaffold
    rows a *different-axis* prior, then soft-blend it into the nb3200 anchor
    gated by embedding max-similarity.

    The anchor nb3200 lives on the clip-on-nb3090 / chemprop_aux axis. The
    ChemBERTa kNN lives on a learned-language-model latent axis. For the
    novel-scaffold rows where the anchor has no scaffold support, a high
    embedding-similarity neighbour in the 4139 train carries real signal the
    fingerprint/Mordred trees compress away (cf. feedback_rank_stretch_universal
    variance compression on novel-scaffold OOD).

EMBEDDING (cached, no model call):
    data/processed/chemberta_train_emb.npy  (4139, 768) float32
    data/processed/chemberta_test_emb.npy   (513,  768) float32
    (fallback to Morgan-2048 proxy only if the cache is missing.)

kNN CORRECTION (leak-free):
    Reference set = 4139 TRAIN rows (load_train pec50). Query = the 513 test
    embedding (and its unb_idx subset for OOF). kNN-5 by cosine similarity,
    similarity-weighted train pEC50 -> knn_pred. The unblind LABELS are NEVER
    used as kNN reference -- only the 4139 train labels -- so OOF is honest.

SOFT-BLEND (per-fold learned, novel-scaffold-gated):
    For row i:
        if scaf_train_freq[i] == 0  (novel scaffold, no train support):
            blend = (1-w(i)) * anchor[i] + w(i) * knn_pred[i]
            where w(i) = w_max * clip((maxsim[i] - sim_floor) /
                                      (1 - sim_floor), 0, 1)
            -- only trust the kNN when the nearest train neighbour is
               embedding-similar; w ramps from 0 at sim_floor to w_max at 1.
        else:
            blend = anchor[i]   (anchor already has scaffold support)
    Per outer fold we grid-search (w_max, sim_floor) on the fold-TRAIN rows
    (RAE), then apply those params to fold-VAL rows. No val leak.

CV / GATE:
    15 FRESH kf_seeds {1216..1230}, 5-fold scaffold split on the 253 unblind.
    Per-fold-mean of val RAE is the decision number.
        per_fold_mean < 0.4423 -> "BETTER"
        else                   -> "FAIL"
    Deploy te uses mean-of-fold params (averaged across all 75 folds).

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3200_pred_oof.npy   data/processed/te_nb3200.npy
    data/processed/chemberta_train_emb.npy  data/processed/chemberta_test_emb.npy

Outputs:
    data/processed/nb3302_summary.json
    data/processed/nb3302_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3302.npy         (513,) float32 -- deploy te
    submissions/nb3302_chembl_embed_knn.csv  (only on BETTER)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko, morgan_fp_batch
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3302"

# -- Anchor (blend target) -----------------------------------------------------
ANCHOR_NAME = "nb3200"
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb3200_pred_oof.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb3200.npy"

# -- Embedding cache -----------------------------------------------------------
EMB_TRAIN_PATH = DATA_PROCESSED / "chemberta_train_emb.npy"
EMB_TEST_PATH = DATA_PROCESSED / "chemberta_test_emb.npy"

# -- kNN -----------------------------------------------------------------------
K_NN = 5
SIM_POWER = 1.0  # similarity weighting exponent on the kNN neighbours

# -- Soft-blend grid (learned per fold) ----------------------------------------
W_MAX_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
SIM_FLOOR_GRID = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Gate ----------------------------------------------------------------------
GATE_BETTER = 0.4423

# -- References ----------------------------------------------------------------
REF_NB3200 = 0.4424  # deep-verify of nb3190 clip winner
REF_NB1191 = 0.4718  # PRE-pyramid wide-seed
REF_NB2171 = 0.4682  # cycle-167 anchor-swap ceiling
REF_CLIP_BEST = 0.4422  # nb3173 best clip


def _safe_scaffold(smi: str) -> str | None:
    try:
        return bemis_murcko(smi) or None
    except Exception:
        return None


def _l2norm(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize so dot product == cosine similarity."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _knn_predict(
    query_emb: np.ndarray,
    ref_emb_norm: np.ndarray,
    ref_y: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Similarity-weighted kNN regression by cosine similarity.

    Returns (knn_pred (n_query,), max_sim (n_query,)).
    query_emb is raw (will be L2-normed here); ref_emb_norm is pre-normed.
    """
    q = _l2norm(query_emb.astype(np.float64))
    sims = q @ ref_emb_norm.T  # (n_query, n_ref) cosine in [-1, 1]
    n_q = sims.shape[0]
    knn_pred = np.empty(n_q, dtype=np.float64)
    max_sim = np.empty(n_q, dtype=np.float64)
    for i in range(n_q):
        row = sims[i]
        # top-k by similarity (descending)
        idx = np.argpartition(row, -k)[-k:]
        nbr_sim = row[idx]
        nbr_y = ref_y[idx]
        max_sim[i] = float(nbr_sim.max())
        # weight = max(sim, 0) ** power  (clamp negatives to 0 weight)
        w = np.clip(nbr_sim, 0.0, None) ** SIM_POWER
        sw = w.sum()
        if sw < 1e-12:
            knn_pred[i] = float(nbr_y.mean())
        else:
            knn_pred[i] = float((w * nbr_y).sum() / sw)
    return knn_pred, max_sim


def _apply_blend(
    anchor: np.ndarray,
    knn_pred: np.ndarray,
    max_sim: np.ndarray,
    is_novel: np.ndarray,
    w_max: float,
    sim_floor: float,
) -> np.ndarray:
    """Novel-scaffold-gated soft-blend of anchor with kNN prediction."""
    out = anchor.copy()
    if w_max <= 0.0:
        return out
    denom = max(1.0 - sim_floor, 1e-6)
    ramp = np.clip((max_sim - sim_floor) / denom, 0.0, 1.0)
    w = w_max * ramp
    w = np.where(is_novel, w, 0.0)  # only novel-scaffold rows get kNN
    out = (1.0 - w) * anchor + w * knn_pred
    return out


def _best_params_on_train(
    anchor_tr: np.ndarray,
    knn_tr: np.ndarray,
    sim_tr: np.ndarray,
    novel_tr: np.ndarray,
    y_tr: np.ndarray,
) -> tuple[float, float, float]:
    """Grid-search (w_max, sim_floor) minimizing RAE on fold-train."""
    best = (0.0, SIM_FLOOR_GRID[0])
    best_r = float(rae(y_tr, anchor_tr))  # w_max=0 baseline (anchor only)
    for w_max in W_MAX_GRID:
        for sim_floor in SIM_FLOOR_GRID:
            pred = _apply_blend(
                anchor_tr, knn_tr, sim_tr, novel_tr, w_max, sim_floor
            )
            r = float(rae(y_tr, pred))
            if r < best_r - 1e-12:
                best_r = r
                best = (w_max, sim_floor)
    return best[0], best[1], best_r


def _run_one_seed(
    anchor_unb: np.ndarray,
    knn_unb: np.ndarray,
    sim_unb: np.ndarray,
    novel_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Per-fold grid-learned soft-blend at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_val_raes: list[float] = []
    fold_params: list[tuple[float, float]] = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        w_max, sim_floor, _ = _best_params_on_train(
            anchor_unb[tr_loc], knn_unb[tr_loc], sim_unb[tr_loc],
            novel_unb[tr_loc], y_unb[tr_loc],
        )
        fold_params.append((w_max, sim_floor))
        val_pred = _apply_blend(
            anchor_unb[va_loc], knn_unb[va_loc], sim_unb[va_loc],
            novel_unb[va_loc], w_max, sim_floor,
        )
        oof_blend[va_loc] = val_pred
        fold_val_raes.append(float(rae(y_unb[va_loc], val_pred)))
    if np.isnan(oof_blend).any():
        raise RuntimeError(
            f"kf_seed={kf_seed}: scaffold splits did not cover all rows"
        )
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "per_fold_val_rae_std": float(np.std(fold_val_raes, ddof=1)),
        "fold_params": fold_params,  # list of (w_max, sim_floor)
        "oof": oof_blend,
    }


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- ChemBERTa-embedding kNN correction on novel-scaffold tail")
    print(f"          anchor   = {ANCHOR_NAME}")
    print(f"          k_nn     = {K_NN}")
    print(
        f"          kf_seeds = {len(KF_SEEDS)} fresh "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print(
        f"          gate: per_fold_mean < {GATE_BETTER:.4f} "
        f"-> BETTER, else FAIL"
    )
    print("=" * 78)

    # -- Load test, truth, unblind idx ---------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values if "name" in te.columns else te["Molecule Name"].values
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Load anchor ----------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 1: load anchor {ANCHOR_NAME} (pred_oof + te)")
    print("-" * 78)
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)  # (253,)
    anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)  # (513,)
    if anchor_oof.shape != (n_unb,):
        raise ValueError(f"anchor oof shape {anchor_oof.shape} != ({n_unb},)")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"anchor te shape {anchor_te.shape} != ({n_test},)")
    anchor_unb = anchor_te[unb_idx]  # 513-deploy te restricted to unb rows
    rae_anchor_oof = float(rae(y_unb, anchor_oof))
    rae_anchor_te_unb = float(rae(y_unb, anchor_unb))
    leak_anchor = float(np.mean(np.isclose(anchor_oof, y_unb, atol=1e-6)))
    print(
        f"   {ANCHOR_NAME}: oof_RAE={rae_anchor_oof:.4f}  "
        f"te[unb]_RAE={rae_anchor_te_unb:.4f}  leak_eq={leak_anchor:.2%}"
    )
    if leak_anchor > 0.05:
        print(f"   WARN: {leak_anchor:.1%} anchor rows == truth -- possible leak")
    # The OOF-blend uses the honest cross-fit anchor_oof as the per-row anchor.
    anchor_for_oof = anchor_oof

    # -- Load / build embedding ----------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: load embedding (ChemBERTa-768; Morgan-2048 fallback)")
    print("-" * 78)
    used_fallback = False
    if EMB_TRAIN_PATH.exists() and EMB_TEST_PATH.exists():
        emb_train = np.load(EMB_TRAIN_PATH).astype(np.float64)
        emb_test = np.load(EMB_TEST_PATH).astype(np.float64)
        emb_name = "chemberta_768"
        print(
            f"   cached ChemBERTa: train {emb_train.shape}  test {emb_test.shape}"
        )
    else:
        used_fallback = True
        emb_name = "morgan_2048_fallback"
        tr_smiles_fb = load_train()["smiles"].astype(str).tolist()
        emb_train = morgan_fp_batch(tr_smiles_fb).astype(np.float64)
        emb_test = morgan_fp_batch(te_smiles).astype(np.float64)
        print(
            f"   FALLBACK Morgan-2048: train {emb_train.shape}  "
            f"test {emb_test.shape}"
        )

    # -- Train labels (kNN reference) ----------------------------------------
    tr = load_train()
    if len(tr) != emb_train.shape[0]:
        raise ValueError(
            f"train rows {len(tr)} != emb_train rows {emb_train.shape[0]}"
        )
    train_smiles = tr["smiles"].astype(str).tolist()
    train_y = tr["pec50"].to_numpy(dtype=np.float64)
    print(
        f"   train pEC50: n={len(train_y)} mean={train_y.mean():.3f} "
        f"std={train_y.std():.3f}"
    )

    # -- Scaffolds + novel mask (scaf_train_freq == 0 on test/unb) -----------
    print("\n" + "-" * 78)
    print("STEP 3: Bemis-Murcko scaffolds + novel mask (scaf_train_freq==0)")
    print("-" * 78)
    train_sc = [_safe_scaffold(s) for s in train_smiles]
    test_sc = [_safe_scaffold(s) for s in te_smiles]
    freq_map: dict[str, int] = {}
    for s in train_sc:
        if s is None:
            continue
        freq_map[s] = freq_map.get(s, 0) + 1
    test_freq = np.array([freq_map.get(s, 0) for s in test_sc], dtype=int)
    novel_test = test_freq == 0  # (513,)
    unb_scaffolds = [test_sc[i] or "" for i in unb_idx]
    novel_unb = novel_test[unb_idx]  # (253,)
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(
        f"   TEST novel(scaf_freq==0)={int(novel_test.sum())}/{n_test} "
        f"({novel_test.mean():.1%})"
    )
    print(
        f"   UNB  novel(scaf_freq==0)={int(novel_unb.sum())}/{n_unb} "
        f"({novel_unb.mean():.1%})"
    )
    print(f"   UNB  n_unique_scaffolds={n_unique_scaf}")

    # -- Build kNN predictions (reference = 4139 train) ----------------------
    print("\n" + "-" * 78)
    print("STEP 4: kNN-5 cosine on ChemBERTa (ref=4139 train, query=513 test)")
    print("-" * 78)
    ref_norm = _l2norm(emb_train)
    knn_test, sim_test = _knn_predict(emb_test, ref_norm, train_y, K_NN)
    knn_unb = knn_test[unb_idx]  # (253,)
    sim_unb = sim_test[unb_idx]  # (253,)
    print(
        f"   kNN test: pred mean={knn_test.mean():.3f} std={knn_test.std():.3f}  "
        f"maxsim mean={sim_test.mean():.3f}"
    )
    # Diagnostics on the novel-scaffold unblind rows
    if novel_unb.any():
        knn_raw_rae = float(rae(y_unb[novel_unb], knn_unb[novel_unb]))
        anc_novel_rae = float(rae(y_unb[novel_unb], anchor_for_oof[novel_unb]))
        print(
            f"   [novel-unb] anchor_RAE={anc_novel_rae:.4f}  "
            f"raw_kNN_RAE={knn_raw_rae:.4f}  "
            f"maxsim mean={sim_unb[novel_unb].mean():.3f} "
            f"median={np.median(sim_unb[novel_unb]):.3f}"
        )
    # Correlation of kNN residual-correction direction vs anchor error
    anchor_err = anchor_for_oof - y_unb
    knn_minus_anchor = knn_unb - anchor_for_oof
    if novel_unb.any():
        corr_dir = float(
            np.corrcoef(anchor_err[novel_unb], knn_minus_anchor[novel_unb])[0, 1]
        )
        print(
            f"   [novel-unb] corr(anchor_err, kNN-anchor)={corr_dir:+.3f} "
            f"(negative => kNN pulls toward truth)"
        )

    # -- Multi-seed sweep -----------------------------------------------------
    print("\n" + "-" * 78)
    print(
        f"SWEEP: {len(KF_SEEDS)} FRESH kf_seeds "
        f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}"
    )
    print("-" * 78)
    seed_records = []
    pooled_raes: list[float] = []
    per_fold_means: list[float] = []
    oof_stack: list[np.ndarray] = []
    all_params: list[tuple[float, float]] = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(
            anchor_for_oof, knn_unb, sim_unb, novel_unb, y_unb,
            unb_scaffolds, s,
        )
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        all_params.extend(res["fold_params"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "fold_params": [
                {"w_max": round(wm, 3), "sim_floor": round(sf, 3)}
                for (wm, sf) in res["fold_params"]
            ],
        })
        wm_arr = np.array([p[0] for p in res["fold_params"]])
        sf_arr = np.array([p[1] for p in res["fold_params"]])
        print(
            f"   kf={s}: pooled_RAE={res['pooled_rae']:.4f}  "
            f"perfold_mean={res['per_fold_val_rae_mean']:.4f}  "
            f"w_max_mean={wm_arr.mean():.2f}  "
            f"sim_floor_mean={sf_arr.mean():.2f}  "
            f"wall={time.time()-ts:.2f}s"
        )

    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    pf_arr = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(pooled_arr)

    mean_pooled = float(pooled_arr.mean())
    std_pooled = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0

    mean_pf = float(pf_arr.mean())
    std_pf = float(pf_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem_pf = std_pf / np.sqrt(n_s) if n_s > 1 else 0.0
    t_mult = 2.145  # df=14, two-sided 95%
    ci_low_pf = mean_pf - t_mult * sem_pf
    ci_high_pf = mean_pf + t_mult * sem_pf
    median_pf = float(np.median(pf_arr))

    # Mean-of-fold deploy params (across all 75 folds)
    wm_all = np.array([p[0] for p in all_params], dtype=np.float64)
    sf_all = np.array([p[1] for p in all_params], dtype=np.float64)
    deploy_w_max = float(wm_all.mean())
    deploy_sim_floor = float(sf_all.mean())
    frac_zero_w = float(np.mean(wm_all == 0.0))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled  mean    = {mean_pooled:.4f}  std = {std_pooled:.4f}")
    print(f"   perfold mean    = {mean_pf:.4f}  std = {std_pf:.4f}")
    print(f"   perfold sem     = {sem_pf:.4f}")
    print(f"   perfold 95% CI  = [{ci_low_pf:.4f}, {ci_high_pf:.4f}]")
    print(f"   perfold median  = {median_pf:.4f}")
    print(f"   perfold min/max = [{pf_arr.min():.4f}, {pf_arr.max():.4f}]")
    print(
        f"\n   anchor {ANCHOR_NAME} oof_RAE          = {rae_anchor_oof:.4f}"
    )
    print(f"   delta vs anchor oof (perfold)    = {mean_pf - rae_anchor_oof:+.4f}")
    print(f"   ref nb3200 (deep-verify)         = {REF_NB3200:.4f}")
    print(f"   delta vs nb3200 (perfold mean)    = {mean_pf - REF_NB3200:+.4f}")
    print(
        f"\n   deploy params (mean-of-folds): "
        f"w_max={deploy_w_max:.3f}  sim_floor={deploy_sim_floor:.3f}"
    )
    print(f"   fraction folds with w_max==0     = {frac_zero_w:.1%}")

    # -- Deploy te ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("DEPLOY te (mean-of-folds params applied to 513 anchor + kNN)")
    print("-" * 78)
    te_pred = _apply_blend(
        anchor_te, knn_test, sim_test, novel_test,
        deploy_w_max, deploy_sim_floor,
    ).astype(np.float32)
    n_changed_te = int(np.sum(~np.isclose(te_pred, anchor_te.astype(np.float32))))
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(
        f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
        f"min={te_pred.min():.3f}  max={te_pred.max():.3f}"
    )
    print(f"   te rows changed from anchor       = {n_changed_te}/{n_test}")
    print(f"   te[unb] in-sample RAE             = {te_unb_in_rae:.4f}")

    # Median-seed OOF for storage (by perfold mean)
    med_seed_idx = int(np.argsort(pf_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(
        f"   median seed = {median_seed} "
        f"(perfold_mean={pf_arr[med_seed_idx]:.4f})"
    )

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_pf < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3302 15-seed per-fold-mean {mean_pf:.4f} "
            f"beats BETTER gate {GATE_BETTER:.4f} ({mean_pf - GATE_BETTER:+.4f}). "
            f"ChemBERTa-embedding kNN ({emb_name}) on the novel-scaffold tail "
            f"adds cross-axis signal the chemprop_aux/clip anchor (nb3200 "
            f"{REF_NB3200:.4f}) compresses away. Deploy params "
            f"w_max={deploy_w_max:.3f} sim_floor={deploy_sim_floor:.3f}; "
            f"{n_changed_te} of {n_test} te rows corrected. This is a genuine "
            f"SUBSTRATE-CHANGE (foundation-embedding manifold, not train "
            f"fingerprints). Re-verify with deep-30 before any PRIMARY swap; "
            f"confirm OOF honesty (kNN ref is 4139 train only, no unblind "
            f"labels)."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3302 15-seed per-fold-mean {mean_pf:.4f} fails BETTER "
            f"gate {GATE_BETTER:.4f} ({mean_pf - GATE_BETTER:+.4f}). The "
            f"ChemBERTa-embedding kNN ({emb_name}) on the novel-scaffold tail "
            f"did not beat the anchor nb3200 ({REF_NB3200:.4f}); per-fold "
            f"grid {'mostly selected w_max=0 (anchor-only)' if frac_zero_w > 0.5 else 'selected kNN weight but it did not transfer to fold-val'} "
            f"(frac w_max==0 = {frac_zero_w:.1%}, deploy w_max={deploy_w_max:.3f}). "
            f"Consistent with feedback_unblind_augmentation (OOD wall set by "
            f"scaffold support, not embedding proximity) and cycle-160/163 "
            f"foundation-embedding collapse (ChemBERTa-77M-MTR -> 0 weight). "
            f"Embedding-similarity at the novel-scaffold tail is too low "
            f"(novel-unb maxsim mean shown above) for a kNN prior to carry "
            f"pEC50 signal. Close the embedding-kNN-correction axis on this "
            f"anchor; pivot to abstention or scaffold-diverse aux TRAINING "
            f"(not post-hoc kNN). delta vs anchor oof = "
            f"{mean_pf - rae_anchor_oof:+.4f}."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_chembl_embed_knn.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    # novel-tail diagnostics for the summary
    novel_diag = {}
    if novel_unb.any():
        novel_diag = {
            "n_novel_unb": int(novel_unb.sum()),
            "anchor_oof_rae_novel": round(
                float(rae(y_unb[novel_unb], anchor_for_oof[novel_unb])), 4
            ),
            "raw_knn_rae_novel": round(
                float(rae(y_unb[novel_unb], knn_unb[novel_unb])), 4
            ),
            "maxsim_novel_mean": round(float(sim_unb[novel_unb].mean()), 4),
            "maxsim_novel_median": round(
                float(np.median(sim_unb[novel_unb])), 4
            ),
            "corr_anchorerr_knnminusanchor_novel": round(
                float(
                    np.corrcoef(
                        anchor_err[novel_unb], knn_minus_anchor[novel_unb]
                    )[0, 1]
                ), 4
            ),
        }

    summary = {
        "tag": TAG,
        "method": "chemberta_embedding_knn_correction_on_novel_scaffold_tail",
        "paradigm": (
            "route foundation embeddings (ChemBERTa-768) as a kNN prior on the "
            "novel-scaffold tail; soft-blend into nb3200 anchor gated by "
            "embedding max-similarity; per-fold grid-learned (w_max,sim_floor)"
        ),
        "anchor": ANCHOR_NAME,
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": True,
        "anchor_oof_rae": round(rae_anchor_oof, 4),
        "anchor_te_unb_rae": round(rae_anchor_te_unb, 4),
        "anchor_leak_eq_truth_frac": round(leak_anchor, 4),
        "embedding_name": emb_name,
        "embedding_used_fallback": used_fallback,
        "embedding_train_path": str(EMB_TRAIN_PATH),
        "embedding_test_path": str(EMB_TEST_PATH),
        "embedding_dim": int(emb_train.shape[1]),
        "k_nn": K_NN,
        "sim_power": SIM_POWER,
        "w_max_grid": W_MAX_GRID,
        "sim_floor_grid": SIM_FLOOR_GRID,
        "n_novel_test": int(novel_test.sum()),
        "n_novel_unb": int(novel_unb.sum()),
        "novel_frac_test": round(float(novel_test.mean()), 4),
        "novel_frac_unb": round(float(novel_unb.mean()), 4),
        "novel_tail_diagnostics": novel_diag,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": n_s,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "per_fold_mean_array": [round(float(v), 4) for v in per_fold_means],
        "mean_pooled_rae": round(mean_pooled, 4),
        "std_pooled_rae": round(std_pooled, 4),
        "mean_per_fold_rae": round(mean_pf, 4),
        "std_per_fold_rae": round(std_pf, 4),
        "sem_per_fold_rae": round(sem_pf, 4),
        "ci95_per_fold_low": round(ci_low_pf, 4),
        "ci95_per_fold_high": round(ci_high_pf, 4),
        "median_per_fold_rae": round(median_pf, 4),
        "min_per_fold_rae": round(float(pf_arr.min()), 4),
        "max_per_fold_rae": round(float(pf_arr.max()), 4),
        "deploy_w_max": round(deploy_w_max, 4),
        "deploy_sim_floor": round(deploy_sim_floor, 4),
        "frac_folds_w_max_zero": round(frac_zero_w, 4),
        "n_te_rows_changed": n_changed_te,
        "ref_nb3200_deep_verify": REF_NB3200,
        "ref_nb1191_pre_pyramid": REF_NB1191,
        "ref_nb2171_anchor_swap": REF_NB2171,
        "ref_clip_best": REF_CLIP_BEST,
        "delta_vs_nb3200_perfold_mean": round(mean_pf - REF_NB3200, 4),
        "delta_vs_anchor_oof_perfold_mean": round(mean_pf - rae_anchor_oof, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "median_seed": int(median_seed),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": (str(sub_csv) if verdict == "BETTER" else None),
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   embedding             = {emb_name} (fallback={used_fallback})")
    print(f"   per_fold_mean ({n_s} seeds) = {mean_pf:.4f} +/- {std_pf:.4f}")
    print(f"   95% CI                = [{ci_low_pf:.4f}, {ci_high_pf:.4f}]")
    print(f"   delta vs nb3200       = {mean_pf - REF_NB3200:+.4f}")
    print(f"   delta vs anchor oof   = {mean_pf - rae_anchor_oof:+.4f}")
    print(
        f"   deploy params         = w_max={deploy_w_max:.3f} "
        f"sim_floor={deploy_sim_floor:.3f} (frac w0={frac_zero_w:.0%})"
    )
    print(f"   te rows changed       = {n_changed_te}/{n_test}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "embedding_name", "embedding_used_fallback",
        "mean_per_fold_rae", "std_per_fold_rae",
        "ci95_per_fold_low", "ci95_per_fold_high",
        "delta_vs_nb3200_perfold_mean", "delta_vs_anchor_oof_perfold_mean",
        "anchor_oof_rae", "deploy_w_max", "deploy_sim_floor",
        "frac_folds_w_max_zero", "n_te_rows_changed",
        "novel_tail_diagnostics", "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
