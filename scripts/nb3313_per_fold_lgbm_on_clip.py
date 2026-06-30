"""nb3313 -- LGBM on (y - nb3200) residual using Avalon-512 features ONLY.

NEW PARADIGM (orthogonal substrate):
    nb3262 tried LGBM(K=20) on the (y - nb3200) residual but used the SAME
    117-col 5-way chemprop_aux feature matrix that already drives the
    chemprop_aux-anchored ceiling -- redundant substrate, so the residual
    model had nothing orthogonal to find (the anchor already absorbed it).

    nb3313 swaps the substrate: train the residual LGBM on Avalon-512
    fingerprint bits ONLY (no Mordred / no chemprop embedding / no AtomPair /
    no MACCS / no ChEMBL-kNN). Avalon bits encode a DIFFERENT structural
    hashing than the K=20 RFE slice, so any residual structure they capture is
    orthogonal to the chemprop_aux stack that produced nb3200.

ANCHOR (PRE-clean chain via nb3090 -> nb3200):
    nb3200 deep-30 OOF RAE  = 0.4424 (cycle-160 PRIMARY-1 candidate)
    target gate (BETTER)    < 0.4423 (per-fold-mean)

PROTOCOL (per kf_seed, 5-fold scaffold split on 253 unblind):
    residual = y_unb - nb3200_oof
    For each fold:
        mdl = LGBM(Avalon-512, max_depth=3, n_est=200, lr=0.03)
        mdl.fit(X_unb_avalon[tr_loc], residual[tr_loc])
        oof[va_loc] = nb3200_oof[va_loc] + mdl.predict(X_unb_avalon[va_loc])
    pooled = rae(y_unb, oof)
    per_fold_mean = mean(per-fold val RAE)
    Repeat for 15 fresh kf_seeds {1216..1230}.

DEPLOY:
    Refit LGBM on (X_unb_avalon, residual) once per kf_seed; mean-bag
    prediction on X_te_avalon; te_final = te_nb3200 + mean-bag residual pred.

GATE (on 15-seed per-fold-mean):
    per_fold_mean < 0.4423 -> "BETTER" (beats nb3200 deep-30 mean)
    else                   -> "FAIL"

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb3200_pred_oof.npy   data/processed/te_nb3200.npy
    data/processed/te_avalon512.npy      (513, 512) uint8 Avalon fingerprint

Outputs:
    data/processed/nb3313_summary.json
    data/processed/nb3313_pred_oof.npy   (253,) float32 -- median-seed OOF
    data/processed/te_nb3313.npy         (513,) float32 -- deploy te
    submissions/nb3313_per_fold_lgbm_on_clip.csv  (only on BETTER)
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
import lightgbm as lgb
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb3313"
PARENT_TAG = "nb3200"

# -- Anchor (nb3200) ---------------------------------------------------------
ANCHOR_OOF_PATH = DATA_PROCESSED / "nb3200_pred_oof.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_nb3200.npy"

# -- Avalon-512 feature matrix (orthogonal substrate vs chemprop_aux 117col) -
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"

# -- CV protocol -------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1216, 1231))  # 15 FRESH seeds {1216..1230}

# -- Gate --------------------------------------------------------------------
GATE_BETTER = 0.4423  # per-fold-mean < this -> BETTER (beats nb3200 0.4424)

# -- References --------------------------------------------------------------
REF_NB3200 = 0.4424     # cycle-160 deep-30 mean (PRIMARY-1 candidate)
REF_NB3200_STD = 0.0023
REF_NB3090 = 0.4472     # parent of nb3200
REF_NB3262 = 0.4424     # chemprop_aux-feature residual sibling (redundant substrate)
REF_NB2171 = 0.4682     # prior PRIMARY-1 anchor swap
REF_NB1191 = 0.4718     # PRE-pyramid wide-seed mean
CHEMPROP_AUX_REF = 0.6216


def _lgbm_params(seed):
    """Residual LGBM per task spec: max_depth=3, n_est=200, lr=0.03.

    Shallow trees (depth=3) + modest n_est on 512 sparse binary Avalon bits.
    """
    return dict(
        objective="regression",
        max_depth=3,
        num_leaves=7,            # 2**3 - 1 to honor max_depth=3
        n_estimators=200,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _load_avalon_test(path, n_test_expected):
    if not path.exists():
        raise FileNotFoundError(f"missing Avalon cache: {path}")
    X = np.load(path)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"Avalon shape mismatch {path}: {X.shape}")
    if X.shape[1] != 512:
        raise ValueError(f"Avalon dim {X.shape[1]} != 512")
    X = X.astype(np.float32)
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


# ============================================================================
# core honest CV: LGBM(Avalon-512) on (y - nb3200) residual per kf_seed
# ============================================================================

def _run_one_seed(
    X_unb_av, anchor_oof, residual, y_unb, unb_scaffolds, kf_seed, n_folds,
):
    """One kf_seed honest n-fold scaffold CV; LGBM(seed=kf_seed) on residual."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=n_folds, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    per_fold_val_rae = []
    per_fold_train_rae = []
    for fi, (tr_loc, va_loc) in enumerate(splits):
        mdl = lgb.LGBMRegressor(**_lgbm_params(kf_seed))
        mdl.fit(X_unb_av[tr_loc], residual[tr_loc])
        resid_pred_va = mdl.predict(X_unb_av[va_loc])
        oof[va_loc] = anchor_oof[va_loc] + resid_pred_va
        per_fold_val_rae.append(float(rae(y_unb[va_loc], oof[va_loc])))
        resid_pred_tr = mdl.predict(X_unb_av[tr_loc])
        per_fold_train_rae.append(
            float(rae(y_unb[tr_loc], anchor_oof[tr_loc] + resid_pred_tr))
        )
    if np.isnan(oof).any():
        raise RuntimeError(
            f"scaffold splits did not cover all rows (kf_seed={kf_seed})"
        )
    pooled = float(rae(y_unb, oof))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(per_fold_val_rae)),
        "per_fold_val_rae_std": (
            float(np.std(per_fold_val_rae, ddof=1))
            if len(per_fold_val_rae) > 1 else 0.0
        ),
        "per_fold_val_rae": per_fold_val_rae,
        "per_fold_train_rae_mean": float(np.mean(per_fold_train_rae)),
        "oof": oof,
    }


# ============================================================================
# main
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- LGBM on (y - nb3200) residual, Avalon-512 features ONLY")
    print(f"          parent     : {PARENT_TAG} (deep-30 mean {REF_NB3200:.4f})")
    print(f"          substrate  : Avalon-512 (orthogonal to chemprop_aux 117col)")
    print(f"          LGBM params: max_depth=3, n_est=200, lr=0.03")
    print(f"          kf_seeds   : {len(KF_SEEDS)} FRESH "
          f"{{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}}")
    print(f"          gate       : per_fold_mean < {GATE_BETTER:.4f}"
          f" -> BETTER else FAIL")
    print("=" * 78)

    # -- Load test + truth + unblind idx -------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns
                 else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values if "name" in te.columns
                else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Scaffolds for honest CV ---------------------------------------------
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique = {n_unique_scaf}")

    # -- Load nb3200 anchor (PRE-clean chain via nb3090) ---------------------
    anchor_oof = np.load(ANCHOR_OOF_PATH).astype(np.float64)  # (253,)
    anchor_te = np.load(ANCHOR_TE_PATH).astype(np.float64)    # (513,)
    if anchor_oof.shape != (n_unb,):
        raise ValueError(f"nb3200 oof shape {anchor_oof.shape} != ({n_unb},)")
    if anchor_te.shape != (n_test,):
        raise ValueError(f"nb3200 te shape {anchor_te.shape} != ({n_test},)")
    rae_anchor = float(rae(y_unb, anchor_oof))
    leak_eq = float(np.mean(np.isclose(anchor_oof, y_unb, atol=1e-6)))
    residual = y_unb - anchor_oof
    print(f"[anchor] nb3200 oof RAE = {rae_anchor:.4f}  (ref {REF_NB3200:.4f}, "
          f"d={rae_anchor - REF_NB3200:+.4f})")
    print(f"[anchor] leak_eq_truth_frac = {leak_eq:.2%}")
    print(f"[residual] mean={residual.mean():+.4f}  std={residual.std():.4f}  "
          f"min={residual.min():+.4f}  max={residual.max():+.4f}")

    # -- Build Avalon-512 substrate ------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load Avalon-512 feature matrix (orthogonal substrate)")
    print("-" * 78)
    X_te_av = _load_avalon_test(AVALON_TE_PATH, n_test)
    X_unb_av = X_te_av[unb_idx]
    assert X_unb_av.shape == (n_unb, 512)
    n_active_bits = int((X_te_av.sum(axis=0) > 0).sum())
    print(f"   X_te_av  = {X_te_av.shape}  X_unb_av = {X_unb_av.shape}")
    print(f"   active bits (nonzero over 513) = {n_active_bits}/512")

    # -- Multi-seed honest cross-fit -----------------------------------------
    print("\n" + "-" * 78)
    print(f"STEP 2: HONEST {N_FOLDS}-fold scaffold CV over {len(KF_SEEDS)}"
          f" kf_seeds")
    print("-" * 78)
    seed_records = []
    pooled_raes = []
    per_fold_means = []
    oof_stack = []
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(
            X_unb_av, anchor_oof, residual, y_unb, unb_scaffolds,
            kf_seed=s, n_folds=N_FOLDS,
        )
        pooled_raes.append(res["pooled_rae"])
        per_fold_means.append(res["per_fold_val_rae_mean"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "per_fold_val_rae_std": round(res["per_fold_val_rae_std"], 4),
            "per_fold_train_rae_mean": round(res["per_fold_train_rae_mean"], 4),
        })
        print(f"   kf={s}: pooled={res['pooled_rae']:.4f}  "
              f"perfold_mean={res['per_fold_val_rae_mean']:.4f}  "
              f"perfold_std={res['per_fold_val_rae_std']:.4f}  "
              f"train_mean={res['per_fold_train_rae_mean']:.4f}  "
              f"wall={time.time()-ts:.2f}s")

    pooled_arr = np.asarray(pooled_raes, dtype=np.float64)
    pf_arr = np.asarray(per_fold_means, dtype=np.float64)
    n_s = len(pooled_arr)

    mean_pooled = float(pooled_arr.mean())
    std_pooled = float(pooled_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem_pooled = std_pooled / np.sqrt(n_s) if n_s > 1 else 0.0

    mean_pf = float(pf_arr.mean())
    std_pf = float(pf_arr.std(ddof=1)) if n_s > 1 else 0.0
    sem_pf = std_pf / np.sqrt(n_s) if n_s > 1 else 0.0

    t_mult = 2.145  # df=14, two-sided 95%
    ci_low_pf = mean_pf - t_mult * sem_pf
    ci_high_pf = mean_pf + t_mult * sem_pf
    median_pf = float(np.median(pf_arr))

    print("\n" + "-" * 78)
    print(f"AGGREGATE ({n_s} seeds)")
    print("-" * 78)
    print(f"   pooled  mean    = {mean_pooled:.4f}  std = {std_pooled:.4f}")
    print(f"   perfold mean    = {mean_pf:.4f}  std = {std_pf:.4f}")
    print(f"   perfold sem     = {sem_pf:.4f}")
    print(f"   perfold 95% CI  = [{ci_low_pf:.4f}, {ci_high_pf:.4f}]")
    print(f"   perfold median  = {median_pf:.4f}")
    print(f"   perfold min/max = [{pf_arr.min():.4f}, {pf_arr.max():.4f}]")
    print(f"\n   ref nb3200 (deep-30 mean) = {REF_NB3200:.4f} +/- {REF_NB3200_STD:.4f}")
    print(f"   delta vs nb3200           = {mean_pf - REF_NB3200:+.4f}")
    print(f"   ref nb3262 (chemprop-feat)= {REF_NB3262:.4f}")
    print(f"   ref nb3090 (parent)       = {REF_NB3090:.4f}")
    print(f"   ref nb2171 (anchor-swap)  = {REF_NB2171:.4f}")

    # -- Deploy: refit per-kf_seed on ALL 253, mean-bag predict on te ---------
    print("\n" + "-" * 78)
    print(f"STEP 3: deploy refit on all 253 unblind, mean-bag {len(KF_SEEDS)}"
          f"-seed LGBM, te = nb3200_te + resid_pred_te")
    print("-" * 78)
    sum_te_resid = np.zeros(n_test, dtype=np.float64)
    for s in KF_SEEDS:
        mdl = lgb.LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_unb_av, residual)
        sum_te_resid += mdl.predict(X_te_av).astype(np.float64)
    mean_te_resid = sum_te_resid / len(KF_SEEDS)
    te_pred = (anchor_te + mean_te_resid).astype(np.float32)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te_resid mean={mean_te_resid.mean():+.4f}  "
          f"std={mean_te_resid.std():.4f}")
    print(f"   te(513) final mean={te_pred.mean():.3f}  std={te_pred.std():.3f}  "
          f"min={te_pred.min():.3f}  max={te_pred.max():.3f}")
    print(f"   te[unb] in-sample RAE = {te_unb_in_rae:.4f}  "
          f"(expected NEGATIVE gap vs honest mean -- deploy sees 253 labels)")
    print(f"   gap (in_sample - honest) = {te_unb_in_rae - mean_pf:+.4f}")

    # Median-seed OOF for storage (by perfold mean)
    med_seed_idx = int(np.argsort(pf_arr)[n_s // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)
    print(f"   median seed = {median_seed} "
          f"(perfold_mean={pf_arr[med_seed_idx]:.4f})")

    # -- Gate ----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 4: GATE")
    print("-" * 78)
    if mean_pf < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE-CANDIDATE. nb3313 15-seed per-fold-mean "
            f"{mean_pf:.4f} beats BETTER gate {GATE_BETTER:.4f} "
            f"({mean_pf - GATE_BETTER:+.4f}) and nb3200 deep-30 "
            f"{REF_NB3200:.4f} ({mean_pf - REF_NB3200:+.4f}). "
            f"LGBM on (y - nb3200) residual using an ORTHOGONAL substrate "
            f"(Avalon-512 bits, NOT the chemprop_aux 117col K=20 slice) "
            f"extracts structure the clip+pyramid chemprop_aux stack missed. "
            f"This is the substrate-change lever (cf. cycle-134/169): nb3262 "
            f"failed because chemprop_aux features were redundant; Avalon bits "
            f"are a different structural hashing. PRE-clean anchor chain "
            f"(nb3090 -> nb3200). Re-verify with deep-30 (kf_seeds 30+) before "
            f"any PRIMARY-1 swap; same under-dispersion-risk root as cycle-160."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3313 15-seed per-fold-mean {mean_pf:.4f} fails "
            f"BETTER gate {GATE_BETTER:.4f} ({mean_pf - GATE_BETTER:+.4f}). "
            f"Delta vs nb3200 (deep-30 mean {REF_NB3200:.4f}) = "
            f"{mean_pf - REF_NB3200:+.4f}. LGBM on (y - nb3200) residual using "
            f"Avalon-512 bits ONLY cannot find new structure on top of the "
            f"learned-clip chemprop_aux stack -- the orthogonal-substrate "
            f"hypothesis does not hold here either (Avalon bits add no residual "
            f"signal nb3200 lacks, mirroring nb3262's chemprop_aux-feature "
            f"failure). Residual-on-residual paradigm on the nb3200 anchor "
            f"remains closed across BOTH feature substrates; the bottleneck is "
            f"the anchor's n=253 capacity, not the feature hashing. "
            f"Substrate change must come from a DIFFERENT ANCHOR axis."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts ------------------------------------------------------
    print("\n" + "-" * 78)
    print("STEP 5: SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}  (median-seed honest OOF, 253,)")
    print(f"   [save] {te_path}   (deploy mean-bag te, 513,)")

    sub_csv = SUBMISSIONS / f"{TAG}_per_fold_lgbm_on_clip.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": ("lgbm_on_y_minus_nb3200_residual_with_avalon512_features_only_"
                   "honest_5fold_scaffold_cv_15_fresh_seeds"),
        "substrate": "avalon512_only_orthogonal_to_chemprop_aux_117col",
        "anchor_oof_path": str(ANCHOR_OOF_PATH),
        "anchor_te_path": str(ANCHOR_TE_PATH),
        "anchor_pre_unblind": True,
        "anchor_oof_rae": round(rae_anchor, 4),
        "anchor_leak_eq_truth_frac": round(leak_eq, 4),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_min": float(residual.min()),
        "residual_max": float(residual.max()),
        "feat_dim": 512,
        "n_active_avalon_bits": n_active_bits,
        "lgbm_params_seed0": _lgbm_params(0),
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
        "sem_pooled_rae": round(sem_pooled, 4),
        "mean_per_fold_rae": round(mean_pf, 4),
        "std_per_fold_rae": round(std_pf, 4),
        "sem_per_fold_rae": round(sem_pf, 4),
        "ci95_per_fold_low": round(ci_low_pf, 4),
        "ci95_per_fold_high": round(ci_high_pf, 4),
        "median_per_fold_rae": round(median_pf, 4),
        "min_per_fold_rae": round(float(pf_arr.min()), 4),
        "max_per_fold_rae": round(float(pf_arr.max()), 4),
        "ref_nb3200_deep30_mean": REF_NB3200,
        "ref_nb3200_deep30_std": REF_NB3200_STD,
        "ref_nb3262_chemprop_feat": REF_NB3262,
        "ref_nb3090": REF_NB3090,
        "ref_nb2171": REF_NB2171,
        "ref_nb1191": REF_NB1191,
        "ref_chemprop_aux": CHEMPROP_AUX_REF,
        "delta_vs_nb3200_perfold_mean": round(mean_pf - REF_NB3200, 4),
        "delta_vs_nb3262": round(mean_pf - REF_NB3262, 4),
        "delta_vs_anchor_in_sample": round(mean_pf - rae_anchor, 4),
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_min": float(te_pred.min()),
        "te_max": float(te_pred.max()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "te_unb_in_sample_minus_honest_gap": round(te_unb_in_rae - mean_pf, 4),
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
    print(f"   per_fold_mean ({n_s} seeds) = {mean_pf:.4f} +/- {std_pf:.4f}")
    print(f"   95% CI                = [{ci_low_pf:.4f}, {ci_high_pf:.4f}]")
    print(f"   delta vs nb3200       = {mean_pf - REF_NB3200:+.4f}")
    print(f"   verdict               = {verdict}")
    print(f"   wall                  = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_per_fold_rae", "std_per_fold_rae",
        "ci95_per_fold_low", "ci95_per_fold_high",
        "delta_vs_nb3200_perfold_mean", "delta_vs_nb3262",
        "anchor_oof_rae", "anchor_leak_eq_truth_frac",
        "te_unb_in_sample_rae", "te_unb_in_sample_minus_honest_gap",
        "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
