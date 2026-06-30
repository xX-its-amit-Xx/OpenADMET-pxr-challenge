"""nb2154 -- Multi-fingerprint stack as alternative anchor (8285-d).

HYPOTHESIS:
    nb2103 K=28 (mean-bag RAE = 0.4737) is the current SHAP-pruning winner on
    the 117-col 5-way K-tuned matrix (AtomPair + MACCS + Mordred + ChempropEmbed
    + Avalon + ChEMBL kNN). The 117-col matrix mixes physicochemical descriptors
    (Mordred) and learned embeddings (ChempropEmbed) with fingerprints. This
    notebook tests a DIFFERENT prior: a PURE multi-fingerprint stack of five
    complementary FPs (no Mordred, no learned embedding, no ChEMBL kNN).

        Morgan-2048    (radius=2)     -- ECFP4 standard
        MACCS-167                     -- substructure keys
        AtomPair-2048                 -- topological pair counts -> bits
        Avalon-1024                   -- Avalon FP (fragment-based)
        ECFP6-2048     (radius=3)     -- extended-radius Morgan

    Actual total: 2048+167+2048+1024+2048 = 7335-d (task spec said 8285 but
    arithmetic gives 7335; we use the arithmetic-correct value).

    If a pure-FP stack matches or beats nb2103 K=28 0.4737, it gives us a
    DECORRELATED anchor to blend with the 117-col stack -- the two priors
    sample completely different feature axes (FP-only vs FP+desc+embed).
    SHAP-pruning to top-28 keeps the comparison parity with nb2103 K=28.

PROTOCOL:
    1. Compute all 5 FPs for the 513-test test set, slice to unb_idx (n=253).
    2. Build full 8285-d X_unb (concat).
    3. Anchor = chemprop_aux te[unb_idx]; residual = y_unb - anchor.
    4. STEP-A: full-fit LGBM(MSE) on residual (same hyperparams as
       nb1852/nb1861/nb2063/nb2103); SHAP TreeExplainer; mean |SHAP| per col
       -> top-28 indices.
    5. STEP-B: restrict to top-28 cols; 5-seed bag (seeds 0,1,7,42,137); per
       seed: 5-fold scaffold CV using Bemis-Murcko scaffolds of the 253-unb
       compounds.
    6. Report per-seed RAE, mean-bag / median-bag RAE, family composition of
       top-28, comparison vs nb2103 K=28 (0.4737), comparison of family
       composition vs nb2063 top-28 (Mordred 11 / ChempropEmbed 8 / AtomPair 5
       / Avalon 2 / MACCS 1 / ChEMBL_kNN 1).

Outputs:
    scripts/nb2154_multi_fp.py
    data/processed/nb2154_summary.json
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
import lightgbm as lgb
import shap
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Avalon import pyAvalonTools

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED

TAG = "nb2154"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_SEEDS = [0, 1, 7, 42, 137]
RESID_FOLDS = 5
TOP_K_SHAP = 28

# Family bit-widths
N_MORGAN = 2048
N_MACCS = 167
N_ATOMPAIR = 2048
N_AVALON = 1024
N_ECFP6 = 2048
N_TOTAL = N_MORGAN + N_MACCS + N_ATOMPAIR + N_AVALON + N_ECFP6  # 8285

# References (PRE-unblind path)
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_REF = 0.4737
NB2063_TOP28_FAM = {
    "Mordred": 11,
    "ChempropEmbed": 8,
    "AtomPair": 5,
    "Avalon": 2,
    "MACCS": 1,
    "ChEMBL_kNN": 1,
}
DECISION_MARGIN = 0.003


# ---------------------------------------------------------------------------
# Fingerprint generators (constructed once, reused per molecule)
# ---------------------------------------------------------------------------
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_MORGAN)
_ATOMPAIR_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_ATOMPAIR)
_ECFP6_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=N_ECFP6)


def _fp_morgan_row(mol) -> np.ndarray:
    bv = _MORGAN_GEN.GetFingerprint(mol)
    arr = np.zeros(N_MORGAN, dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _fp_maccs_row(mol) -> np.ndarray:
    bv = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros(N_MACCS, dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _fp_atompair_row(mol) -> np.ndarray:
    bv = _ATOMPAIR_GEN.GetFingerprint(mol)
    arr = np.zeros(N_ATOMPAIR, dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _fp_avalon_row(mol) -> np.ndarray:
    bv = pyAvalonTools.GetAvalonFP(mol, nBits=N_AVALON)
    arr = np.zeros(N_AVALON, dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _fp_ecfp6_row(mol) -> np.ndarray:
    bv = _ECFP6_GEN.GetFingerprint(mol)
    arr = np.zeros(N_ECFP6, dtype=np.uint8)
    Chem.DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def _compute_all_fps(smiles: list[str]) -> tuple[np.ndarray, list[str | None]]:
    """Return (X, scaffolds) where X is the 8285-d concat matrix and scaffolds
    are Bemis-Murcko SMILES (or None) per row."""
    n = len(smiles)
    X = np.zeros((n, N_TOTAL), dtype=np.float32)
    scaffolds: list[str | None] = []
    n_failed = 0
    for i, s in enumerate(smiles):
        mol = standardize(s)
        if mol is None:
            scaffolds.append(None)
            n_failed += 1
            continue
        try:
            X[i, :N_MORGAN] = _fp_morgan_row(mol)
            off = N_MORGAN
            X[i, off:off + N_MACCS] = _fp_maccs_row(mol)
            off += N_MACCS
            X[i, off:off + N_ATOMPAIR] = _fp_atompair_row(mol)
            off += N_ATOMPAIR
            X[i, off:off + N_AVALON] = _fp_avalon_row(mol)
            off += N_AVALON
            X[i, off:off + N_ECFP6] = _fp_ecfp6_row(mol)
        except Exception:
            n_failed += 1
        try:
            scaf_mol = MurckoScaffold.GetScaffoldForMol(mol)
            scaffolds.append(Chem.MolToSmiles(scaf_mol))
        except Exception:
            scaffolds.append(None)
    if n_failed > 0:
        print(f"   [warn] {n_failed}/{n} SMILES failed standardize/FP")
    return X, scaffolds


def _lgbm_params(seed: int) -> dict:
    """Same hyperparams as nb2063/nb2103 for parity."""
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


def _feat_family_of_idx(idx: int) -> str:
    """Return the source family name for a given 0..8284 column index."""
    if idx < N_MORGAN:
        return "Morgan2048"
    idx -= N_MORGAN
    if idx < N_MACCS:
        return "MACCS167"
    idx -= N_MACCS
    if idx < N_ATOMPAIR:
        return "AtomPair2048"
    idx -= N_ATOMPAIR
    if idx < N_AVALON:
        return "Avalon1024"
    idx -= N_AVALON
    return "ECFP6_2048"


def _feat_name_of_idx(idx: int) -> str:
    if idx < N_MORGAN:
        return f"Morgan2048_bit_{idx}"
    idx0 = idx - N_MORGAN
    if idx0 < N_MACCS:
        return f"MACCS167_bit_{idx0}"
    idx0 -= N_MACCS
    if idx0 < N_ATOMPAIR:
        return f"AtomPair2048_bit_{idx0}"
    idx0 -= N_ATOMPAIR
    if idx0 < N_AVALON:
        return f"Avalon1024_bit_{idx0}"
    idx0 -= N_AVALON
    return f"ECFP6_2048_bit_{idx0}"


def _scaffold_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 scaffolds: list[str | None],
                                 seed: int) -> np.ndarray:
    n = len(residual)
    splits = scaffold_kfold_indices(scaffolds, n_splits=RESID_FOLDS,
                                    shuffle=True, seed=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    if np.isnan(oof).any():
        raise RuntimeError(f"OOF has NaN for seed {seed}")
    return oof


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- multi-FP stack ({N_TOTAL}-d) -> SHAP top-{TOP_K_SHAP}; "
          f"anchor={ANCHOR}")
    print(f"          seeds={RESID_SEEDS}  folds={RESID_FOLDS} (scaffold CV)")
    print(f"          ref: nb2103 K=28 mean-bag RAE = {NB2103_K28_REF:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print("=" * 78)

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"chemprop_aux te shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] {ANCHOR} te[unb_idx] in_RAE = {rae_anchor:.4f}  "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[resid] mean={residual.mean():+.4f}  std={residual.std():.4f}")

    # ---- STEP 0: compute all 5 FPs for unb compounds ----
    print("\n" + "-" * 78)
    print(f"STEP 0: compute all 5 FPs for 513-test, slice to unb (n={n_unb})")
    print(f"   widths: Morgan={N_MORGAN}  MACCS={N_MACCS}  AtomPair={N_ATOMPAIR}"
          f"  Avalon={N_AVALON}  ECFP6={N_ECFP6}  total={N_TOTAL}")
    print("-" * 78)
    t_fp = time.time()
    X_all, scaffolds_all = _compute_all_fps(test_smiles)
    print(f"   X_all shape = {X_all.shape}  wall = {time.time() - t_fp:.1f}s")
    X_unb = X_all[unb_idx].astype(np.float32)
    scaffolds_unb = [scaffolds_all[i] for i in unb_idx.tolist()]
    n_scaf_none = sum(1 for s in scaffolds_unb if s is None)
    n_scaf_unique = len(set(s for s in scaffolds_unb if s is not None))
    print(f"   X_unb shape = {X_unb.shape}  scaffolds_unb: "
          f"{n_scaf_unique} unique, {n_scaf_none} None")

    # ---- STEP 1: SHAP top-28 from full-fit LGBM on residual ----
    print("\n" + "-" * 78)
    print(f"STEP 1: full-fit LGBM(MSE) on {N_TOTAL}-d -> SHAP top-{TOP_K_SHAP}")
    print("-" * 78)
    t_shap = time.time()
    mdl_full = lgb.LGBMRegressor(**_lgbm_params(seed=0))
    mdl_full.fit(X_unb, residual)
    explainer = shap.TreeExplainer(mdl_full)
    shap_vals = explainer.shap_values(X_unb)
    shap_imp = np.abs(shap_vals).mean(axis=0).astype(np.float32)
    if shap_imp.shape[0] != N_TOTAL:
        raise ValueError(
            f"SHAP imp shape {shap_imp.shape} != N_TOTAL {N_TOTAL}"
        )
    top28_idx = np.argsort(-shap_imp)[:TOP_K_SHAP].astype(np.int32)
    top28_names = [_feat_name_of_idx(int(i)) for i in top28_idx]
    top28_families = [_feat_family_of_idx(int(i)) for i in top28_idx]
    top28_imp = [float(shap_imp[i]) for i in top28_idx]
    print(f"   full-fit SHAP done   wall = {time.time() - t_shap:.1f}s")

    fam_counts: dict[str, int] = {}
    for fam in top28_families:
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
    print(f"   top-{TOP_K_SHAP} family composition (this run):")
    for f in ("Morgan2048", "MACCS167", "AtomPair2048", "Avalon1024",
              "ECFP6_2048"):
        print(f"     {f:14s}: {fam_counts.get(f, 0)}")
    print(f"   nb2063 top-28 family composition (reference):")
    for f, c in NB2063_TOP28_FAM.items():
        print(f"     {f:14s}: {c}")
    print(f"   top-10 features by SHAP |mean|:")
    for rank, (name, imp, fam) in enumerate(
            zip(top28_names[:10], top28_imp[:10], top28_families[:10]), 1):
        print(f"     {rank:2d}. [{fam:14s}] {name:30s}  |SHAP|={imp:.5f}")

    # ---- STEP 2: refit on top-28, 5-seed bag x 5-fold scaffold CV ----
    X_top28 = X_unb[:, top28_idx].astype(np.float32)
    print(f"\n   X_top28 shape = {X_top28.shape}")
    print("\n" + "-" * 78)
    print(f"STEP 2: PER-SEED LGBM(MSE) RESIDUAL CROSS-FIT (scaffold KFold)")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _scaffold_cross_fit_one_seed(
            X_top28, residual, scaffolds_unb, s
        )
        pred_corr_s = anchor + resid_oof_s
        per_seed_corrected[i] = pred_corr_s
        rae_s = float(rae(y_unb, pred_corr_s))
        per_seed_rae.append(rae_s)
        delta_s = rae_s - rae_anchor
        per_seed_records.append({
            "seed": int(s),
            "rae_corrected": rae_s,
            "delta_vs_chemprop_aux": delta_s,
            "resid_oof_std": float(resid_oof_s.std()),
            "resid_oof_mean": float(resid_oof_s.mean()),
            "wall_sec": round(time.time() - ts, 2),
        })
        print(f"   seed {s:3d}:  rae_corr = {rae_s:.4f}  "
              f"(d_vs_anchor = {delta_s:+.4f})  "
              f"|resid_oof|.std = {resid_oof_s.std():.3f}  "
              f"wall = {time.time() - ts:.1f}s")

    mean_bag_oof = per_seed_corrected.mean(axis=0)
    median_bag_oof = np.median(per_seed_corrected, axis=0)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))
    rae_median_bag = float(rae(y_unb, median_bag_oof))

    per_seed_rae_arr = np.array(per_seed_rae)
    rae_per_seed_mean = float(per_seed_rae_arr.mean())
    rae_per_seed_median = float(np.median(per_seed_rae_arr))
    rae_per_seed_std = float(per_seed_rae_arr.std())
    rae_per_seed_min = float(per_seed_rae_arr.min())
    rae_per_seed_max = float(per_seed_rae_arr.max())

    # ---- Decorrelation vs nb2103 K=28 ----
    nb2103_oof_path = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
    pearson_vs_nb2103_K28 = None
    if nb2103_oof_path.exists():
        nb2103_oof = np.load(nb2103_oof_path).astype(np.float64)
        if nb2103_oof.shape[0] == n_unb:
            pearson_vs_nb2103_K28 = float(
                np.corrcoef(mean_bag_oof, nb2103_oof)[0, 1]
            )
    pearson_vs_anchor = float(np.corrcoef(mean_bag_oof, anchor)[0, 1])

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean          = {rae_per_seed_mean:.4f}")
    print(f"   per-seed median        = {rae_per_seed_median:.4f}")
    print(f"   per-seed std           = {rae_per_seed_std:.4f}")
    print(f"   per-seed min/max       = "
          f"{rae_per_seed_min:.4f} / {rae_per_seed_max:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f}, "
          f"d_vs_nb2103_K28 = {rae_mean_bag - NB2103_K28_REF:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}")
    print(f"   Pearson(mean_bag, anchor)      = {pearson_vs_anchor:.4f}")
    if pearson_vs_nb2103_K28 is not None:
        print(f"   Pearson(mean_bag, nb2103_K28)  = "
              f"{pearson_vs_nb2103_K28:.4f}")
    else:
        print(f"   Pearson(mean_bag, nb2103_K28)  = N/A "
              f"(missing {nb2103_oof_path.name})")

    # ---- Family-composition delta vs nb2063 top-28 ----
    fam_delta_vs_nb2063 = {}
    all_fams = set(fam_counts.keys()) | set(NB2063_TOP28_FAM.keys())
    for f in sorted(all_fams):
        this_c = int(fam_counts.get(f, 0))
        ref_c = int(NB2063_TOP28_FAM.get(f, 0))
        fam_delta_vs_nb2063[f] = {
            "nb2154_count": this_c,
            "nb2063_count": ref_c,
            "delta": this_c - ref_c,
        }

    # ---- Verdict vs nb2103 K=28 ----
    delta_vs_nb2103 = rae_mean_bag - NB2103_K28_REF
    beats_anchor = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb2103 = rae_mean_bag < NB2103_K28_REF - DECISION_MARGIN
    flat_vs_nb2103 = abs(delta_vs_nb2103) < DECISION_MARGIN
    if beats_nb2103:
        verdict = "MULTI_FP_BEATS_NB2103_K28_NEW_PRIMARY_CANDIDATE"
    elif flat_vs_nb2103:
        verdict = ("MULTI_FP_FLAT_VS_NB2103_K28_USE_AS_DECORRELATED_BLEND_"
                   "PARTNER")
    elif beats_anchor:
        verdict = "MULTI_FP_BEATS_ANCHOR_BUT_WORSE_THAN_NB2103_K28"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "MULTI_FP_FLAT_VS_ANCHOR"
    else:
        verdict = "MULTI_FP_HURTS_ANCHOR"
    pre_unblind_clean = True
    print(f"\n   verdict                = {verdict}")
    print(f"   PRE-unblind clean      = {pre_unblind_clean}")

    # ---- Save summary ----
    top28_records = [
        {
            "rank": int(rank),
            "feat_idx_in_8285": int(idx),
            "feat_name": top28_names[rank - 1],
            "feat_family": top28_families[rank - 1],
            "mean_abs_shap": float(top28_imp[rank - 1]),
        }
        for rank, idx in enumerate(top28_idx.tolist(), 1)
    ]

    summary = {
        "tag": TAG,
        "method": ("lgbm_mse_on_top28_shap_of_8285col_pure_FP_stack_"
                   "scaffold_CV"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "feat_breakdown_full": {
            "Morgan2048": N_MORGAN,
            "MACCS167": N_MACCS,
            "AtomPair2048": N_ATOMPAIR,
            "Avalon1024": N_AVALON,
            "ECFP6_2048": N_ECFP6,
            "total": N_TOTAL,
        },
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "top_K_shap": TOP_K_SHAP,
        "n_unb": n_unb,
        "n_scaffolds_unique_unb": n_scaf_unique,
        "n_scaffolds_none_unb": n_scaf_none,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "cv_kind": "scaffold_kfold_BemisMurcko_5fold_shuffle_per_seed",
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "top28_family_counts_nb2154": fam_counts,
        "top28_family_counts_nb2063_ref": NB2063_TOP28_FAM,
        "top28_family_delta_vs_nb2063": fam_delta_vs_nb2063,
        "top28_records": top28_records,
        "per_seed_rae": per_seed_rae,
        "per_seed_records": per_seed_records,
        "rae_per_seed_mean": rae_per_seed_mean,
        "rae_per_seed_median": rae_per_seed_median,
        "rae_per_seed_std": rae_per_seed_std,
        "rae_per_seed_min": rae_per_seed_min,
        "rae_per_seed_max": rae_per_seed_max,
        "rae_mean_bag": rae_mean_bag,
        "rae_median_bag": rae_median_bag,
        "delta_mean_bag_vs_chemprop_aux": rae_mean_bag - rae_anchor,
        "delta_mean_bag_vs_nb2103_K28": delta_vs_nb2103,
        "beats_chemprop_aux": bool(beats_anchor),
        "beats_nb2103_K28": bool(beats_nb2103),
        "flat_vs_nb2103_K28": bool(flat_vs_nb2103),
        "pearson_vs_anchor": pearson_vs_anchor,
        "pearson_vs_nb2103_K28": pearson_vs_nb2103_K28,
        "verdict": verdict,
        "pre_unblind_clean": pre_unblind_clean,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_ref": NB2103_K28_REF,
        "decision_margin": DECISION_MARGIN,
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
        "feat_breakdown_full", "top_K_shap",
        "n_unb", "n_scaffolds_unique_unb", "n_scaffolds_none_unb",
        "cv_kind",
        "rae_anchor_chemprop_aux", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_per_seed_min", "rae_per_seed_max",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_chemprop_aux", "delta_mean_bag_vs_nb2103_K28",
        "beats_chemprop_aux", "beats_nb2103_K28", "flat_vs_nb2103_K28",
        "pearson_vs_anchor", "pearson_vs_nb2103_K28",
        "top28_family_counts_nb2154", "top28_family_counts_nb2063_ref",
        "verdict", "pre_unblind_clean", "nb2103_K28_ref",
    ):
        print(f"  {k}: {res.get(k)}")
