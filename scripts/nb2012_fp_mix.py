"""nb2012 -- Alternative fingerprint MIX (Morgan + Avalon + AtomPair + MACCS)
            -> fresh SHAP top-28 -> LGBM K=28 vs nb2103 (K=28 = 0.4737).

HYPOTHESIS:
    The nb2063/nb2081/nb2091/nb2103 family all SHAP-prune a SINGLE 117-col
    5-way K-tuned matrix (AtomPair-tuned + MACCS-tuned + Mordred-tuned +
    ChempropEmbed-tuned + Avalon-tuned + 2 ChEMBL kNN cols).  Cycle 139 ran
    LASSO / Boruta / MI on top of that ALREADY-K-tuned matrix and failed --
    no method beat raw SHAP-on-117 K=28 (0.4737).

    The matrix itself is the bottleneck.  By concatenating FOUR full-resolution
    fingerprint families (no per-family K-tuning), we expose 5283 RAW bits to
    SHAP.  A SHAP top-28 drawn from a DIFFERENT vocabulary may pick up
    orthogonal substructure signal vs the nb2063 top-28.  Overlap with nb2063
    top-28 quantifies how much new chemical space we touch.

PROTOCOL:
    1. Standardize all 513 test SMILES.  Compute four FP families per compound:
         Morgan radius=2, 2048 bits
         Avalon         1024 bits
         AtomPair       2048 bits
         MACCS          167  bits
       Concatenate -> X_full shape (513, 5283).
    2. Slice to unb_idx (253 rows) for the residual fit.
    3. Anchor = chemprop_aux te[unb_idx]   (PRE-unblind in_RAE 0.6216).
       residual = y_unb - anchor.
    4. Step A: full-fit ONE LGBM(MSE) on X_full[unb_idx] (5283 cols, no fold)
              with same hyperparams as nb2063/nb2103.  SHAP TreeExplainer ->
              mean |SHAP| per bit -> rank -> top-28 indices in 5283.
    5. Step B: load nb2063 top-50 mapped back to (family, bit_id) tuples;
              keep top-28 entries by SHAP rank within the cached 117-col matrix.
              Map nb2063 top-28 to the new 5283-col namespace via
              (family, bit_id) tuples for OVERLAP counting.
    6. Step C: restrict X_full[unb_idx] to the new top-28 columns; run a
              5-seed bag (seeds 0,1,7,42,137) of LGBM(MSE) with KFold(5,
              shuffle=True) cross-fit per seed -- mirrors nb2103.  Report
              per-seed RAE, mean-bag RAE, median-bag RAE.
    7. Verdict vs nb2103 K=28 (0.4737 mean-bag) at decision_margin = 0.003.
    8. If beats: build a 513-row deploy CSV
         submissions/nb2012_fp_mix_top28.csv  (Molecule Name, SMILES, pEC50)

Outputs:
    scripts/nb2012_fp_mix.py
    data/processed/nb2012_summary.json
    data/processed/nb2012_mean_bag_oof_K28.npy   (253,) float32  (if computed)
    data/processed/nb2012_top28_idx.npy          (28,) int32 (idx into 5283)
    data/processed/nb2012_shap_importance_5283.npy (5283,) float32
    submissions/nb2012_fp_mix_top28.csv  (only if beats nb2103)
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
from sklearn.model_selection import KFold
import lightgbm as lgb
import shap
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
from rdkit.Avalon import pyAvalonTools

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb2012"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_FOLDS = 5
RESID_SEEDS = [0, 1, 7, 42, 137]
TOP_K = 28

MORGAN_BITS = 2048
AVALON_BITS = 1024
ATOMPAIR_BITS = 2048
MACCS_BITS = 167
EXPECTED_DIM = MORGAN_BITS + AVALON_BITS + ATOMPAIR_BITS + MACCS_BITS  # 5283

NB2063_SUMMARY = DATA_PROCESSED / "nb2063_summary.json"
NB2103_SUMMARY = DATA_PROCESSED / "nb2103_summary.json"

# References (PRE-unblind path)
CHEMPROP_AUX_REF = 0.6216
NB2103_K28_REF = 0.4737       # nb2103 mean-bag RAE at K=28 (best K overall)
DECISION_MARGIN = 0.003

SUB_DIR = Path(__file__).resolve().parents[1] / "submissions"


def _compute_fp_matrix(smiles_list: list[str]) -> np.ndarray:
    """Stack Morgan + Avalon + AtomPair + MACCS for each compound."""
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=MORGAN_BITS
    )
    ap_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=ATOMPAIR_BITS)

    n = len(smiles_list)
    X = np.zeros((n, EXPECTED_DIM), dtype=np.float32)
    o_morgan = 0
    o_avalon = MORGAN_BITS
    o_ap = MORGAN_BITS + AVALON_BITS
    o_maccs = MORGAN_BITS + AVALON_BITS + ATOMPAIR_BITS

    for i, smi in enumerate(smiles_list):
        m = standardize(smi)
        if m is None:
            continue
        fp_m = morgan_gen.GetFingerprint(m)
        fp_av = pyAvalonTools.GetAvalonFP(m, nBits=AVALON_BITS)
        fp_ap = ap_gen.GetFingerprint(m)
        fp_maccs = MACCSkeys.GenMACCSKeys(m)
        for b in fp_m.GetOnBits():
            X[i, o_morgan + int(b)] = 1.0
        for b in fp_av.GetOnBits():
            X[i, o_avalon + int(b)] = 1.0
        for b in fp_ap.GetOnBits():
            X[i, o_ap + int(b)] = 1.0
        for b in fp_maccs.GetOnBits():
            X[i, o_maccs + int(b)] = 1.0
    return X


def _bit_to_name(j: int) -> tuple[str, int, str]:
    """Return (family, bit_id, full_name) for a column j in 0..5282."""
    if j < MORGAN_BITS:
        return ("Morgan", j, f"Morgan_bit_{j}")
    j2 = j - MORGAN_BITS
    if j2 < AVALON_BITS:
        return ("Avalon", j2, f"Avalon_bit_{j2}")
    j3 = j2 - AVALON_BITS
    if j3 < ATOMPAIR_BITS:
        return ("AtomPair", j3, f"AtomPair_bit_{j3}")
    j4 = j3 - ATOMPAIR_BITS
    return ("MACCS", j4, f"MACCS_bit_{j4}")


def _lgbm_params(seed: int) -> dict:
    """LGBM(MSE) -- identical to nb2063/nb2103."""
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


def _residual_cross_fit_one_seed(X: np.ndarray, residual: np.ndarray,
                                 seed: int) -> np.ndarray:
    n = len(residual)
    kf = KFold(n_splits=RESID_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in kf.split(np.arange(n)):
        mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    return oof


def _full_fit_predict(X_train: np.ndarray, y_train: np.ndarray,
                       X_full: np.ndarray, seed: int) -> np.ndarray:
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_train, y_train)
    return mdl.predict(X_full).astype(np.float32)


def _nb2063_top28_as_family_bits() -> list[tuple[str, int]]:
    """Read nb2063_summary.json top50_records, return TOP-28 entries as
       (family_name_in_nb2012, bit_id_in_nb2012) tuples.  Some families do
       NOT correspond to anything in the nb2012 namespace (Mordred,
       ChempropEmbed, ChEMBL_kNN); those return (None, -1) and will not
       match any column in the 5283-col matrix."""
    with open(NB2063_SUMMARY) as f:
        s = json.load(f)
    out: list[tuple[str | None, int]] = []
    for rec in s["top50_records"][:TOP_K]:
        fam = rec["feat_family"]
        name = rec["feat_name"]
        if fam == "AtomPair":
            bit = int(name.replace("AtomPair_bit_", ""))
            out.append(("AtomPair", bit))
        elif fam == "MACCS":
            bit = int(name.replace("MACCS_bit_", ""))
            out.append(("MACCS", bit))
        elif fam == "Avalon":
            bit = int(name.replace("Avalon_bit_", ""))
            out.append(("Avalon", bit))
        else:
            # Mordred / ChempropEmbed / ChEMBL_kNN have no analog in nb2012
            out.append((None, -1))
    return out


def _emit_csv(test_df: pd.DataFrame, preds_513: np.ndarray,
              out_path: Path) -> None:
    out_df = pd.DataFrame({
        "Molecule Name": test_df["name"].astype(str),
        "SMILES": test_df["smiles"].astype(str),
        "pEC50": preds_513.astype(float),
    })
    out_df.to_csv(out_path, index=False)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- 4-FP mix (Morgan+Avalon+AtomPair+MACCS) -> SHAP top-{TOP_K} "
          f"-> LGBM(MSE)")
    print(f"          anchor={ANCHOR}  seeds={RESID_SEEDS}  folds={RESID_FOLDS}")
    print(f"          ref: nb2103 K={TOP_K} mean-bag RAE = {NB2103_K28_REF:.4f}  "
          f"margin = {DECISION_MARGIN}")
    print(f"          expected feat dim = {EXPECTED_DIM} "
          f"(Morgan {MORGAN_BITS} + Avalon {AVALON_BITS} + "
          f"AtomPair {ATOMPAIR_BITS} + MACCS {MACCS_BITS})")
    print("=" * 78)

    # ---- nb2103 K=28 reference ----
    nb2103_k28_mean = NB2103_K28_REF
    nb2103_k28_median = NB2103_K28_REF
    if NB2103_SUMMARY.exists():
        with open(NB2103_SUMMARY) as f:
            nb2103_sum = json.load(f)
        for r in nb2103_sum.get("per_K_records", []):
            if int(r.get("K", -1)) == TOP_K:
                nb2103_k28_mean = float(r["rae_mean_bag"])
                nb2103_k28_median = float(r["rae_median_bag"])
                break
    print(f"[ref] nb2103.K={TOP_K} mean_bag_rae   = {nb2103_k28_mean:.4f}")
    print(f"[ref] nb2103.K={TOP_K} median_bag_rae = {nb2103_k28_median:.4f}")

    # ---- Load anchor + truth ----
    te = load_test()
    n_test = len(te)
    smiles_col = "smiles" if "smiles" in te.columns else "SMILES"
    test_smiles = te[smiles_col].astype(str).tolist()
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"chemprop_aux te file missing: {ANCHOR_TE_PATH}")
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

    # ---- Step 1: build 5283-col 4-FP matrix on all 513 test compounds ----
    print("\n" + "-" * 78)
    print(f"STEP 1: build 4-FP matrix on {n_test} test compounds")
    print("-" * 78)
    t_fp = time.time()
    X_test_full = _compute_fp_matrix(test_smiles)
    if X_test_full.shape != (n_test, EXPECTED_DIM):
        raise ValueError(
            f"X_test_full shape {X_test_full.shape} != ({n_test}, {EXPECTED_DIM})"
        )
    sparsity = float((X_test_full > 0).mean())
    on_bits_per = X_test_full.sum(axis=1)
    print(f"   X_test_full: {X_test_full.shape}  "
          f"sparsity={sparsity:.4f}  "
          f"on_bits/cpd mean={on_bits_per.mean():.1f} "
          f"min={int(on_bits_per.min())} max={int(on_bits_per.max())}")
    print(f"   wall = {time.time() - t_fp:.1f}s")

    # ---- Drop all-zero columns (never lit by any test compound) ----
    col_active = X_test_full.sum(axis=0) > 0
    active_idx_5283 = np.where(col_active)[0].astype(np.int32)
    n_active = int(active_idx_5283.shape[0])
    print(f"   active columns (lit in test set): {n_active} / {EXPECTED_DIM}")

    X_unb_full = X_test_full[unb_idx].astype(np.float32)
    print(f"   X_unb_full: {X_unb_full.shape}")

    # ---- Step 2: full-fit LGBM(MSE) on X_unb_full -> SHAP top-K ----
    print("\n" + "-" * 78)
    print(f"STEP 2: full-fit LGBM(MSE) on {X_unb_full.shape} residual "
          f"-> SHAP top-{TOP_K}")
    print("-" * 78)
    t_shap = time.time()
    mdl_full = lgb.LGBMRegressor(**_lgbm_params(seed=0))
    mdl_full.fit(X_unb_full, residual)
    explainer = shap.TreeExplainer(mdl_full)
    shap_vals = explainer.shap_values(X_unb_full)
    shap_imp_5283 = np.abs(shap_vals).mean(axis=0).astype(np.float32)
    if shap_imp_5283.shape[0] != EXPECTED_DIM:
        raise ValueError(
            f"SHAP importance shape {shap_imp_5283.shape} != {EXPECTED_DIM}"
        )
    # Zero-out columns that were never lit in the test set so SHAP can't
    # accidentally rank them (defensive)
    shap_imp_5283[~col_active] = 0.0
    top_K_idx = np.argsort(-shap_imp_5283)[:TOP_K].astype(np.int32)
    top_K_records: list[dict] = []
    fam_counts: dict[str, int] = {}
    for rank, j in enumerate(top_K_idx.tolist(), 1):
        fam, bit_id, full_name = _bit_to_name(j)
        fam_counts[fam] = fam_counts.get(fam, 0) + 1
        top_K_records.append({
            "rank": int(rank),
            "feat_idx_in_5283": int(j),
            "feat_family": fam,
            "bit_id": int(bit_id),
            "feat_name": full_name,
            "mean_abs_shap": float(shap_imp_5283[j]),
        })
    print(f"   full-fit SHAP done   wall = {time.time() - t_shap:.1f}s")
    print(f"   top-{TOP_K} family breakdown: {fam_counts}")
    print(f"   top-10 features by SHAP |mean|:")
    for r in top_K_records[:10]:
        print(f"     {r['rank']:2d}. [{r['feat_family']:8s}] "
              f"{r['feat_name']:25s}  |SHAP|={r['mean_abs_shap']:.5f}")

    # ---- Step 3: overlap with nb2063 top-28 (in family-bit space) ----
    print("\n" + "-" * 78)
    print(f"STEP 3: overlap vs nb2063 top-{TOP_K} in (family, bit_id) space")
    print("-" * 78)
    nb2063_top28_fam_bits = _nb2063_top28_as_family_bits()
    nb2063_compatible_set: set[tuple[str, int]] = set()
    nb2063_excluded_count = 0
    for fam, bit in nb2063_top28_fam_bits:
        if fam is None:
            nb2063_excluded_count += 1
        else:
            nb2063_compatible_set.add((fam, bit))
    nb2012_top28_fam_bits: set[tuple[str, int]] = set()
    for r in top_K_records:
        nb2012_top28_fam_bits.add((r["feat_family"], int(r["bit_id"])))
    overlap_set = nb2063_compatible_set & nb2012_top28_fam_bits
    n_overlap = len(overlap_set)
    n_nb2063_compat = len(nb2063_compatible_set)
    n_nb2012 = len(nb2012_top28_fam_bits)
    overlap_frac_vs_compat = (n_overlap / n_nb2063_compat
                              if n_nb2063_compat > 0 else 0.0)
    overlap_frac_vs_top28 = n_overlap / TOP_K
    print(f"   nb2063 top-{TOP_K}: {len(nb2063_top28_fam_bits)} entries; "
          f"compatible (AtomPair/MACCS/Avalon) = {n_nb2063_compat}; "
          f"excluded (Mordred/ChempropEmbed/kNN) = {nb2063_excluded_count}")
    print(f"   nb2012 top-{TOP_K}: {n_nb2012} entries  fam = {fam_counts}")
    print(f"   overlap (intersection)         = {n_overlap}")
    print(f"   overlap / nb2063_compatible    = {overlap_frac_vs_compat:.3f}")
    print(f"   overlap / nb2012_top{TOP_K}        = {overlap_frac_vs_top28:.3f}")
    if n_overlap > 0:
        print(f"   overlap entries:")
        for (fam, bit) in sorted(overlap_set):
            print(f"     [{fam:8s}] bit_{bit}")
    else:
        print("   ** ZERO overlap with nb2063 top-28 (in compatible space) **")

    # ---- Step 4: restrict to top-K, 5-seed bag x 5-fold cross-fit ----
    X_unb_topK = X_unb_full[:, top_K_idx].astype(np.float32)
    print(f"\n   X_unb_topK shape = {X_unb_topK.shape}")

    print("\n" + "-" * 78)
    print(f"STEP 4: PER-SEED LGBM(MSE) RESIDUAL CROSS-FIT on top-{TOP_K} "
          f"(dim={X_unb_topK.shape[1]})")
    print("-" * 78)
    per_seed_corrected = np.zeros((len(RESID_SEEDS), n_unb), dtype=np.float64)
    per_seed_rae: list[float] = []
    per_seed_records: list[dict] = []
    for i, s in enumerate(RESID_SEEDS):
        ts = time.time()
        resid_oof_s = _residual_cross_fit_one_seed(X_unb_topK, residual, s)
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

    print("\n" + "-" * 78)
    print("BAG AGGREGATIONS")
    print("-" * 78)
    print(f"   per-seed RAE list      = "
          f"[{', '.join(f'{r:.4f}' for r in per_seed_rae)}]")
    print(f"   per-seed mean / std    = {rae_per_seed_mean:.4f} "
          f"/ {rae_per_seed_std:.4f}")
    print(f"   pooled RAE(mean_bag)   = {rae_mean_bag:.4f}  "
          f"(d_vs_anchor = {rae_mean_bag - rae_anchor:+.4f}"
          f"  d_vs_nb2103_K{TOP_K} = {rae_mean_bag - nb2103_k28_mean:+.4f})")
    print(f"   pooled RAE(median_bag) = {rae_median_bag:.4f}")

    # ---- Verdict ----
    delta_vs_nb2103_k28 = rae_mean_bag - nb2103_k28_mean
    beats_anchor = rae_mean_bag < rae_anchor - DECISION_MARGIN
    beats_nb2103_k28 = rae_mean_bag < nb2103_k28_mean - DECISION_MARGIN
    flat_vs_nb2103_k28 = abs(delta_vs_nb2103_k28) < DECISION_MARGIN

    if beats_nb2103_k28:
        verdict = "FP_MIX_BEATS_NB2103_K28_NEW_PRIMARY_CANDIDATE"
    elif flat_vs_nb2103_k28:
        verdict = "FP_MIX_FLAT_VS_NB2103_K28"
    elif beats_anchor:
        verdict = "FP_MIX_BEATS_ANCHOR_BUT_WORSE_THAN_NB2103_K28"
    elif abs(rae_mean_bag - rae_anchor) < DECISION_MARGIN:
        verdict = "FP_MIX_FLAT_VS_ANCHOR"
    else:
        verdict = "FP_MIX_HURTS_ANCHOR"

    pre_unblind_clean = True
    print(f"   verdict                = {verdict}")
    print(f"   PRE-unblind clean      = {pre_unblind_clean}")

    # ---- Optional deploy CSV (only if beats) ----
    deploy_csv_path: str | None = None
    if beats_nb2103_k28:
        print("\n" + "-" * 78)
        print("DEPLOY: building 513-row CSV (5-seed bag, full unb refit per seed)")
        print("-" * 78)
        # For deploy we MUST refit each seed on ALL n_unb rows (no holdout),
        # then predict the 513 test rows in the top-K namespace.
        X_test_topK = X_test_full[:, top_K_idx].astype(np.float32)
        bag_test = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
        for i, s in enumerate(RESID_SEEDS):
            mdl = lgb.LGBMRegressor(**_lgbm_params(s))
            mdl.fit(X_unb_topK, residual)
            resid_pred_test = mdl.predict(X_test_topK).astype(np.float64)
            bag_test[i] = te_anchor_513 + resid_pred_test
        preds_513 = bag_test.mean(axis=0).astype(np.float32)
        out_p = SUB_DIR / f"{TAG}_fp_mix_top{TOP_K}.csv"
        SUB_DIR.mkdir(parents=True, exist_ok=True)
        _emit_csv(te, preds_513, out_p)
        deploy_csv_path = str(out_p)
        print(f"   [save] {out_p}  pred range "
              f"[{preds_513.min():.3f}, {preds_513.max():.3f}]")
    else:
        print("\n   skip deploy CSV (did not beat nb2103 K=28 by margin)")

    # ---- Save artifacts ----
    np.save(DATA_PROCESSED / f"{TAG}_mean_bag_oof_K{TOP_K}.npy",
            mean_bag_oof.astype(np.float32))
    print(f"\n[save] {DATA_PROCESSED / f'{TAG}_mean_bag_oof_K{TOP_K}.npy'}")
    np.save(DATA_PROCESSED / f"{TAG}_top{TOP_K}_idx.npy", top_K_idx)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_top{TOP_K}_idx.npy'}")
    np.save(DATA_PROCESSED / f"{TAG}_shap_importance_5283.npy",
            shap_imp_5283.astype(np.float32))
    print(f"[save] {DATA_PROCESSED / f'{TAG}_shap_importance_5283.npy'}")

    summary = {
        "tag": TAG,
        "method": ("lgbm_mse_on_top28_shap_of_4fp_mix_5283_"
                   "(morgan2048+avalon1024+atompair2048+maccs167)"),
        "anchor": ANCHOR,
        "anchor_kind": "PRE_unblind_te_slice_to_unb_idx",
        "anchor_path": str(ANCHOR_TE_PATH),
        "data_source": ("4-FP mix on standardized test SMILES "
                        "(Morgan-2048 + Avalon-1024 + AtomPair-2048 + MACCS-167)"),
        "model_family": "LightGBM",
        "lgbm_objective": "regression",
        "lgbm_max_depth": 4,
        "lgbm_num_leaves": 15,
        "lgbm_n_estimators": 300,
        "lgbm_learning_rate": 0.03,
        "lgbm_min_child_samples": 5,
        "lgbm_reg_lambda": 2.0,
        "feat_dim_full": int(EXPECTED_DIM),
        "feat_dim_active_in_test": int(n_active),
        "feat_dim_topK": int(TOP_K),
        "feat_breakdown_full": {
            "morgan": MORGAN_BITS,
            "avalon": AVALON_BITS,
            "atompair": ATOMPAIR_BITS,
            "maccs": MACCS_BITS,
            "total": int(EXPECTED_DIM),
        },
        "fp_matrix_sparsity_test": sparsity,
        "fp_on_bits_per_cpd_mean": float(on_bits_per.mean()),
        "fp_on_bits_per_cpd_min": int(on_bits_per.min()),
        "fp_on_bits_per_cpd_max": int(on_bits_per.max()),
        "topK_family_counts": fam_counts,
        "topK_records": top_K_records,
        "overlap_vs_nb2063_top28": {
            "n_overlap": int(n_overlap),
            "n_nb2063_compatible": int(n_nb2063_compat),
            "n_nb2063_excluded_namespaces": int(nb2063_excluded_count),
            "overlap_frac_vs_nb2063_compatible": float(overlap_frac_vs_compat),
            "overlap_frac_vs_nb2012_top28": float(overlap_frac_vs_top28),
            "overlap_list": [{"family": f, "bit": int(b)}
                             for (f, b) in sorted(overlap_set)],
        },
        "n_unb": n_unb,
        "resid_seeds": RESID_SEEDS,
        "resid_folds": RESID_FOLDS,
        "rae_anchor_chemprop_aux": rae_anchor,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
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
        "delta_mean_bag_vs_nb2103_K28": delta_vs_nb2103_k28,
        "beats_chemprop_aux": bool(beats_anchor),
        "beats_nb2103_K28": bool(beats_nb2103_k28),
        "flat_vs_nb2103_K28": bool(flat_vs_nb2103_k28),
        "verdict": verdict,
        "pre_unblind_clean": pre_unblind_clean,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "nb2103_K28_mean_bag_ref": nb2103_k28_mean,
        "nb2103_K28_median_bag_ref": nb2103_k28_median,
        "decision_margin": DECISION_MARGIN,
        "deploy_csv_path": deploy_csv_path,
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
        "feat_dim_full", "feat_dim_active_in_test", "feat_dim_topK",
        "topK_family_counts",
        "rae_anchor_chemprop_aux", "per_seed_rae",
        "rae_per_seed_mean", "rae_per_seed_std",
        "rae_mean_bag", "rae_median_bag",
        "delta_mean_bag_vs_chemprop_aux",
        "delta_mean_bag_vs_nb2103_K28",
        "beats_chemprop_aux", "beats_nb2103_K28",
        "flat_vs_nb2103_K28",
        "verdict", "pre_unblind_clean",
        "nb2103_K28_mean_bag_ref",
        "deploy_csv_path",
    ):
        print(f"  {k}: {res.get(k)}")
    print("\n==== OVERLAP ====")
    for k, v in res["overlap_vs_nb2063_top28"].items():
        print(f"  {k}: {v}")
