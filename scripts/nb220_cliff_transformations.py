"""nb220 -- Activity cliff transformation mining.

For each MMP cliff in our training set (10 pairs with |ΔpEC50|>1.0, Tan>=0.7),
identify the SMIRKS transformation (R-group swap). Then search external NR
datasets and our own train MMPs (looser threshold) for the SAME transformation
and compute the empirical mean ΔpEC50 for that motif change.

This is the user's "chain-of-thought analogy" approach for cliffs: zoom out
to find structurally identical transformations elsewhere, learn what they
typically do to activity, then apply that knowledge to predict our test
compounds when they sit on the cliff side of an MMP with a training compound.

Use cases:
  1. Test compound + training analog form an MMP with a known transformation
     → predict pEC50 = anchor_pEC50 + transformation_prior_delta
  2. Use as auxiliary feature: "expected delta from nearest MMP transformation"
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdMMPA, AllChem, DataStructs

from pxr.data import load_train, load_test
from pxr.chem import add_standard_columns
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

import lightgbm as lgb

COLLAPSE_THRESH = 0.58

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)


def get_mmp_transforms(df, val_col="pec50", min_cores=10):
    """For each compound, fragment via single-cut MMPA, group by core, return transform pairs.

    Returns: list of (core_smiles, smi_A, smi_B, R_A, R_B, val_A, val_B, delta)
    """
    print(f"  Fragmenting {len(df):,} compounds for MMP...")
    core_to_rgroups = defaultdict(list)  # core -> [(smi, R-group, val), ...]

    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue
            frags = rdMMPA.FragmentMol(mol, resultsAsMols=False)  # default maxCuts=3
        except Exception:
            continue
        for frag in frags:
            if not isinstance(frag, tuple) or len(frag) != 2:
                continue
            core, rg = frag
            # core can be empty string (terminal cut); rg is always non-empty if frag is valid
            if rg:
                core_key = core if core else "*TERMINAL*"
                core_to_rgroups[core_key].append((row["smiles"], rg, row[val_col]))

    print(f"  Distinct cores: {len(core_to_rgroups):,}")

    # Heuristic caps to control memory:
    #   - skip terminal cores (everything fragments to a terminal at some cut, far too noisy)
    #   - skip cores with > MAX_RGS R-groups (overly promiscuous → low signal pairs)
    #   - cap total transforms collected
    MAX_RGS = 12
    MAX_TRANSFORMS = 2_000_000

    transforms = []
    skipped_terminal = skipped_large = 0
    for core, rgs in core_to_rgroups.items():
        if core == "*TERMINAL*":
            skipped_terminal += 1
            continue
        if len(rgs) < 2:
            continue
        if len(rgs) > MAX_RGS:
            skipped_large += 1
            continue
        for i in range(len(rgs)):
            for j in range(i+1, len(rgs)):
                smi_a, rg_a, v_a = rgs[i]
                smi_b, rg_b, v_b = rgs[j]
                if rg_a == rg_b:
                    continue
                transforms.append((core, smi_a, smi_b, rg_a, rg_b, v_a, v_b, v_b - v_a))
                if len(transforms) >= MAX_TRANSFORMS:
                    break
            if len(transforms) >= MAX_TRANSFORMS:
                break
        if len(transforms) >= MAX_TRANSFORMS:
            break

    print(f"  Skipped terminal cores: {skipped_terminal:,}")
    print(f"  Skipped over-{MAX_RGS}-R-group cores: {skipped_large:,}")
    print(f"  Distinct MMP transforms: {len(transforms):,}")
    return transforms


def build_transform_prior(transforms):
    """For each (R_A->R_B) transform, compute mean and median Δ across all cores."""
    rg_pair_deltas = defaultdict(list)
    for core, sa, sb, ra, rb, va, vb, d in transforms:
        # symmetric key (ra, rb) sorted
        key = tuple(sorted([ra, rb]))
        # signed delta: positive if rb has higher activity
        sign = 1 if ra <= rb else -1
        rg_pair_deltas[key].append(d * sign)

    prior = {}
    for key, deltas in rg_pair_deltas.items():
        if len(deltas) >= 3:
            prior[key] = {
                "mean": np.mean(deltas),
                "median": np.median(deltas),
                "std": np.std(deltas),
                "count": len(deltas),
            }
    return prior


def compute_test_cliff_feature(test_smiles, train_df, transform_prior, k=3):
    """For each test compound, find k closest training analogs via MMPA,
    look up R-group transformations in the prior, return expected delta."""
    print(f"  Computing cliff features for {len(test_smiles)} test compounds...")

    # Fragment all training compounds, index by core
    train_core_index = defaultdict(list)
    for idx, row in train_df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row["smiles"])
            if mol is None:
                continue
            frags = rdMMPA.FragmentMol(mol, resultsAsMols=False)
            for frag in frags:
                if not isinstance(frag, tuple) or len(frag) != 2:
                    continue
                core, rg = frag
                if rg:
                    core_key = core if core else "*TERMINAL*"
                    train_core_index[core_key].append((row["smiles"], rg, row["pec50"]))
        except Exception:
            continue

    expected_deltas = np.full(len(test_smiles), 0.0)
    expected_anchors = np.full(len(test_smiles), np.nan)
    n_matched = 0

    for i, smi in enumerate(test_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        try:
            frags = rdMMPA.FragmentMol(mol, resultsAsMols=False)
        except Exception:
            continue

        deltas, anchors = [], []
        for frag in frags:
            if not isinstance(frag, tuple) or len(frag) != 2:
                continue
            core, rg_q = frag
            if not rg_q: continue
            core_key = core if core else "*TERMINAL*"
            # Find train analogs with same core
            analogs = train_core_index.get(core_key, [])
            for tr_smi, tr_rg, tr_val in analogs:
                if tr_rg == rg_q:
                    # exact match → anchor pEC50 directly
                    anchors.append(tr_val)
                    deltas.append(0.0)
                else:
                    key = tuple(sorted([rg_q, tr_rg]))
                    if key in transform_prior:
                        sign = 1 if tr_rg <= rg_q else -1
                        deltas.append(transform_prior[key]["median"] * sign)
                        anchors.append(tr_val)

        if anchors:
            anchors = np.array(anchors)
            deltas = np.array(deltas)
            preds = anchors + deltas
            # Use top-k closest by sim... here just take median (anchors share core)
            expected_anchors[i] = np.median(preds)
            expected_deltas[i]  = np.median(deltas)
            n_matched += 1

    print(f"  {n_matched}/{len(test_smiles)} test compounds matched a training core")
    return expected_anchors, expected_deltas, n_matched


def main():
    print("=== nb220: Activity cliff transformation mining ===\n")

    tr = load_train()
    te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    oof_base = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_base  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    base_rae = rae(y_tr, oof_base)
    print(f"Base nb197: OOF RAE={base_rae:.4f}\n")

    # ── A. Build transformation prior from training MMPs ──────────────────────
    print("[A] Building MMP transformation prior from PXR training set...")
    train_simple = tr[["std_smiles", "pec50"]].rename(columns={"std_smiles": "smiles"})
    train_transforms = get_mmp_transforms(train_simple)
    prior_train = build_transform_prior(train_transforms)
    print(f"  Prior covers {len(prior_train):,} R-group pairs (n>=3)\n")

    # ── B. Augment prior with external NR data ────────────────────────────────
    print("[B] Adding external NR MMPs to prior...")
    nr = pd.read_parquet(DATA_EXTERNAL / "chembl_nr_extended.parquet")
    nr_simple = nr[["std_smiles", "pec50"]].dropna().rename(columns={"std_smiles": "smiles"})
    nr_transforms = get_mmp_transforms(nr_simple)
    prior_nr = build_transform_prior(nr_transforms)
    print(f"  External NR prior: {len(prior_nr):,} R-group pairs")

    # Merge: keep train prior if available, fall back to NR
    combined_prior = dict(prior_train)
    for k, v in prior_nr.items():
        if k not in combined_prior:
            combined_prior[k] = v
        else:
            # Weighted: train counts as 3x external
            n_tr = combined_prior[k]["count"]
            n_ex = v["count"]
            combined_prior[k] = {
                "mean": (3 * n_tr * combined_prior[k]["mean"] + n_ex * v["mean"]) / (3*n_tr + n_ex),
                "median": combined_prior[k]["median"],  # keep train median
                "std": combined_prior[k]["std"],
                "count": n_tr + n_ex,
            }
    print(f"  Merged prior: {len(combined_prior):,} R-group pairs\n")

    # ── C. Compute cliff features for test set ────────────────────────────────
    print("[C] Computing test cliff features...")
    test_pred_from_transform, test_delta, n_matched = compute_test_cliff_feature(
        smiles_te, train_simple, combined_prior)
    print(f"  Matched {n_matched}/{len(smiles_te)} test compounds")
    finite_te = np.isfinite(test_pred_from_transform)
    print(f"  Transform-based test predictions: {finite_te.sum()} valid")
    if finite_te.sum() > 0:
        print(f"    range: [{test_pred_from_transform[finite_te].min():.2f}, "
              f"{test_pred_from_transform[finite_te].max():.2f}]")
        print(f"    mean ± std: {test_pred_from_transform[finite_te].mean():.2f} ± "
              f"{test_pred_from_transform[finite_te].std():.2f}")
    else:
        print("    (no valid transform-based predictions)")

    # ── D. Compute cliff features for train (in-fold validity check) ──────────
    print("\n[D] Computing train cliff features (out-of-fold)...")
    # For each fold, use only training compounds NOT in val fold to build prior
    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    oof_transform_pred = np.full(len(y_tr), np.nan)
    oof_transform_delta = np.zeros(len(y_tr))

    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        fold_tr = train_simple.iloc[tr_idx].reset_index(drop=True)
        fold_transforms = get_mmp_transforms(fold_tr)
        fold_prior = build_transform_prior(fold_transforms)
        # Merge with external NR
        merged = dict(fold_prior)
        for k, v in prior_nr.items():
            if k not in merged: merged[k] = v
        # Apply to validation compounds
        va_smiles = train_simple.iloc[va_idx]["smiles"].tolist()
        va_preds, va_deltas, n_m = compute_test_cliff_feature(va_smiles, fold_tr, merged)
        oof_transform_pred[va_idx] = va_preds
        oof_transform_delta[va_idx] = va_deltas
        print(f"  fold {fold_idx+1}: {n_m}/{len(va_idx)} validation compounds matched")

    finite_oof = np.isfinite(oof_transform_pred)
    print(f"\n  Transform OOF coverage: {finite_oof.sum()}/{len(y_tr)} ({finite_oof.mean()*100:.1f}%)")
    if finite_oof.sum() > 100:
        from scipy.stats import spearmanr
        rho = spearmanr(y_tr[finite_oof], oof_transform_pred[finite_oof]).correlation
        mae_t = np.mean(np.abs(y_tr[finite_oof] - oof_transform_pred[finite_oof]))
        print(f"  Transform OOF: Spearman ρ = {rho:.3f}  MAE = {mae_t:.4f}")

    # ── E. Use as feature in augmented LGBM ───────────────────────────────────
    print("\n[E] Augmented LGBM with transform features...")
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)

    # Fill NaN with column median
    oof_pred_filled = np.where(np.isfinite(oof_transform_pred), oof_transform_pred,
                                np.nanmedian(oof_transform_pred))
    te_pred_filled = np.where(np.isfinite(test_pred_from_transform), test_pred_from_transform,
                                np.nanmedian(oof_transform_pred))

    X_tr_aug = np.column_stack([X_tr_base, oof_pred_filled, oof_transform_delta])
    X_te_aug = np.column_stack([X_te_base, te_pred_filled, test_delta])

    cv_results = {}
    for name, X_tr_use, X_te_use in [
        ("base_only",  X_tr_base, X_te_base),
        ("cliff_aug",  X_tr_aug,  X_te_aug),
    ]:
        oof = np.zeros(len(y_tr))
        te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(X_tr_use[tr_idx], y_tr[tr_idx],
                  eval_set=[(X_tr_use[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(X_tr_use[va_idx])
            te_preds.append(m.predict(X_te_use))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof)
        ratio = te_pred.std() / oof.std()
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {name:12s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  [{flag}]")
        cv_results[name] = (oof, te_pred, r, ratio)

    # ── F. Blend with nb197 ──────────────────────────────────────────────────
    oof_aug, te_aug, r_aug, ratio_aug = cv_results["cliff_aug"]
    print("\n[F] Blend with nb197:")
    best_blend, best_r_bl = None, 999
    for w in np.arange(0.05, 0.75, 0.05):
        oof_bl = (1-w)*oof_base + w*oof_aug
        te_bl  = (1-w)*te_base  + w*te_aug
        r_bl   = rae(y_tr, oof_bl)
        ratio_bl = te_bl.std() / oof_bl.std()
        flag = "PASS" if ratio_bl >= COLLAPSE_THRESH else "FAIL"
        print(f"  w={w:.2f}: OOF={r_bl:.4f}  ratio={ratio_bl:.3f}  [{flag}]")
        if ratio_bl >= COLLAPSE_THRESH and r_bl < best_r_bl:
            best_r_bl = r_bl; best_blend = (w, oof_bl, te_bl, ratio_bl)

    saved = []
    if ratio_aug >= COLLAPSE_THRESH and r_aug < base_rae:
        np.save(DATA_PROCESSED / "oof_nb220_cliff_aug.npy", oof_aug)
        np.save(DATA_PROCESSED / "te_nb220_cliff_aug.npy", te_aug)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_aug})
        sub.to_csv(SUBMISSIONS / "220_cliff_aug.csv", index=False)
        saved.append(f"220_cliff_aug OOF={r_aug:.4f}")

    if best_blend and best_r_bl < base_rae:
        w_b, oof_b, te_b, ratio_b = best_blend
        name = f"220_blend_w{int(w_b*100):02d}"
        np.save(DATA_PROCESSED / f"oof_{name}.npy", oof_b)
        np.save(DATA_PROCESSED / f"te_{name}.npy", te_b)
        sub = pd.DataFrame({"Molecule Name": te_df["name"], "pEC50": te_b})
        sub.to_csv(SUBMISSIONS / f"{name}.csv", index=False)
        saved.append(f"{name} OOF={best_r_bl:.4f}")

    print(f"\n=== Saved: {saved or ['none']}")


if __name__ == "__main__":
    main()
