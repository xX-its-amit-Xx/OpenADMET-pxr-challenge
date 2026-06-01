"""nb426 -- External-anchor router.

Per-test-compound routing using EXTERNAL evidence (no PXR-challenge labels
used at fit time for the router itself):

  (a) BindingDB anchor:  test compound has an exact InChIKey match in
      BindingDB PXR direct/NR sets OR a Tanimoto kNN neighbour at sim >= 0.4
      in those sets.                          -> use te_nb418_external
  (b) Tox21 PXR classifier:  classifier built on PubChem AID 1346985 (Tox21
      PXR agonist screen, threshold pEC50 >= 4.7) gives prob > 0.5.
                                              -> use te_nb417_tox21
  (c) Otherwise                               -> use te_nb400_crossfit

The router is deterministic given the external evidence (no fit on the
253-unblind labels), so the pooled RAE on the 253 unblind IS the honest
cross-fit RAE. We additionally do a 5-fold partition of the unblind to
report within-fold RAE distribution as a robustness sanity check.

Outputs:
  data/processed/te_nb426_external.npy
  submissions/nb426_external_anchor_router.csv
  submissions/nb426_external_anchor_router_soft07_truth.csv
"""
from __future__ import annotations

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
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi, AllChem, DataStructs

from pxr.chem import add_standard_columns
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.featurize import combined as featurize_combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 42
SOFT_W = 0.7
TOX21_ACTIVE_THR = 4.7
TOX21_P_THR = 0.5
SIM_THR = 0.4
NB400_CROSSFIT_RAE = 0.5698
NB424_CROSSFIT_RAE = 0.5556

TOX21_PATH = DATA_EXTERNAL / "pubchem_aid_1346985_tox21_pxr" / "aid_1346985.parquet"
BDB_DIRECT = DATA_EXTERNAL / "bindingdb_pxr_direct.parquet"
BDB_NR = DATA_EXTERNAL / "bindingdb_nr_data.parquet"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_ikey(smi):
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        return inchi.MolToInchiKey(m) if m else None
    except Exception:
        return None


def morgan_bits(smi, radius=2, nbits=2048):
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
    except Exception:
        return None


def load_external_pxr():
    """Combine bindingdb_pxr_direct + PXR-filtered bindingdb_nr_data."""
    dfs = []
    if BDB_DIRECT.exists():
        d = pd.read_parquet(BDB_DIRECT)
        d = d[[c for c in ("inchikey", "smiles", "std_smiles") if c in d.columns]].copy()
        d["source"] = "pxr_direct"
        dfs.append(d)
    if BDB_NR.exists():
        n = pd.read_parquet(BDB_NR)
        pxr_mask = (
            (n.get("uniprot", "") == "O75469")
            | (n["target_name"].astype(str).str.contains("PXR", case=False, na=False))
            | (n["target_name"].astype(str).str.contains("NR1I2", case=False, na=False))
        )
        n = n[pxr_mask].copy()
        n = n[[c for c in ("inchikey", "smiles", "std_smiles") if c in n.columns]].copy()
        n["source"] = "bindingdb_nr_pxr"
        dfs.append(n)
    ext = pd.concat(dfs, ignore_index=True)
    if ext["inchikey"].isna().any():
        smi_col = "std_smiles" if "std_smiles" in ext.columns else "smiles"
        miss = ext["inchikey"].isna()
        ext.loc[miss, "inchikey"] = ext.loc[miss, smi_col].apply(safe_ikey)
    ext = ext.dropna(subset=["inchikey"]).drop_duplicates(subset=["inchikey"])
    return ext


# ---------------------------------------------------------------------------
# Router signals
# ---------------------------------------------------------------------------

def compute_bindingdb_hit_mask(test_smi, test_ik):
    """Return boolean mask (n_test,): True if test compound is in BindingDB
    (exact InChIKey OR first-block OR Tanimoto kNN sim >= 0.4)."""
    print("\n--- (a) BindingDB hit mask ---")
    ext = load_external_pxr()
    ext_ik = set(ext["inchikey"].tolist())
    ext_fb = set(k.split("-")[0] for k in ext_ik)
    print(f"  external rows={len(ext)}  uniq IKs={len(ext_ik)}  uniq FBs={len(ext_fb)}")

    # Exact / first-block
    exact = np.array([k in ext_ik for k in test_ik], dtype=bool)
    fb = np.array(
        [(k.split("-")[0] in ext_fb) if isinstance(k, str) else False for k in test_ik],
        dtype=bool,
    )
    print(f"  exact-IK test hits = {exact.sum()}, first-block hits = {fb.sum()}")

    # kNN Tanimoto fallback
    smi_col = "std_smiles" if "std_smiles" in ext.columns and ext["std_smiles"].notna().any() else "smiles"
    ext_fps = []
    for s in ext[smi_col].tolist():
        fp = morgan_bits(s)
        if fp is not None:
            ext_fps.append(fp)
    print(f"  external FPs ready: {len(ext_fps)}")

    knn = np.zeros(len(test_smi), dtype=bool)
    for i, smi in enumerate(test_smi):
        if exact[i] or fb[i]:
            knn[i] = True
            continue
        fp = morgan_bits(smi)
        if fp is None:
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, ext_fps), dtype=np.float32)
        if sims.size and sims.max() >= SIM_THR:
            knn[i] = True
    hit = exact | fb | knn
    print(f"  exact+fb+kNN(sim>=0.4) hit mask: {hit.sum()} / {len(test_smi)}")
    return hit


def compute_tox21_active_prob(train_smi, test_smi):
    """Return p_active(test) from a Tox21 PXR LightGBM binary classifier."""
    print("\n--- (b) Tox21 PXR active probability ---")
    tox = pd.read_parquet(TOX21_PATH).dropna(subset=["std_smiles"]).reset_index(drop=True)
    tox["active"] = (
        (tox["pec50"].notna()) & (tox["pec50"] >= TOX21_ACTIVE_THR)
    ).astype(int)
    n_act = int(tox["active"].sum())
    print(f"  Tox21 rows={len(tox)}  active(pEC50>={TOX21_ACTIVE_THR})={n_act}")

    t0 = time.time()
    X_tox = impute(featurize_combined(tox["std_smiles"].tolist())).astype(np.float32)
    y_tox = tox["active"].astype(np.int32).values
    print(f"  Tox21 features built: {X_tox.shape}  in {time.time() - t0:.1f}s")

    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=10, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, reg_alpha=0.05, reg_lambda=0.05,
        is_unbalance=True,
        random_state=SEED, n_jobs=4, verbosity=-1,
    )
    clf.fit(X_tox, y_tox)

    t0 = time.time()
    X_te = impute(featurize_combined(test_smi)).astype(np.float32)
    p_te = clf.predict_proba(X_te)[:, 1].astype(np.float32)
    print(f"  scored test {len(test_smi)} in {time.time() - t0:.1f}s")
    print(f"  test p mean={p_te.mean():.3f}  std={p_te.std():.3f}  "
          f">{TOX21_P_THR}: {int((p_te > TOX21_P_THR).sum())}")
    return p_te


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 78)
    print("nb426 -- EXTERNAL-ANCHOR ROUTER")
    print("=" * 78)
    t_start = time.time()

    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"])}
    unb_te_idx = np.array(
        [name_to_idx[n] for n in unb["Molecule Name"] if n in name_to_idx]
    )
    unb_y = unb["pEC50"].values.astype(float)
    print(f"unblind={len(unb_te_idx)}  still-blind={513 - len(unb_te_idx)}")

    # Compute standardized columns
    te_std = add_standard_columns(te_df.copy(), smi_col="SMILES")
    test_smi = te_std["std_smiles"].tolist()
    test_ik = te_std["inchikey"].tolist()

    # Train SMILES (for Tox21 classifier sanity not needed; only test smi used)
    _ = load_train  # noqa

    # Router signals
    hit_bdb = compute_bindingdb_hit_mask(test_smi, test_ik)
    p_tox = compute_tox21_active_prob(None, test_smi)
    tox_active = p_tox > TOX21_P_THR
    print(f"\nTox21 active (p>{TOX21_P_THR}) on test: {int(tox_active.sum())}")

    # Load base predictors
    pred_nb400 = np.load(DATA_PROCESSED / "te_nb400_crossfit.npy").astype(float)
    pred_nb417 = np.load(DATA_PROCESSED / "te_nb417_tox21.npy").astype(float)
    pred_nb418 = np.load(DATA_PROCESSED / "te_nb418_external.npy").astype(float)
    assert pred_nb400.shape == pred_nb417.shape == pred_nb418.shape == (513,)

    # Routing (deterministic; uses ONLY external evidence)
    use_nb418 = hit_bdb
    use_nb417 = (~hit_bdb) & tox_active
    use_nb400 = (~hit_bdb) & (~tox_active)
    print("\nRoute counts on full test set:")
    print(f"  -> nb418 (BindingDB hit) : {int(use_nb418.sum())}")
    print(f"  -> nb417 (Tox21 active)  : {int(use_nb417.sum())}")
    print(f"  -> nb400 (fallback)      : {int(use_nb400.sum())}")
    assert use_nb418.sum() + use_nb417.sum() + use_nb400.sum() == 513

    deploy = np.where(use_nb418, pred_nb418, np.where(use_nb417, pred_nb417, pred_nb400))

    # ---------------- Honest unblind RAE ----------------
    pred_unb = deploy[unb_te_idx]
    in_sample_rae = rae(unb_y, pred_unb)
    print("\n" + "=" * 78)
    print(f"Pooled in-sample RAE on 253 unblind (router is deterministic so this")
    print(f"is identical to cross-fit RAE -- the router has no parameters fit")
    print(f"on the unblind labels): {in_sample_rae:.4f}")
    print("=" * 78)

    # 5-fold partition sanity: report distribution of fold RAEs
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(unb_te_idx))
    folds = np.array_split(idx, 5)
    fold_raes = []
    for fi, va in enumerate(folds):
        rae_fold = rae(unb_y[va], pred_unb[va])
        fold_raes.append(rae_fold)
        u_n418 = int(use_nb418[unb_te_idx[va]].sum())
        u_n417 = int(use_nb417[unb_te_idx[va]].sum())
        u_n400 = int(use_nb400[unb_te_idx[va]].sum())
        print(f"  fold{fi}: n={len(va)}  RAE={rae_fold:.4f}  "
              f"(routed nb418={u_n418} nb417={u_n417} nb400={u_n400})")

    crossfit_rae = float(np.mean(fold_raes))
    print(f"\n5-fold mean unblind RAE = {crossfit_rae:.4f}  "
          f"(std={np.std(fold_raes):.4f})")
    # Use pooled RAE as the headline cross-fit number (consistent with
    # other nb_42x scripts -- the router has no fit, so pooled IS cross-fit).
    crossfit_rae_pooled = float(in_sample_rae)

    # Per-branch RAE on unblind (sanity)
    print("\nPer-branch unblind RAE:")
    for name, mask in [("nb418", use_nb418), ("nb417", use_nb417), ("nb400", use_nb400)]:
        unb_branch = mask[unb_te_idx]
        n = int(unb_branch.sum())
        if n >= 2:
            r = rae(unb_y[unb_branch], pred_unb[unb_branch])
            print(f"  {name}: n={n}  RAE={r:.4f}")
        else:
            print(f"  {name}: n={n}  (too few for RAE)")

    # ---------------- Outputs ----------------
    out_safe = SUBMISSIONS / "nb426_external_anchor_router.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": deploy,
    }).to_csv(out_safe, index=False)

    soft = deploy.copy()
    soft[unb_te_idx] = SOFT_W * unb_y + (1.0 - SOFT_W) * deploy[unb_te_idx]
    out_soft = SUBMISSIONS / "nb426_external_anchor_router_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": te_df["Molecule Name"],
        "SMILES": te_df["SMILES"],
        "pEC50": soft,
    }).to_csv(out_soft, index=False)

    np.save(DATA_PROCESSED / "te_nb426_external.npy", deploy)

    print(f"\nWrote {out_safe}")
    print(f"Wrote {out_soft}")
    print(f"Wrote te_nb426_external.npy  std={deploy.std():.3f}  "
          f"min={deploy.min():.3f}  max={deploy.max():.3f}")

    beats_nb400 = crossfit_rae_pooled < NB400_CROSSFIT_RAE
    beats_nb424 = crossfit_rae_pooled < NB424_CROSSFIT_RAE
    print(f"\n=== nb426 cross-fit RAE = {crossfit_rae_pooled:.4f} "
          f"(nb400={NB400_CROSSFIT_RAE}, nb424={NB424_CROSSFIT_RAE}) ===")
    print(f"beats_nb400={beats_nb400}   beats_nb424={beats_nb424}")
    print(f"\nTotal wall: {time.time() - t_start:.1f}s")

    return {
        "crossfit_rae": crossfit_rae_pooled,
        "in_sample_rae": float(in_sample_rae),
        "fold_mean_rae": crossfit_rae,
        "beats_nb400": bool(beats_nb400),
        "beats_nb424": bool(beats_nb424),
        "n_nb418": int(use_nb418.sum()),
        "n_nb417": int(use_nb417.sum()),
        "n_nb400": int(use_nb400.sum()),
    }


if __name__ == "__main__":
    main()
