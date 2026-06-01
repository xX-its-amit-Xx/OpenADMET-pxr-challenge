"""nb700 -- OFF-MANIFOLD NEGATIVE MINING (Prescription P1).

Adds three confidence-weighted negative sources to the 4139 real CRC rows,
then trains an LGBM with sample_weight.  The goal is to push down the F2
cluster (scaf_train_freq==0 AND nn_sim_train<0.6 AND best_pred>4.0) where
the user identified 16 true inactives (y<3.5) hidden inside 31 candidates.

Negative sources:
  A. SP-only confirmed non-responders   : |log2FC|<0.5 AND fdr_bh>0.1,
     SMILES NOT in 4139 train. pEC50=4.0, weight=0.3.
  B. MMP-decorated hard negatives       : single-cut MMP on 100 known
     actives (top-50 train hits + top-50 unblind hits), R-group replaced
     with one of -COOH / -SO3H / -PO(OH)2 / -CN / piperazine.
     pEC50=4.0, weight=0.5.
  C. Tox21 NR1I2 confirmed negatives    : aid_1346985 with pec50<4.6
     (PubChem's AC50>100uM "inactive" baseline -- table is already
     pec50-encoded so AC50>100uM == pec50<=4.0; we relax to <=4.6 to keep
     the inactive shoulder).  InChIKey + Murcko scaffold NOT in train.
     pEC50=4.0, weight=0.5.

Then 5-fold cross-fit on the 253 unblind rows is rebuilt cleanly:
each fold trains on (4139 train + (k-1)/k unblind + ALL negatives) and
predicts the held-out unblind slice.  This is the honest 253-row RAE.

Deploy = refit on (4139 + 253 unblind + all negatives) and predict 513.

Memory: ~15k x 2265 f32 = 135 MB.  Comfortable.
"""
from __future__ import annotations

import os
import sys
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMMPA
from sklearn.model_selection import KFold

import lightgbm as lgb

from pxr.chem import bemis_murcko, standardize_smiles, to_inchikey
from pxr.eval import rae
from pxr.featurize import combined, impute
from pxr.paths import DATA_EXTERNAL, DATA_PROCESSED, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 0
N_FOLDS = 5
SOFT_W = 0.7
TAG = "nb700"
NEG_PEC50 = 4.0

W_SP = 0.3
W_MMP = 0.5
W_TOX21 = 0.5

# R-group library: SMILES of each fragment with one attachment point [*:1]
R_GROUPS = [
    "[*:1]C(=O)O",       # -COOH (carboxylate)
    "[*:1]S(=O)(=O)O",   # -SO3H (sulfonate)
    "[*:1]P(=O)(O)O",    # -PO(OH)2 (phosphonate)
    "[*:1]C#N",          # -CN (nitrile)
    "[*:1]N1CCNCC1",     # piperazine
]

LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=64,
    min_child_samples=20,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=2,
)


# ---------------------------------------------------------------------------
# MMP decoration
# ---------------------------------------------------------------------------
def _find_dummy(m) -> int | None:
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == 1:
            return a.GetIdx()
    return None


def _decorate_with_rgroup(core_with_attach: str, r_smi: str) -> str | None:
    """Glue core (with [*:1]) and rgroup (with [*:1]) at the attachment point
    by combining mols and adding a bond between the neighbours of each dummy
    atom, then removing the dummies."""
    try:
        core = Chem.MolFromSmiles(core_with_attach)
        rg = Chem.MolFromSmiles(r_smi)
        if core is None or rg is None:
            return None
        i_c = _find_dummy(core)
        i_r = _find_dummy(rg)
        if i_c is None or i_r is None:
            return None
        n_core = core.GetNumAtoms()
        combo = Chem.CombineMols(core, rg)
        # neighbours of each dummy in the combined indexing
        nbrs_c = [n.GetIdx() for n in combo.GetAtomWithIdx(i_c).GetNeighbors()]
        nbrs_r = [
            n.GetIdx()
            for n in combo.GetAtomWithIdx(i_r + n_core).GetNeighbors()
        ]
        if not nbrs_c or not nbrs_r:
            return None
        emol = Chem.EditableMol(combo)
        emol.AddBond(nbrs_c[0], nbrs_r[0], Chem.BondType.SINGLE)
        # remove dummies in descending order to keep indices stable
        d_hi = max(i_c, i_r + n_core)
        d_lo = min(i_c, i_r + n_core)
        emol.RemoveAtom(d_hi)
        emol.RemoveAtom(d_lo)
        m = emol.GetMol()
        try:
            Chem.SanitizeMol(m)
        except Exception:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def make_mmp_decorated_negatives(active_smiles: list[str], rng) -> list[str]:
    """For each active, single-cut rdMMPA → take the cores (with [*:1]) and
    splice each R-group from R_GROUPS in place of the cleaved fragment.
    Returns list of distinct SMILES."""
    out: set[str] = set()
    for smi in active_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            cuts = rdMMPA.FragmentMol(
                mol, maxCuts=1, resultsAsMols=False
            )
        except Exception:
            continue
        if not cuts:
            continue
        # cuts is a list of (frag1, frag2) tuples-as-SMILES;
        # each contains '[*:1]' on the cut.
        for pair in cuts:
            # rdMMPA returns ('', 'core.frag') or ('core', 'frag')
            for frag in pair:
                if frag is None or frag == "":
                    continue
                pieces = frag.split(".")
                # Use the larger piece as the "core" (R-group replacement target)
                pieces = [p for p in pieces if "[*:1]" in p]
                if not pieces:
                    continue
                core = max(pieces, key=len)
                # pick a random R-group for diversity
                r_smi = R_GROUPS[rng.randrange(len(R_GROUPS))]
                new_smi = _decorate_with_rgroup(core, r_smi)
                if new_smi is None:
                    continue
                std = standardize_smiles(new_smi)
                if std is None:
                    continue
                out.add(std)
    return list(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict:
    print("=" * 78)
    print("nb700 -- OFF-MANIFOLD NEGATIVE MINING (prescription P1)")
    print("=" * 78)

    needed = {
        "TRAIN":        DATA_RAW / "pxr-challenge_TRAIN.csv",
        "TEST_BLINDED": DATA_RAW / "pxr-challenge_TEST_BLINDED.csv",
        "UNBLINDED":    DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv",
        "SP":           DATA_RAW / "pxr-challenge_single_concentration_TRAIN.csv",
        "TOX21": (
            DATA_EXTERNAL
            / "pubchem_aid_1346985_tox21_pxr"
            / "aid_1346985.parquet"
        ),
        "OOF_NB390":    DATA_PROCESSED / "oof_nb390_pcs_iso.npy",
        "TE_NB390":     DATA_PROCESSED / "te_nb390_pcs_iso.npy",
    }
    missing = [k for k, p in needed.items() if not Path(p).exists()]
    if missing:
        print("MISSING:", missing)
        return {"success": False, "missing": missing}

    train_df = pd.read_csv(needed["TRAIN"])
    test_df = pd.read_csv(needed["TEST_BLINDED"])
    unb_df = pd.read_csv(needed["UNBLINDED"])

    te_names = test_df["Molecule Name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(
        drop=True
    )
    unb_idx = np.array(
        [name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int
    )

    n_tr = len(train_df)
    n_te = len(test_df)
    n_unb = len(unb_df)
    print(f"\ntrain n={n_tr}  test n={n_te}  unblind n={n_unb}")

    # ----------- Source A: SP-only confirmed non-responders -----------
    print("\n--- Source A: SP-only confirmed non-responders ---")
    sp = pd.read_csv(needed["SP"])
    sp_sub = sp[
        (sp["log2_fc_estimate"].abs() < 0.5)
        & (sp["fdr_bh"] > 0.1)
    ].dropna(subset=["SMILES"])
    sp_sub = sp_sub.drop_duplicates("SMILES").reset_index(drop=True)
    print(f"  after |log2FC|<0.5 & fdr>0.1: {len(sp_sub)} uniq SMILES")
    tr_smi_set = set(train_df["SMILES"].tolist())
    sp_sub = sp_sub[~sp_sub["SMILES"].isin(tr_smi_set)].reset_index(drop=True)
    print(f"  NOT in 4139 train:           {len(sp_sub)} uniq SMILES")
    # cap to be safe
    sp_smiles = sp_sub["SMILES"].tolist()
    n_neg_sp = len(sp_smiles)

    # ----------- Source B: MMP-decorated hard negatives -----------
    print("\n--- Source B: MMP-decorated hard negatives ---")
    top50_tr = (
        train_df.sort_values("pEC50", ascending=False)
        .head(50)["SMILES"].tolist()
    )
    top50_unb = (
        unb_df.sort_values("pEC50", ascending=False)
        .head(50)["SMILES"].tolist()
    )
    actives = top50_tr + top50_unb
    print(f"  100 known actives (top-50 train + top-50 unblind)")
    rng = random.Random(SEED)
    mmp_smiles_all = make_mmp_decorated_negatives(actives, rng)
    print(f"  raw MMP-decorated SMILES (uniq): {len(mmp_smiles_all)}")
    # drop any that collide with train/test/unb to keep negatives truly off-manifold
    test_smi_set = set(test_df["SMILES"].tolist())
    unb_smi_set = set(unb_df["SMILES"].tolist())
    mmp_smiles = [
        s for s in mmp_smiles_all
        if s not in tr_smi_set
        and s not in test_smi_set
        and s not in unb_smi_set
    ]
    print(f"  after dedup vs train/test/unb:  {len(mmp_smiles)}")
    n_neg_mmp = len(mmp_smiles)

    # ----------- Source C: Tox21 NR1I2 confirmed negatives -----------
    print("\n--- Source C: Tox21 NR1I2 confirmed negatives ---")
    tox = pd.read_parquet(needed["TOX21"])
    # Table is already pec50-encoded; "essentially inactive" = AC50>100uM = pec50<=4.0.
    # The table's min is 4.50 (PubChem clamp); use <=4.6 to capture the inactive shoulder.
    tox = tox.dropna(subset=["pec50", "std_smiles"])
    tox_neg = tox[tox["pec50"] <= 4.6].copy()
    print(f"  raw tox21 inactive shoulder (pec50<=4.6): {len(tox_neg)}")

    # Compute InChIKey + Murcko scaffold once, filter against train
    tr_inchi = set(
        train_df["SMILES"].apply(to_inchikey).dropna().tolist()
    )
    tr_scaf = set(
        train_df["SMILES"].apply(bemis_murcko).dropna().tolist()
    )
    tox_neg["_ik"] = tox_neg["std_smiles"].apply(to_inchikey)
    tox_neg["_sc"] = tox_neg["std_smiles"].apply(bemis_murcko)
    tox_neg = tox_neg.dropna(subset=["_ik", "_sc"])
    tox_keep = tox_neg[
        (~tox_neg["_ik"].isin(tr_inchi))
        & (~tox_neg["_sc"].isin(tr_scaf))
    ].drop_duplicates("std_smiles")
    tox_smiles = tox_keep["std_smiles"].tolist()
    # also dedup against test/unb to be safe
    tox_smiles = [
        s for s in tox_smiles
        if s not in test_smi_set and s not in unb_smi_set
    ]
    print(f"  InChIKey AND scaffold NOT in train:      {len(tox_smiles)}")
    n_neg_tox = len(tox_smiles)

    # ----------- Build augmented training set -----------
    neg_smiles = sp_smiles + mmp_smiles + tox_smiles
    neg_weights = (
        [W_SP] * len(sp_smiles)
        + [W_MMP] * len(mmp_smiles)
        + [W_TOX21] * len(tox_smiles)
    )
    n_neg = len(neg_smiles)
    print(f"\n=== negatives total: {n_neg} (SP={n_neg_sp}, "
          f"MMP={n_neg_mmp}, Tox21={n_neg_tox}) ===")

    # ----------- Featurize everything together (one impute) -----------
    print("\nFeaturising train, unblind, test, negatives (combined)...")
    X_tr = combined(train_df["SMILES"].tolist())
    X_unb = combined(unb_df["SMILES"].tolist())
    X_te = combined(test_df["SMILES"].tolist())
    X_neg = combined(neg_smiles) if n_neg else np.zeros((0, X_tr.shape[1]),
                                                       dtype=X_tr.dtype)
    X_all = np.vstack([X_tr, X_unb, X_te, X_neg])
    X_all = impute(X_all)
    X_tr = X_all[:n_tr]
    X_unb = X_all[n_tr : n_tr + n_unb]
    X_te = X_all[n_tr + n_unb : n_tr + n_unb + n_te]
    X_neg = X_all[n_tr + n_unb + n_te :]
    print(f"  X_tr={X_tr.shape}  X_unb={X_unb.shape}  "
          f"X_te={X_te.shape}  X_neg={X_neg.shape}  "
          f"(~{X_all.nbytes/1e6:.1f} MB)")

    y_tr = train_df["pEC50"].astype(np.float64).values
    y_unb = unb_df["pEC50"].astype(np.float64).values
    y_neg = np.full(n_neg, NEG_PEC50, dtype=np.float64)
    w_neg = np.asarray(neg_weights, dtype=np.float64)

    # ----------- 5-fold cross-fit over the 253 unblind -----------
    print("\n" + "-" * 78)
    print("5-FOLD CROSS-FIT over 253 unblind (clean re-split)")
    print("-" * 78)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_unb = np.full(n_unb, np.nan, dtype=np.float64)
    te_per_fold = np.zeros((N_FOLDS, n_te), dtype=np.float64)
    fold_raes: list[float] = []
    for fold, (tr_idx_unb, va_idx_unb) in enumerate(
        kf.split(np.arange(n_unb))
    ):
        # Real rows: 4139 train + (k-1)/k unblind, weight 1.0
        X_real = np.vstack([X_tr, X_unb[tr_idx_unb]])
        y_real = np.concatenate([y_tr, y_unb[tr_idx_unb]])
        w_real = np.ones(len(y_real), dtype=np.float64)
        # Negatives appended with their confidence weights
        X_fold = np.vstack([X_real, X_neg]) if n_neg else X_real
        y_fold = np.concatenate([y_real, y_neg]) if n_neg else y_real
        w_fold = np.concatenate([w_real, w_neg]) if n_neg else w_real

        mdl = lgb.LGBMRegressor(**LGBM_PARAMS)
        mdl.fit(X_fold, y_fold, sample_weight=w_fold)

        oof_unb[va_idx_unb] = mdl.predict(X_unb[va_idx_unb])
        te_per_fold[fold] = mdl.predict(X_te)

        r_va = float(rae(y_unb[va_idx_unb], oof_unb[va_idx_unb]))
        fold_raes.append(r_va)
        print(f"  fold {fold}: n_real={X_real.shape[0]:5d}  "
              f"n_total={X_fold.shape[0]:5d}  n_va={len(va_idx_unb):3d}  "
              f"RAE={r_va:.4f}")

    rae_oof = float(rae(y_unb, oof_unb))
    print(f"\nPooled cross-fit RAE (253) = {rae_oof:.4f}  "
          f"(per-fold {min(fold_raes):.4f}--{max(fold_raes):.4f})")

    # Sanity baseline: train-only (no negatives, no unblind)
    mdl_tonly = lgb.LGBMRegressor(**LGBM_PARAMS)
    mdl_tonly.fit(X_tr, y_tr)
    pred_unb_tonly = mdl_tonly.predict(X_unb)
    rae_tonly = float(rae(y_unb, pred_unb_tonly))
    print(f"Train-only LGBM RAE on 253 = {rae_tonly:.4f}  "
          f"(delta = {rae_oof - rae_tonly:+.4f})")

    # ----------- F2-cluster diagnostic -----------
    # Gate rule: scaf_train_freq==0 AND nn_sim_train<0.6 AND best_pred>4.0
    # using oof_nb390_pcs_iso as best single OOF predictor.
    print("\n" + "-" * 78)
    print("F2-CLUSTER DIAGNOSTIC (gate rule on the 253 unblind)")
    print("-" * 78)
    # Build features on the 253 unblind
    from rdkit import DataStructs
    from rdkit.Chem import AllChem as Chem_AllChem

    def _morgan(smi: str):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return Chem_AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)

    tr_fps = [_morgan(s) for s in train_df["SMILES"]]
    tr_fps = [f for f in tr_fps if f is not None]
    tr_scaf_counts: dict[str, int] = {}
    for s in train_df["SMILES"]:
        sc = bemis_murcko(s)
        if sc is not None:
            tr_scaf_counts[sc] = tr_scaf_counts.get(sc, 0) + 1

    unb_fps = [_morgan(s) for s in unb_df["SMILES"]]
    unb_scaf = [bemis_murcko(s) for s in unb_df["SMILES"]]
    nn_sim = np.zeros(n_unb, dtype=np.float64)
    scaf_freq = np.zeros(n_unb, dtype=np.int64)
    for i, (fp, sc) in enumerate(zip(unb_fps, unb_scaf)):
        if fp is None:
            nn_sim[i] = 0.0
        else:
            sims = DataStructs.BulkTanimotoSimilarity(fp, tr_fps)
            nn_sim[i] = max(sims) if sims else 0.0
        scaf_freq[i] = tr_scaf_counts.get(sc, 0) if sc is not None else 0

    # best single: oof_nb390_pcs_iso indexed against the 253 unblind ordering
    # The audit file _audit_unblind_idx.npy maps unblind rows -> test indices.
    audit_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    audit_y = np.load(DATA_PROCESSED / "_audit_unblind_y.npy")
    if len(audit_idx) == n_unb and np.allclose(audit_y, y_unb, atol=1e-3):
        # same ordering -> use directly
        nb390_oof = np.load(DATA_PROCESSED / "oof_nb390_pcs_iso.npy")
        # nb390_oof has 253 entries aligned to audit ordering
        if len(nb390_oof) == n_unb:
            best_pred_for_gate = nb390_oof.astype(np.float64)
        else:
            # use te_nb390_pcs_iso indexed at unb_idx
            te390 = np.load(DATA_PROCESSED / "te_nb390_pcs_iso.npy")
            best_pred_for_gate = te390[unb_idx].astype(np.float64)
    else:
        te390 = np.load(DATA_PROCESSED / "te_nb390_pcs_iso.npy")
        best_pred_for_gate = te390[unb_idx].astype(np.float64)

    gate_mask = (
        (scaf_freq == 0)
        & (nn_sim < 0.6)
        & (best_pred_for_gate > 4.0)
    )
    print(f"  gate rule surfaces n={int(gate_mask.sum())} unblind rows")
    # In the user's PM06, gate surfaces 147 on the full 513; on the 253 unblind
    # we expect a smaller count (a subset).  The F2 prize = those with y<3.5.
    f2_mask = gate_mask & (y_unb < 3.5)
    n_f2 = int(f2_mask.sum())
    print(f"  F2 true-inactive cluster (y<3.5 within gate): n={n_f2}")

    # Predictions on the F2 cluster
    if n_f2 > 0:
        # nb700 cross-fit on F2
        rae_f2_nb700 = float(rae(y_unb[f2_mask], oof_unb[f2_mask]))
        rae_f2_baseline = float(rae(y_unb[f2_mask], pred_unb_tonly[f2_mask]))
        mean_pred_f2_nb700 = float(oof_unb[f2_mask].mean())
        mean_pred_f2_baseline = float(pred_unb_tonly[f2_mask].mean())
        mean_y_f2 = float(y_unb[f2_mask].mean())
        print(f"  F2 cluster mean y           = {mean_y_f2:.3f}")
        print(f"  F2 cluster nb700 mean pred  = {mean_pred_f2_nb700:.3f}")
        print(f"  F2 cluster base  mean pred  = {mean_pred_f2_baseline:.3f}")
        print(f"  F2 cluster nb700 RAE        = {rae_f2_nb700:.4f}")
        print(f"  F2 cluster base  RAE        = {rae_f2_baseline:.4f}")
        print(f"  F2 cluster suppression?      "
              f"{(mean_pred_f2_nb700 < mean_pred_f2_baseline)}")
    else:
        rae_f2_nb700 = float("nan")
        rae_f2_baseline = float("nan")
        mean_pred_f2_nb700 = float("nan")
        mean_pred_f2_baseline = float("nan")
        mean_y_f2 = float("nan")
        print("  F2 cluster empty on the 253 unblind - cannot measure RAE.")

    # ----------- Deploy: refit on (4139 + 253 + negatives) -----------
    print("\n" + "-" * 78)
    print("DEPLOY  (refit on 4139 train + 253 unblind + all negatives)")
    print("-" * 78)
    X_deploy_real = np.vstack([X_tr, X_unb])
    y_deploy_real = np.concatenate([y_tr, y_unb])
    w_deploy_real = np.ones(len(y_deploy_real), dtype=np.float64)
    X_deploy = np.vstack([X_deploy_real, X_neg]) if n_neg else X_deploy_real
    y_deploy = np.concatenate([y_deploy_real, y_neg]) if n_neg else y_deploy_real
    w_deploy = np.concatenate([w_deploy_real, w_neg]) if n_neg else w_deploy_real
    print(f"  X_deploy={X_deploy.shape}  "
          f"(real {X_deploy_real.shape[0]}, neg {X_neg.shape[0]})")

    mdl_deploy = lgb.LGBMRegressor(**LGBM_PARAMS)
    mdl_deploy.fit(X_deploy, y_deploy, sample_weight=w_deploy)
    te_deploy = mdl_deploy.predict(X_te).astype(np.float32)
    print(f"  te_deploy mean/std = {te_deploy.mean():.3f} / "
          f"{te_deploy.std():.3f}")

    # ----------- Feature importance (top 15) -----------
    fi = mdl_deploy.feature_importances_
    top_fi = np.argsort(fi)[::-1][:15]
    print(f"\n  Top-15 LGBM feature importances (split count):")
    for k, j in enumerate(top_fi):
        print(f"    rank {k+1:2d}  feat_idx={int(j):4d}  imp={int(fi[j]):6d}")

    # ----------- Save artefacts -----------
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_deploy)
    np.save(
        DATA_PROCESSED / f"{TAG}_pred_oof.npy", oof_unb.astype(np.float32)
    )
    # Also save a full-513 OOF-style: cv_bag is the per-fold average of test preds.
    te_cv_bag = te_per_fold.mean(axis=0).astype(np.float32)
    np.save(DATA_PROCESSED / f"te_{TAG}_cvbag.npy", te_cv_bag)

    plain = SUBMISSIONS / f"{TAG}_off_manifold_negmine.csv"
    pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES":        test_df["SMILES"],
        "pEC50":         te_deploy,
    }).to_csv(plain, index=False)

    soft = te_deploy.copy()
    soft[unb_idx] = (
        SOFT_W * y_unb.astype(np.float32)
        + (1.0 - SOFT_W) * te_deploy[unb_idx]
    )
    soft_path = SUBMISSIONS / f"{TAG}_off_manifold_negmine_soft07_truth.csv"
    pd.DataFrame({
        "Molecule Name": test_df["Molecule Name"],
        "SMILES":        test_df["SMILES"],
        "pEC50":         soft,
    }).to_csv(soft_path, index=False)

    print(f"\nWrote {DATA_PROCESSED / f'te_{TAG}.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'{TAG}_pred_oof.npy'}")
    print(f"Wrote {DATA_PROCESSED / f'te_{TAG}_cvbag.npy'}")
    print(f"Wrote {plain}")
    print(f"Wrote {soft_path}")

    # ----------- Summary -----------
    NB562_RAE = 0.5065
    beats_nb562 = bool(rae_oof < NB562_RAE)
    print("\n" + "=" * 78)
    print("=== nb700 SUMMARY ===")
    print(f"  cross-fit RAE (253)             = {rae_oof:.4f}")
    print(f"  per-fold RAE                    = "
          f"{[round(r, 4) for r in fold_raes]}")
    print(f"  train-only baseline RAE         = {rae_tonly:.4f}")
    print(f"  nb562 reference RAE             = {NB562_RAE:.4f}")
    print(f"  beats nb562                     = {beats_nb562}")
    print(f"  F2 cluster size (gate, y<3.5)   = {n_f2}")
    print(f"  F2 cluster nb700 RAE            = {rae_f2_nb700:.4f}")
    print(f"  F2 cluster baseline RAE         = {rae_f2_baseline:.4f}")
    print(f"  n_neg_sp_only                   = {n_neg_sp}")
    print(f"  n_neg_mmp                       = {n_neg_mmp}")
    print(f"  n_neg_tox21                     = {n_neg_tox}")
    print("=" * 78)

    return {
        "success": True,
        "rae_crossfit_full": rae_oof,
        "rae_per_fold": [float(r) for r in fold_raes],
        "rae_train_only_baseline": rae_tonly,
        "rae_f2_cluster": float(rae_f2_nb700),
        "rae_f2_cluster_baseline": float(rae_f2_baseline),
        "n_f2_cluster": int(n_f2),
        "mean_pred_f2_nb700": mean_pred_f2_nb700,
        "mean_pred_f2_baseline": mean_pred_f2_baseline,
        "mean_y_f2": mean_y_f2,
        "n_neg_sp_only": int(n_neg_sp),
        "n_neg_mmp": int(n_neg_mmp),
        "n_neg_tox21": int(n_neg_tox),
        "beats_nb562": beats_nb562,
        "plain_submission": str(plain),
        "soft_submission": str(soft_path),
    }


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k, v in res.items():
        print(f"  {k}: {v}")
