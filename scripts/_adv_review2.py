"""Deeper checks: KS test, MW/logP shift significance, nb358 calibration density."""
import pandas as pd
import numpy as np
from scipy import stats
from rdkit import Chem
from rdkit.Chem import Descriptors

RAW = "d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/data/raw"
SUB = "d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge/submissions"

unblind = pd.read_csv(f"{RAW}/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
test = pd.read_csv(f"{RAW}/pxr-challenge_TEST_BLINDED.csv")
nb320 = pd.read_csv(f"{SUB}/nb320_phase2_top50_slsqp.csv")
nb358 = pd.read_csv(f"{SUB}/nb358_iso_cal_per_pred_truth.csv")
nb346 = pd.read_csv(f"{SUB}/nb346_rank_iso_truth.csv")
nb380 = pd.read_csv(f"{SUB}/nb380_calibrated_bma_truth.csv")

unblind_ids = set(unblind["Molecule Name"])
test_ids = set(test["Molecule Name"])
still_blind_ids = test_ids - unblind_ids

test_smi = dict(zip(test["Molecule Name"], test["SMILES"]))


def mol(s):
    return Chem.MolFromSmiles(s)


def mw_logp(s):
    m = mol(s)
    return Descriptors.MolWt(m), Descriptors.MolLogP(m)


mw_un = []
lp_un = []
for mid in unblind_ids:
    a, b = mw_logp(test_smi[mid])
    mw_un.append(a); lp_un.append(b)

mw_sb = []
lp_sb = []
for mid in still_blind_ids:
    a, b = mw_logp(test_smi[mid])
    mw_sb.append(a); lp_sb.append(b)

# KS tests on MW, logP, nb320 preds
print("=== Distribution shift tests (unblind vs still_blind) ===")
ks_mw = stats.ks_2samp(mw_un, mw_sb)
ks_lp = stats.ks_2samp(lp_un, lp_sb)
print(f"MW   KS stat={ks_mw.statistic:.4f} p={ks_mw.pvalue:.2e}")
print(f"logP KS stat={ks_lp.statistic:.4f} p={ks_lp.pvalue:.2e}")

# nb320 preds
nb320_map = dict(zip(nb320["Molecule Name"], nb320["pEC50"]))
preds_un = np.array([nb320_map[m] for m in unblind_ids])
preds_sb = np.array([nb320_map[m] for m in still_blind_ids])
ks_pred = stats.ks_2samp(preds_un, preds_sb)
print(f"nb320 preds KS stat={ks_pred.statistic:.4f} p={ks_pred.pvalue:.2e}")
print(f"nb320 unblind mean/std: {preds_un.mean():.3f} / {preds_un.std():.3f}")
print(f"nb320 stillblind mean/std: {preds_sb.mean():.3f} / {preds_sb.std():.3f}")

# nb358 calibration sparsity: for each still-blind pred, find # of unblind preds within 0.05
neighbors_close = []
for p in preds_sb:
    n = np.sum(np.abs(preds_un - p) <= 0.05)
    neighbors_close.append(n)
neighbors_close = np.array(neighbors_close)
print(f"\nIsotonic calibration density check:")
print(f"  median # of unblind anchors within 0.05 of each still-blind pred: {np.median(neighbors_close):.1f}")
print(f"  fraction still-blind with <3 anchors within 0.05: {np.mean(neighbors_close<3):.3f}")
print(f"  fraction still-blind with 0 anchors within 0.05: {np.mean(neighbors_close==0):.3f}")

# Effective sample size for per-predictor isotonic: 253 points / number of monotone constraints
# Density at high-pEC50 end (this is where the hits live)
hi_un = np.sum(preds_un >= 5.5)
hi_sb = np.sum(preds_sb >= 5.5)
print(f"\npreds>=5.5: unblind={hi_un}, still_blind={hi_sb}")
print(f"preds>=5.0: unblind={np.sum(preds_un>=5.0)}, still_blind={np.sum(preds_sb>=5.0)}")

# How do nb358/nb346/nb380 preds shift between groups?
for name, df in [("nb358", nb358), ("nb346", nb346), ("nb380", nb380)]:
    m = dict(zip(df["Molecule Name"], df["pEC50"]))
    pu = np.array([m[x] for x in unblind_ids])
    psb = np.array([m[x] for x in still_blind_ids])
    print(f"{name}: unblind mean={pu.mean():.3f} std={pu.std():.3f} | stillblind mean={psb.mean():.3f} std={psb.std():.3f}")

# Sampling bias check: compare unblind pEC50 to expected baseline mean
# RAE of mean predictor on unblind
y_un = unblind["pEC50"].values
mae_mean = np.mean(np.abs(y_un - y_un.mean()))
rae_norm = mae_mean
print(f"\nunblind mean-abs-deviation (RAE denominator surrogate): {mae_mean:.4f}")
print(f"unblind label std: {y_un.std():.3f}")
print(f"compare: train pEC50 mean abs dev from train mean")
# If still_blind label std differs, RAE comparisons are not apples-to-apples
