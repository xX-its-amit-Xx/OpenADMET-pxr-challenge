"""
nb721_precision_classifier.py -- Binary LGBM: F2_dead vs F2_gate_active.

Trains on the labeled subset of the pm06 nb390 gate cohort (147 unblind in gate),
labels follow the user's literal spec:
  F2_dead         : truth_if_unblind <  3.5   (expected ~16)
  F2_gate_active  : truth_if_unblind >= 5.0   (expected ~22; actual 59 with literal cut)

LOOCV is used because n_train is small (~16 + n_active).

Features (per pm06 distinguishers):
  logP, fsp3, MW, TPSA, formal_charge, num_aromatic_rings,
  sa_score, qed, chemprop_disagreement_std, nn_sim_train

Outputs:
  data/processed/nb721_classifier_scores.csv -- score for all 147 gate members
                                                (in_gate_nb390==True)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import lightgbm as lgb

ROOT = r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge"
PROC = os.path.join(ROOT, "data/processed")
RAW = os.path.join(ROOT, "data/raw")

# ---------- 1. Load cohort + SMILES + chemprop uncertainty ----------
coh = pd.read_csv(os.path.join(PROC, "nb720_gate_cohort.csv"))
test = pd.read_csv(os.path.join(RAW, "pxr-challenge_TEST_BLINDED.csv")).rename(
    columns={"Molecule Name": "name", "SMILES": "smiles"}
)
unc = pd.read_csv(os.path.join(PROC, "nb422_compound_uncertainty.csv"))[
    ["compound_name", "chemprop_disagreement_std"]
].rename(columns={"compound_name": "name"})

df = coh.merge(test, on="name", how="left").merge(unc, on="name", how="left")
assert df["smiles"].notna().all(), "missing SMILES"

# ---------- 2. Compute RDKit features ----------
def featurize(smi: str) -> dict:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {k: np.nan for k in
                ["logp", "fsp3", "mw", "tpsa", "formal_charge",
                 "num_aromatic_rings", "sa_score", "qed"]}
    return {
        "logp": Descriptors.MolLogP(m),
        "fsp3": Lipinski.FractionCSP3(m),
        "mw": Descriptors.MolWt(m),
        "tpsa": Descriptors.TPSA(m),
        "formal_charge": Chem.GetFormalCharge(m),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
        "sa_score": sascorer.calculateScore(m),
        "qed": QED.qed(m),
    }

feats = pd.DataFrame([featurize(s) for s in df["smiles"]])
# nb720 already has logp/mw; recompute to be consistent and to add missing ones
for c in feats.columns:
    df[c] = feats[c].values

FEATS = [
    "logp", "fsp3", "mw", "tpsa", "formal_charge",
    "num_aromatic_rings", "sa_score", "qed",
    "chemprop_disagreement_std", "nn_sim_train",
]

# ---------- 3. Restrict to pm06 gate (nb390 cohort = 147 unblind + 170 blind = 317 total) ----------
gate = df[df["in_gate_nb390"]].copy().reset_index(drop=True)
print(f"in_gate_nb390 total: {len(gate)}   unblind: {int(gate.is_unblind.sum())}")

# Labeled set: unblind & (truth<3.5 or truth>=5.0)
mask_dead = gate["is_unblind"] & (gate["truth_if_unblind"] < 3.5)
mask_active = gate["is_unblind"] & (gate["truth_if_unblind"] >= 5.0)
labeled = gate[mask_dead | mask_active].copy().reset_index(drop=True)
labeled["y"] = (labeled["truth_if_unblind"] >= 5.0).astype(int)  # 1 = active, 0 = dead

n_dead = int((labeled.y == 0).sum())
n_active = int((labeled.y == 1).sum())
print(f"Training labels: F2_dead={n_dead}  F2_gate_active={n_active}  (total={len(labeled)})")

X_lab = labeled[FEATS].astype(float).values
y_lab = labeled["y"].values

# Median-impute on labeled medians (chemprop_disagreement_std may be NaN for blind tail)
col_med = np.nanmedian(X_lab, axis=0)
inds = np.where(np.isnan(X_lab))
X_lab[inds] = np.take(col_med, inds[1])

# ---------- 4. LOOCV ----------
LGB_PARAMS = dict(
    objective="binary",
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=15,
    min_child_samples=5,
    reg_lambda=1.0,
    class_weight="balanced",
    verbose=-1,
    n_jobs=1,
)

loo = LeaveOneOut()
oof = np.zeros(len(y_lab), dtype=float)
for tr, te in loo.split(X_lab):
    clf = lgb.LGBMClassifier(**LGB_PARAMS)
    clf.fit(X_lab[tr], y_lab[tr])
    oof[te] = clf.predict_proba(X_lab[te])[:, 1]

auc = roc_auc_score(y_lab, oof)
thr = 0.7
pred_thr = (oof >= thr).astype(int)
# precision/recall for the POSITIVE class = F2_gate_active
prec = precision_score(y_lab, pred_thr, zero_division=0)
rec = recall_score(y_lab, pred_thr, zero_division=0)

print()
print("LOOCV results")
print(f"  AUC                       : {auc:.4f}")
print(f"  precision @ thr=0.70 (act): {prec:.4f}")
print(f"  recall    @ thr=0.70 (act): {rec:.4f}")

# ---------- 5. Refit on all labeled, score ALL gate members ----------
final = lgb.LGBMClassifier(**LGB_PARAMS)
final.fit(X_lab, y_lab)

X_all = gate[FEATS].astype(float).values
inds_all = np.where(np.isnan(X_all))
X_all[inds_all] = np.take(col_med, inds_all[1])
scores = final.predict_proba(X_all)[:, 1]

out = gate[[
    "name", "is_unblind", "truth_if_unblind", "group_label",
    "scaf_train_freq", "nn_sim_train", "best_pred_nb390",
] + FEATS].copy()
out["classifier_score"] = scores
out["predicted_class"] = np.where(scores >= 0.7, "F2_gate_active",
                          np.where(scores <= 0.3, "F2_dead", "uncertain"))

# Mark training rows
labeled_names = set(labeled["name"])
out["used_in_training"] = out["name"].isin(labeled_names)

out_path = os.path.join(PROC, "nb721_classifier_scores.csv")
out.to_csv(out_path, index=False)

n_total = len(out)
n_blind = int((~out["is_unblind"]).sum())
n_pred_dead_blind = int(((~out["is_unblind"]) & (out["predicted_class"] == "F2_dead")).sum())
# Also: how many of the 513 (full test) would the classifier flag F2_dead? Score the 513 too.
# Apply final on all 513:
X513 = df[FEATS].astype(float).values
i513 = np.where(np.isnan(X513))
X513[i513] = np.take(col_med, i513[1])
s513 = final.predict_proba(X513)[:, 1]
pred513_dead = int((s513 <= 0.3).sum())

print()
print(f"Scored {n_total} gate members  (blind={n_blind})")
print(f"Predicted F2_dead among blind gate     : {n_pred_dead_blind}")
print(f"Predicted F2_dead among full 513 test  : {pred513_dead}")
print()
print(f"Saved -> {out_path}")

# Echo a short JSON-ish summary for the orchestrator
import json
summary = dict(
    loocv_auc=round(float(auc), 4),
    precision_at_0p70=round(float(prec), 4),
    recall_at_0p70=round(float(rec), 4),
    n_train_dead=n_dead,
    n_train_active=n_active,
    n_gate_total=n_total,
    n_predicted_F2_dead_in_513=pred513_dead,
    n_predicted_F2_dead_blind_in_gate=n_pred_dead_blind,
)
print("SUMMARY_JSON " + json.dumps(summary))
