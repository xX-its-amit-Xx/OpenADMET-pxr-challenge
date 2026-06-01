"""
nb720_gate_build.py -- Apply the pm06 F2 OOD-inactive gate to the 513 test compounds.

Gate rule (from pm06): scaf_train_freq == 0  AND  nn_sim_train < 0.6  AND  best_pred > 4.0.

Uses nb562 (rank-stretched blend) as `best_pred` by default; also reports overlap
with the pm06 cohort (which used nb390_pcs_iso, n=147 on the 253 unblind).

Outputs:
  data/processed/nb720_gate_cohort.csv
    columns: gate_member_id, name, in_gate, in_gate_nb390, truth_if_unblind,
             group_label, scaf_train_freq, nn_sim_train, best_pred,
             best_pred_nb390, logp, mw, is_unblind
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

ROOT = r"d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge"
PM = os.path.join(ROOT, "data/processed/postmortem")
PROC = os.path.join(ROOT, "data/processed")
SUBS = os.path.join(ROOT, "submissions")

# ---- 1. Load per-test-compound features for all 513 ----
chem = pd.read_parquet(os.path.join(PM, "pm_test_chem_all513.parquet"))
# columns include: name, scaf_train_freq, nn_sim_train, is_unblind, logp, mw, scaffold
assert len(chem) == 513, chem.shape

# ---- 2. Load both predictions on 513 test (nb562 = primary best_pred; nb390 = pm06 reference) ----
nb562 = pd.read_csv(os.path.join(SUBS, "nb562_rank_stretch_grid_s1.10.csv"))
nb390 = pd.read_csv(os.path.join(SUBS, "nb390_pcs-iso_per-compound_co.csv"))

nb562 = nb562.rename(columns={"Molecule Name": "name", "pEC50": "best_pred"})[["name", "best_pred"]]
nb390 = nb390.rename(columns={"Molecule Name": "name", "pEC50": "best_pred_nb390"})[["name", "best_pred_nb390"]]

df = chem.merge(nb562, on="name", how="left").merge(nb390, on="name", how="left")
assert df["best_pred"].notna().all(), "nb562 missing rows"
assert df["best_pred_nb390"].notna().all(), "nb390 missing rows"

# ---- 3. Apply the pm06 gate (with nb562 as best, and the original nb390 for comparison) ----
df["in_gate"] = (df.scaf_train_freq == 0) & (df.nn_sim_train < 0.6) & (df.best_pred > 4.0)
df["in_gate_nb390"] = (df.scaf_train_freq == 0) & (df.nn_sim_train < 0.6) & (df.best_pred_nb390 > 4.0)

n_gate_513 = int(df.in_gate.sum())
n_gate_nb390_513 = int(df.in_gate_nb390.sum())
overlap_513 = int((df.in_gate & df.in_gate_nb390).sum())

# ---- 4. Attach truth for the 253 unblind ----
unblind_y = np.load(os.path.join(PM, "pm_unblind_y.npy"))
cu = pd.read_parquet(os.path.join(PM, "pm_compounds.parquet"))[["name", "truth"]]
df = df.merge(cu, on="name", how="left").rename(columns={"truth": "truth_if_unblind"})
assert df.loc[df.is_unblind, "truth_if_unblind"].notna().all(), "truth missing on unblind"

# ---- 5. Within the gate cohort, identify F2_dead (truth<3.5) and F2_gate_active (truth>=5.0) on the 253 unblind ----
unb_gate = df[df.in_gate & df.is_unblind].copy()
unb_gate_nb390 = df[df.in_gate_nb390 & df.is_unblind].copy()

def label(row):
    if not row.in_gate or not row.is_unblind or pd.isna(row.truth_if_unblind):
        return ""
    t = row.truth_if_unblind
    if t < 3.5:
        return "F2_dead"
    if t >= 5.0:
        return "F2_gate_active"
    return "mid"

df["group_label"] = df.apply(label, axis=1)

n_dead = int((df.group_label == "F2_dead").sum())
n_active = int((df.group_label == "F2_gate_active").sum())
n_mid = int((df.group_label == "mid").sum())
n_in_gate_unblind = int((df.in_gate & df.is_unblind).sum())

# overlap with the pm06 nb390 cohort (147 on 253 unblind)
pm06_unblind_gate = set(df.loc[df.in_gate_nb390 & df.is_unblind, "name"])
nb562_unblind_gate = set(df.loc[df.in_gate & df.is_unblind, "name"])
overlap_unblind = len(pm06_unblind_gate & nb562_unblind_gate)
pm06_n = len(pm06_unblind_gate)
nb562_n = len(nb562_unblind_gate)
# Jaccard
jac = overlap_unblind / max(1, len(pm06_unblind_gate | nb562_unblind_gate))
# Overlap as fraction of pm06
overlap_frac_of_pm06 = overlap_unblind / max(1, pm06_n)

# ---- 6. Save the cohort CSV ----
out = df[
    [
        "name",
        "in_gate",
        "in_gate_nb390",
        "is_unblind",
        "truth_if_unblind",
        "group_label",
        "scaf_train_freq",
        "nn_sim_train",
        "best_pred",
        "best_pred_nb390",
        "logp",
        "mw",
        "scaffold",
    ]
].copy()
out.insert(0, "gate_member_id", np.arange(len(out)))
out_path = os.path.join(PROC, "nb720_gate_cohort.csv")
out.to_csv(out_path, index=False)

print("=" * 60)
print("nb720 -- pm06 F2 gate applied to 513 test compounds")
print("=" * 60)
print(f"best_pred = nb562 (rank-stretched blend, s=1.10)")
print(f"reference = nb390_pcs_iso (the original pm06 best)")
print()
print(f"Gate (novel scaffold & nn_sim<0.6 & best_pred>4.0):")
print(f"  on 513 test (nb562)            : {n_gate_513}")
print(f"  on 513 test (nb390 reference)  : {n_gate_nb390_513}  [pm06 reported 147 on 253 unblind]")
print(f"  513-overlap (nb562 & nb390)    : {overlap_513}")
print()
print(f"On 253 unblind subset:")
print(f"  in_gate (nb562)                : {nb562_n}")
print(f"  in_gate_nb390 (pm06 cohort)    : {pm06_n}")
print(f"  overlap                        : {overlap_unblind}  (Jaccard {jac:.3f}, "
      f"{overlap_frac_of_pm06*100:.1f}% of pm06)")
print()
print(f"Within nb562 gate cohort on 253 unblind (n={nb562_n}):")
print(f"  F2_dead         (truth<3.5)    : {n_dead}")
print(f"  F2_gate_active  (truth>=5.0)   : {n_active}")
print(f"  mid             (3.5..5.0)     : {n_mid}")
print()
print(f"Saved -> {out_path}")
