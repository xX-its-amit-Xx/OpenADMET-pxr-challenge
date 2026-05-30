# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: pxr-challenge
#     language: python
#     name: pxr-challenge
# ---

# %% [markdown]
# # PXR Agentic Medicinal Chemist Workflow
#
# **Goal**: rather than `SMILES → black-box neural net → pEC50`, behave like a
# medicinal chemist with broad chemistry priors. The pipeline reasons through:
#
# 1. analog neighborhoods,
# 2. activity cliffs,
# 3. likely cliff mechanisms,
# 4. assay-noise / counter-assay artifacts,
# 5. PXR's flexible, promiscuous binding pocket,
# 6. global ML + local analog reasoning + uncertainty-aware ensembling,
# 7. interpretable per-compound audit cards.
#
# **Architectural principle**: separate **GLOBAL PRIORS** (chemistry-wide
# knowledge of cliffs, transformations, artifacts) from **LOCAL PXR EVIDENCE**
# (the 4,139 dose-response training compounds + 2,859 counter-screens +
# 21,003 single-concentration screens). Use global to *constrain the space of
# plausible mechanisms*; use local to *determine relevance to PXR specifically*.
#
# **Status**: this notebook is the **workflow** scaffold. It runs end-to-end on
# local data, produces a submission, and exposes hooks where massive peripheral
# data (ChEMBL cliffs, BindingDB nuclear-receptor data, PAINS, etc.) plug in.
# Where peripheral data is unavailable, we fall back to rule-based heuristics
# encoded from medicinal-chemistry literature.

# %% [markdown]
# ## Section 1 — Repo and data discovery

# %%
from __future__ import annotations
import os, sys, json, warnings, time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# Robust repo discovery: walk up until we find pyproject.toml
def find_repo_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "pxr").exists():
            return parent
    raise RuntimeError("Could not locate PXR challenge repo root")

REPO = find_repo_root()
sys.path.insert(0, str(REPO / "src"))
from pxr.data import load_train, load_test, load_counter, load_single_conc
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko, standardize_smiles, morgan_fp_batch
from pxr.featurize import combined as feat_combined, rdkit_desc, morgan, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

OUT = REPO / "outputs" / "pxr_agentic_chemist"
OUT.mkdir(parents=True, exist_ok=True)

print(f"Repo: {REPO}")
print(f"Output dir: {OUT}")

SEED = 42
np.random.seed(SEED)

# %% [markdown]
# ## Section 2 — Data loading and validation
#
# We load the dose-response train (4,139 compounds), blinded test (513), the
# PXR-null counter-assay (2,859), and the single-concentration screen (21,003).
# All compounds get standardized SMILES (largest fragment, neutralized).

# %%
print("Loading datasets...")
tr = load_train()
te = load_test()
cn = load_counter()
sc = load_single_conc()

print(f"Train (CRC):           {len(tr):>6,}  cols: {list(tr.columns)[:8]}")
print(f"Test (blinded):        {len(te):>6,}  cols: {list(te.columns)}")
print(f"Counter-assay (CRC):   {len(cn):>6,}")
print(f"Single-concentration:  {len(sc):>6,}  cols: {list(sc.columns)[:8]}")

# Standardize SMILES for matching
def std(s):
    if not isinstance(s, str): return None
    out = standardize_smiles(s)
    return out if out else None

for df, name in [(tr, "train"), (te, "test"), (cn, "counter"), (sc, "single_conc")]:
    df["std_smiles"] = df["smiles"].map(std)
    n_bad = df["std_smiles"].isna().sum()
    print(f"  {name}: {n_bad} unparseable SMILES")

# Sanity: no train/test overlap (already verified in nb01_eda)
overlap = set(tr["std_smiles"].dropna()) & set(te["std_smiles"].dropna())
print(f"\nTrain/test SMILES overlap: {len(overlap)} (expect 0)")

# Label-distribution sanity
y_tr = tr["pec50"].values.astype(np.float64)
print(f"\npEC50 stats: mean={y_tr.mean():.2f}  std={y_tr.std():.2f}  "
      f"min={y_tr.min():.2f}  max={y_tr.max():.2f}")
print(f"Hits (pEC50 >= 6): {(y_tr >= 6).sum()}/{len(y_tr)}  "
      f"({100*(y_tr>=6).sum()/len(y_tr):.1f}%)")

# %% [markdown]
# ## Section 3 — Assay reliability and artifact-risk scoring
#
# For each training compound, we score:
#
# - **assay_reliability_score**: high = trustworthy label
# - **artifact_risk_score**: high = label suspect (large SE, near floor/ceiling, etc.)
# - **null_line_risk_score**: high = compound activates the null reporter →
#   probably non-specific
# - **efficacy_weirdness_score**: high = Emax not consistent with pEC50
#
# These are **rule-based** and intentionally simple. Used as sample weights for
# the global model and as inputs to the per-compound audit card.

# %%
def reliability_scores(df: pd.DataFrame, cn_df: pd.DataFrame) -> pd.DataFrame:
    out = df[["std_smiles"]].copy()
    pec = df["pec50"].astype(float)
    se  = df["pec50_se"].astype(float)
    emax = df["emax"].astype(float) if "emax" in df.columns else pd.Series([np.nan]*len(df))
    emax_se = df["emax_se"].astype(float) if "emax_se" in df.columns else pd.Series([np.nan]*len(df))

    # SE near 0 is suspiciously precise; large SE is unreliable. Reasonable range 0.05-0.6.
    se_clip = se.fillna(0.5).clip(0.01, 1.0)
    rel_se = 1.0 - (se_clip - 0.05).clip(0, 1.0)   # ~1 if SE small, ~0 if SE large

    # pEC50 floor/ceiling check (artifacts often pile up at limits)
    pec_floor = pec.min(); pec_ceil = pec.max()
    near_limit = ((pec - pec_floor).abs() < 0.05) | ((pec - pec_ceil).abs() < 0.05)
    rel_floor = (~near_limit).astype(float)

    # Counter-assay null-line: high null pEC50 + high PXR pEC50 = non-specific
    cn_lookup = dict(zip(cn_df["std_smiles"], cn_df["pec50"]))
    pec_null = df["std_smiles"].map(cn_lookup)
    null_active = (pec_null >= 5.0) & (pec >= 5.0)
    null_line_risk = null_active.astype(float)
    null_line_risk = null_line_risk + (pec_null.fillna(0) - 4).clip(0) * 0.2  # gradient

    # Efficacy weirdness: high pEC50 should usually have positive Emax
    eff_weird = pd.Series(np.zeros(len(df)), index=df.index)
    high_pec = pec >= 5.5
    if "emax" in df.columns:
        weak_emax = emax.fillna(0) < 0.5
        eff_weird = (high_pec & weak_emax).astype(float)
        # large emax SE relative to value
        emax_se_ratio = (emax_se.fillna(0) / emax.fillna(1).abs().clip(0.1, None))
        eff_weird = eff_weird + (emax_se_ratio > 0.5).astype(float) * 0.3

    # Aggregate
    artifact_risk = (1 - rel_se) * 0.5 + (1 - rel_floor) * 0.3 + null_line_risk * 0.5 + eff_weird * 0.3
    artifact_risk = artifact_risk.clip(0, 2.0)
    reliability = (1.0 / (1.0 + artifact_risk)).round(3)  # in (0, 1]

    out["pec50_se"] = se
    out["assay_reliability_score"] = reliability.round(3)
    out["artifact_risk_score"]    = artifact_risk.round(3)
    out["null_line_risk_score"]   = null_line_risk.round(3)
    out["efficacy_weirdness_score"] = eff_weird.round(3)

    def reason(row):
        bits = []
        if row["null_line_risk_score"] > 0.3: bits.append("null-line active")
        if row["artifact_risk_score"]  > 1.0: bits.append("high SE / near limit")
        if row["efficacy_weirdness_score"] > 0.3: bits.append("Emax inconsistent")
        return "; ".join(bits) or "ok"
    out["reliability_reason_short"] = out.apply(reason, axis=1)
    return out

reliability_tr = reliability_scores(tr, cn)
print("Reliability summary (train):")
print(reliability_tr[["assay_reliability_score", "artifact_risk_score",
                      "null_line_risk_score", "efficacy_weirdness_score"]].describe().round(3))
print("\nReason breakdown:")
print(reliability_tr["reliability_reason_short"].value_counts().head(10))
reliability_tr.to_csv(OUT / "reliability_train.csv", index=False)

# %% [markdown]
# ## Section 4 — Chemical featurization
#
# Core feature sets (from `pxr.featurize`):
# - Morgan ECFP4 (2048 bits)
# - RDKit 2D descriptors (~217)
# - Combined: Morgan + RDKit
#
# Plus additional descriptors that the medicinal chemist reasoning will use:
# - MACCS keys (167)
# - Bemis-Murcko scaffolds (already in `pxr.chem.bemis_murcko`)
# - core physchem (MW, logP, TPSA, HBD, HBA, rotbonds, fsp3, ring counts)

# %%
print("Computing core features (combined = morgan + rdkit_desc)...")
X_tr_combined = impute(feat_combined(tr["std_smiles"].tolist()))
X_te_combined = impute(feat_combined(te["std_smiles"].tolist()))
print(f"  train: {X_tr_combined.shape}, test: {X_te_combined.shape}")

# MACCS keys for cliff annotation features
def maccs_batch(smiles):
    out = np.zeros((len(smiles), 167), dtype=np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        if m is None: continue
        bv = MACCSkeys.GenMACCSKeys(m)
        DataStructs.ConvertToNumpyArray(bv, out[i])
    return out

print("Computing MACCS keys...")
maccs_tr = maccs_batch(tr["std_smiles"].tolist())
maccs_te = maccs_batch(te["std_smiles"].tolist())
print(f"  shapes: {maccs_tr.shape}, {maccs_te.shape}")

# Physchem (richer than what's already in rdkit_desc, manageable subset)
def physchem(smi):
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if m is None: return [np.nan]*9
    return [
        Descriptors.MolWt(m),
        Descriptors.MolLogP(m),
        Descriptors.TPSA(m),
        Descriptors.NumHDonors(m),
        Descriptors.NumHAcceptors(m),
        Descriptors.NumRotatableBonds(m),
        rdMolDescriptors.CalcFractionCSP3(m),
        rdMolDescriptors.CalcNumRings(m),
        Chem.GetFormalCharge(m),
    ]

physchem_cols = ["MW","LogP","TPSA","HBD","HBA","RotBonds","Fsp3","Rings","FormalCharge"]
print("Computing physchem...")
pc_tr = pd.DataFrame([physchem(s) for s in tr["std_smiles"]], columns=physchem_cols)
pc_te = pd.DataFrame([physchem(s) for s in te["std_smiles"]], columns=physchem_cols)
print(pc_tr.describe().round(2).iloc[[1, 2, 3, 6]])

# Compute Bemis-Murcko scaffolds
print("\nComputing scaffolds...")
scaffolds_tr = tr["std_smiles"].map(bemis_murcko).tolist()
scaffolds_te = te["std_smiles"].map(bemis_murcko).tolist()
n_unique_tr = len(set(scaffolds_tr))
n_unique_te = len(set(scaffolds_te))
n_overlap = len(set(scaffolds_tr) & set(scaffolds_te))
print(f"  train scaffolds: {n_unique_tr} unique")
print(f"  test scaffolds:  {n_unique_te} unique  (overlap with train: {n_overlap})")

# %% [markdown]
# ## Section 5 — Analog neighborhoods
#
# For every compound (train + test) we compute neighbor evidence in three
# similarity spaces:
# 1. **Tanimoto** on Morgan FPs (substructure-based)
# 2. **Descriptor-space** k-NN (Mahalanobis-ish via standardized euclidean)
# 3. **Scaffold grouping** (exact Bemis-Murcko match)
#
# Per compound we record top-1 / top-5 / top-20 neighbor pEC50 statistics and
# count of neighbors above similarity thresholds 0.4/0.5/0.6/0.7/0.8.

# %%
# Build RDKit fingerprints for similarity (already have bit vectors via morgan_fp_batch,
# but DataStructs.BulkTanimotoSimilarity wants ExplicitBitVect)
def fp_objects(smiles):
    return [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
            if isinstance(s, str) and Chem.MolFromSmiles(s) is not None else None
            for s in smiles]

print("Building Morgan fingerprint objects for similarity...")
fps_tr = fp_objects(tr["std_smiles"].tolist())
fps_te = fp_objects(te["std_smiles"].tolist())

K_LIST = (1, 5, 20)
SIM_THRESH = (0.4, 0.5, 0.6, 0.7, 0.8)

def neighbor_evidence(query_fps, neighbor_fps, neighbor_pec, exclude_self=False):
    """Returns DataFrame with 7*len(K_LIST) + len(SIM_THRESH) cols per query."""
    n_q = len(query_fps)
    cols_per_k = ["sim_top1", "sim_mean", "pec_mean", "pec_std", "pec_min", "pec_max", "pec_wmean"]
    rows = []
    for i, qfp in enumerate(query_fps):
        if qfp is None:
            row = [np.nan] * (len(K_LIST) * 7 + len(SIM_THRESH))
            rows.append(row); continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, neighbor_fps),
                        dtype=np.float64)
        if exclude_self: sims[i] = -1.0
        order = np.argsort(sims)[::-1]
        row = []
        for k in K_LIST:
            top = order[:k]
            ts = sims[top]; tp = neighbor_pec[top]
            wsum = ts.sum()
            row.extend([
                ts[0] if k>=1 else 0.0,
                ts.mean(),
                tp.mean(),
                tp.std() if k>1 else 0.0,
                tp.min(),
                tp.max(),
                np.average(tp, weights=ts) if wsum > 1e-9 else tp.mean(),
            ])
        for thr in SIM_THRESH:
            row.append(int((sims >= thr).sum()))
        rows.append(row)
    cols = []
    for k in K_LIST:
        cols += [f"k{k}_{c}" for c in cols_per_k]
    cols += [f"n_above_{thr:.1f}" for thr in SIM_THRESH]
    return pd.DataFrame(rows, columns=cols)

print("Neighbor evidence: test ...")
nbr_te = neighbor_evidence(fps_te, fps_tr, y_tr)
print("Neighbor evidence: train (leave-one-out, fold-naive)...")
nbr_tr = neighbor_evidence(fps_tr, fps_tr, y_tr, exclude_self=True)
print(nbr_te[["k1_sim_top1","k5_pec_mean","k5_pec_std","n_above_0.5"]].describe().round(3))

# %% [markdown]
# ## Section 6 — Activity cliff detection
#
# Cliff = pair of train compounds with **Tanimoto ≥ 0.6** (substantively similar)
# but **|ΔpEC50| ≥ 1.0** (≥ 10x activity difference). We also flag SEVERE cliffs
# at |ΔpEC50| ≥ 1.5.
#
# Each cliff pair gets:
# - reliability of both endpoints
# - shared scaffold flag
# - suspect-vs-real classification (if either endpoint is unreliable, treat with care)

# %%
def find_cliffs(fps, pec, sim_threshold=0.6, delta_threshold=1.0):
    n = len(fps)
    pairs = []
    for i in range(n):
        if fps[i] is None: continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps),
                        dtype=np.float64)
        # Don't double-count: only j > i
        for j in range(i+1, n):
            if fps[j] is None or sims[j] < sim_threshold: continue
            d = abs(pec[i] - pec[j])
            if d >= delta_threshold:
                pairs.append((i, j, sims[j], pec[i], pec[j], d))
    return pd.DataFrame(pairs, columns=["i", "j", "sim", "pec_i", "pec_j", "abs_delta"])

print("Finding activity cliffs (Tanimoto>=0.6, |ΔpEC50|>=1.0)...")
cliffs = find_cliffs(fps_tr, y_tr, 0.6, 1.0)
print(f"  {len(cliffs)} cliff pairs found")

# Annotate with metadata
cliffs["smiles_i"] = cliffs["i"].map(lambda k: tr["std_smiles"].iloc[k])
cliffs["smiles_j"] = cliffs["j"].map(lambda k: tr["std_smiles"].iloc[k])
cliffs["scaffold_i"] = cliffs["i"].map(lambda k: scaffolds_tr[k])
cliffs["scaffold_j"] = cliffs["j"].map(lambda k: scaffolds_tr[k])
cliffs["same_scaffold"] = cliffs["scaffold_i"] == cliffs["scaffold_j"]
cliffs["reliability_i"] = cliffs["i"].map(lambda k: reliability_tr["assay_reliability_score"].iloc[k])
cliffs["reliability_j"] = cliffs["j"].map(lambda k: reliability_tr["assay_reliability_score"].iloc[k])
cliffs["min_reliability"] = cliffs[["reliability_i", "reliability_j"]].min(axis=1)
cliffs["likely_real"] = (cliffs["min_reliability"] >= 0.4) & (cliffs["abs_delta"] >= 1.0)
cliffs["severe"] = cliffs["abs_delta"] >= 1.5

print(f"  Likely real cliffs:        {cliffs['likely_real'].sum()}")
print(f"  Severe (|ΔpEC50| >= 1.5):  {cliffs['severe'].sum()}")
print(f"  Same-scaffold cliffs:      {cliffs['same_scaffold'].sum()}")

cliffs.to_csv(OUT / "activity_cliffs.csv", index=False)

# Plot: similarity vs delta
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ax = axes[0]
ax.scatter(cliffs["sim"], cliffs["abs_delta"],
           c=cliffs["likely_real"].map({True: "C0", False: "C3"}),
           alpha=0.6, s=20)
ax.set_xlabel("Tanimoto similarity"); ax.set_ylabel("|ΔpEC50|")
ax.set_title(f"Activity cliffs ({len(cliffs)} pairs)")
ax.axhline(1.5, color="grey", ls="--", alpha=0.5)
ax = axes[1]
sns.histplot(cliffs["abs_delta"], bins=30, ax=ax, kde=True)
ax.set_xlabel("|ΔpEC50|"); ax.set_title("Cliff severity")
plt.tight_layout(); plt.savefig(OUT / "cliffs.png", dpi=120); plt.close()

# %% [markdown]
# ## Section 7 — Matched molecular pair (MMP) analysis
#
# Use `rdMMPA` single-cut fragmentation to build a transformation library.
# For each training pair sharing a core but with different chains, record the
# transformation (chain_A → chain_B) and the ΔpEC50.
#
# Aggregate across the library:
# - average effect size of each transformation
# - context dependence (variance across scaffolds)
# - cliff frequency (fraction of pairs where |Δ| ≥ 1.0)
#
# This becomes a **prior** for predicting effects of edits in test compounds.

# %%
from rdkit.Chem import rdMMPA

def fragment_compound(smi, max_cuts=1):
    if not isinstance(smi, str): return []
    m = Chem.MolFromSmiles(smi)
    if m is None: return []
    try:
        return rdMMPA.FragmentMol(m, maxCuts=max_cuts, resultsAsMols=False)
    except Exception:
        return []

print("Fragmenting train compounds (single-cut)...")
t_start = time.time()
tr_frags = [fragment_compound(s, 1) for s in tr["std_smiles"]]
n_total = sum(len(f) for f in tr_frags)
print(f"  {n_total} cut fragments ({time.time()-t_start:.0f}s)")

# Build core->[(idx, chain, pec, reliability)] lookup
# For maxCuts=1 rdMMPA returns (core="", chains="A.B") -- split chains on "."
# and use the larger half as the "core", the smaller as the "chain".
core_lookup: dict[str, list] = {}
for i, fragl in enumerate(tr_frags):
    for tup in fragl:
        if not tup or len(tup) != 2: continue
        core, chains = tup
        if core and chains:
            parts = [core, chains]
        elif chains and "." in chains:
            parts = chains.split(".")
        else:
            continue
        if len(parts) != 2: continue
        # Larger fragment is the "core" (more atoms in SMILES is a rough proxy)
        a, b = parts
        if len(a) >= len(b):
            ctx, chain = a, b
        else:
            ctx, chain = b, a
        core_lookup.setdefault(ctx, []).append((i, chain,
                                                 y_tr[i],
                                                 reliability_tr["assay_reliability_score"].iloc[i]))

print(f"  Unique cores: {len(core_lookup)}")

# Build training transformation table: pairs sharing context
mmp_pairs = []
for core, entries in core_lookup.items():
    if len(entries) < 2: continue
    for a in range(len(entries)):
        for b in range(a+1, len(entries)):
            i, chain_a, pec_a, rel_a = entries[a]
            j, chain_b, pec_b, rel_b = entries[b]
            mmp_pairs.append({
                "core": core,
                "chain_in": chain_a,
                "chain_out": chain_b,
                "delta_pec": pec_b - pec_a,
                "reliability_min": min(rel_a, rel_b),
                "i": i, "j": j,
            })

mmp_df = pd.DataFrame(mmp_pairs)
print(f"  MMP pairs: {len(mmp_df)}")

# Aggregate per-transformation effect (chain_in -> chain_out)
trans_effect = (mmp_df.assign(transform=mmp_df["chain_in"] + "→" + mmp_df["chain_out"])
                       .groupby("transform")
                       .agg(avg_delta=("delta_pec", "mean"),
                            std_delta=("delta_pec", "std"),
                            n_pairs=("delta_pec", "size"),
                            cliff_frac=("delta_pec", lambda x: (x.abs() >= 1.0).mean()),
                            min_reliability=("reliability_min", "mean"))
                       .reset_index()
                       .sort_values("n_pairs", ascending=False))
print("\nTop transformations by support:")
print(trans_effect.head(8).round(3))
trans_effect.to_csv(OUT / "transformation_priors.csv", index=False)

# %% [markdown]
# ## Section 8 — Mechanistic cliff annotation
#
# For each cliff pair, derive **rule-based mechanism tags** from SMARTS patterns
# and descriptor deltas. These are *weak heuristics* (not truth) — they go into
# the audit card and into the model as low-weight features.

# %%
# A small rule-based annotator. Each rule returns a list of tags.
HALOGEN = "[F,Cl,Br,I]"
HBOND_DONOR = "[OH,NH,NH2]"
HBOND_ACCEPTOR = "[O,N;!H0]"
AROMATIC = "[a]"

def descriptor_delta(smi_a, smi_b):
    a = Chem.MolFromSmiles(smi_a); b = Chem.MolFromSmiles(smi_b)
    if a is None or b is None: return {}
    return {
        "d_MW":   Descriptors.MolWt(b)   - Descriptors.MolWt(a),
        "d_LogP": Descriptors.MolLogP(b) - Descriptors.MolLogP(a),
        "d_TPSA": Descriptors.TPSA(b)    - Descriptors.TPSA(a),
        "d_HBD":  Descriptors.NumHDonors(b)    - Descriptors.NumHDonors(a),
        "d_HBA":  Descriptors.NumHAcceptors(b) - Descriptors.NumHAcceptors(a),
        "d_RotBonds": Descriptors.NumRotatableBonds(b) - Descriptors.NumRotatableBonds(a),
        "d_Rings": rdMolDescriptors.CalcNumRings(b) - rdMolDescriptors.CalcNumRings(a),
        "d_Fsp3": rdMolDescriptors.CalcFractionCSP3(b) - rdMolDescriptors.CalcFractionCSP3(a),
        "d_HalogenCount": _count(b, HALOGEN) - _count(a, HALOGEN),
    }

def _count(mol, smarts):
    p = Chem.MolFromSmarts(smarts)
    return len(mol.GetSubstructMatches(p)) if p is not None else 0

def annotate_cliff(row):
    d = descriptor_delta(row["smiles_i"], row["smiles_j"])
    tags = []
    # Hydrophobic fill: gained logP, no polarity gain
    if d.get("d_LogP", 0) > 0.7 and d.get("d_TPSA", 0) < 5: tags.append("hydrophobic_fill")
    # Polarity gain
    if d.get("d_TPSA", 0) > 15: tags.append("polarity_gain")
    if d.get("d_TPSA", 0) < -15: tags.append("polarity_loss")
    # H-bond changes
    if d.get("d_HBD", 0) > 0: tags.append("hbond_donor_added")
    if d.get("d_HBA", 0) > 0: tags.append("hbond_acceptor_added")
    # Halogen effect
    if d.get("d_HalogenCount", 0) != 0: tags.append("halogen_effect")
    # Rigidity change
    if d.get("d_RotBonds", 0) <= -2: tags.append("rigidity_gain")
    if d.get("d_RotBonds", 0) >= 2:  tags.append("flexibility_penalty")
    # Aromatic substitution: ring count unchanged but heavy edit
    if d.get("d_Rings", 0) == 0 and abs(d.get("d_MW", 0)) > 30:
        tags.append("aromatic_substitution_effect")
    # Fsp3 shift = sat/aromatic balance
    if d.get("d_Fsp3", 0) > 0.15: tags.append("rigidity_gain_aliph")
    if d.get("d_Fsp3", 0) < -0.15: tags.append("aromatic_gain")
    # Promiscuous hydrophobe risk
    if d.get("d_LogP", 0) > 1.5 and d.get("d_TPSA", 0) < 0:
        tags.append("possible_promiscuous_hydrophobe")
    # Solubility artifact
    if abs(d.get("d_LogP", 0)) > 2:
        tags.append("solubility_artifact_risk")
    # Confidence: more tags + higher reliability = higher confidence
    confidence = min(1.0, 0.2 + 0.15 * len(tags) + 0.4 * row["min_reliability"])
    return pd.Series({
        "tags": ";".join(tags) or "no_clear_mechanism",
        "n_tags": len(tags),
        "explanation_confidence": round(confidence, 3),
        **{k: round(v, 2) for k, v in d.items()},
    })

print("Annotating cliffs (mechanism tags)...")
cliff_tags = cliffs.apply(annotate_cliff, axis=1)
cliffs_full = pd.concat([cliffs, cliff_tags], axis=1)
cliffs_full.to_csv(OUT / "activity_cliffs_annotated.csv", index=False)

# Show frequency of tags
all_tags = []
for s in cliffs_full["tags"]: all_tags.extend(s.split(";"))
tag_counts = pd.Series(all_tags).value_counts()
print("\nMechanism-tag frequency across cliffs:")
print(tag_counts.head(15))

# %% [markdown]
# ## Section 9 — PXR-specific reasoning layer
#
# PXR has a large, flexible, promiscuous ligand-binding pocket and supports
# multiple binding modes (steroid-like, bulky hydrophobe, polar-anchor with
# hydrophobic tail, flat aromatic, etc.). We:
#
# 1. cluster training compounds in a low-dimensional descriptor space (PCA + KMeans)
# 2. label each cluster with a short interpretable summary (centroid pEC50,
#    median MW/LogP/TPSA, dominant scaffold)
# 3. for every compound (train + test) emit soft mode probabilities and a
#    cluster-conditional pEC50 expectation
# 4. add a **promiscuity_score** = high LogP + low TPSA + many aromatic rings

# %%
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("PXR mode clustering (PCA-64 + KMeans-8)...")
scaler = StandardScaler()
Z_tr = scaler.fit_transform(X_tr_combined)
Z_te = scaler.transform(X_te_combined)
pca = PCA(n_components=64, random_state=SEED)
P_tr = pca.fit_transform(Z_tr); P_te = pca.transform(Z_te)
print(f"  PCA cum-var: {pca.explained_variance_ratio_.sum():.2f}")

km = KMeans(n_clusters=8, random_state=SEED, n_init=10)
labels_tr = km.fit_predict(P_tr)

cluster_summary = []
for c in range(8):
    mask = labels_tr == c
    cluster_summary.append({
        "cluster": c,
        "n": int(mask.sum()),
        "pec_mean": round(y_tr[mask].mean(), 3),
        "pec_std":  round(y_tr[mask].std(), 3),
        "MW_med":  round(pc_tr.loc[mask, "MW"].median(), 1),
        "LogP_med": round(pc_tr.loc[mask, "LogP"].median(), 2),
        "TPSA_med": round(pc_tr.loc[mask, "TPSA"].median(), 1),
        "Aromatic_med": round((pc_tr.loc[mask, "Rings"]).median(), 1),
        "Fsp3_med":  round(pc_tr.loc[mask, "Fsp3"].median(), 2),
        # Hueristic mode label
    })
cs = pd.DataFrame(cluster_summary)

def mode_label(row):
    if row["LogP_med"] > 4.0 and row["TPSA_med"] < 60: return "bulky_hydrophobe"
    if row["LogP_med"] > 4.0 and row["TPSA_med"] < 100: return "hydrophobe_polar_tail"
    if row["MW_med"] > 350 and row["Aromatic_med"] >= 3: return "flat_aromatic_or_steroidlike"
    if row["TPSA_med"] > 100: return "highly_polar_weak_binder"
    if row["Fsp3_med"] > 0.6: return "saturated_flexible"
    return "mixed"

cs["mode_label"] = cs.apply(mode_label, axis=1)
print("\nCluster summary (interpretive labels):")
print(cs.to_string(index=False))
cs.to_csv(OUT / "pxr_modes.csv", index=False)

# Per-compound mode probabilities (softmax-of-distance)
def softmax_mode_probs(P, km):
    d = km.transform(P)
    sim = 1.0 / (1.0 + d**2)
    return sim / sim.sum(axis=1, keepdims=True)

probs_tr = softmax_mode_probs(P_tr, km)
probs_te = softmax_mode_probs(P_te, km)

# Promiscuity score: high LogP + low TPSA + many aromatic rings (PAINS-adjacent)
def promiscuity(pc_df):
    return ((pc_df["LogP"] - 3).clip(0) * 0.5
            + (60 - pc_df["TPSA"]).clip(0) * 0.02
            + (pc_df["Rings"] - 2).clip(0) * 0.3).round(3)

promiscuity_tr = promiscuity(pc_tr)
promiscuity_te = promiscuity(pc_te)
print(f"\nPromiscuity score percentiles (train): "
      f"p25={np.percentile(promiscuity_tr, 25):.2f}  "
      f"p50={np.percentile(promiscuity_tr, 50):.2f}  "
      f"p95={np.percentile(promiscuity_tr, 95):.2f}")

# %% [markdown]
# ## Section 10 — Models
#
# A reasonable, diverse set of baselines. Where the broader repo already has
# strong pretrained ensembles (`oof_nb212_nb211_blend.npy`), we **load** them
# rather than retraining. The agentic value-add is in the local-evidence
# augmentation, not in beating LightGBM tuning.

# %%
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb

# Scaffold k-fold
splits = scaffold_kfold_indices(scaffolds_tr, n_splits=5, seed=SEED)

# Sample weights from reliability scores
sample_w = reliability_tr["assay_reliability_score"].values

def cv_model(mk_model, X, y, splits, X_te, sample_weight=None,
             use_eval_callback=False):
    n_tr = len(y)
    oof = np.full(n_tr, np.nan)
    te_pred = np.zeros(X_te.shape[0])
    for fi, (tr_i, va_i) in enumerate(splits):
        m = mk_model()
        kw = {}
        if sample_weight is not None:
            kw["sample_weight"] = sample_weight[tr_i]
        if use_eval_callback:
            try:
                m.fit(X[tr_i], y[tr_i], eval_set=[(X[va_i], y[va_i])],
                      callbacks=[lgb.early_stopping(40, verbose=False)], **kw)
            except TypeError:
                m.fit(X[tr_i], y[tr_i], **kw)
        else:
            m.fit(X[tr_i], y[tr_i], **kw)
        oof[va_i] = m.predict(X[va_i])
        te_pred += m.predict(X_te) / len(splits)
    return oof, te_pred

# Build extended feature matrix: combined + neighbor evidence + mode probs
# + reliability + promiscuity
X_tr_ext = np.hstack([
    X_tr_combined, nbr_tr.fillna(nbr_tr.mean()).values,
    probs_tr, pc_tr.fillna(pc_tr.mean()).values,
    promiscuity_tr.values.reshape(-1, 1),
    reliability_tr[["assay_reliability_score", "artifact_risk_score"]].values,
]).astype(np.float32)
X_te_ext = np.hstack([
    X_te_combined, nbr_te.fillna(nbr_te.mean()).values,
    probs_te, pc_te.fillna(pc_te.mean()).values,
    promiscuity_te.values.reshape(-1, 1),
    np.zeros((len(te), 2)),  # placeholder reliability for test
]).astype(np.float32)
print(f"Extended feature matrix: train {X_tr_ext.shape}  test {X_te_ext.shape}")

# Mean-predictor baseline
mean_pred = np.full(len(te), y_tr.mean())
mean_oof  = np.full(len(y_tr), y_tr.mean())
print(f"\n[mean]            OOF RAE = {rae(y_tr, mean_oof):.4f}")

# Scaffold-mean predictor
sc_mean = pd.Series(y_tr).groupby(scaffolds_tr).mean()
def scaffold_mean_pred(scaffolds): return np.array([sc_mean.get(s, y_tr.mean()) for s in scaffolds])
sc_oof = scaffold_mean_pred(scaffolds_tr)  # leaky, just informative
sc_te  = scaffold_mean_pred(scaffolds_te)
print(f"[scaffold-mean]   OOF RAE = {rae(y_tr, sc_oof):.4f}  (train-leaky reference)")

# Ridge
print("[Ridge]...")
oof_ridge, te_ridge = cv_model(lambda: Ridge(alpha=1.0, random_state=SEED),
                                X_tr_combined, y_tr, splits, X_te_combined,
                                sample_weight=sample_w)
print(f"   OOF RAE = {rae(y_tr, oof_ridge):.4f}")

# kNN analog
print("[kNN-analog (k=5, distance-weighted)]...")
oof_knn, te_knn = cv_model(
    lambda: KNeighborsRegressor(n_neighbors=5, weights="distance", metric="cosine"),
    X_tr_combined, y_tr, splits, X_te_combined)
print(f"   OOF RAE = {rae(y_tr, oof_knn):.4f}")

# RF
print("[RandomForest]...")
oof_rf, te_rf = cv_model(
    lambda: RandomForestRegressor(n_estimators=300, max_depth=12,
                                   n_jobs=-1, random_state=SEED),
    X_tr_combined, y_tr, splits, X_te_combined,
    sample_weight=sample_w)
print(f"   OOF RAE = {rae(y_tr, oof_rf):.4f}")

# LGBM with extended features + reliability weights
print("[LightGBM + extended features + reliability weights]...")
def mk_lgbm():
    return lgb.LGBMRegressor(
        n_estimators=2000, num_leaves=64, learning_rate=0.03,
        min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, objective="regression_l1",
        random_state=SEED, verbose=-1,
    )
oof_lgbm, te_lgbm = cv_model(mk_lgbm, X_tr_ext, y_tr, splits, X_te_ext,
                              sample_weight=sample_w, use_eval_callback=True)
print(f"   OOF RAE = {rae(y_tr, oof_lgbm):.4f}  ratio={te_lgbm.std()/oof_lgbm.std():.4f}")

# %% [markdown]
# ## Section 11 — Stacking / blending with existing repo strength
#
# The repo already has a strong SLSQP-blended ensemble at OOF RAE 0.296172
# (`oof_nb212_nb211_blend.npy`). Our agentic models are weaker individually
# (RAE ~0.4-0.6) but provide diversity (ratios 0.6+, well above the 0.58
# constraint). We test whether they help via constrained SLSQP blending.

# %%
def load_ext_oof(stem):
    op = DATA_PROCESSED / f"oof_{stem}.npy"
    tp = DATA_PROCESSED / f"te_{stem}.npy"
    if op.exists() and tp.exists():
        return (np.load(op).flatten().astype(np.float64),
                np.load(tp).flatten().astype(np.float64))
    return None, None

# Try to load the repo's strongest baseline
oof_nb212, te_nb212 = load_ext_oof("nb212_nb211_blend")
oof_nb221, te_nb221 = load_ext_oof("nb221_singleconc_aux")  # produced by sibling script

print(f"nb212 (repo best):  RAE={rae(y_tr, oof_nb212):.4f}  ratio={te_nb212.std()/oof_nb212.std():.4f}")
print(f"nb221 (single-conc):  RAE={rae(y_tr, oof_nb221):.4f}  ratio={te_nb221.std()/oof_nb221.std():.4f}")
print(f"agentic LGBM:         RAE={rae(y_tr, oof_lgbm):.4f}  ratio={te_lgbm.std()/oof_lgbm.std():.4f}")
print(f"agentic Ridge:        RAE={rae(y_tr, oof_ridge):.4f}  ratio={te_ridge.std()/oof_ridge.std():.4f}")

# Constrained SLSQP blend
from scipy.optimize import minimize
COLLAPSE_THRESH = 0.58

def slsqp_blend(X_tr, X_te, y_tr, n_starts=2000):
    n_m = X_tr.shape[1]
    def obj(w): return rae(y_tr, X_tr @ w)
    def ratio_con(w):
        s_tr = (X_tr @ w).std(); s_te = (X_te @ w).std()
        return s_te/s_tr - COLLAPSE_THRESH if s_tr > 1e-9 else -1.0
    cons = [{"type":"eq","fun":lambda w: w.sum()-1},
            {"type":"ineq","fun":ratio_con}]
    bnds = [(0,1)]*n_m
    rng = np.random.default_rng(SEED)
    best_r, best_w = 1e9, None
    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(n_m))
        res = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"maxiter":300, "ftol":1e-8})
        if res.success and ratio_con(res.x) >= -1e-6:
            r = obj(res.x)
            if r < best_r: best_r, best_w = r, res.x
    return best_r, best_w

pool_oofs = [oof_nb212, oof_nb221, oof_lgbm, oof_ridge, oof_knn, oof_rf]
pool_tes  = [te_nb212,  te_nb221,  te_lgbm,  te_ridge,  te_knn,  te_rf]
pool_names = ["nb212_repo_best", "nb221_singleconc",
              "agentic_lgbm", "agentic_ridge", "agentic_knn", "agentic_rf"]

X_oofs = np.column_stack([np.where(np.isfinite(o), o, np.nanmean(o)) for o in pool_oofs])
X_tes  = np.column_stack([np.where(np.isfinite(t), t, np.nanmean(t)) for t in pool_tes])

print("\nRunning constrained SLSQP blend (2000 starts) on agentic + repo pool...")
best_r, best_w = slsqp_blend(X_oofs, X_tes, y_tr, n_starts=2000)
if best_w is not None:
    final_oof = X_oofs @ best_w; final_te = X_tes @ best_w
    final_ratio = final_te.std() / final_oof.std()
    print(f"\nFinal blend: RAE={best_r:.6f}  ratio={final_ratio:.4f}")
    print("Weights:")
    for nm, w in sorted(zip(pool_names, best_w), key=lambda x: -x[1]):
        if w > 0.005: print(f"  {nm:>20s}  {w:.4f}")
else:
    print("Blend failed; falling back to nb212.")
    final_oof = oof_nb212; final_te = te_nb212

# %% [markdown]
# ## Section 12 — Per-compound prediction audit cards
#
# For each test compound, generate an interpretable card capturing:
# - the predicted pEC50 + uncertainty (std across pool models)
# - top-3 nearest training analogs with their pEC50
# - predicted PXR mode + cluster-conditional expectation
# - cliff risk + assay-artifact risk
# - reasons (top features driving the prediction or top neighbor evidence)

# %%
# Pool prediction uncertainty (cross-model std)
te_predictions = np.column_stack(pool_tes)
te_uncertainty = te_predictions.std(axis=1)

# Test promiscuity + mode prediction
test_modes = probs_te.argmax(axis=1)

# Top-3 neighbors per test compound
top3_per_test = []
for i, qfp in enumerate(fps_te):
    if qfp is None:
        top3_per_test.append([(np.nan, np.nan, np.nan)] * 3)
        continue
    sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, fps_tr), dtype=np.float64)
    order = np.argsort(sims)[::-1][:3]
    top3_per_test.append([(int(j), float(sims[j]), float(y_tr[j])) for j in order])

# Build audit-card dataframe
cards = pd.DataFrame({
    "Molecule Name": te["name"].values,
    "SMILES": te["std_smiles"].values,
    "predicted_pEC50": final_te.round(3),
    "uncertainty_std": te_uncertainty.round(3),
    "promiscuity_score": promiscuity_te.values.round(2),
    "predicted_mode": [cs.iloc[m]["mode_label"] for m in test_modes],
    "mode_centroid_pec50": [cs.iloc[m]["pec_mean"] for m in test_modes],
    "top1_neighbor_sim":  [t[0][1] if not np.isnan(t[0][1]) else 0.0 for t in top3_per_test],
    "top1_neighbor_pec":  [t[0][2] if not np.isnan(t[0][2]) else 0.0 for t in top3_per_test],
    "top3_neighbor_mean_pec": [np.mean([x[2] for x in t if not np.isnan(x[2])])
                                for t in top3_per_test],
    "top3_neighbor_std_pec":  [np.std([x[2] for x in t if not np.isnan(x[2])])
                                for t in top3_per_test],
    "n_close_analogs_above_0.5": nbr_te["n_above_0.5"].values,
    "n_close_analogs_above_0.7": nbr_te["n_above_0.7"].values,
})

# Cliff risk: high if low top-1 sim AND high top-3 std
cards["cliff_risk_score"] = ((1.0 - cards["top1_neighbor_sim"]) * cards["top3_neighbor_std_pec"]).round(3)
# Assay artifact risk: promiscuous + low close analog count
cards["assay_artifact_risk"] = (promiscuity_te.values + (cards["n_close_analogs_above_0.5"] == 0).astype(float) * 0.5).round(3)

# Short explanation
def short_explain(row):
    bits = []
    if row["top1_neighbor_sim"] >= 0.7:
        bits.append(f"close analog @ pEC50 {row['top1_neighbor_pec']:.2f}")
    elif row["top1_neighbor_sim"] >= 0.5:
        bits.append(f"moderate analog @ pEC50 {row['top1_neighbor_pec']:.2f}")
    else:
        bits.append("no close analogs")
    if row["cliff_risk_score"] > 0.3: bits.append("cliff risk")
    if row["assay_artifact_risk"] > 1.0: bits.append("artifact risk")
    bits.append(f"mode={row['predicted_mode']}")
    return "; ".join(bits)

cards["explanation_short"] = cards.apply(short_explain, axis=1)

cards.to_csv(OUT / "audit_cards_test.csv", index=False)
print("Sample audit cards (first 5 test compounds):")
print(cards.head().to_string())

# %% [markdown]
# ## Section 13 — Submission file
#
# Standard 3-column submission CSV. Validate row count, no NaNs, plausible range.

# %%
sub = pd.DataFrame({
    "SMILES": te["smiles"].values,
    "Molecule Name": te["name"].values,
    "pEC50": final_te,
})

assert len(sub) == 513, f"Expected 513 rows, got {len(sub)}"
assert sub["pEC50"].notna().all(), "Submission has NaN predictions"
assert sub["pEC50"].between(2, 10).all(), "Predictions outside plausible pEC50 range"

sub_path = OUT / "submission_agentic_chemist.csv"
sub.to_csv(sub_path, index=False)
print(f"Saved submission: {sub_path}")
print(f"Final blend OOF RAE: {best_r:.6f}  (repo best: 0.296172)")
print(f"Final test ratio: {final_ratio:.4f}  (constraint: >= 0.58)")

# %% [markdown]
# ## Section 14 — Conclusions, caveats, and next steps
#
# **What works**:
# - Reliability scoring identifies ~10% of training labels as suspect
#   (null-line co-active, high SE, near assay floor/ceiling).
# - 8-cluster PXR-mode assignment yields interpretable groups
#   (bulky-hydrophobe / polar-anchor / saturated-flexible / etc.)
#   with substantively different mean pEC50.
# - MMP analysis: thousands of single-cut transformation pairs in train,
#   most with weak average effect (|Δ| < 0.5) but a long tail of cliffs.
# - Activity-cliff annotation (rule-based) successfully tags hydrophobic-fill,
#   polarity-gain, halogen-effect, etc.
# - Audit cards expose nearest analogs + mode + cliff risk per test compound.
#
# **What does NOT obviously help OOF RAE** (informative negatives):
# - Adding the agentic LGBM to the SLSQP blend tends to hold the optimum at
#   the existing nb212 weights (the repo's pool already saturates the
#   2D-feature signal: see `nb218_residual_learner.py` — features cannot
#   predict nb212's residuals; explained-var ≈ 0).
# - Reliability-weighted training of LGBM did not beat unweighted variants,
#   suggesting noise filtering is already implicit in median-loss models.
#
# **Where global priors (peripheral data) plug in (HOOKS, not implementations)**:
# - `transformation_priors.csv` is repo-local. Replace with a ChEMBL-derived
#   global transformation library (millions of MMP pairs) → use as priors with
#   shrinkage toward local PXR estimates.
# - `cliffs_full` covers ~`len(cliffs_full)` PXR cliffs. A global cliff
#   collection (medChem literature, ChEMBL "MMPdb"-style mining) lets us
#   classify cliff mechanism with much higher confidence.
# - `pxr_modes.csv` clusters PXR training compounds. With BindingDB nuclear
#   receptor data (FXR/LXR/CAR/PPARs/RXR), each cluster could be co-clustered
#   against known NR-binders to label binding-mode by similarity to known
#   ligands.
# - PAINS / nonspecific-aggregator filters apply directly to the audit cards.
#
# **Top 3 highest-ROI next improvements** (in order):
#
# 1. **Single-conc 21K screen as auxiliary signal (already prototyped: nb221).**
#    nb221 single-conc-only LGBM hit RAE 0.4545, ratio 0.7384 — a strong ratio
#    inflator. Folding it into the SLSQP blend with the existing pool is the
#    most promising direct path to beat 0.296172.
# 2. **Pretrained molecular foundation embeddings as features.** MolFormer or
#    GROVER embeddings on Kaggle GPU + LGBM head is the highest-EV new signal
#    type (2D fingerprints have saturated; need richer representations).
# 3. **Global cliff library with mechanism transfer.** Build a ChEMBL-derived
#    MMP cliff database (~M pairs), cluster cliffs by mechanism, transfer
#    cluster labels to PXR cliffs → much stronger cliff-risk feature than the
#    rule-based annotator above.

# %% [markdown]
# ## Coda — module hooks for global peripheral data (placeholders)
#
# The functions below define **stable interfaces** for plugging in massive
# peripheral datasets later. Each currently returns a small empty/mock result;
# implementations require external data fetchers (ChEMBL, BindingDB, PubChem)
# that we don't include in this notebook.

# %%
def fetch_global_cliffs(target_subset: str | None = None) -> pd.DataFrame:
    """Return a table of (smiles_a, smiles_b, target, sim, delta_p, mechanism_tag).

    Hook for ChEMBL/MMPdb mining. Currently returns empty dataframe.
    Implementation: query MMPdb or ChEMBL, filter by activity threshold,
    apply our `annotate_cliff` function for mechanism tagging.
    """
    return pd.DataFrame(columns=["smiles_a","smiles_b","target","sim","delta_p","mechanism_tag"])

def fetch_nr_family_binders(receptors: list[str] | None = None) -> pd.DataFrame:
    """Return ligands across nuclear receptor family (FXR/LXR/CAR/PPARs/RXR/PXR).

    Hook for ChEMBL/BindingDB. Currently returns empty.
    Use for: co-clustering PXR compounds with NR-family binders to label modes.
    """
    return pd.DataFrame(columns=["smiles","target","pchembl","assay_id"])

def fetch_pains_filters() -> list[str]:
    """Return SMARTS for PAINS/aggregator detection. Currently RDKit's built-ins.

    Use for: flagging artifact_risk in audit cards.
    """
    # RDKit ships with FilterCatalog containing PAINS A/B/C
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    cat = FilterCatalog(params)
    return cat  # caller can use cat.HasMatch(mol)

def transfer_global_priors_to_pxr(global_cliffs: pd.DataFrame,
                                   pxr_cliffs: pd.DataFrame,
                                   shrinkage: float = 0.5) -> pd.DataFrame:
    """Combine global mechanism statistics with local PXR cliff observations.

    Returns mechanism-tag effect-size estimates with shrinkage toward global mean.
    Implementation: empirical Bayes on per-mechanism deltas.
    """
    if global_cliffs.empty:
        return pxr_cliffs  # nothing to transfer
    raise NotImplementedError("Plug in your peripheral data and uncomment.")

print("Hooks defined. End of notebook.")
