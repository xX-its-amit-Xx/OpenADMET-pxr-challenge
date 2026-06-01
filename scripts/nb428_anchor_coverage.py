"""nb428 -- Per-compound external anchor coverage on the 513 test set.

For each test compound, quantify presence/strength of:
  (1) BindingDB exact inchikey hit (nb418 logic)
  (2) BindingDB Tanimoto >= 0.5 NN hit
  (3) Tox21 AID 1346985 inchikey match
  (4) nb318 Tanimoto-OOD top-1 sim to train (anchor = sim >= 0.5)
  (5) nb324 distillation agreement (compounds where student pred deviates
      meaningfully from nb320 -- low agreement => low anchor)

Cross-tabulate against nb320 RAE on the 253 unblind subset:
  - RAE on compounds WITH any anchor vs WITHOUT
  - Top-30 chemprop-error compounds: how many have anchor coverage?

Saves data/processed/nb428_anchor_coverage.csv.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi, AllChem, DataStructs

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, DATA_RAW

RDLogger.DisableLog("rdApp.*")


def safe_ikey(smi):
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        return inchi.MolToInchiKey(m) if m else None
    except Exception:
        return None


def morgan_bits(smi):
    try:
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
    except Exception:
        return None


def main():
    print("=== nb428: External anchor coverage on the 513 test ===\n")

    # --- core data ---
    te_df = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    te_names = te_df["Molecule Name"].tolist()
    te_smis = te_df["SMILES"].tolist()
    n_te = len(te_smis)
    print(f"Test rows: {n_te}")

    tr = load_train()
    tr = add_standard_columns(tr)
    smi_tr = tr["std_smiles"].tolist()
    ik_tr = set(safe_ikey(s) for s in smi_tr)
    n_tr = len(smi_tr)
    print(f"Train rows: {n_tr}")

    ik_te = [safe_ikey(s) for s in te_smis]

    # ============================================================
    # (1) BindingDB exact inchikey hits
    # ============================================================
    print("\n[1] BindingDB exact inchikey lookup...")
    direct_p = DATA_EXTERNAL / "bindingdb_pxr_direct.parquet"
    nr_p = DATA_EXTERNAL / "bindingdb_nr_data.parquet"
    bdb_frames = []
    if direct_p.exists():
        d = pd.read_parquet(direct_p)
        if "inchikey" in d.columns and "pec50" in d.columns:
            bdb_frames.append(d[["inchikey", "pec50"]].copy())
            print(f"  pxr_direct: {len(d)} rows")
    if nr_p.exists():
        n = pd.read_parquet(nr_p)
        pxr_mask = (
            (n.get("uniprot", "") == "O75469")
            | (n["target_name"].astype(str).str.contains("PXR", case=False, na=False))
            | (n["target_name"].astype(str).str.contains("NR1I2", case=False, na=False))
        )
        n = n[pxr_mask]
        if "inchikey" in n.columns and "pec50" in n.columns:
            bdb_frames.append(n[["inchikey", "pec50"]].copy())
            print(f"  bindingdb_nr PXR-only: {len(n)} rows")
    bdb = pd.concat(bdb_frames, ignore_index=True).dropna(subset=["pec50", "inchikey"])
    bdb["pec50"] = bdb["pec50"].clip(3.0, 10.0)
    bdb_by_ik = bdb.groupby("inchikey")["pec50"].median().to_dict()
    print(f"  unique BindingDB inchikeys (PXR): {len(bdb_by_ik)}")

    bdb_exact_hit = np.array(
        [1 if (k is not None and k in bdb_by_ik) else 0 for k in ik_te],
        dtype=np.int8,
    )
    bdb_exact_pec50 = np.array(
        [bdb_by_ik.get(k, np.nan) if k else np.nan for k in ik_te],
        dtype=np.float32,
    )
    print(f"  test exact inchikey hits: {int(bdb_exact_hit.sum())}/{n_te}")

    # ============================================================
    # (2) BindingDB Tanimoto >= 0.5 NN hits
    # ============================================================
    print("\n[2] BindingDB Tanimoto >= 0.5 NN lookup...")
    # Build FPs over the unique-inchikey BindingDB set -- re-load originals for smiles
    bdb_with_smi = []
    if direct_p.exists():
        d2 = pd.read_parquet(direct_p)
        cols = [c for c in ("inchikey", "smiles", "std_smiles") if c in d2.columns]
        bdb_with_smi.append(d2[cols])
    if nr_p.exists():
        n2 = pd.read_parquet(nr_p)
        pxr_mask = (
            (n2.get("uniprot", "") == "O75469")
            | (n2["target_name"].astype(str).str.contains("PXR", case=False, na=False))
            | (n2["target_name"].astype(str).str.contains("NR1I2", case=False, na=False))
        )
        n2 = n2[pxr_mask]
        cols = [c for c in ("inchikey", "smiles", "std_smiles") if c in n2.columns]
        bdb_with_smi.append(n2[cols])
    bsmi = pd.concat(bdb_with_smi, ignore_index=True).dropna(subset=["inchikey"])
    bsmi = bsmi.drop_duplicates(subset=["inchikey"])
    smi_col = "std_smiles" if "std_smiles" in bsmi.columns and bsmi["std_smiles"].notna().any() else "smiles"
    bsmi = bsmi[bsmi["inchikey"].isin(bdb_by_ik.keys())].copy()

    print(f"  building FPs for {len(bsmi)} BindingDB compounds...")
    bdb_fps = []
    for s in bsmi[smi_col].tolist():
        fp = morgan_bits(s)
        if fp is not None:
            bdb_fps.append(fp)
        else:
            bdb_fps.append(None)
    bdb_valid_idx = [i for i, fp in enumerate(bdb_fps) if fp is not None]
    bdb_fps_valid = [bdb_fps[i] for i in bdb_valid_idx]
    print(f"  valid FPs: {len(bdb_fps_valid)}")

    bdb_nn_hit = np.zeros(n_te, dtype=np.int8)
    bdb_top1_sim = np.zeros(n_te, dtype=np.float32)
    for i, smi in enumerate(te_smis):
        fp = morgan_bits(smi)
        if fp is None or not bdb_fps_valid:
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, bdb_fps_valid),
                        dtype=np.float32)
        top = float(sims.max())
        bdb_top1_sim[i] = top
        if top >= 0.5:
            bdb_nn_hit[i] = 1
    print(f"  test NN(Tanimoto>=0.5) hits: {int(bdb_nn_hit.sum())}/{n_te}")

    # ============================================================
    # (3) Tox21 AID 1346985 inchikey matches
    # ============================================================
    print("\n[3] Tox21 AID 1346985 inchikey lookup...")
    tox_p = DATA_EXTERNAL / "pubchem_aid_1346985_tox21_pxr" / "aid_1346985.parquet"
    tox_hit = np.zeros(n_te, dtype=np.int8)
    if tox_p.exists():
        tox = pd.read_parquet(tox_p)
        if "inchikey" in tox.columns:
            tox_iks = set(tox["inchikey"].dropna().tolist())
        else:
            tox_iks = set(safe_ikey(s) for s in tox.get("std_smiles", tox.get("smiles", [])).tolist())
        print(f"  Tox21 unique inchikeys: {len(tox_iks)}")
        tox_hit = np.array([1 if (k and k in tox_iks) else 0 for k in ik_te],
                           dtype=np.int8)
    print(f"  test Tox21 inchikey hits: {int(tox_hit.sum())}/{n_te}")

    # ============================================================
    # (4) nb318 Tanimoto-OOD: top-1 sim to TRAIN
    # ============================================================
    print("\n[4] Top-1 Tanimoto to train (nb318 OOD distance)...")
    tr_fps = []
    for s in smi_tr:
        fp = morgan_bits(s)
        if fp is not None:
            tr_fps.append(fp)
    print(f"  train FPs: {len(tr_fps)}")
    te_top1_tr_sim = np.zeros(n_te, dtype=np.float32)
    for i, smi in enumerate(te_smis):
        fp = morgan_bits(smi)
        if fp is None:
            continue
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, tr_fps),
                        dtype=np.float32)
        te_top1_tr_sim[i] = float(sims.max())
    train_nn_anchor = (te_top1_tr_sim >= 0.5).astype(np.int8)
    print(f"  test compounds with train top-1 sim >= 0.5: {int(train_nn_anchor.sum())}/{n_te}")
    print(f"  median test top-1 sim to train: {np.median(te_top1_tr_sim):.3f}")

    # ============================================================
    # (5) nb324 distillation coverage: agreement w/ nb320 ensemble
    # ============================================================
    print("\n[5] nb324 distillation vs nb320 agreement...")
    te_nb324 = np.load(DATA_PROCESSED / "te_nb324_distilled.npy")
    te_nb320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    # nonzero agreement = both predictors fall in a reasonable band
    distill_dev = np.abs(te_nb324 - te_nb320)
    # Define "covered" as |dev| <= 0.5 log-units (predictors agree)
    distill_cov = (distill_dev <= 0.5).astype(np.int8)
    print(f"  median |nb324 - nb320|: {np.median(distill_dev):.3f}")
    print(f"  agreement (<=0.5 logU): {int(distill_cov.sum())}/{n_te}")

    # ============================================================
    # Build per-compound coverage table
    # ============================================================
    cov = pd.DataFrame({
        "molecule_name": te_names,
        "smiles": te_smis,
        "inchikey": ik_te,
        "bdb_exact_hit": bdb_exact_hit,
        "bdb_exact_pec50": bdb_exact_pec50,
        "bdb_nn_hit_tanimoto_ge_0p5": bdb_nn_hit,
        "bdb_top1_sim": bdb_top1_sim,
        "tox21_aid1346985_hit": tox_hit,
        "train_top1_sim": te_top1_tr_sim,
        "train_nn_anchor_ge_0p5": train_nn_anchor,
        "distill_dev_nb324_vs_nb320": distill_dev,
        "distill_agreement_le_0p5": distill_cov,
    })
    # any-anchor flag = exact bdb OR nn bdb OR tox21 hit OR strong train neighbor
    cov["any_external_anchor"] = (
        (cov["bdb_exact_hit"] == 1)
        | (cov["bdb_nn_hit_tanimoto_ge_0p5"] == 1)
        | (cov["tox21_aid1346985_hit"] == 1)
    ).astype(np.int8)
    cov["any_anchor_or_strong_train_nn"] = (
        (cov["any_external_anchor"] == 1)
        | (cov["train_nn_anchor_ge_0p5"] == 1)
    ).astype(np.int8)

    print("\nCoverage summary:")
    print(f"  bdb_exact      : {cov['bdb_exact_hit'].sum()}")
    print(f"  bdb_nn>=0.5    : {cov['bdb_nn_hit_tanimoto_ge_0p5'].sum()}")
    print(f"  tox21_hit      : {cov['tox21_aid1346985_hit'].sum()}")
    print(f"  any_external   : {cov['any_external_anchor'].sum()}")
    print(f"  any_or_trainNN : {cov['any_anchor_or_strong_train_nn'].sum()}")

    # ============================================================
    # Cross-tab vs unblind: nb320 RAE anchored vs unanchored
    # ============================================================
    print("\n[6] Cross-tab vs 253-compound unblind...")
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(te_names)}
    keep = unb["Molecule Name"].isin(name_to_idx)
    unb_kept = unb[keep].copy()
    unb_idx = np.array([name_to_idx[n] for n in unb_kept["Molecule Name"]])
    unb_y = unb_kept["pEC50"].values.astype(np.float64)
    print(f"  unblind matched: {len(unb_idx)}")

    # nb320 errors on unblind
    nb320_unb = te_nb320[unb_idx]
    nb320_err = np.abs(nb320_unb - unb_y)
    overall_rae = rae(unb_y, nb320_unb)
    print(f"\n  nb320 unblind RAE (all 253): {overall_rae:.4f}")

    cov_sub = cov.iloc[unb_idx].reset_index(drop=True)
    anchored_mask = cov_sub["any_external_anchor"].values == 1
    print(f"  unblind compounds with any external anchor: {int(anchored_mask.sum())}/{len(unb_y)}")
    if anchored_mask.sum() >= 2:
        anchored_rae = rae(unb_y[anchored_mask], nb320_unb[anchored_mask])
        print(f"  nb320 RAE on ANCHORED unblind subset:    {anchored_rae:.4f}  "
              f"(n={anchored_mask.sum()}, mean |err|={nb320_err[anchored_mask].mean():.3f})")
    else:
        anchored_rae = float("nan")
        print("  (anchored subset too small for RAE)")
    if (~anchored_mask).sum() >= 2:
        unanchored_rae = rae(unb_y[~anchored_mask], nb320_unb[~anchored_mask])
        print(f"  nb320 RAE on UNANCHORED unblind subset:  {unanchored_rae:.4f}  "
              f"(n={(~anchored_mask).sum()}, mean |err|={nb320_err[~anchored_mask].mean():.3f})")
    else:
        unanchored_rae = float("nan")

    # Same cross-tab using ANY anchor OR strong train neighbor
    am2 = cov_sub["any_anchor_or_strong_train_nn"].values == 1
    print(f"\n  unblind with any anchor OR train sim>=0.5: {int(am2.sum())}/{len(unb_y)}")
    if am2.sum() >= 2:
        print(f"  nb320 RAE on (any or trainNN) subset:    "
              f"{rae(unb_y[am2], nb320_unb[am2]):.4f}  (n={am2.sum()})")
    if (~am2).sum() >= 2:
        print(f"  nb320 RAE on neither subset:             "
              f"{rae(unb_y[~am2], nb320_unb[~am2]):.4f}  (n={(~am2).sum()})")

    # ============================================================
    # Chemprop mean error on anchored subset (systematic bias?)
    # ============================================================
    print("\n[7] Is chemprop mean WRONG on anchored compounds?")
    chemprop_te = np.load(DATA_PROCESSED / "te_chemprop_aux.npy")
    chemprop_unb = chemprop_te[unb_idx]
    chemprop_err = chemprop_unb - unb_y  # signed
    chemprop_abs = np.abs(chemprop_err)
    if anchored_mask.sum() >= 2:
        print(f"  chemprop mean signed error (anchored)  : {chemprop_err[anchored_mask].mean():+.3f}")
        print(f"  chemprop mean |err|        (anchored)  : {chemprop_abs[anchored_mask].mean():.3f}")
        print(f"  chemprop RAE              (anchored)  : "
              f"{rae(unb_y[anchored_mask], chemprop_unb[anchored_mask]):.4f}")
    if (~anchored_mask).sum() >= 2:
        print(f"  chemprop mean signed error (unanchored): {chemprop_err[~anchored_mask].mean():+.3f}")
        print(f"  chemprop mean |err|        (unanchored): {chemprop_abs[~anchored_mask].mean():.3f}")
        print(f"  chemprop RAE              (unanchored): "
              f"{rae(unb_y[~anchored_mask], chemprop_unb[~anchored_mask]):.4f}")

    # ============================================================
    # Top-30 chemprop-error compounds: anchor coverage?
    # ============================================================
    print("\n[8] Top-30 chemprop-error unblind compounds & their anchor coverage:")
    order = np.argsort(chemprop_abs)[::-1][:30]
    rows = []
    for k in order:
        cov_row = cov_sub.iloc[k]
        rows.append({
            "name": unb_kept["Molecule Name"].iloc[k],
            "true": float(unb_y[k]),
            "chemprop": float(chemprop_unb[k]),
            "abs_err": float(chemprop_abs[k]),
            "nb320": float(nb320_unb[k]),
            "nb320_err": float(np.abs(nb320_unb[k] - unb_y[k])),
            "bdb_exact": int(cov_row["bdb_exact_hit"]),
            "bdb_nn05": int(cov_row["bdb_nn_hit_tanimoto_ge_0p5"]),
            "tox21": int(cov_row["tox21_aid1346985_hit"]),
            "train_top1": float(cov_row["train_top1_sim"]),
            "any_anchor": int(cov_row["any_external_anchor"]),
        })
    top30 = pd.DataFrame(rows)
    print(top30.to_string(index=False))
    n_top30_anchored = int(top30["any_anchor"].sum())
    print(f"\n  of top-30 chemprop-error compounds, {n_top30_anchored}/30 have any external anchor")
    n_top30_any_or_train = int(
        (
            (top30["any_anchor"] == 1)
            | (top30["train_top1"] >= 0.5)
        ).sum()
    )
    print(f"  of top-30, {n_top30_any_or_train}/30 have any anchor OR train sim>=0.5")

    # ============================================================
    # Save the table
    # ============================================================
    out_csv = DATA_PROCESSED / "nb428_anchor_coverage.csv"
    cov.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Final scalar summary for parent agent
    print("\n" + "=" * 60)
    print("SCALAR SUMMARY")
    print("=" * 60)
    print(f"bdb_exact_hits        = {int(cov['bdb_exact_hit'].sum())}")
    print(f"bdb_nn_ge_0p5_hits    = {int(cov['bdb_nn_hit_tanimoto_ge_0p5'].sum())}")
    print(f"tox21_hits            = {int(cov['tox21_aid1346985_hit'].sum())}")
    print(f"any_external_anchor   = {int(cov['any_external_anchor'].sum())}")
    print(f"any_or_train_nn       = {int(cov['any_anchor_or_strong_train_nn'].sum())}")
    print(f"unblind anchored RAE  = {anchored_rae:.4f}  (n={int(anchored_mask.sum())})")
    print(f"unblind unanchored RAE= {unanchored_rae:.4f}  (n={int((~anchored_mask).sum())})")
    print(f"top30 chemprop-err with any_anchor = {n_top30_anchored}/30")


if __name__ == "__main__":
    main()
