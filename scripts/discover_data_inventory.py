"""Inventory the NEW data files (CHANGELOG 2026-05-27) for modeling impact.

Three new raw drops:
  - pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv  (96 rows, semi-pure DRC)
  - pxr-challenge_htchem-libraries_TRAIN.csv              (456 rows, crude DRC)
  - pxr-challenge_TEST_PHASE_1_UNBLINDED.csv              (253 rows, unblinded test labels)

For each source:
  * row count, schema, value ranges (pEC50/Emax/log2FC), SMILES validation rate, unique InChIKeys
Cross-source set ops vs existing TRAIN (4139) and TEST_513:
  * compounds NOT in existing train (additive label gain)
  * compounds NOT in test-513    (safe to use as train; no leakage)
  * compounds overlapping test-513 (LEAKAGE if used as train -- exclude!)
Scaffold-coverage shift: novel Murcko scaffolds added by each source.
Tanimoto shift: median top-1 sim of test-513 to OLD train vs OLD+NEW.
Plausible LB lift projection from delta-coverage + recent loss-curve calibration.

Output: data/processed/new_data_inventory.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/pxr importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxr import data as pxr_data  # noqa: E402
from pxr.chem import (  # noqa: E402
    bemis_murcko,
    morgan_fp_batch,
    standardize_smiles,
    to_inchikey,
)
from pxr.paths import DATA_PROCESSED  # noqa: E402


def _summarize_numeric(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if not len(s):
        return {"n": 0}
    return {
        "n": int(len(s)),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "std": float(s.std()),
    }


def _smiles_qc(smiles: pd.Series) -> tuple[pd.Series, pd.Series, dict]:
    """Standardize + InChIKey each SMILES. Return std_smiles, inchikey, qc-dict."""
    std = smiles.fillna("").map(standardize_smiles)
    ik = std.map(lambda s: to_inchikey(s) if s else None)
    n = len(smiles)
    n_parsed = int(std.notna().sum())
    n_ik = int(ik.notna().sum())
    qc = {
        "n_rows": n,
        "n_smiles_parsed": n_parsed,
        "smiles_valid_rate": round(n_parsed / max(n, 1), 4),
        "n_unique_inchikeys": int(ik.dropna().nunique()),
        "n_inchikeys_resolved": n_ik,
    }
    return std, ik, qc


def _inventory_source(name: str, df: pd.DataFrame, smiles_col: str = "smiles") -> dict:
    out = {
        "name": name,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "n_columns": int(df.shape[1]),
    }
    if smiles_col not in df.columns:
        out["error"] = f"missing smiles col '{smiles_col}'"
        return out
    std, ik, qc = _smiles_qc(df[smiles_col])
    out.update(qc)
    # Numeric summaries on any plausible label-bearing column.
    for col in ("pec50", "pec50_raw", "pec50_se", "emax", "emax_norm",
                "emax_raw", "emax_rel", "log2FC", "log2_fc"):
        if col in df.columns:
            out.setdefault("value_ranges", {})[col] = _summarize_numeric(df[col])
    return out, std, ik


def _build_fp_index(smiles: list[str]):
    fps = morgan_fp_batch(smiles)
    keep = fps.sum(axis=1) > 0
    return fps[keep], np.where(keep)[0]


def _max_tanimoto_to_set(query_fps: np.ndarray, ref_fps: np.ndarray, batch: int = 64) -> np.ndarray:
    """Vectorized max-Tanimoto for each query row against ref pool. Uint8 bit FPs."""
    q = query_fps.astype(np.float32)
    r = ref_fps.astype(np.float32)
    q_sum = q.sum(axis=1, keepdims=True)
    r_sum = r.sum(axis=1, keepdims=True).T  # (1, R)
    out = np.zeros(len(q), dtype=np.float32)
    for i in range(0, len(q), batch):
        chunk = q[i:i + batch]
        inter = chunk @ r.T  # (b, R)
        denom = q_sum[i:i + batch] + r_sum - inter
        denom = np.where(denom == 0, 1.0, denom)
        tan = inter / denom
        out[i:i + batch] = tan.max(axis=1)
    return out


def main() -> None:
    t0 = time.time()
    print("[1/6] Loading existing TRAIN + TEST...")
    train = pxr_data.load_train()
    test = pxr_data.load_test()
    counter = pxr_data.load_counter()
    print(f"  existing train rows: {len(train)} | test rows: {len(test)}")

    train_std = train["smiles"].fillna("").map(standardize_smiles)
    train_ik = train_std.map(lambda s: to_inchikey(s) if s else None)
    test_std = test["smiles"].fillna("").map(standardize_smiles)
    test_ik = test_std.map(lambda s: to_inchikey(s) if s else None)

    train_ik_set = set(train_ik.dropna().unique())
    test_ik_set = set(test_ik.dropna().unique())
    print(f"  unique train InChIKeys: {len(train_ik_set)} | test InChIKeys: {len(test_ik_set)}")

    print("[2/6] Inventorying new sources...")
    new_sources = {
        "semi_pure": pxr_data.load_semi_pure(),
        "crudes": pxr_data.load_crudes(),
        "phase1_unblinded": pxr_data.load_phase1_unblinded(),
    }

    per_source = {}
    std_by_src = {}
    ik_by_src = {}
    for name, df in new_sources.items():
        print(f"  - {name}: {len(df)} rows")
        info, std, ik = _inventory_source(name, df)
        per_source[name] = info
        std_by_src[name] = std
        ik_by_src[name] = ik

    print("[3/6] Computing set differences vs existing train + test-513...")
    set_diffs = {}
    for name, ik in ik_by_src.items():
        ik_valid = ik.dropna()
        ik_set = set(ik_valid.unique())
        not_in_train = ik_set - train_ik_set
        overlap_test = ik_set & test_ik_set
        usable_as_train = ik_set - test_ik_set  # safe to add to train (no leakage)
        novel_overall = ik_set - train_ik_set - test_ik_set
        set_diffs[name] = {
            "n_unique_compounds": len(ik_set),
            "n_overlap_existing_train": len(ik_set & train_ik_set),
            "n_NOT_in_existing_train": len(not_in_train),
            "n_overlap_test_513_LEAKAGE": len(overlap_test),
            "n_usable_as_train_no_leakage": len(usable_as_train),
            "n_novel_vs_train_and_test": len(novel_overall),
        }

    print("[4/6] Computing Murcko scaffold coverage shift...")
    train_scaffolds = train["smiles"].map(bemis_murcko)
    test_scaffolds = test["smiles"].map(bemis_murcko)
    train_scaf_set = set(train_scaffolds.dropna().unique())
    test_scaf_set = set(test_scaffolds.dropna().unique())
    test_scaf_missing_from_train = test_scaf_set - train_scaf_set

    scaffold_shift = {
        "existing_train_n_unique_scaffolds": len(train_scaf_set),
        "test_513_n_unique_scaffolds": len(test_scaf_set),
        "test_scaffolds_missing_from_train_BEFORE": len(test_scaf_missing_from_train),
    }

    new_scaf_union = set()
    new_scaf_test_rescued = {}
    for name, std in std_by_src.items():
        scafs = std.map(lambda s: bemis_murcko(s) if s else None)
        scafs_valid = set(scafs.dropna().unique())
        novel_vs_train = scafs_valid - train_scaf_set
        rescued = scafs_valid & test_scaf_missing_from_train
        per_source[name]["n_unique_scaffolds"] = len(scafs_valid)
        per_source[name]["n_NOVEL_scaffolds_vs_train"] = len(novel_vs_train)
        per_source[name]["n_test_scaffolds_RESCUED"] = len(rescued)
        new_scaf_union |= novel_vs_train
        new_scaf_test_rescued[name] = len(rescued)

    test_scaf_after = test_scaf_missing_from_train - new_scaf_union
    scaffold_shift.update({
        "novel_scaffolds_added_total": len(new_scaf_union),
        "test_scaffolds_missing_from_train_AFTER": len(test_scaf_after),
        "test_scaffolds_recovered": len(test_scaf_missing_from_train) - len(test_scaf_after),
        "per_source_rescued": new_scaf_test_rescued,
    })

    print("[5/6] Computing Tanimoto-similarity shift on test-513...")
    test_smi = test["smiles"].dropna().tolist()
    train_smi = train["smiles"].dropna().tolist()
    # union of all usable new compounds (excluding leakage with test-513)
    new_safe_std = []
    for name, std in std_by_src.items():
        ik = ik_by_src[name]
        mask = ik.notna() & ~ik.isin(test_ik_set)
        new_safe_std.extend(std[mask].dropna().tolist())
    # dedup by string
    new_safe_std = list(dict.fromkeys(s for s in new_safe_std if s))

    print(f"  fingerprinting test ({len(test_smi)}), old-train ({len(train_smi)}), new-safe ({len(new_safe_std)})")
    test_fps, _ = _build_fp_index(test_smi)
    old_fps, _ = _build_fp_index(train_smi)
    new_fps, _ = _build_fp_index(new_safe_std)

    sim_old = _max_tanimoto_to_set(test_fps, old_fps)
    if len(new_fps):
        combined_fps = np.vstack([old_fps, new_fps])
    else:
        combined_fps = old_fps
    sim_after = _max_tanimoto_to_set(test_fps, combined_fps)

    tanimoto_shift = {
        "test_513_top1_sim_median_OLD": float(np.median(sim_old)),
        "test_513_top1_sim_median_AFTER": float(np.median(sim_after)),
        "test_513_top1_sim_mean_OLD": float(np.mean(sim_old)),
        "test_513_top1_sim_mean_AFTER": float(np.mean(sim_after)),
        "delta_median": float(np.median(sim_after) - np.median(sim_old)),
        "n_test_with_sim_gain": int((sim_after > sim_old + 1e-6).sum()),
        "n_test_pushed_above_0.50_OLD": int((sim_old >= 0.50).sum()),
        "n_test_pushed_above_0.50_AFTER": int((sim_after >= 0.50).sum()),
        "n_test_pushed_above_0.35_OLD": int((sim_old >= 0.35).sum()),
        "n_test_pushed_above_0.35_AFTER": int((sim_after >= 0.35).sum()),
    }

    print("[6/6] Projecting RAE lift...")
    # Calibration anchors (from memory / leaderboard log):
    #  - chemprop_aux LB ~0.6246 (PRIMARY-1); honest cross-fit floor ~0.5065 (nb562)
    #  - Phase-1 unblinding alone gave +0 LB lift unless rescaffold support changes (feedback_unblind_augmentation)
    #  - F2 (greasy-novel-inactive) tail carries ~0.10 RAE prize
    # Heuristic: each rescued test scaffold deshrinks ~0.03 RAE on its row,
    #            distributed across 513 -> ~6e-5 RAE per rescued scaffold-row.
    #            Phase-1 unblinded labels enter directly as 253 additional truth rows
    #            -> for refit-on-aug, expected LB delta ~-0.01 to -0.03 (nb590 evidence).

    n_phase1_truth_rows = int(per_source["phase1_unblinded"]["n_rows"])  # 253 LABELS for test
    n_semi_pure_novel_train = set_diffs["semi_pure"]["n_NOT_in_existing_train"]
    n_crudes_novel_train = set_diffs["crudes"]["n_NOT_in_existing_train"]
    n_test_rescued_by_phase1 = scaffold_shift["per_source_rescued"]["phase1_unblinded"]
    n_test_rescued_by_aux = (
        scaffold_shift["per_source_rescued"]["semi_pure"]
        + scaffold_shift["per_source_rescued"]["crudes"]
    )

    # Anchors from memory
    rae_chemprop_aux_lb = 0.6246
    rae_floor_estimate = 0.5065  # honest cross-fit, OOD wall (nb562)
    # Phase-1 isn't augmentation -> it IS 253/513 of the leaderboard truth.
    # For the remaining 260 still-blinded rows, nb590 evidence: aug LGBM 0.5869 vs
    # baseline 0.5065 cross-fit (+0.08 worse). So augmenting train w/ phase1 alone
    # does NOT shift LB; scaffold-bridge is the real lift mechanism.
    # Conservative per-scaffold rescue lift from semi-pure+crudes (NOT phase1, that
    # only rescues scaffolds already inside the LB-revealed half):
    rescue_lift_per_aux_scaffold = 0.001
    aux_scaffold_lift = rescue_lift_per_aux_scaffold * n_test_rescued_by_aux
    # Crude/semi-pure as auxiliary heads (per memory: rt-aux head, not direct pEC50)
    crudes_aux_head_lift = 0.003 if n_crudes_novel_train > 100 else 0.001
    semi_pure_aux_head_lift = 0.001 if n_semi_pure_novel_train > 50 else 0.0005

    estimated_lift = aux_scaffold_lift + crudes_aux_head_lift + semi_pure_aux_head_lift
    projected_lb = max(rae_floor_estimate, rae_chemprop_aux_lb - estimated_lift)

    projection = {
        "anchor_current_lb_chemprop_aux": rae_chemprop_aux_lb,
        "anchor_honest_floor_nb562": rae_floor_estimate,
        "phase1_truth_rows_for_validation": n_phase1_truth_rows,
        "phase1_test_scaffolds_rescued": n_test_rescued_by_phase1,
        "components": {
            "aux_scaffold_lift": round(aux_scaffold_lift, 4),
            "crudes_aux_head_lift": crudes_aux_head_lift,
            "semi_pure_aux_head_lift": semi_pure_aux_head_lift,
        },
        "estimated_rae_reduction_blinded_half": round(estimated_lift, 4),
        "projected_lb_rae_after_full_use": round(projected_lb, 4),
        "caveats": [
            "Phase-1 unblinded = 253/513 LEADERBOARD TRUTH. Use as HONEST validation, not train augmentation.",
            "nb590 evidence: augmenting train w/ phase1 alone -> 0.5869 worse than nb562 0.5065 (OOD wall set by scaffold support, not label count).",
            "Crude/semi-pure pEC50 noisier (median SE 0.23 vs 0.13); plug as auxiliary head, not primary target.",
            "Crudes add 386 NOVEL scaffolds vs train but only 3 overlap with test-513 scaffolds -> limited bridge.",
            "F2 (greasy-novel-inactive) RAE tail ~0.10 needs off-manifold neg-mining; aux compounds unlikely to cover unless promiscuity-routed.",
            "Realistic 1-2 week ceiling: LB 0.60-0.62; floor 0.55 only with structural bridges + chemprop-aux 6-head.",
        ],
    }

    out_path = DATA_PROCESSED / "new_data_inventory.json"
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "existing_anchors": {
            "train_rows": int(len(train)),
            "train_unique_inchikeys": len(train_ik_set),
            "test_513_rows": int(len(test)),
            "test_513_unique_inchikeys": len(test_ik_set),
            "counter_rows": int(len(counter)),
        },
        "new_sources": per_source,
        "set_differences": set_diffs,
        "scaffold_shift": scaffold_shift,
        "tanimoto_shift": tanimoto_shift,
        "lift_projection": projection,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")
    print(json.dumps({k: v for k, v in payload.items() if k != "new_sources"}, indent=2)[:2000])


if __name__ == "__main__":
    main()
