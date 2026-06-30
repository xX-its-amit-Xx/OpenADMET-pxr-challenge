"""nb1562 -- DEPLOY 5-way K-tuned naive 1/5 mean (nb1553) to 513.

PROTOCOL (handoff from nb1553):
    1. Anchor = te_chemprop_aux.npy on 513   (PRE-unblind, in_RAE 0.6216 @ unb_idx).
    2. Build 5 independent deploy residuals, each:
         - features = top-K family columns + ChEMBL kNN-5 (pred_chembl_pec50 + mean_sim) on 513
         - residual target on 253 = y_unb - anchor[unb_idx]
         - 5-seed LGBM-Huber bag (alpha=1.0, d=3, n_est=80, lr=0.05, leaves=7)
         - fit on ALL 253 unblind, predict residual on 513
         - mean over 5 seeds = mean_resid_513_family   (shape (513,))
       Families (K-tuned, taken from nb1553_summary.json):
         A  top-25 AtomPair bits      (nb1553 AtomPair top_idx_ranked[:25])
         B  top-20 MACCS bits         (nb1553 MACCS top_idx_ranked[:20])
         C  top-20 Mordred cols       (nb1553 Mordred top_idx_ranked[:20])
         D  top-20 chemprop-embed     (nb1553 ChempropEmbed top_idx_ranked[:20])
         E  top-30 Avalon bits        (nb1553 Avalon top_idx_ranked[:30])
    3. te_nb1562 = anchor + (1/5) * sum of 5 mean_resid_513_family vectors.
    4. Save:
         data/processed/te_nb1562.npy                 (513,) float32
         submissions/nb1562_deploy_nb1553.csv         SMILES, Molecule Name, pEC50

Honest LB anchor (from nb1553 naive 1/5 mean cross-fit): 0.5225
Predicted LB ~ 0.526.
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
from lightgbm import LGBMRegressor

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, morgan_fp_batch
from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED

TAG = "nb1562"
ANCHOR = "chemprop_aux"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"

RESID_SEEDS = [0, 1, 7, 42, 137]

ATOMPAIR_TE_PATH = DATA_PROCESSED / "te_atompair.npy"
MACCS_TE_PATH = DATA_PROCESSED / "te_maccs.npy"
CHEMPROP_EMBED_TE_PATH = DATA_PROCESSED / "te_chemprop_embed_300.npy"
AVALON_TE_PATH = DATA_PROCESSED / "te_avalon512.npy"
MORDRED_DIR = Path("C:/pxr_artifacts/nb1030")

NB1553_SUMMARY = DATA_PROCESSED / "nb1553_summary.json"

PRED_CHEMBL_513_PATH = DATA_PROCESSED / "pred_chembl_pec50_513.npy"
SIM_CHEMBL_513_PATH = DATA_PROCESSED / "sim_chembl_513.npy"

SUBMISSIONS = Path(__file__).resolve().parents[1] / "submissions"

HONEST_LB_ANCHOR_NB1553 = 0.5225

K_AP = 25
K_MACCS = 20
K_MORD = 20
K_EMBED = 20
K_AVALON = 30


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="huber",
        alpha=1.0,
        learning_rate=0.05,
        n_estimators=80,
        max_depth=3,
        num_leaves=7,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        verbosity=-1,
        random_state=seed,
        n_jobs=2,
    )


def _load_mordred_test(n_test_expected: int) -> np.ndarray:
    mte_p = MORDRED_DIR / "X_mordred_test.npy"
    if not mte_p.exists():
        raise FileNotFoundError(f"Mordred cache missing: {mte_p}")
    X_te_m = np.load(mte_p).astype(np.float32)
    if X_te_m.shape[0] != n_test_expected:
        raise ValueError(
            f"Mordred test shape mismatch: {X_te_m.shape} vs n_test={n_test_expected}"
        )
    X_te_m = np.where(np.isfinite(X_te_m), X_te_m, np.nan).astype(np.float32)
    col_med = np.nanmedian(X_te_m, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0).astype(np.float32)
    bad = ~np.isfinite(X_te_m)
    if bad.any():
        idx_r, idx_c = np.where(bad)
        X_te_m[idx_r, idx_c] = col_med[idx_c]
    return X_te_m


def _load_chemprop_embed_test(n_test_expected: int) -> np.ndarray:
    if not CHEMPROP_EMBED_TE_PATH.exists():
        raise FileNotFoundError(f"Chemprop embed cache missing: {CHEMPROP_EMBED_TE_PATH}")
    X = np.load(CHEMPROP_EMBED_TE_PATH).astype(np.float32)
    if X.shape[0] != n_test_expected:
        raise ValueError(f"Chemprop embed shape mismatch: {X.shape}")
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return X


def _extract_family_top_idx(sum_1553: dict, family: str, K: int) -> np.ndarray:
    for f in sum_1553["families"]:
        if f["family"] == family:
            arr = np.array(f["top_idx_ranked"], dtype=int)
            if len(arr) < K:
                raise ValueError(
                    f"nb1553 {family} ranked list has {len(arr)} < K={K}"
                )
            return arr[:K]
    raise KeyError(f"{family} entry not found in nb1553_summary.json")


def _build_family_residual_deploy(
    family_name: str,
    X_full_unb: np.ndarray,   # (253, d) feature matrix for this family on unblind rows
    X_full_513: np.ndarray,   # (513, d) feature matrix for this family on full test
    residual_unb: np.ndarray, # (253,) residual target y_unb - anchor[unb_idx]
    y_unb: np.ndarray,
    anchor_unb: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Fit 5-seed LGBM-Huber bag on full 253, predict residual on 513.

    Returns mean_resid_513 (513,) and per-seed diagnostics.
    """
    print(f"\n[fam ] {family_name}  X_unb={X_full_unb.shape}  X_te={X_full_513.shape}")
    n_test = X_full_513.shape[0]
    seed_resid_513 = np.zeros((len(RESID_SEEDS), n_test), dtype=np.float64)
    per_seed_records = []
    for si, s in enumerate(RESID_SEEDS):
        mdl = LGBMRegressor(**_lgbm_params(s))
        mdl.fit(X_full_unb, residual_unb)
        resid_in = mdl.predict(X_full_unb)
        corr_in = anchor_unb + resid_in
        in_rae_s = float(rae(y_unb, corr_in))
        resid_513 = mdl.predict(X_full_513)
        seed_resid_513[si] = resid_513
        per_seed_records.append({
            "seed": int(s),
            "in_sample_rae_253": in_rae_s,
            "resid_513_mean": float(resid_513.mean()),
            "resid_513_std": float(resid_513.std()),
        })
        print(f"   seed {s:3d}:  in-sample RAE_253={in_rae_s:.4f}  "
              f"resid_513 mean={resid_513.mean():+.4f}  std={resid_513.std():.4f}")
    mean_resid_513 = seed_resid_513.mean(axis=0)
    info = {
        "family": family_name,
        "n_features": int(X_full_unb.shape[1]),
        "per_seed_records": per_seed_records,
        "mean_resid_513_stats": {
            "mean": float(mean_resid_513.mean()),
            "std": float(mean_resid_513.std()),
            "min": float(mean_resid_513.min()),
            "max": float(mean_resid_513.max()),
        },
    }
    return mean_resid_513, info


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- DEPLOY 5-way K-tuned naive 1/5 mean (chemprop_aux anchor) -> 513")
    print(f"          honest LB anchor (nb1553 naive_1_5_mean): {HONEST_LB_ANCHOR_NB1553}")
    print(f"          K_AP={K_AP}  K_MACCS={K_MACCS}  K_MORD={K_MORD}  "
          f"K_EMBED={K_EMBED}  K_AVALON={K_AVALON}")
    print("=" * 78)

    # ---- Load test metadata ----
    te = load_test()
    n_test = len(te)
    if "smiles" in te.columns:
        test_smiles = te["smiles"].astype(str).tolist()
    else:
        test_smiles = te["SMILES"].astype(str).tolist()
    if "molecule_name" in te.columns:
        mol_names = te["molecule_name"].astype(str).tolist()
    elif "Molecule Name" in te.columns:
        mol_names = te["Molecule Name"].astype(str).tolist()
    else:
        cand = [c for c in te.columns if "name" in c.lower()]
        if not cand:
            raise KeyError(
                f"No Molecule Name column found in test ({te.columns.tolist()})"
            )
        mol_names = te[cand[0]].astype(str).tolist()
    print(f"[load] n_test={n_test}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_unb={n_unb}")

    # ---- Anchor ----
    if not ANCHOR_TE_PATH.exists():
        raise FileNotFoundError(f"Anchor missing: {ANCHOR_TE_PATH}")
    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    if te_anchor_513.shape[0] != n_test:
        raise ValueError(
            f"anchor shape mismatch: {te_anchor_513.shape} vs {n_test}"
        )
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    print(f"[anchor] te_{ANCHOR}  in_RAE(unb_idx)={rae_anchor:.4f}  "
          f"(ref 0.6216)")
    residual_unb = y_unb - anchor_unb
    print(f"[resid] mean={residual_unb.mean():+.4f}  std={residual_unb.std():.4f}")

    # ---- Top-K indices from nb1553 summary ----
    if not NB1553_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB1553_SUMMARY}")
    with open(NB1553_SUMMARY) as f:
        sum_1553 = json.load(f)

    top_ap_idx = _extract_family_top_idx(sum_1553, "AtomPair", K=K_AP)
    top_maccs_idx = _extract_family_top_idx(sum_1553, "MACCS", K=K_MACCS)
    top_mord_idx = _extract_family_top_idx(sum_1553, "Mordred", K=K_MORD)
    top_embed_idx = _extract_family_top_idx(sum_1553, "ChempropEmbed", K=K_EMBED)
    top_avalon_idx = _extract_family_top_idx(sum_1553, "Avalon", K=K_AVALON)

    print(f"[reuse] AtomPair       top-{len(top_ap_idx)}     from nb1553")
    print(f"[reuse] MACCS          top-{len(top_maccs_idx)}     from nb1553")
    print(f"[reuse] Mordred        top-{len(top_mord_idx)}     from nb1553")
    print(f"[reuse] ChempropEmbed  top-{len(top_embed_idx)}     from nb1553")
    print(f"[reuse] Avalon         top-{len(top_avalon_idx)}     from nb1553")

    # ---- ChEMBL kNN features on 513 (cached) ----
    if not PRED_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(f"missing {PRED_CHEMBL_513_PATH}")
    if not SIM_CHEMBL_513_PATH.exists():
        raise FileNotFoundError(f"missing {SIM_CHEMBL_513_PATH}")
    pred_chembl_513 = np.load(PRED_CHEMBL_513_PATH).astype(np.float32)
    sim_chembl_513 = np.load(SIM_CHEMBL_513_PATH).astype(np.float32)
    print(f"[chembl] pred_chembl_513 mean={pred_chembl_513.mean():.3f}  "
          f"std={pred_chembl_513.std():.3f}")
    print(f"[chembl] sim_chembl_513  mean={sim_chembl_513.mean():.3f}  "
          f"std={sim_chembl_513.std():.3f}")

    # ---- Raw feature caches (513) ----
    X_ap_te = np.load(ATOMPAIR_TE_PATH).astype(np.float32)
    X_maccs_te = np.load(MACCS_TE_PATH).astype(np.float32)
    X_mord_te = _load_mordred_test(n_test_expected=n_test)
    X_emb_te = _load_chemprop_embed_test(n_test_expected=n_test)
    X_avalon_te = np.load(AVALON_TE_PATH).astype(np.float32)

    # ---- Build per-family feature matrices: family-block + ChEMBL kNN (2 cols) ----
    # Family A: AtomPair-25
    X_ap_te_top = X_ap_te[:, top_ap_idx]
    X_ap_te_full = np.concatenate(
        [X_ap_te_top,
         pred_chembl_513.reshape(-1, 1),
         sim_chembl_513.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    X_ap_unb_full = X_ap_te_full[unb_idx]

    # Family B: MACCS-20
    X_maccs_te_top = X_maccs_te[:, top_maccs_idx]
    X_maccs_te_full = np.concatenate(
        [X_maccs_te_top,
         pred_chembl_513.reshape(-1, 1),
         sim_chembl_513.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    X_maccs_unb_full = X_maccs_te_full[unb_idx]

    # Family C: Mordred-20
    X_mord_te_top = X_mord_te[:, top_mord_idx]
    X_mord_te_full = np.concatenate(
        [X_mord_te_top,
         pred_chembl_513.reshape(-1, 1),
         sim_chembl_513.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    X_mord_unb_full = X_mord_te_full[unb_idx]

    # Family D: ChempropEmbed-20
    X_emb_te_top = X_emb_te[:, top_embed_idx]
    X_emb_te_full = np.concatenate(
        [X_emb_te_top,
         pred_chembl_513.reshape(-1, 1),
         sim_chembl_513.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    X_emb_unb_full = X_emb_te_full[unb_idx]

    # Family E: Avalon-30
    X_avalon_te_top = X_avalon_te[:, top_avalon_idx]
    X_avalon_te_full = np.concatenate(
        [X_avalon_te_top,
         pred_chembl_513.reshape(-1, 1),
         sim_chembl_513.reshape(-1, 1)], axis=1
    ).astype(np.float32)
    X_avalon_unb_full = X_avalon_te_full[unb_idx]

    # ---- 5 independent 5-seed deploy residuals ----
    print("\n" + "-" * 78)
    print("BUILD 5 DEPLOY RESIDUALS (each 5-seed LGBM-Huber bag, fit on full 253)")
    print("-" * 78)

    mean_resid_ap, info_ap = _build_family_residual_deploy(
        "AtomPair-25+ChEMBL", X_ap_unb_full, X_ap_te_full,
        residual_unb, y_unb, anchor_unb
    )
    mean_resid_maccs, info_maccs = _build_family_residual_deploy(
        "MACCS-20+ChEMBL", X_maccs_unb_full, X_maccs_te_full,
        residual_unb, y_unb, anchor_unb
    )
    mean_resid_mord, info_mord = _build_family_residual_deploy(
        "Mordred-20+ChEMBL", X_mord_unb_full, X_mord_te_full,
        residual_unb, y_unb, anchor_unb
    )
    mean_resid_emb, info_emb = _build_family_residual_deploy(
        "ChempropEmbed-20+ChEMBL", X_emb_unb_full, X_emb_te_full,
        residual_unb, y_unb, anchor_unb
    )
    mean_resid_avalon, info_avalon = _build_family_residual_deploy(
        "Avalon-30+ChEMBL", X_avalon_unb_full, X_avalon_te_full,
        residual_unb, y_unb, anchor_unb
    )

    # ---- Per-family in_RAE diagnostics on 253 ----
    print("\n" + "-" * 78)
    print("PER-FAMILY DEPLOY RESIDUAL DIAGNOSTICS (in-sample on 253)")
    print("-" * 78)
    family_te = {}
    for name, mean_resid_513 in (
        ("AtomPair-25", mean_resid_ap),
        ("MACCS-20", mean_resid_maccs),
        ("Mordred-20", mean_resid_mord),
        ("ChempropEmbed-20", mean_resid_emb),
        ("Avalon-30", mean_resid_avalon),
    ):
        te_fam = te_anchor_513 + mean_resid_513
        rae_fam = float(rae(y_unb, te_fam[unb_idx]))
        print(f"   {name:<20s}  in_RAE(unb)={rae_fam:.4f}  "
              f"d_vs_anchor={rae_fam - rae_anchor:+.4f}")
        family_te[name] = (te_fam, rae_fam)

    # ---- 5-way naive 1/5 mean blend: te_nb1562 = anchor + (1/5) sum of 5 mean_resid_513 ----
    mean_resid_5way_513 = (
        mean_resid_ap + mean_resid_maccs + mean_resid_mord
        + mean_resid_emb + mean_resid_avalon
    ) / 5.0
    te_nb1562 = te_anchor_513 + mean_resid_5way_513

    in_rae_blend = float(rae(y_unb, te_nb1562[unb_idx]))
    print("\n" + "-" * 78)
    print("BLEND DIAGNOSTICS  te_nb1562 = anchor + (1/5) sum_5 mean_resid_513")
    print("-" * 78)
    print(f"   te_nb1562  mean={te_nb1562.mean():.4f}  "
          f"std={te_nb1562.std():.4f}  "
          f"min={te_nb1562.min():.4f}  max={te_nb1562.max():.4f}")
    print(f"   in_RAE(unb_idx)       = {in_rae_blend:.4f}")
    print(f"   d_vs_anchor           = {in_rae_blend - rae_anchor:+.4f}")
    print(f"   honest LB anchor      = {HONEST_LB_ANCHOR_NB1553} (from nb1553 naive_1_5_mean)")

    # ---- Save NPY ----
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(te_path, te_nb1562.astype(np.float32))
    print(f"\n[save] {te_path}")

    # ---- Save CSV ----
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    csv_path = SUBMISSIONS / f"{TAG}_deploy_nb1553.csv"
    df = pd.DataFrame({
        "SMILES": test_smiles,
        "Molecule Name": mol_names,
        "pEC50": te_nb1562.astype(np.float64),
    })
    df.to_csv(csv_path, index=False)
    print(f"[save] {csv_path}  rows={len(df)}  cols={list(df.columns)}")

    # ---- Summary ----
    summary = {
        "tag": TAG,
        "parent_method": "nb1553",
        "parent_variant": "naive_1_5_mean",
        "anchor": ANCHOR,
        "anchor_path": str(ANCHOR_TE_PATH),
        "anchor_kind": "PRE_unblind_te_513",
        "rae_anchor": rae_anchor,
        "blend_recipe": "mean_of_5_family_deploy_residuals_added_to_anchor",
        "families": [
            {"name": "AtomPair-25+ChEMBL", "K": K_AP, "info": info_ap},
            {"name": "MACCS-20+ChEMBL", "K": K_MACCS, "info": info_maccs},
            {"name": "Mordred-20+ChEMBL", "K": K_MORD, "info": info_mord},
            {"name": "ChempropEmbed-20+ChEMBL", "K": K_EMBED, "info": info_emb},
            {"name": "Avalon-30+ChEMBL", "K": K_AVALON, "info": info_avalon},
        ],
        "resid_seeds": RESID_SEEDS,
        "lgbm_params": {
            "objective": "huber", "alpha": 1.0, "max_depth": 3,
            "num_leaves": 7, "n_estimators": 80, "learning_rate": 0.05,
        },
        "n_test": int(n_test),
        "n_unb": int(n_unb),
        "te_nb1562_stats": {
            "mean": float(te_nb1562.mean()),
            "std": float(te_nb1562.std()),
            "min": float(te_nb1562.min()),
            "max": float(te_nb1562.max()),
        },
        "per_family_in_rae_unb": {
            name: rae_fam for name, (_, rae_fam) in family_te.items()
        },
        "in_rae_unb_blend": in_rae_blend,
        "delta_blend_vs_anchor": in_rae_blend - rae_anchor,
        "honest_lb_anchor_nb1553": HONEST_LB_ANCHOR_NB1553,
        "te_npy_path": str(te_path),
        "csv_path": str(csv_path),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_test", "n_unb",
        "rae_anchor",
        "te_nb1562_stats",
        "per_family_in_rae_unb",
        "in_rae_unb_blend",
        "delta_blend_vs_anchor",
        "honest_lb_anchor_nb1553",
        "csv_path", "te_npy_path",
    ):
        print(f"  {k}: {res.get(k)}")
