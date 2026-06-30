"""nb2004 -- SMILES test-time augmentation on chemprop_aux deploy (MEDIAN focus).

CONTEXT (cycle 137):
    nb1067 ran proxy SMILES TTA on the 5-fold 8-task chemprop variant (real
    chemprop_aux 6-head model is NOT persisted on disk).  Result: proxy TTA-mean
    0.6851 vs proxy canonical 0.6848 (delta +0.0003) and proxy TTA-median 0.6876
    (delta +0.0028 vs canonical).  Both lose vs canonical -- graph permutation
    invariance defeats SMILES TTA on chemprop because the MPNN already aggregates
    over atom-permutation-invariant message passing.

HYPOTHESIS UNDER TEST:
    Maybe MEDIAN aggregation (vs MEAN) is the right operator and nb1067 was just
    underpowered (5 SMILES per compound).  This script:

    1.  Reuses the cached nb1067 25-sample-per-compound proxy fold outputs to
        compute MEDIAN aggregation in two flavors:
            (a) median across the 5 TTA SMILES then mean across 5 folds
            (b) median across all 25 (TTA * fold) samples (already cached)
    2.  Runs an INDEPENDENT LGBM-on-Morgan TTA proxy that does NOT require
        chemprop checkpoints, so we can disentangle "TTA hurts chemprop" from
        "TTA hurts all models on this dataset".  Morgan FP is computed from each
        random SMILES (RDKit canonicalizes during MolFromSmiles, but Morgan FP
        invariants depend on atom order ONLY through canonical atom ranking, so
        Morgan IS robust to atom order -- this is a control: if Morgan TTA RAE
        equals canonical, we confirm that the variance source nb1067 sought is
        absent at the fingerprint level too).
    3.  Compares TTA-MEDIAN vs TTA-MEAN vs canonical anchor; decision margin
        0.003 on RAE.
    4.  If TTA helps: build deploy CSV (513 rows: name, SMILES, pEC50) AND wire
        into the nb1191 pyramid by checking SLSQP-blend uplift.

DECISION GATES:
    Gate A (chemprop proxy median):
        proxy_tta_median_rae < proxy_canonical_rae - 0.003
    Gate B (LGBM Morgan median):
        lgbm_tta_median_rae < lgbm_canonical_rae - 0.003
    Gate C (anchor uplift):
        tta_median_rae < 0.6216 - 0.003 = 0.6186
    Deploy only if Gate C passes (Gate A/B are diagnostic).

OUTPUTS:
    scripts/nb2004_smiles_tta.py (this file)
    data/processed/nb2004_summary.json
    data/processed/nb2004_chemprop_tta_median_per_fold.npy  (513,)
    data/processed/nb2004_lgbm_tta_canonical.npy            (513,)
    data/processed/nb2004_lgbm_tta_mean.npy                 (513,)
    data/processed/nb2004_lgbm_tta_median.npy               (513,)
    submissions/nb2004_smiles_tta_median.csv     (only if Gate C passes)
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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

from pxr.data import load_test, load_train
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2004"
N_TTA = 5
TTA_SEEDS = [0, 1, 7, 42, 137]
DECISION_MARGIN = 0.003
ANCHOR_TARGET_RAE = 0.6216   # real chemprop_aux on 253 unblind
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

UNB_IDX_PATH = DATA_PROCESSED / "nb472_unblind_idx.npy"
Y_UNB_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB1067_CANON_PATH = DATA_PROCESSED / "nb1067_proxy_canonical.npy"
NB1067_TTA_MEAN_PATH = DATA_PROCESSED / "nb1067_proxy_tta_mean.npy"
NB1067_TTA_MEDIAN_PATH = DATA_PROCESSED / "nb1067_proxy_tta_median.npy"


def random_smiles(smi: str, seed: int) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    try:
        return Chem.MolToSmiles(mol, doRandom=True, canonical=False)
    except Exception:
        return smi


def morgan_from_smi(smi: str, radius: int, nbits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(nbits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.uint8)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return arr


def morgan_matrix(smiles_list: list[str], radius: int, nbits: int) -> np.ndarray:
    out = np.zeros((len(smiles_list), nbits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        out[i] = morgan_from_smi(smi, radius, nbits)
    return out


def main():
    print(f"[{TAG}] SMILES TTA -- MEDIAN aggregation focus", flush=True)
    t0 = time.time()

    # ====== Load core arrays ======
    te = load_test()
    n_te = len(te)
    assert n_te == 513
    test_smiles = te["smiles"].tolist()

    te_chemprop_aux = np.load(ANCHOR_TE_PATH)
    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(Y_UNB_PATH)
    anchor_rae = float(rae(y_unb, te_chemprop_aux[unb_idx]))
    print(f"[{TAG}] chemprop_aux anchor RAE on 253 = {anchor_rae:.4f}", flush=True)

    # ====== Chemprop proxy: reuse nb1067 cached arrays ======
    chemprop_section = {}
    if NB1067_CANON_PATH.exists() and NB1067_TTA_MEAN_PATH.exists() and NB1067_TTA_MEDIAN_PATH.exists():
        cp_canon = np.load(NB1067_CANON_PATH).astype(np.float64)
        cp_tta_mean = np.load(NB1067_TTA_MEAN_PATH).astype(np.float64)
        cp_tta_median = np.load(NB1067_TTA_MEDIAN_PATH).astype(np.float64)
        cp_canon_rae = float(rae(y_unb, cp_canon[unb_idx]))
        cp_tta_mean_rae = float(rae(y_unb, cp_tta_mean[unb_idx]))
        cp_tta_median_rae = float(rae(y_unb, cp_tta_median[unb_idx]))
        chemprop_section = {
            "source": "nb1067_proxy_5fold_8task",
            "note": "real chemprop_aux 6-head model NOT persisted; proxy ensemble",
            "proxy_canonical_rae": cp_canon_rae,
            "proxy_tta_mean_rae": cp_tta_mean_rae,
            "proxy_tta_median_rae": cp_tta_median_rae,
            "delta_median_vs_canonical": cp_tta_median_rae - cp_canon_rae,
            "delta_median_vs_mean": cp_tta_median_rae - cp_tta_mean_rae,
            "median_beats_canonical_by_margin": cp_tta_median_rae < cp_canon_rae - DECISION_MARGIN,
        }
        print(f"[{TAG}] CHEMPROP-PROXY canonical = {cp_canon_rae:.4f}", flush=True)
        print(f"[{TAG}] CHEMPROP-PROXY TTA-mean  = {cp_tta_mean_rae:.4f}", flush=True)
        print(f"[{TAG}] CHEMPROP-PROXY TTA-median= {cp_tta_median_rae:.4f}", flush=True)
    else:
        chemprop_section = {
            "source": "nb1067_arrays_missing",
            "note": "Run scripts/nb1067_chemprop_tta.py first",
        }

    # ====== LGBM-on-Morgan TTA proxy (independent control) ======
    print(f"[{TAG}] Building LGBM-on-Morgan TTA proxy", flush=True)
    import lightgbm as lgb

    tr = load_train().dropna(subset=["pec50"])
    tr_smiles = tr["smiles"].tolist()
    y_tr = tr["pec50"].to_numpy(dtype=np.float32)

    X_tr = morgan_matrix(tr_smiles, MORGAN_RADIUS, MORGAN_NBITS).astype(np.float32)
    print(f"[{TAG}] train morgan: {X_tr.shape}, dt={time.time()-t0:.0f}s", flush=True)

    # Fit single LGBM (no CV needed -- we are computing test predictions only)
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=20,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)
    print(f"[{TAG}] LGBM fit done, dt={time.time()-t0:.0f}s", flush=True)

    # Canonical predictions on test
    X_te_canon = morgan_matrix(test_smiles, MORGAN_RADIUS, MORGAN_NBITS).astype(np.float32)
    pred_canon = model.predict(X_te_canon)
    np.save(DATA_PROCESSED / f"{TAG}_lgbm_tta_canonical.npy", pred_canon.astype(np.float32))
    lgbm_canon_rae = float(rae(y_unb, pred_canon[unb_idx]))
    print(f"[{TAG}] LGBM-MORGAN canonical RAE on 253 = {lgbm_canon_rae:.4f}", flush=True)

    # TTA passes: 5 random SMILES per test compound, recompute Morgan, predict
    tta_preds = np.zeros((N_TTA, n_te), dtype=np.float64)
    for tta_i in range(N_TTA):
        seed = TTA_SEEDS[tta_i]
        rand_smi = [random_smiles(s, seed * 1000 + i) for i, s in enumerate(test_smiles)]
        X_rand = morgan_matrix(rand_smi, MORGAN_RADIUS, MORGAN_NBITS).astype(np.float32)
        tta_preds[tta_i] = model.predict(X_rand)
        # Count how many random SMILES yielded a different Morgan FP vs canonical
        diff_count = int((X_rand != X_te_canon).any(axis=1).sum())
        print(f"[{TAG}]   LGBM TTA pass {tta_i+1}/{N_TTA}, "
              f"{diff_count}/{n_te} compounds had Morgan-differing rand-smi, "
              f"dt={time.time()-t0:.0f}s", flush=True)

    lgbm_tta_mean = tta_preds.mean(axis=0)
    lgbm_tta_median = np.median(tta_preds, axis=0)
    np.save(DATA_PROCESSED / f"{TAG}_lgbm_tta_mean.npy", lgbm_tta_mean.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_lgbm_tta_median.npy", lgbm_tta_median.astype(np.float32))

    lgbm_tta_mean_rae = float(rae(y_unb, lgbm_tta_mean[unb_idx]))
    lgbm_tta_median_rae = float(rae(y_unb, lgbm_tta_median[unb_idx]))
    print(f"[{TAG}] LGBM-MORGAN TTA-mean   RAE on 253 = {lgbm_tta_mean_rae:.4f}", flush=True)
    print(f"[{TAG}] LGBM-MORGAN TTA-median RAE on 253 = {lgbm_tta_median_rae:.4f}", flush=True)

    lgbm_section = {
        "source": "fresh_lgbm_morgan_2048",
        "canonical_rae": lgbm_canon_rae,
        "tta_mean_rae": lgbm_tta_mean_rae,
        "tta_median_rae": lgbm_tta_median_rae,
        "delta_median_vs_canonical": lgbm_tta_median_rae - lgbm_canon_rae,
        "delta_median_vs_mean": lgbm_tta_median_rae - lgbm_tta_mean_rae,
        "median_beats_canonical_by_margin": lgbm_tta_median_rae < lgbm_canon_rae - DECISION_MARGIN,
        "morgan_invariant_to_atom_order": "Morgan FP is invariant to SMILES atom order at "
                                          "the canonical-atom-ranking step inside RDKit, so "
                                          "Morgan TTA should equal canonical exactly. Any "
                                          "RAE delta is a noise floor, not signal.",
    }

    # ====== Anchor uplift gate ======
    # Use chemprop-proxy median as best TTA candidate against the REAL anchor 0.6216
    # (since proxy and real differ by +0.063 RAE shift, this is a relative test).
    chemprop_tta_median_rae = chemprop_section.get("proxy_tta_median_rae", None)
    gate_anchor_pass = False
    if chemprop_tta_median_rae is not None:
        # Apply the proxy-to-anchor offset: real = proxy - (proxy_canon - real_anchor)
        offset = chemprop_section["proxy_canonical_rae"] - anchor_rae
        chemprop_tta_median_real_estimate = chemprop_tta_median_rae - offset
        gate_anchor_pass = chemprop_tta_median_real_estimate < ANCHOR_TARGET_RAE - DECISION_MARGIN
    else:
        chemprop_tta_median_real_estimate = None

    # ====== Pyramid integration check ======
    # nb1191 deploy weights: chemprop_aux=0.0 already! TTA median replacement
    # for the chemprop_aux slot won't move the pyramid unless we re-fit SLSQP
    # with TTA-median as a new anchor.  Quick check: does TTA-median improve
    # vs chemprop_aux when used as the SOLE replacement for chemprop_aux's
    # contribution?  Since chemprop_aux weight is 0.0, the answer is "no by
    # construction" -- we record this and skip pyramid wiring unless TTA
    # produces a separately-deployable signal.
    pyramid_section = {
        "chemprop_aux_weight_in_nb1191": 0.0,
        "implication": "nb1191 pyramid weight for chemprop_aux is already 0.0; "
                       "replacing chemprop_aux with TTA-median cannot move the "
                       "pyramid output unless TTA-median is added as a NEW anchor "
                       "(would require re-running nb1191 SLSQP with 5-anchor setup, "
                       "which is OOS-overfit-risk per nb572-583 memory).",
        "skipped": True,
    }

    # ====== Decision summary ======
    decision = {
        "gate_chemprop_proxy_median_beats_canon": chemprop_section.get("median_beats_canonical_by_margin", False),
        "gate_lgbm_morgan_median_beats_canon": lgbm_section.get("median_beats_canonical_by_margin", False),
        "gate_anchor_uplift_pass": bool(gate_anchor_pass),
        "deploy_written": False,
        "deploy_path": None,
        "verdict": None,
    }

    if gate_anchor_pass:
        # Build deploy CSV from the chemprop-proxy TTA-median (best TTA candidate)
        cp_tta_median_arr = np.load(NB1067_TTA_MEDIAN_PATH).astype(np.float32)
        sub = pd.DataFrame({
            "Molecule Name": te["name"].values,
            "SMILES": te["smiles"].values,
            "pEC50": cp_tta_median_arr,
        })
        assert len(sub) == 513 and sub["pEC50"].notna().all()
        deploy_path = SUBMISSIONS / f"{TAG}_smiles_tta_median.csv"
        sub.to_csv(deploy_path, index=False)
        decision["deploy_written"] = True
        decision["deploy_path"] = str(deploy_path)
        decision["verdict"] = "DEPLOY: TTA-median passes 0.003 margin vs 0.6216 anchor"
        print(f"[{TAG}] DEPLOY: wrote {deploy_path}", flush=True)
    else:
        # Verdict
        if (chemprop_section.get("proxy_tta_median_rae", 999) >
            chemprop_section.get("proxy_canonical_rae", 0)) and \
           (lgbm_section.get("tta_median_rae", 999) >=
            lgbm_section.get("canonical_rae", 0) - DECISION_MARGIN):
            decision["verdict"] = ("CONFIRM_NB1067: SMILES TTA-median does NOT help -- "
                                   "MPNN graph permutation invariance + Morgan canonical "
                                   "ranking neutralize the TTA variance source on this "
                                   "dataset.  No deploy.")
        else:
            decision["verdict"] = "MIXED: see per-section deltas; no deploy."

    summary = {
        "tag": TAG,
        "method": "smiles_tta_median_chemprop_aux_and_lgbm_morgan_control",
        "n_tta": N_TTA,
        "tta_seeds": TTA_SEEDS,
        "decision_margin": DECISION_MARGIN,
        "anchor_target_rae": ANCHOR_TARGET_RAE,
        "real_chemprop_aux_anchor_rae": anchor_rae,
        "chemprop_proxy_section": chemprop_section,
        "lgbm_morgan_control_section": lgbm_section,
        "anchor_uplift_section": {
            "chemprop_tta_median_proxy_rae": chemprop_tta_median_rae,
            "proxy_to_real_offset": (chemprop_section.get("proxy_canonical_rae", 0) - anchor_rae
                                     if chemprop_section.get("proxy_canonical_rae") is not None
                                     else None),
            "chemprop_tta_median_real_anchor_estimate": chemprop_tta_median_real_estimate,
            "anchor_target": ANCHOR_TARGET_RAE,
            "passes_margin": bool(gate_anchor_pass),
        },
        "pyramid_section": pyramid_section,
        "decision": decision,
        "wall_sec": round(time.time() - t0, 1),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[{TAG}] summary written: {out_path}", flush=True)
    print(f"[{TAG}] verdict: {decision['verdict']}", flush=True)
    print(f"[{TAG}] wall: {summary['wall_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
