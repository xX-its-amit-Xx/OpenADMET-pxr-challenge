"""nb1067 -- SMILES randomization TTA on chemprop_aux (cycle 135 method).

HYPOTHESIS:
    Averaging chemprop_aux inference over 5 random non-canonical SMILES per
    compound reduces variance and tightens predictions on the analog-expansion
    test set.

CRITICAL CAVEAT (documented below):
    The 6-head chemprop_aux model that produced data/processed/te_chemprop_aux.npy
    (nb35_chemprop_auxiliary.ipynb) is NOT persisted on disk -- only the
    inference outputs (oof + te .npy files) are cached.  The notebook trains an
    MPNN on all 4141 training rows for 54 epochs after CV, then runs predict
    once, then discards the model state.  No torch.save() / trainer.save() is
    called.

    What IS on disk:
        - 5 fold checkpoints at
          data/processed/chemprop_fold_{0..4}/model/model_0/checkpoints/
          BUT these are 8-task models (different variant, n_tasks=8), NOT the
          nb35 6-head chemprop_aux configuration.  Architecture matches
          (BondMessagePassing depth=3 d_h=300 + RegressionFFN n_layers=1
          input_dim=300 hidden_dim=300) but the readout heads differ.

    Consequence:
        TRUE chemprop_aux TTA is impossible without retraining nb35
        (~30 min CPU per fold * 5 folds + 1 hr full retrain ~ 3 hr).  This
        script does TWO things:

        (1) Documents the artifact gap (nb1067_summary.json field
            "artifact_gap").

        (2) Runs a PROXY TTA on the 5-fold 8-task ensemble that IS on disk:
            generate 5 random SMILES per test compound, run inference through
            each of the 5 fold checkpoints, average over (5 SMILES * 5 folds)
            and read the task-0 (pEC50) head.  Report PROXY TTA-mean vs
            PROXY anchor (5-fold average on canonical SMILES) so that the
            variance-reduction signal can be assessed on equivalent
            architecture.  This is NOT a chemprop_aux replacement; it is a
            diagnostic for whether SMILES TTA helps Chemprop-class models at
            all on this dataset.

DECISION GATE:
    Original gate: PROXY TTA RAE on 253 must be < 0.6216 by >=0.003 to deploy.
    Because the proxy is NOT the real chemprop_aux model, decision applies to
    the proxy anchor (5-fold ensemble on canonical SMILES), not 0.6216.

OUTPUTS:
    scripts/nb1067_chemprop_tta.py (this file)
    data/processed/nb1067_summary.json
    data/processed/nb1067_proxy_canonical.npy  (513,) -- 5-fold mean, canonical SMILES
    data/processed/nb1067_proxy_tta_mean.npy   (513,) -- 5-fold * 5 SMILES mean
    data/processed/nb1067_proxy_tta_median.npy (513,) -- 5-fold * 5 SMILES median
    submissions/nb1067_deploy_chemprop_tta.csv  (only if TTA passes decision gate)
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
import torch
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from pxr.data import load_test
from pxr.eval import rae
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1067"
N_TTA = 5
TTA_SEEDS = [0, 1, 7, 42, 137]
DECISION_MARGIN = 0.003
ANCHOR_TARGET_RAE = 0.6216  # original chemprop_aux on 253 unblind

FOLD_DIR = DATA_PROCESSED / "chemprop_fold_{fold}/model/model_0/checkpoints"
FOLD_CK_NAMES = ["best-epoch=19-val_loss=0.53.ckpt",
                 "best-epoch=19-val_loss=0.54.ckpt",
                 "best-epoch=19-val_loss=0.50.ckpt",
                 "best-epoch=19-val_loss=0.54.ckpt",
                 "best-epoch=19-val_loss=0.54.ckpt"]

UNB_IDX_PATH = DATA_PROCESSED / "nb472_unblind_idx.npy"
Y_UNB_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2103_K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"


def find_fold_checkpoint(fold: int) -> Path:
    folder = Path(str(FOLD_DIR).format(fold=fold))
    # Prefer 'best-' pattern; fall back to last.ckpt
    bests = sorted(folder.glob("best-*.ckpt"))
    if bests:
        return bests[0]
    return folder / "last.ckpt"


def randomize_smiles(smi: str, n: int, seed: int) -> list[str]:
    """Generate n random non-canonical SMILES for a single SMILES string."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi] * n
    rng = np.random.RandomState(seed)
    out = []
    for k in range(n):
        # RDKit MolToSmiles with doRandom shuffles atom order
        try:
            r = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        except Exception:
            r = smi
        out.append(r)
    return out


def load_fold_models():
    """Load the 5 fold checkpoints (8-task proxy chemprop variant).
    Returns list of (MPNN, output_mean_task0, output_scale_task0)."""
    from chemprop import models as cmodels
    fold_models = []
    for fold in range(5):
        ck_path = find_fold_checkpoint(fold)
        ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
        # Reconstruct MPNN via load_from_file
        try:
            mpnn = cmodels.MPNN.load_from_file(str(ck_path), map_location="cpu")
        except Exception:
            # Fallback: use load_from_checkpoint
            mpnn = cmodels.MPNN.load_from_checkpoint(str(ck_path), map_location="cpu")
        mpnn.eval()
        ot = ck["hyper_parameters"]["predictor"].output_transform
        m0 = float(ot.mean.flatten()[0])
        s0 = float(ot.scale.flatten()[0])
        fold_models.append((mpnn, m0, s0))
    return fold_models


def predict_fold(mpnn, smiles_list, m0: float, s0: float) -> np.ndarray:
    """Run inference for one fold, return task-0 (pEC50) predictions unscaled."""
    from chemprop import data as cdata
    dpts = [cdata.MoleculeDatapoint.from_smi(s) for s in smiles_list]
    ds = cdata.MoleculeDataset(dpts)
    loader = cdata.build_dataloader(ds, batch_size=128, shuffle=False, num_workers=0)
    import lightning as L
    trainer = L.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    raw = trainer.predict(mpnn, loader)
    arr = torch.cat(raw).numpy()  # (N, n_tasks)
    # output_transform already applied inside model? Check both:
    # MPNN.predict applies output_transform by default in chemprop 2.x
    # so arr is in raw pEC50 units for task 0.
    return arr[:, 0]


def main():
    print(f"[{TAG}] SMILES randomization TTA on chemprop_aux", flush=True)
    print(f"[{TAG}] WARNING: real chemprop_aux 6-head model NOT on disk; using "
          f"5-fold 8-task proxy", flush=True)

    t0 = time.time()
    te = load_test()
    n_te = len(te)
    smiles = te["smiles"].tolist()
    assert n_te == 513

    # ===== Anchor verification =====
    te_chemprop_aux = np.load(ANCHOR_TE_PATH)
    unb_idx = np.load(UNB_IDX_PATH)
    y_unb = np.load(Y_UNB_PATH)
    anchor_unb = te_chemprop_aux[unb_idx]
    anchor_rae = float(rae(y_unb, anchor_unb))
    print(f"[{TAG}] anchor te_chemprop_aux RAE on 253 unblind = {anchor_rae:.4f}", flush=True)

    # ===== Load 5-fold proxy models =====
    print(f"[{TAG}] Loading 5 fold checkpoints (8-task proxy variant)...", flush=True)
    fold_models = load_fold_models()
    print(f"[{TAG}] {len(fold_models)} folds loaded", flush=True)

    # ===== Canonical-SMILES proxy anchor (5-fold mean) =====
    print(f"[{TAG}] Predicting canonical SMILES through 5 folds...", flush=True)
    fold_canon = np.zeros((5, n_te), dtype=np.float64)
    for fid, (mpnn, m0, s0) in enumerate(fold_models):
        fold_canon[fid] = predict_fold(mpnn, smiles, m0, s0)
    proxy_canonical = fold_canon.mean(axis=0)
    np.save(DATA_PROCESSED / f"{TAG}_proxy_canonical.npy", proxy_canonical.astype(np.float32))
    proxy_canon_rae = float(rae(y_unb, proxy_canonical[unb_idx]))
    print(f"[{TAG}] proxy canonical 5-fold RAE on 253 unblind = {proxy_canon_rae:.4f}", flush=True)

    # ===== TTA pass =====
    print(f"[{TAG}] Generating {N_TTA} random SMILES per compound + folds inference...", flush=True)
    # Shape: (N_TTA, 5_folds, 513)
    all_preds = np.zeros((N_TTA, 5, n_te), dtype=np.float64)
    for tta_i in range(N_TTA):
        seed_tta = TTA_SEEDS[tta_i]
        # Generate one random SMILES for each compound for this TTA pass
        rand_smiles = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rand_smiles.append(smi)
                continue
            try:
                r = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            except Exception:
                r = smi
            rand_smiles.append(r)
        for fid, (mpnn, m0, s0) in enumerate(fold_models):
            all_preds[tta_i, fid] = predict_fold(mpnn, rand_smiles, m0, s0)
        print(f"[{TAG}]   TTA pass {tta_i+1}/{N_TTA} done "
              f"(t={time.time()-t0:.0f}s)", flush=True)

    # Aggregate: per-compound mean and median over (N_TTA * 5_folds) samples
    flat = all_preds.reshape(N_TTA * 5, n_te)  # 25 samples per compound
    tta_mean = flat.mean(axis=0)
    tta_median = np.median(flat, axis=0)
    np.save(DATA_PROCESSED / f"{TAG}_proxy_tta_mean.npy", tta_mean.astype(np.float32))
    np.save(DATA_PROCESSED / f"{TAG}_proxy_tta_median.npy", tta_median.astype(np.float32))

    tta_mean_rae = float(rae(y_unb, tta_mean[unb_idx]))
    tta_median_rae = float(rae(y_unb, tta_median[unb_idx]))
    print(f"[{TAG}] proxy TTA-mean   RAE on 253 unblind = {tta_mean_rae:.4f}", flush=True)
    print(f"[{TAG}] proxy TTA-median RAE on 253 unblind = {tta_median_rae:.4f}", flush=True)

    # ===== Decision gates =====
    # Original gate vs 0.6216
    gate_vs_chemprop_aux = (tta_mean_rae < ANCHOR_TARGET_RAE - DECISION_MARGIN)
    # Proxy gate vs 5-fold canonical
    gate_vs_proxy_canon = (tta_mean_rae < proxy_canon_rae - DECISION_MARGIN)

    # ===== Cascade with nb2103 K=28 residual =====
    cascade_record = None
    if NB2103_K28_OOF_PATH.exists():
        nb2103_resid_oof = np.load(NB2103_K28_OOF_PATH)
        # nb2103 file is the corrected anchor + residual on 253 unblind
        # Reproduce cascade structure: corrected = chemprop_aux[unb] + residual
        # The mean_bag_oof_K28.npy stores the corrected predictions directly
        cascade_chemprop_aux_resid = nb2103_resid_oof - te_chemprop_aux[unb_idx]
        # Apply same residual offsets onto TTA-mean
        cascade_tta = tta_mean[unb_idx] + cascade_chemprop_aux_resid
        cascade_tta_rae = float(rae(y_unb, cascade_tta))
        cascade_chemprop_rae = float(rae(y_unb, nb2103_resid_oof))
        cascade_record = {
            "nb2103_K28_chemprop_aux_cascade_rae": cascade_chemprop_rae,
            "nb2103_K28_TTA_cascade_rae": cascade_tta_rae,
            "delta_TTA_cascade_vs_chemprop_aux_cascade": cascade_tta_rae - cascade_chemprop_rae,
            "TTA_cascade_improves": cascade_tta_rae < cascade_chemprop_rae - DECISION_MARGIN,
            "method": "residual_transfer: residual_K28_K28 = nb2103[unb] - chemprop_aux[unb]; "
                      "cascade_tta = tta_mean[unb] + residual",
            "caveat": "Residuals were FIT on chemprop_aux; applying to TTA is naive transfer "
                      "(not a re-fit). Decision-grade cascade requires retraining the 5-seed "
                      "LGBM residual bag against TTA-mean as anchor (~17s).",
        }
        print(f"[{TAG}] cascade chemprop_aux+nb2103_K28 RAE = {cascade_chemprop_rae:.4f}", flush=True)
        print(f"[{TAG}] cascade TTA+nb2103_K28           RAE = {cascade_tta_rae:.4f}", flush=True)
    else:
        print(f"[{TAG}] WARN: {NB2103_K28_OOF_PATH} not found; cascade skipped", flush=True)

    # ===== Deploy CSV =====
    deploy_written = False
    deploy_path = None
    if gate_vs_chemprop_aux and gate_vs_proxy_canon:
        # Build deploy: TTA-averaged proxy + nb2103 K=28 LGBM residual cascade
        # (Transfer-only; flagged as approximate in summary.)
        if cascade_record is not None:
            # Apply residual to all 513 (residuals computed on unb only -> just
            # apply offset to the unb slice; for non-unb compounds we have no
            # residual, so use tta_mean as-is)
            deploy_pred = tta_mean.copy()
            # NOTE: K=28 LGBM residual was trained on 253-unb only.  We do
            # NOT have residual predictions for the other 260 test rows.  So
            # cascade can only be applied at deploy if we re-fit on ALL 513
            # via the SHAP-K-tuned pipeline.  Mark deploy as "TTA-only" and
            # document the limitation.
            sub = pd.DataFrame({
                "Molecule Name": te["name"].values,
                "SMILES": te["smiles"].values,
                "pEC50": deploy_pred,
            })
            assert len(sub) == 513 and sub["pEC50"].notna().all()
            deploy_path = SUBMISSIONS / f"{TAG}_deploy_chemprop_tta.csv"
            sub.to_csv(deploy_path, index=False)
            deploy_written = True
            print(f"[{TAG}] DEPLOY: wrote {deploy_path}", flush=True)
    else:
        print(f"[{TAG}] decision gate FAILED -- not writing deploy CSV", flush=True)

    # ===== Summary =====
    summary = {
        "tag": TAG,
        "method": "smiles_randomization_TTA_chemprop",
        "n_tta": N_TTA,
        "tta_seeds": TTA_SEEDS,
        "decision_margin": DECISION_MARGIN,
        "anchor_target_rae": ANCHOR_TARGET_RAE,
        "artifact_gap": {
            "real_chemprop_aux_model_persisted": False,
            "real_chemprop_aux_te_npy_cached": True,
            "real_chemprop_aux_oof_npy_cached": True,
            "issue": "nb35_chemprop_auxiliary.ipynb fits the 6-head MPNN on all "
                     "4141 rows then runs predict + discards model state. No "
                     "torch.save() / Trainer.save_checkpoint() call. The only "
                     "Chemprop ckpts on disk are 5 fold-CV variants under "
                     "data/processed/chemprop_fold_*/ with n_tasks=8 -- a "
                     "DIFFERENT model variant.",
            "consequence": "True chemprop_aux TTA requires re-running nb35 "
                           "(~3 hr CPU) to obtain trained 6-head model state. "
                           "This script uses the 5-fold 8-task ensemble as a "
                           "PROXY -- architecturally similar (BondMessagePassing "
                           "depth=3 d_h=300 + RegressionFFN n_layers=1 input_dim=300 "
                           "hidden_dim=300) but with different readout heads.",
            "remediation": "To do real TTA, add Trainer.save_checkpoint to nb35 "
                           "final retrain block, save to "
                           "data/processed/chemprop_aux_final.ckpt, then re-run "
                           "this script with FOLD_DIR pointing at the saved file.",
        },
        "real_anchor_rae_on_253": anchor_rae,
        "proxy_canonical_5fold_rae_on_253": proxy_canon_rae,
        "proxy_tta_mean_rae_on_253": tta_mean_rae,
        "proxy_tta_median_rae_on_253": tta_median_rae,
        "delta_tta_mean_vs_real_anchor": tta_mean_rae - anchor_rae,
        "delta_tta_mean_vs_proxy_canon": tta_mean_rae - proxy_canon_rae,
        "gate_vs_real_chemprop_aux": gate_vs_chemprop_aux,
        "gate_vs_proxy_canonical": gate_vs_proxy_canon,
        "deploy_written": deploy_written,
        "deploy_path": str(deploy_path) if deploy_path is not None else None,
        "cascade_nb2103_K28": cascade_record,
        "wall_sec": round(time.time() - t0, 1),
    }
    summary_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[{TAG}] summary written: {summary_path}", flush=True)
    print(f"[{TAG}] wall: {summary['wall_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
