"""nb3350 -- Fresh 5-fold multitask chemprop anchor (chemprop_aux_v3).

Goal
----
The frozen ``te_chemprop_aux.npy`` is the ONLY PRE-clean anchor on this project
(trained on the original 4139, nb<320 refit, RAE 0.6216 on the 253 unblind). A
fresh, INDEPENDENT chemprop refit -- same nb35 architecture but a NEW 5-fold
scaffold CV ensemble rather than nb950's single 80/20 split -- could decorrelate
enough to be useful in a blend.

Architecture (matches notebooks/03_multitask.ipynb == nb35_chemprop_auxiliary):
  MPNN(
    message_passing = BondMessagePassing(depth=3, d_h=300),
    agg             = MeanAggregation(),
    predictor       = RegressionFFN(n_tasks=2, n_layers=2, dropout=0.1),
  )
  Two heads, MULTITASK (single shared encoder):
    task 0 = PXR  pEC50  (4139 labels)
    task 1 = counter-assay pEC50 (NaN-masked where absent; ~2858/4139 present)
  chemprop 2.x masks NaN targets out of the loss natively, so the multitask
  head is safe here (the augmented-corpus NaN-poisoning that forced nb950 to
  split the heads does not bite on the clean 4139 train, where the null head is
  dense). We early-stop on the PXR-only val RMSE of the held-out fold.

Protocol (the crucial difference from nb950 single-split)
---------------------------------------------------------
  * Train corpus = 4139 ORIGINAL train ONLY (counter labels merged as task 1).
  * 5-fold scaffold CV (src.pxr.eval.scaffold_kfold_indices, seed=42) on the 4139.
  * For each fold k: fit on the other 4 folds (early-stop on the held-out fold's
    PXR rows), then
      - OOF-predict the held-out fold  -> fills oof_4139[fold_k]   (genuine OOF)
      - predict the 253 unblind         -> accumulate, averaged over folds
      - predict the 513 test            -> accumulate, averaged over folds
  * The 253 unblind + 513 test are HELD OUT of every fit (eval only). The
    253-unblind anchor = mean of the 5 fold-models' predictions on the 253.
  * 513-test deploy anchor = mean of the 5 fold-models' predictions on the 513,
    re-indexed to load_test() row order (so it aligns bit-for-bit with the
    frozen te_chemprop_aux and with _audit_unblind_idx).

Alignment (verified before writing this script)
-----------------------------------------------
  * load_test() == 513 rows, 513 unique std-SMILES (no dedup collapse).
  * _audit_unblind_idx.npy (253,) indexes load_test() row order and selects
    exactly the 253 phase1-unblinded compounds (InChIKey-matched 253/253).
  * te_chemprop_aux[_audit_unblind_idx] vs pm_unblind_y == RAE 0.6216 (frozen).
  => nb3350_chemprop_v3_oof (253) := te_513[_audit_unblind_idx]; both are in
     load_test() order, so the RAE / correlation comparison is apples-to-apples.

Gates (reported)
----------------
  * fresh-anchor RAE on the 253 unblind < 0.65  -> "USABLE_ANCHOR"
  * Pearson r( fresh_253 , frozen_253 ) < 0.95  -> "DECORRELATED"

Wall budget: 5 folds CPU. nb3340 measured ~7-10 min for a full-corpus refit;
each fold trains on ~80% so ~6-9 min/fold => ~30-45 min for 5 folds. If the
elapsed time after fold 3 projects the 5-fold total over WALL_BUDGET_MIN, we
finalize as a 3-fold ensemble (OOF then only covers 3 folds; the 253/513
anchors average the folds actually run). 3-fold still yields a valid, if
slightly noisier, independent anchor.

Outputs
-------
  data/processed/nb3350_chemprop_v3_oof.npy   (253,)  fresh anchor on unblind
  data/processed/te_nb3350_chemprop_v3.npy    (513,)  fresh anchor on test (deploy)
  data/processed/nb3350_oof_train.npy         (4139,) genuine OOF on train (diag)
  data/processed/nb3350_summary.json
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxr.data import load_train, load_test, load_counter, load_phase1_unblinded  # noqa: E402
from pxr.chem import standardize_smiles, to_inchikey, bemis_murcko  # noqa: E402
from pxr.eval import rae, scaffold_kfold_indices  # noqa: E402
from pxr.paths import DATA_PROCESSED  # noqa: E402

SEED = 42
DEPTH = 3
HIDDEN_DIM = 300
FFN_LAYERS = 2
DROPOUT = 0.1
MAX_EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 64
N_SPLITS = 5
WALL_BUDGET_MIN = 30.0          # if 5-fold projects over this after fold 3, cut to 3-fold
PER_FOLD_TIMEOUT_MIN = 7.0      # hard per-fold wall: skip a fold if elapsed grows past total box
TOTAL_BOX_MIN = 17.0            # finalize/stop accepting new folds once total elapsed exceeds this
FROZEN_ANCHOR = DATA_PROCESSED / "te_chemprop_aux.npy"
AUDIT_IDX = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNBLIND_Y = DATA_PROCESSED / "postmortem" / "pm_unblind_y.npy"


# ---------------------------------------------------------------------------
# Corpus assembly: clean 4139 train + counter task; 253/513 held out
# ---------------------------------------------------------------------------

def build_corpus():
    print("[corpus] loading train / counter / test / phase1...", flush=True)
    tr = load_train()
    co = load_counter()
    te = load_test()
    ph = load_phase1_unblinded()

    assert len(tr) == 4139, f"TRAIN expected 4139, got {len(tr)}"
    assert len(te) == 513, f"TEST expected 513, got {len(te)}"
    assert len(ph) == 253, f"PHASE1 expected 253, got {len(ph)}"

    # standardize
    for df in (tr, co, te, ph):
        df["std_smiles"] = df["smiles"].apply(standardize_smiles)
        df.dropna(subset=["std_smiles"], inplace=True)

    # --- main train (4139), keep rows with a real PXR pec50 ---
    tr_main = tr.loc[tr["pec50"].notna(), ["std_smiles", "pec50"]].copy()

    # --- merge counter pec50 as the second (null) task; median over dup SMILES ---
    co_clean = co.loc[co["pec50"].notna(), ["std_smiles", "pec50"]].copy()
    co_clean = co_clean.rename(columns={"pec50": "pec50_null"})
    co_grp = co_clean.groupby("std_smiles", as_index=False)["pec50_null"].median()

    # Dedupe train by std_smiles (median pec50 if collisions), then attach null head.
    tr_main = tr_main.groupby("std_smiles", as_index=False)["pec50"].median()
    corpus = tr_main.merge(co_grp, on="std_smiles", how="left")
    corpus["scaffold"] = corpus["std_smiles"].apply(bemis_murcko).fillna("")

    n_null = int(corpus["pec50_null"].notna().sum())
    print(f"[corpus] train unique SMILES: {len(corpus)} | counter(null) labels: {n_null}",
          flush=True)

    # Test in load_test() ORDER (no dedup -- 513 unique verified) so the output
    # te_513 aligns with the frozen anchor and _audit_unblind_idx.
    te_out = te[["std_smiles", "name"]].reset_index(drop=True)
    assert len(te_out) == 513, f"test rows changed to {len(te_out)} after std"

    ph_out = ph.loc[ph["pec50"].notna(), ["std_smiles", "name", "pec50"]].reset_index(drop=True)
    print(f"[corpus] test rows: {len(te_out)} (load_test order) | phase1 val rows: {len(ph_out)}",
          flush=True)
    return corpus, te_out, ph_out, n_null


# ---------------------------------------------------------------------------
# chemprop multitask fold fit
# ---------------------------------------------------------------------------

def _make_mt_dataset(cdata, smi_list, y2_scaled):
    """y2_scaled: (n,2) standardized targets, NaN preserved on the null task."""
    dpts = [
        cdata.MoleculeDatapoint.from_smi(s, y=np.asarray(y, dtype=np.float32))
        for s, y in zip(smi_list, y2_scaled)
    ]
    return cdata.MoleculeDataset(dpts)


def run_cv(corpus, te_df, ph_df, t_start):
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping
    import chemprop
    from chemprop import data as cdata, models as cmodels, nn as cnn

    print(f"[chemprop] versions: chemprop={chemprop.__version__} torch={torch.__version__} "
          f"lightning={L.__version__}", flush=True)

    smiles = corpus["std_smiles"].tolist()
    y_pxr = corpus["pec50"].to_numpy(dtype=np.float32)              # (N,)
    y_null = corpus["pec50_null"].to_numpy(dtype=np.float32)        # (N,) NaN where absent
    scaffolds = corpus["scaffold"].tolist()
    N = len(smiles)

    # Standardize each task on observed values only (NaN preserved on null task).
    pxr_mean = float(np.nanmean(y_pxr))
    pxr_std = float(np.nanstd(y_pxr, ddof=1)) or 1.0
    null_mean = float(np.nanmean(y_null))
    null_std = float(np.nanstd(y_null, ddof=1)) or 1.0
    Y2 = np.column_stack([
        (y_pxr - pxr_mean) / pxr_std,
        (y_null - null_mean) / null_std,
    ]).astype(np.float32)                                          # (N,2), NaN on null kept

    smi_te = te_df["std_smiles"].tolist()
    smi_ph = ph_df["std_smiles"].tolist()

    folds = scaffold_kfold_indices(scaffolds, n_splits=N_SPLITS, seed=SEED)

    oof = np.full(N, np.nan, dtype=np.float64)                     # OOF PXR pred on train
    te_acc = np.zeros(len(smi_te), dtype=np.float64)               # sum of fold preds (513)
    ph_acc = np.zeros(len(smi_ph), dtype=np.float64)               # sum of fold preds (253)
    folds_done = 0
    fold_epochs = []

    # ----- RESUME: load per-fold checkpoint if present -----------------------
    # The accumulators (ph_acc/te_acc) are SUMS of fold-model predictions on the
    # held-out 253/513; resuming means seeding them with the cached folds' sums
    # and continuing to ADD new fold-models. For the held-out inference ensemble
    # this is valid bagging even if the cached folds came from a different split
    # geometry (the OOF-on-train diagnostic is then only partially consistent).
    _ckd = ROOT / "data" / "processed"
    _fd_path = _ckd / "nb3350_ckpt_folds_done.npy"
    resume_n = 0
    if _fd_path.exists():
        try:
            resume_n = int(np.load(_fd_path)[0])
            if resume_n >= 1:
                oof = np.load(_ckd / "nb3350_ckpt_oof.npy").astype(np.float64)
                ph_acc = np.load(_ckd / "nb3350_ckpt_ph_acc.npy").astype(np.float64)
                te_acc = np.load(_ckd / "nb3350_ckpt_te_acc.npy").astype(np.float64)
                assert ph_acc.shape[0] == len(smi_ph), "ckpt ph_acc len mismatch"
                assert te_acc.shape[0] == len(smi_te), "ckpt te_acc len mismatch"
                folds_done = resume_n
                fold_epochs = [None] * resume_n   # epochs of cached folds unknown
                print(f"[resume] loaded checkpoint: {resume_n} folds done "
                      f"(oof finite={int(np.isfinite(oof).sum())}, "
                      f"ph_acc mean={ph_acc.mean():.4f}); continuing to fold {resume_n}+",
                      flush=True)
        except Exception as _e:
            print(f"[resume] WARN could not load checkpoint ({_e}); starting fresh",
                  flush=True)
            resume_n = 0
            oof = np.full(N, np.nan, dtype=np.float64)
            te_acc = np.zeros(len(smi_te), dtype=np.float64)
            ph_acc = np.zeros(len(smi_ph), dtype=np.float64)
            folds_done = 0
            fold_epochs = []

    def make_loader(idx, shuffle):
        ds = _make_mt_dataset(cdata, [smiles[i] for i in idx], Y2[idx])
        return cdata.build_dataloader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    def predict(model, trainer, smi_list):
        ds = cdata.MoleculeDataset([cdata.MoleculeDatapoint.from_smi(s) for s in smi_list])
        loader = cdata.build_dataloader(ds, batch_size=128, shuffle=False, num_workers=0)
        raw = trainer.predict(model, loader)
        arr = torch.cat(raw).numpy()                              # (n, 2): [pxr_sc, null_sc]
        pxr_sc = arr[:, 0]
        return pxr_sc * pxr_std + pxr_mean                        # PXR head, de-standardized

    elapsed_new_total = 0.0      # cumulative wall (min) of folds trained THIS run
    for k, (tr_idx, va_idx) in enumerate(folds):
        # RESUME: skip folds already accumulated in the checkpoint.
        if k < resume_n:
            continue
        elapsed = (time.time() - t_start) / 60.0
        # Total-box guard: stop accepting new folds once the overall wall is near.
        if elapsed > TOTAL_BOX_MIN:
            print(f"[box] {elapsed:.1f} min > TOTAL_BOX_MIN {TOTAL_BOX_MIN}; "
                  f"finalizing as {folds_done}-fold ensemble before fold {k}.", flush=True)
            break
        # Project remaining: if one more fold (avg cost) would blow the box, stop.
        if folds_done > resume_n:                       # have a measured new-fold cost
            avg_new = (elapsed_new_total) / (folds_done - resume_n)
            if elapsed + avg_new > TOTAL_BOX_MIN:
                print(f"[box] projecting +{avg_new:.1f} min next fold from {elapsed:.1f} "
                      f"exceeds {TOTAL_BOX_MIN}; finalizing {folds_done}-fold.", flush=True)
                break
        # Legacy wall guard: after >=3 folds, project full total; cut if over budget.
        if folds_done >= 3:
            proj_full = elapsed / max(folds_done, 1) * N_SPLITS
            if proj_full > WALL_BUDGET_MIN:
                print(f"[wall] {elapsed:.1f} min after {folds_done} folds projects "
                      f"{proj_full:.1f} min for {N_SPLITS}-fold (> {WALL_BUDGET_MIN}); "
                      f"finalizing as {folds_done}-fold ensemble.", flush=True)
                break

        print(f"\n[fold {k}] train={len(tr_idx)} val={len(va_idx)} | elapsed {elapsed:.1f} min",
              flush=True)
        loader_tr = make_loader(tr_idx, shuffle=True)
        loader_va = make_loader(va_idx, shuffle=False)

        mpnn = cmodels.MPNN(
            message_passing=cnn.BondMessagePassing(depth=DEPTH, d_h=HIDDEN_DIM),
            agg=cnn.MeanAggregation(),
            predictor=cnn.RegressionFFN(n_tasks=2, n_layers=FFN_LAYERS, dropout=DROPOUT),
        )
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
        trainer = L.Trainer(
            max_epochs=MAX_EPOCHS,
            callbacks=[es],
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        fit_start = time.time()
        trainer.fit(mpnn, loader_tr, loader_va)
        fold_fit_min = (time.time() - fit_start) / 60.0
        elapsed_new_total += fold_fit_min
        fold_epochs.append(int(trainer.current_epoch))
        print(f"[fold {k}] fit done in {fold_fit_min:.1f} min "
              f"(best epoch {trainer.current_epoch})", flush=True)

        # OOF on the held-out fold (genuine out-of-fold PXR predictions)
        oof[va_idx] = predict(mpnn, trainer, [smiles[i] for i in va_idx])
        # Accumulate held-out 253 + 513 predictions
        ph_acc += predict(mpnn, trainer, smi_ph)
        te_acc += predict(mpnn, trainer, smi_te)
        folds_done += 1
        # Per-fold checkpoint to disk so a mid-run kill leaves usable partial output
        try:
            import os as _os
            _ckd = "data/processed"
            np.save(_os.path.join(_ckd, "nb3350_ckpt_oof.npy"), oof)
            np.save(_os.path.join(_ckd, "nb3350_ckpt_ph_acc.npy"), ph_acc)
            np.save(_os.path.join(_ckd, "nb3350_ckpt_te_acc.npy"), te_acc)
            np.save(_os.path.join(_ckd, "nb3350_ckpt_folds_done.npy"),
                    np.array([folds_done], dtype=np.int64))
            print(f"[ckpt] saved after fold {k} (folds_done={folds_done})", flush=True)
        except Exception as _e:
            print(f"[ckpt] WARN save failed: {_e}", flush=True)

    assert folds_done >= 1, "no folds completed"
    ph_pred = ph_acc / folds_done
    te_pred = te_acc / folds_done

    # Clip to training-pec50 range +/- 0.5 (same convention as nb35/nb950)
    lo = float(np.nanmin(y_pxr)) - 0.5
    hi = float(np.nanmax(y_pxr)) + 0.5
    ph_pred = np.clip(ph_pred, lo, hi)
    te_pred = np.clip(te_pred, lo, hi)
    oof = np.clip(oof, lo, hi)

    info = {
        "folds_done": folds_done,
        "n_splits_requested": N_SPLITS,
        "fold_best_epochs": fold_epochs,
        "oof_coverage": int(np.isfinite(oof).sum()),
        "pxr_task_mean": pxr_mean, "pxr_task_std": pxr_std,
        "null_task_mean": null_mean, "null_task_std": null_std,
    }
    return oof, ph_pred, te_pred, info


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 72, flush=True)
    print("nb3350 -- fresh 5-fold multitask chemprop anchor (chemprop_aux_v3)", flush=True)
    print("=" * 72, flush=True)

    corpus, te_df, ph_df, n_null = build_corpus()

    oof, ph_pred, te_pred_inorder, info = run_cv(corpus, te_df, ph_df, t_start)

    # ----- align outputs to load_test() row order (te_df IS in that order) -----
    # te_pred_inorder is already in te_df == load_test() order (513).
    assert len(te_pred_inorder) == 513, f"te pred len {len(te_pred_inorder)} != 513"

    # The honest 253-unblind anchor: two equivalent constructions --
    #   (a) direct: ph_pred (fold-ensemble predictions on the 253 phase1 SMILES)
    #   (b) re-index: te_pred_inorder[_audit_unblind_idx]
    # We persist (b) as nb3350_chemprop_v3_oof so it is bit-aligned with how the
    # frozen anchor's 253 is computed; (a) is reported as a cross-check.
    unb_idx = np.load(AUDIT_IDX)
    fresh_253 = te_pred_inorder[unb_idx]

    # ----- evaluate against truth + frozen anchor -----
    y_unb = np.load(UNBLIND_Y)                                     # (253,)
    frozen_te = np.load(FROZEN_ANCHOR)                             # (513,)
    frozen_253 = frozen_te[unb_idx]

    fresh_rae = float(rae(y_unb, fresh_253))
    direct_rae = float(rae(ph_df["pec50"].to_numpy(dtype=np.float32), ph_pred))
    frozen_rae = float(rae(y_unb, frozen_253))
    pearson = float(np.corrcoef(fresh_253, frozen_253)[0, 1])
    # OOF diagnostic on the train rows that received an OOF prediction
    oof_mask = np.isfinite(oof)
    y_pxr_full = corpus["pec50"].to_numpy(dtype=np.float32)
    oof_rae = float(rae(y_pxr_full[oof_mask], oof[oof_mask]))

    usable = fresh_rae < 0.65
    decorrelated = pearson < 0.95

    # ----- save artifacts -----
    np.save(DATA_PROCESSED / "nb3350_chemprop_v3_oof.npy", fresh_253.astype(np.float32))
    np.save(DATA_PROCESSED / "te_nb3350_chemprop_v3.npy", te_pred_inorder.astype(np.float32))
    np.save(DATA_PROCESSED / "nb3350_oof_train.npy", oof.astype(np.float32))

    summary = {
        "script": "nb3350_fresh_chemprop_5fold",
        "architecture": "MPNN multitask BondMP(depth=3,d_h=300)+MeanAgg+FFN(2,drop=0.1), n_tasks=2",
        "protocol": f"{info['folds_done']}-fold scaffold CV ensemble on 4139; 253+513 held out",
        "corpus_train_rows": int(len(corpus)),
        "counter_null_labels": int(n_null),
        "folds_done": info["folds_done"],
        "n_splits_requested": N_SPLITS,
        "fold_best_epochs": info["fold_best_epochs"],
        "oof_coverage": info["oof_coverage"],
        "oof_train_RAE": oof_rae,
        # headline numbers
        "fresh_anchor_unblind_RAE": fresh_rae,
        "fresh_anchor_unblind_RAE_direct_phase1order": direct_rae,
        "frozen_anchor_unblind_RAE": frozen_rae,
        "pearson_fresh_vs_frozen_253": pearson,
        "fresh_253_mean": float(fresh_253.mean()),
        "fresh_253_std": float(fresh_253.std()),
        "frozen_253_mean": float(frozen_253.mean()),
        "frozen_253_std": float(frozen_253.std()),
        "truth_253_mean": float(y_unb.mean()),
        "truth_253_std": float(y_unb.std()),
        "te_513_mean": float(te_pred_inorder.mean()),
        "te_513_std": float(te_pred_inorder.std()),
        # gates
        "GATE_usable_anchor": bool(usable),
        "GATE_decorrelated": bool(decorrelated),
        "wall_time_min": (time.time() - t_start) / 60.0,
    }
    with open(DATA_PROCESSED / "nb3350_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72, flush=True)
    print(f"folds_done={info['folds_done']}  OOF_train_RAE={oof_rae:.4f}", flush=True)
    print(f"FRESH anchor unblind RAE: {fresh_rae:.4f}  (direct/phase1-order {direct_rae:.4f})",
          flush=True)
    print(f"FROZEN anchor unblind RAE: {frozen_rae:.4f}", flush=True)
    print(f"Pearson(fresh,frozen) on 253: {pearson:.4f}", flush=True)
    print(f"GATE usable(<0.65)={usable}  GATE decorrelated(<0.95)={decorrelated}", flush=True)
    print(f"wall {summary['wall_time_min']:.1f} min", flush=True)
    print("=" * 72, flush=True)
    return summary


if __name__ == "__main__":
    main()
