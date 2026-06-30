"""nb2001 -- chemprop_aux REFIT with augmented dataset (4139 + 552 new).

HYPOTHESIS
----------
Current PRE-unblind anchor `chemprop_aux` (v1, trained on 4139 only) sits at
phase1_in_RAE = 0.6216 and `predicted LB` band 0.55-0.65 (per
feedback_lb_two_regime_calibration).  Adding 96 semi-pure + 456 crudes
(=552 new compounds; corpus 4139+552-leaks ~= 4690 unique post-dedupe)
under the tiered-weight regime TIER_W = {original:1.0, semi_pure:0.7, crudes:0.4}
should retain PRE-unblind status (the 253 phase1_unblinded stays held out)
while pushing the anchor down into the 0.55-0.62 band.

DESIGN
------
1. Reuse Kaggle v4 corpus assembly from notebooks/950_chemprop_aux_v2_kaggle.ipynb.
2. Force TIER_W = {original:1.0, semi_pure:0.7, crudes:0.4} (the explicit task
   spec; differs from notebook v2 which had all-1.0 to fight variance compression).
3. Per memory `feedback_resource_limits` -- check CPU load first.
4. Per memory note "Local Windows CPU silent-dies at chemprop epoch 14
   (v3+v4 both)", default to LGBM(combined) fallback on weighted corpus.
   Chemprop attempt is BEHIND a `--try-chemprop` flag with strict wall guard.
5. PRE-unblind discipline: 253 phase1_unblinded is HELD OUT, never trained on.
6. Gate: phase1_in_RAE must be <= 0.6216 to be promoted as new PRE-unblind anchor.

OUTPUTS
-------
- scripts/nb2001_chemprop_refit.py  (this file)
- data/processed/nb2001_summary.json
- data/processed/te_nb2001_refit.npy             (513,)
- data/processed/nb2001_phase1_pred.npy          (253,)
- submissions/nb2001_chemprop_refit.csv  if gate passes
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxr.data import (  # noqa: E402
    load_train, load_test, load_counter,
    load_crudes, load_semi_pure, load_phase1_unblinded,
)
from pxr.chem import standardize_smiles, to_inchikey, bemis_murcko  # noqa: E402
from pxr.eval import rae  # noqa: E402
from pxr.featurize import rdkit_desc, morgan, impute  # noqa: E402
from pxr.paths import DATA_PROCESSED, SUBMISSIONS  # noqa: E402


SEED = 42
# Task-spec tier weights (DIFFERENT from notebook v2 which used all-1.0).
TIER_W = {"original": 1.0, "semi_pure": 0.7, "crudes": 0.4}
CAD_CV_PENALTY = 0.5
GATE_RAE = 0.6216  # chemprop_aux v1 phase1_in_RAE; must beat to be promoted

_NUMERIC_COLS = ("pec50", "pec50_null", "pec50_se", "emax", "emax_se")
_CRUDE_NUMERIC = (
    "Crude CAD Peak Area CV (%)", "Crude Product Yield (%)",
    "Crude Correction Factor", "Crude CAD Slope CV (%)",
    "Crude CAD Yield SE (log10)",
)


def _coerce(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _drop_bad_pec50(df, name):
    if "pec50" not in df.columns:
        return df, 0
    bad = df["pec50"].isna()
    n = int(bad.sum())
    if n:
        print(f"[clean] {name}: drop {n} NaN-pec50 rows", flush=True)
    return df.loc[~bad].copy(), n


def build_corpus():
    """Returns corpus, te_out, ph_out, drop_log -- mirrors nb950 pipeline."""
    print("[corpus] loading sources...", flush=True)
    tr = load_train()
    sp = load_semi_pure()
    cr = load_crudes()
    co = load_counter()
    te = load_test()
    ph = load_phase1_unblinded()

    raw_shapes = {"train": tr.shape, "semi_pure": sp.shape, "crudes": cr.shape,
                  "counter": co.shape, "test": te.shape, "phase1": ph.shape}
    for k, sh in raw_shapes.items():
        print(f"[assert] {k}={sh}", flush=True)
    assert tr.shape[0] == 4139
    assert co.shape[0] == 2859
    assert te.shape[0] == 513
    assert ph.shape[0] == 253
    assert sp.shape[0] == 96
    assert cr.shape[0] == 456

    drop_log = {}
    for name, df in (("train", tr), ("semi_pure", sp), ("crudes", cr),
                     ("counter", co), ("phase1", ph)):
        _coerce(df, _NUMERIC_COLS)
    _coerce(cr, _CRUDE_NUMERIC)
    tr, drop_log["train"] = _drop_bad_pec50(tr, "train")
    sp, drop_log["semi_pure"] = _drop_bad_pec50(sp, "semi_pure")
    cr, drop_log["crudes"] = _drop_bad_pec50(cr, "crudes")
    co, _ = _drop_bad_pec50(co, "counter")
    ph, _ = _drop_bad_pec50(ph, "phase1")

    print("[corpus] standardizing SMILES...", flush=True)
    for df in (tr, sp, cr, co, te, ph):
        df["std_smiles"] = df["smiles"].apply(standardize_smiles)
        df.dropna(subset=["std_smiles"], inplace=True)

    te["inchikey"] = te["std_smiles"].apply(to_inchikey)
    te_ikeys = set(te["inchikey"].dropna())

    tr_main = tr.loc[tr["pec50"].notna(), ["std_smiles", "pec50"]].copy()
    tr_main["weight"] = TIER_W["original"]
    tr_main["source"] = "original"

    sp["inchikey"] = sp["std_smiles"].apply(to_inchikey)
    leaked = sp["inchikey"].isin(te_ikeys)
    sp_clean = sp.loc[~leaked & sp["pec50"].notna(),
                      ["std_smiles", "pec50"]].copy()
    sp_clean["weight"] = TIER_W["semi_pure"]
    sp_clean["source"] = "semi_pure"
    print(f"[corpus] semi_pure: kept {len(sp_clean)}/{len(sp)} (leaked={int(leaked.sum())})", flush=True)

    cad_cv = cr.get("Crude CAD Peak Area CV (%)")
    cr2 = cr.loc[cr["pec50"].notna(), ["std_smiles", "pec50"]].copy()
    if cad_cv is not None:
        cr2 = cr2.assign(cad_cv=cad_cv.reindex(cr2.index).values)
    else:
        cr2 = cr2.assign(cad_cv=np.nan)
    cr_w = np.full(len(cr2), TIER_W["crudes"], dtype=np.float32)
    high_cv = (cr2["cad_cv"].values > 5) & cr2["cad_cv"].notna().values
    cr_w[high_cv] *= CAD_CV_PENALTY
    cr2["weight"] = cr_w
    cr2["source"] = "crudes"
    cr2 = cr2.drop(columns=["cad_cv"])
    print(f"[corpus] crudes: kept {len(cr2)}; high-CV penalty x{CAD_CV_PENALTY} on {int(high_cv.sum())}", flush=True)

    co_clean = co.loc[co["pec50"].notna(), ["std_smiles", "pec50"]].copy()
    co_clean = co_clean.rename(columns={"pec50": "pec50_null"})
    co_grouped = co_clean.groupby("std_smiles", as_index=False)["pec50_null"].median()

    corpus = pd.concat([tr_main, sp_clean, cr2], ignore_index=True)
    corpus = corpus.sort_values("weight", ascending=False)
    corpus = corpus.drop_duplicates(subset=["std_smiles"], keep="first").reset_index(drop=True)
    corpus = corpus.merge(co_grouped, on="std_smiles", how="left")
    corpus["scaffold"] = corpus["std_smiles"].apply(bemis_murcko).fillna("")

    print(f"[corpus] FINAL: {len(corpus)} unique SMILES "
          f"({(corpus['source']=='original').sum()} orig + "
          f"{(corpus['source']=='semi_pure').sum()} SP + "
          f"{(corpus['source']=='crudes').sum()} crudes)", flush=True)
    print(f"[corpus] null head: {corpus['pec50_null'].notna().sum()} non-null", flush=True)

    te_out = te[["std_smiles", "name"]].drop_duplicates("std_smiles").reset_index(drop=True)
    ph_out = ph.loc[ph["pec50"].notna(),
                    ["std_smiles", "name", "pec50"]].reset_index(drop=True)

    return corpus, te_out, ph_out, drop_log


# ---------------------------------------------------------------------------
# LGBM fallback path (CPU-safe, primary path per memory feedback)
# ---------------------------------------------------------------------------

def fit_lgbm(corpus, te_out, ph_out):
    """LGBM(combined) trained on weighted augmented corpus.

    Per-row sample_weight = TIER_W tier weight (already set in corpus['weight']).
    """
    from lightgbm import LGBMRegressor
    print(f"[lgbm] featurizing combined (Morgan2048 + RDKit~217)...", flush=True)
    smi_tr = corpus["std_smiles"].tolist()
    smi_te = te_out["std_smiles"].tolist()
    smi_ph = ph_out["std_smiles"].tolist()

    X_tr = np.concatenate([morgan(smi_tr), rdkit_desc(smi_tr)], axis=1)
    X_te = np.concatenate([morgan(smi_te), rdkit_desc(smi_te)], axis=1)
    X_ph = np.concatenate([morgan(smi_ph), rdkit_desc(smi_ph)], axis=1)
    X_tr = impute(X_tr); X_te = impute(X_te); X_ph = impute(X_ph)
    print(f"[lgbm] X_tr={X_tr.shape}  X_te={X_te.shape}  X_ph={X_ph.shape}", flush=True)

    y_tr = corpus["pec50"].to_numpy(dtype=np.float32)
    w_tr = corpus["weight"].to_numpy(dtype=np.float32)

    model = LGBMRegressor(
        n_estimators=500, num_leaves=64, learning_rate=0.05,
        n_jobs=-1, random_state=SEED, verbose=-1,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    fit_s = time.time() - t0
    print(f"[lgbm] fit done in {fit_s:.1f}s", flush=True)

    lo = float(y_tr.min()) - 0.5
    hi = float(y_tr.max()) + 0.5
    te_preds = np.clip(model.predict(X_te), lo, hi).astype(np.float32)
    ph_preds = np.clip(model.predict(X_ph), lo, hi).astype(np.float32)
    return te_preds, ph_preds, fit_s


# ---------------------------------------------------------------------------
# Chemprop path (CPU-risky; guarded behind --try-chemprop flag)
# ---------------------------------------------------------------------------

def fit_chemprop(corpus, te_out, ph_out, wall_budget_s=4500):
    """2 single-task MPNNs on weighted corpus. Mirrors notebook 950 v2 cell."""
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping
    from chemprop import data as cdata, models as cmodels, nn as cnn

    DEPTH = 3; HIDDEN_DIM = 300; FFN_LAYERS = 2; DROPOUT = 0.1
    MAX_EPOCHS = 50; PATIENCE = 8; BATCH_SIZE = 64
    ACCEL = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"[chemprop] accel={ACCEL}  wall_budget={wall_budget_s}s", flush=True)

    smiles = corpus["std_smiles"].tolist()
    y_pec50 = corpus["pec50"].to_numpy(dtype=np.float32)
    y_null = corpus["pec50_null"].to_numpy(dtype=np.float32)
    w = corpus["weight"].to_numpy(dtype=np.float32)
    scaffolds = corpus["scaffold"].tolist()

    pec50_mean = float(np.nanmean(y_pec50))
    pec50_std = float(np.nanstd(y_pec50, ddof=1)) or 1.0
    y_pec50_sc = (y_pec50 - pec50_mean) / pec50_std

    from pxr.eval import scaffold_kfold_indices

    def make_ds(idx, smi_arr, y_arr, w_arr):
        dpts = [cdata.MoleculeDatapoint.from_smi(smi_arr[i], y=np.array([y_arr[i]], dtype=np.float32),
                                                  weight=float(w_arr[i])) for i in idx]
        return cdata.MoleculeDataset(dpts)

    def fit_head(name, smi_arr, y_arr, w_arr, scaf_arr):
        folds = scaffold_kfold_indices(scaf_arr, n_splits=5)
        tr_i, va_i = folds[0]
        print(f"[chemprop:{name}] tr={len(tr_i)} va={len(va_i)}", flush=True)
        ds_tr = make_ds(tr_i, smi_arr, y_arr, w_arr)
        ds_va = make_ds(va_i, smi_arr, y_arr, w_arr)
        ld_tr = cdata.build_dataloader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        ld_va = cdata.build_dataloader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        mpnn = cmodels.MPNN(
            message_passing=cnn.BondMessagePassing(depth=DEPTH, d_h=HIDDEN_DIM),
            agg=cnn.MeanAggregation(),
            predictor=cnn.RegressionFFN(n_tasks=1, n_layers=FFN_LAYERS, dropout=DROPOUT),
        )
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
        trainer = L.Trainer(
            max_epochs=MAX_EPOCHS, callbacks=[es], accelerator=ACCEL, devices=1,
            enable_progress_bar=False, enable_model_summary=False,
            logger=False, enable_checkpointing=False,
        )
        t0 = time.time()
        trainer.fit(mpnn, ld_tr, ld_va)
        return mpnn, trainer, time.time() - t0

    def predict(model, trainer_h, smi_list, mean, std):
        dpts = [cdata.MoleculeDatapoint.from_smi(s) for s in smi_list]
        ds = cdata.MoleculeDataset(dpts)
        ld = cdata.build_dataloader(ds, batch_size=128, shuffle=False, num_workers=0)
        raw = trainer_h.predict(model, ld)
        p_sc = torch.cat(raw).numpy().reshape(-1)
        return p_sc * std + mean

    t_start = time.time()
    mpnn_main, tr_main, fit_main_s = fit_head("main", smiles, y_pec50_sc, w, scaffolds)
    if time.time() - t_start > wall_budget_s:
        raise TimeoutError("main head exceeded wall budget")

    null_mask = ~np.isnan(y_null)
    n_null = int(null_mask.sum())
    use_null = n_null >= 500
    mpnn_null = tr_null = None
    null_mean = pec50_mean; null_std = pec50_std; fit_null_s = 0.0
    if use_null:
        nidx = np.where(null_mask)[0]
        smi_n = [smiles[i] for i in nidx]
        y_n = y_null[nidx]; w_n = w[nidx]
        scaf_n = [scaffolds[i] for i in nidx]
        null_mean = float(np.mean(y_n))
        null_std = float(np.std(y_n, ddof=1)) or 1.0
        y_n_sc = (y_n - null_mean) / null_std
        mpnn_null, tr_null, fit_null_s = fit_head("null", smi_n, y_n_sc, w_n, scaf_n)

    te_main = predict(mpnn_main, tr_main, te_out["std_smiles"].tolist(), pec50_mean, pec50_std)
    ph_main = predict(mpnn_main, tr_main, ph_out["std_smiles"].tolist(), pec50_mean, pec50_std)
    if use_null:
        te_null = predict(mpnn_null, tr_null, te_out["std_smiles"].tolist(), null_mean, null_std)
        ph_null = predict(mpnn_null, tr_null, ph_out["std_smiles"].tolist(), null_mean, null_std)
        te_preds = 0.70 * te_main + 0.30 * te_null
        ph_preds = 0.70 * ph_main + 0.30 * ph_null
    else:
        te_preds, ph_preds = te_main, ph_main

    lo = float(y_pec50.min()) - 0.5
    hi = float(y_pec50.max()) + 0.5
    te_preds = np.clip(te_preds, lo, hi).astype(np.float32)
    ph_preds = np.clip(ph_preds, lo, hi).astype(np.float32)
    return te_preds, ph_preds, fit_main_s + fit_null_s, bool(use_null)


# ---------------------------------------------------------------------------
# Resource gate (per memory feedback_resource_limits)
# ---------------------------------------------------------------------------

def cpu_safe_for_chemprop():
    """Return True if CPU load < 50% AND > 8 GB RAM free."""
    try:
        import psutil
        load = psutil.cpu_percent(interval=1.0)
        free_gb = psutil.virtual_memory().available / 1e9
        print(f"[resource] cpu={load:.1f}%  free_ram={free_gb:.1f}GB", flush=True)
        return load < 50 and free_gb > 8
    except Exception as e:
        print(f"[resource] check failed: {e} -- assume unsafe", flush=True)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--try-chemprop", action="store_true",
                        help="Attempt chemprop train (memory warns local CPU silent-dies).")
    parser.add_argument("--wall-budget-s", type=int, default=4500,
                        help="Chemprop wall-time guard in seconds (default 75 min).")
    args = parser.parse_args()

    DATA_PROCESSED.mkdir(exist_ok=True, parents=True)
    SUBMISSIONS.mkdir(exist_ok=True, parents=True)

    t_start = time.time()
    summary = {
        "script": "nb2001_chemprop_refit",
        "tier_weights": TIER_W,
        "cad_cv_penalty": CAD_CV_PENALTY,
        "gate_phase1_in_RAE": GATE_RAE,
        "fallback_used": True,
        "model": "nb2001b_lgbm_combined_refit",
    }

    # ---- corpus
    corpus, te_out, ph_out, drop_log = build_corpus()
    summary["corpus_total"] = int(len(corpus))
    summary["corpus_by_source"] = corpus["source"].value_counts().to_dict()
    summary["null_head_n_labels"] = int(corpus["pec50_null"].notna().sum())
    summary["n_test_513"] = int(len(te_out))
    summary["n_phase1_unblinded"] = int(len(ph_out))
    summary["dirty_dropped_per_source"] = drop_log

    # ---- decide path
    chemprop_attempted = False
    chemprop_error = None
    te_preds = ph_preds = None
    fit_s = 0.0
    used_null = False

    if args.try_chemprop:
        if not cpu_safe_for_chemprop():
            print("[main] CPU not safe -- skipping chemprop, going LGBM fallback", flush=True)
        else:
            try:
                print("[main] CPU safe -- attempting chemprop train", flush=True)
                chemprop_attempted = True
                te_preds, ph_preds, fit_s, used_null = fit_chemprop(
                    corpus, te_out, ph_out, wall_budget_s=args.wall_budget_s
                )
                summary.update({
                    "fallback_used": False,
                    "model": "nb2001_chemprop_aux_refit",
                    "chemprop_fit_minutes": fit_s / 60,
                    "chemprop_null_head_used": used_null,
                })
            except Exception as exc:
                chemprop_error = f"{type(exc).__name__}: {exc}"
                print(f"[main] chemprop FAILED: {chemprop_error} -- LGBM fallback", flush=True)
                te_preds = ph_preds = None

    if te_preds is None:
        te_preds, ph_preds, fit_s = fit_lgbm(corpus, te_out, ph_out)
        summary["lgbm_fit_seconds"] = fit_s

    # ---- honest validation on phase1_unblinded (NEVER trained on)
    y_ph = ph_out["pec50"].to_numpy(dtype=np.float32)
    phase1_rae = float(rae(y_ph, ph_preds))
    summary["phase1_unblinded_RAE"] = phase1_rae
    summary["phase1_unblinded_n"] = int(len(ph_out))
    summary["phase1_pred_mean"] = float(ph_preds.mean())
    summary["phase1_pred_std"] = float(ph_preds.std())
    summary["phase1_true_mean"] = float(y_ph.mean())
    summary["phase1_true_std"] = float(y_ph.std())
    summary["test513_pred_mean"] = float(te_preds.mean())
    summary["test513_pred_std"] = float(te_preds.std())
    summary["chemprop_attempted"] = chemprop_attempted
    summary["chemprop_error"] = chemprop_error

    # ---- gate
    passed = phase1_rae <= GATE_RAE
    summary["gate_passed"] = bool(passed)
    summary["delta_vs_anchor_v1"] = float(GATE_RAE - phase1_rae)

    # ---- save numpy artifacts
    np.save(DATA_PROCESSED / "te_nb2001_refit.npy", te_preds)
    np.save(DATA_PROCESSED / "nb2001_phase1_pred.npy", ph_preds)

    # ---- save submission CSV only if gate passes (PRE-unblind candidate)
    sub_path = None
    if passed:
        raw_test = load_test()
        raw_test["std_smiles"] = raw_test["smiles"].apply(standardize_smiles)
        pred_map = dict(zip(te_out["std_smiles"].tolist(), te_preds.tolist()))
        raw_test["pEC50"] = raw_test["std_smiles"].map(pred_map)
        if raw_test["pEC50"].isna().any():
            fill = float(corpus["pec50"].mean())
            raw_test["pEC50"] = raw_test["pEC50"].fillna(fill)
        sub = raw_test.rename(columns={"smiles": "SMILES", "name": "Molecule Name"})[
            ["SMILES", "Molecule Name", "pEC50"]
        ]
        sub_path = SUBMISSIONS / "nb2001_chemprop_refit.csv"
        sub.to_csv(sub_path, index=False)
        print(f"[save] {sub_path}  rows={len(sub)}", flush=True)
        summary["submission_path"] = str(sub_path)
        summary["pre_unblind_status"] = "PRE-unblind-eligible"
    else:
        summary["submission_path"] = None
        summary["pre_unblind_status"] = "GATE_FAILED_no_promote"

    summary["wall_time_min"] = (time.time() - t_start) / 60

    sum_path = DATA_PROCESSED / "nb2001_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] {sum_path}", flush=True)

    print("=" * 70)
    print(f"phase1_in_RAE = {phase1_rae:.4f}   gate = {GATE_RAE}   "
          f"delta = {GATE_RAE - phase1_rae:+.4f}   "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"model = {summary['model']}   wall = {summary['wall_time_min']:.1f} min")
    print("=" * 70)


if __name__ == "__main__":
    main()
