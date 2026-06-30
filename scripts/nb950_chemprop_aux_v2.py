"""nb950 -- chemprop_aux v2: retrain with augmented corpus.

Training corpus (per data-discovery cycle 2026-06-08):
- 4139 original TRAIN (full-weight, w=1.0)
- 95 semi-pure (drop the 1 row leaked to test-513) (w=0.7)
- 456 crudes/htchem (corrected_crude_pEC50; downweight w=0.4, x0.5 if CAD_CV>5%)
- 253 phase1_unblinded ARE HELD OUT (honest val; never trained on)

Two heads (matches the original chemprop_aux convention):
- pxr pEC50
- counter-assay pEC50 (2858 rows w/ counter labels, rest NaN-masked)

Architecture matches nb35_chemprop_auxiliary: MPNN (depth=3, d_h=300, FFN 2 layers, dropout=0.1).
Per-row sample weights via chemprop's MoleculeDatapoint(weight=...).

CPU-only. Single train/val split (90/10 scaffold) for early-stopping; no K-fold here -- we
already pay 25-50 min/fold for chemprop and the user wants a single deploy model.

Outputs:
- scripts/nb950_chemprop_aux_v2.py (this file)
- data/processed/te_chemprop_aux_v2.npy  (513,)
- data/processed/nb950_summary.json
- submissions/nb950_chemprop_aux_v2.csv

If chemprop refit fails or wall time > 90 min, fall back to LGBM(combined) trained on
the same weighted corpus, called nb950b_lgbm_v2 (outputs te_nb950b_lgbm_v2.npy).
"""

import os
import sys
import json
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["PYTHONIOENCODING"] = "utf-8"
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pxr.data import (  # noqa: E402
    load_train,
    load_test,
    load_counter,
    load_crudes,
    load_semi_pure,
    load_phase1_unblinded,
)
from pxr.chem import standardize_smiles, to_inchikey, bemis_murcko  # noqa: E402
from pxr.eval import rae, scaffold_kfold_indices  # noqa: E402
from pxr.paths import DATA_PROCESSED, SUBMISSIONS  # noqa: E402

SEED = 42
DEPTH = 3
HIDDEN_DIM = 300
FFN_LAYERS = 2
DROPOUT = 0.1
MAX_EPOCHS = 50
PATIENCE = 8
BATCH_SIZE = 64
WALL_BUDGET_S = 90 * 60  # 90 min budget for chemprop fit before triggering fallback

TIER_W = {"original": 1.0, "semi_pure": 0.7, "crudes": 0.4}
CAD_CV_PENALTY = 0.5  # extra x0.5 for crude rows with CAD_CV > 5%

# Columns that MUST be numeric in every dose-response-shaped source. Strings
# like "-" and "no data" are coerced to NaN here. Rows with NaN `pec50` are
# DROPPED (no label, no learning signal); NaN on the other columns is kept
# (Emax / SE are auxiliary and may be masked downstream).
_NUMERIC_COLS = ("pec50", "pec50_null", "pec50_se", "emax", "emax_se")
# Crudes-only extra numeric metadata used by build_corpus / CAD-CV penalty.
_CRUDE_NUMERIC_COLS = (
    "Crude CAD Peak Area CV (%)",
    "Crude Product Yield (%)",
    "Crude Correction Factor",
    "Crude CAD Slope CV (%)",
    "Crude CAD Yield SE (log10)",
)


def _coerce_numeric(df, cols, source_name):
    """Coerce listed cols to numeric in-place; report how many values became NaN.

    Returns the count of newly-NaN cells across all listed cols (for logging).
    """
    new_nan = 0
    for c in cols:
        if c not in df.columns:
            continue
        before = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        after = df[c].isna().sum()
        delta = int(after - before)
        if delta > 0:
            print(f"[clean] {source_name}.{c}: coerced {delta} string-bad cells to NaN",
                  flush=True)
            new_nan += delta
    return new_nan


def _drop_unparseable_pec50(df, source_name):
    """Drop rows where `pec50` is NaN (post-coercion); return (df_clean, n_dropped).
    Source must already have had `_coerce_numeric` applied to `pec50`.
    """
    if "pec50" not in df.columns:
        return df, 0
    bad = df["pec50"].isna()
    n_drop = int(bad.sum())
    if n_drop > 0:
        print(f"[clean] {source_name}: dropping {n_drop} rows with unparseable/missing pec50",
              flush=True)
    return df.loc[~bad].copy(), n_drop


# ---------------------------------------------------------------------------
# Corpus assembly
# ---------------------------------------------------------------------------

def build_corpus():
    """Stack train + semi_pure + crudes (excl. leaked SP); dedupe; merge counter labels.

    Returns
    -------
    corpus : DataFrame with [smiles, inchikey, pec50, pec50_null, weight, source, scaffold]
    test_df : DataFrame with [smiles] for the 513 blind test
    phase1_df : DataFrame with [smiles, pec50] for the 253 honest validation
    """
    print("[corpus] Loading sources...", flush=True)
    tr = load_train()
    sp = load_semi_pure()
    cr = load_crudes()
    co = load_counter()
    te = load_test()
    ph = load_phase1_unblinded()

    # ---- shape assertions on raw loads (unit-test-style) -------------------
    raw_shapes = {"train": tr.shape, "semi_pure": sp.shape, "crudes": cr.shape,
                  "counter": co.shape, "test": te.shape, "phase1": ph.shape}
    for k, sh in raw_shapes.items():
        print(f"[assert] raw {k}.shape={sh}", flush=True)
    assert tr.shape[0] == 4139, f"TRAIN expected 4139 rows, got {tr.shape[0]}"
    assert co.shape[0] == 2859, f"COUNTER expected 2859 rows, got {co.shape[0]}"
    assert te.shape[0] == 513, f"TEST expected 513 rows, got {te.shape[0]}"
    assert ph.shape[0] == 253, f"PHASE1 expected 253 rows, got {ph.shape[0]}"
    assert sp.shape[0] == 96, f"SEMI-PURE expected 96 rows, got {sp.shape[0]}"
    assert cr.shape[0] == 456, f"CRUDES expected 456 rows, got {cr.shape[0]}"

    # ---- coerce-then-drop dirty pec50 cells --------------------------------
    # '-' and 'no data' tokens are silent-NaN now; rows w/ NaN pec50 are
    # dropped. Counter / phase1 / train are already clean (verified
    # 2026-06-08) but we run the coerce defensively so future CSV churn does
    # not crash the script.
    drop_log = {}
    for name, df in (("train", tr), ("semi_pure", sp), ("crudes", cr),
                     ("counter", co), ("phase1_unblinded", ph)):
        _coerce_numeric(df, _NUMERIC_COLS, name)
    # crudes has extra numeric metadata cols with '-' sentinels
    _coerce_numeric(cr, _CRUDE_NUMERIC_COLS, "crudes")

    tr, drop_log["train"] = _drop_unparseable_pec50(tr, "train")
    sp, drop_log["semi_pure"] = _drop_unparseable_pec50(sp, "semi_pure")
    cr, drop_log["crudes"] = _drop_unparseable_pec50(cr, "crudes")
    co, drop_log["counter"] = _drop_unparseable_pec50(co, "counter")
    ph, drop_log["phase1_unblinded"] = _drop_unparseable_pec50(ph, "phase1_unblinded")
    print(f"[clean] dropped_per_source = {drop_log}", flush=True)

    # Standardize SMILES everywhere
    print("[corpus] Standardizing SMILES...", flush=True)
    for df in (tr, sp, cr, co, te, ph):
        df["std_smiles"] = df["smiles"].apply(standardize_smiles)
        df.dropna(subset=["std_smiles"], inplace=True)

    # Build the test inchikey set so we can drop leaked semi-pure rows
    te["inchikey"] = te["std_smiles"].apply(to_inchikey)
    te_ikeys = set(te["inchikey"].dropna())

    # ---- main train (4139, w=1.0)
    tr_main = tr.loc[tr["pec50"].notna(), ["std_smiles", "pec50"]].copy()
    tr_main["weight"] = TIER_W["original"]
    tr_main["source"] = "original"

    # ---- semi-pure (drop 1 leaked to test)
    sp["inchikey"] = sp["std_smiles"].apply(to_inchikey)
    leaked = sp["inchikey"].isin(te_ikeys)
    n_leaked = int(leaked.sum())
    sp_clean = sp.loc[~leaked & sp["pec50"].notna(), ["std_smiles", "pec50"]].copy()
    sp_clean["weight"] = TIER_W["semi_pure"]
    sp_clean["source"] = "semi_pure"
    print(f"[corpus] semi-pure: kept {len(sp_clean)} / {len(sp)} (dropped {n_leaked} test-leaked)", flush=True)

    # ---- crudes (use corrected_crude_pEC50 = 'pec50' col post-rename).
    # CAD-CV was string-coerced upstream by _coerce_numeric on _CRUDE_NUMERIC_COLS.
    cad_cv = cr.get("Crude CAD Peak Area CV (%)")
    cr2 = cr.loc[cr["pec50"].notna(), ["std_smiles", "pec50"]].copy()
    if cad_cv is not None:
        cr2 = cr2.assign(cad_cv=cad_cv.reindex(cr2.index).values)
    else:
        cr2 = cr2.assign(cad_cv=np.nan)
    cr_w = np.full(len(cr2), TIER_W["crudes"], dtype=np.float32)
    # CAD_CV > 5% AND not-NaN gets the extra penalty
    high_cv = (cr2["cad_cv"].values > 5) & cr2["cad_cv"].notna().values
    cr_w[high_cv] *= CAD_CV_PENALTY
    cr2["weight"] = cr_w
    cr2["source"] = "crudes"
    cr2 = cr2.drop(columns=["cad_cv"])
    print(f"[corpus] crudes: kept {len(cr2)} rows; high-CV downweighted {int(high_cv.sum())}", flush=True)

    # ---- merge counter-assay labels as a second head
    co_clean = co.loc[co["pec50"].notna(), ["std_smiles", "pec50"]].copy()
    co_clean = co_clean.rename(columns={"pec50": "pec50_null"})
    # If a SMILES has multiple counter measurements, take the median
    co_grouped = co_clean.groupby("std_smiles", as_index=False)["pec50_null"].median()

    # ---- stack everything
    corpus = pd.concat([tr_main, sp_clean, cr2], ignore_index=True)
    # Dedupe by std_smiles: keep the highest-weight row per SMILES
    corpus = corpus.sort_values("weight", ascending=False)
    corpus = corpus.drop_duplicates(subset=["std_smiles"], keep="first").reset_index(drop=True)

    # Merge null head (left join; rows without counter measurement become NaN -> masked)
    corpus = corpus.merge(co_grouped, on="std_smiles", how="left")
    corpus["inchikey"] = corpus["std_smiles"].apply(to_inchikey)

    # Bemis-Murcko scaffold for the train/val split
    corpus["scaffold"] = corpus["std_smiles"].apply(bemis_murcko)
    corpus["scaffold"] = corpus["scaffold"].fillna("")

    print(f"[corpus] FINAL stacked: {len(corpus)} unique SMILES "
          f"({(corpus['source']=='original').sum()} orig + "
          f"{(corpus['source']=='semi_pure').sum()} SP + "
          f"{(corpus['source']=='crudes').sum()} crudes)", flush=True)
    print(f"[corpus] null head non-null rows: {corpus['pec50_null'].notna().sum()}", flush=True)

    # Test + phase1
    te_out = te[["std_smiles", "name"]].drop_duplicates(subset=["std_smiles"]).reset_index(drop=True)
    ph_out = ph.loc[ph["pec50"].notna(), ["std_smiles", "name", "pec50"]].reset_index(drop=True)
    print(f"[corpus] test-513 std rows: {len(te_out)}; phase1 honest val: {len(ph_out)}", flush=True)

    return corpus, te_out, ph_out, drop_log


# ---------------------------------------------------------------------------
# Chemprop fit
# ---------------------------------------------------------------------------

def fit_chemprop(corpus, te_df, ph_df, t_start):
    """Train chemprop on pec50 main head; null head trained as separate model on
    NaN-filtered subset, then ensembled (40/60) -- avoids Lightning val_loss=NaN
    death when a fully-NaN-null batch hits MSE divide-by-zero (compute = 0/0)."""
    # v4 patch: explicit DEBUG logging + stderr trapping; null-NaN no longer
    # poisons val_loss because we split the heads.
    import logging
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    for _logger_name in ("lightning", "lightning.pytorch", "pytorch_lightning"):
        try:
            logging.getLogger(_logger_name).setLevel(logging.DEBUG)
        except Exception:
            pass

    import torch
    import lightning as L
    from lightning.pytorch.callbacks import EarlyStopping
    import chemprop
    from chemprop import data as cdata, models as cmodels, nn as cnn

    print(f"[chemprop] versions: chemprop={chemprop.__version__} torch={torch.__version__} "
          f"lightning={L.__version__}", flush=True)
    sys.stderr.write(f"[chemprop:stderr] versions confirmed; entering v4 patched fit\n")
    sys.stderr.flush()

    # DEFENSIVE: belt-and-suspenders coercion of every numeric column chemprop
    # will consume. v4: pec50_null no longer needs to be finite per-row because
    # the null head trains on a filtered subset (not multitask masking).
    _required_numeric = ["pec50", "pec50_null", "weight"]
    n_before = len(corpus)
    for c in _required_numeric:
        if c in corpus.columns:
            corpus[c] = pd.to_numeric(corpus[c], errors="coerce")
    bad = corpus["pec50"].isna() | corpus["weight"].isna() | ~np.isfinite(corpus["weight"].to_numpy(dtype=np.float64, na_value=np.nan))
    n_bad = int(bad.sum())
    if n_bad > 0:
        print(f"[chemprop] DEFENSIVE: dropping {n_bad}/{n_before} rows with non-numeric pec50/weight after coercion", flush=True)
        corpus = corpus.loc[~bad].reset_index(drop=True)
    print(f"[chemprop] corpus post-coerce: {len(corpus)} rows "
          f"(pec50 dtype={corpus['pec50'].dtype}, weight dtype={corpus['weight'].dtype}, "
          f"pec50_null dtype={corpus['pec50_null'].dtype})", flush=True)

    smiles = corpus["std_smiles"].tolist()
    y_pec50 = corpus["pec50"].to_numpy(dtype=np.float32)            # (N,)
    y_null = corpus["pec50_null"].to_numpy(dtype=np.float32)        # (N,) NaN where unknown
    w = corpus["weight"].to_numpy(dtype=np.float32)                 # (N,)
    scaffolds = corpus["scaffold"].tolist()

    # Standardize main head only (null head standardized separately on its subset)
    pec50_mean = float(np.nanmean(y_pec50))
    pec50_std = float(np.nanstd(y_pec50, ddof=1)) or 1.0
    y_pec50_sc = (y_pec50 - pec50_mean) / pec50_std

    # 5-fold scaffold split for main head val (early stopping only)
    folds = scaffold_kfold_indices(scaffolds, n_splits=5, seed=SEED)
    tr_idx, va_idx = folds[0]
    print(f"[chemprop] MAIN split: train={len(tr_idx)} val={len(va_idx)} "
          f"(scaffold-disjoint, 80/20)", flush=True)

    def make_dataset_single(idx, smi_arr, y_arr, w_arr):
        dpts = []
        for i in idx:
            yi = np.array([y_arr[i]], dtype=np.float32)
            dpts.append(cdata.MoleculeDatapoint.from_smi(
                smi_arr[i], y=yi, weight=float(w_arr[i])
            ))
        return cdata.MoleculeDataset(dpts)

    def fit_one_head(name, smi_arr, y_arr, w_arr, scaf_arr, mean, std):
        """Fit a single-task MPNN; returns the trained (mpnn, trainer) pair."""
        folds_h = scaffold_kfold_indices(scaf_arr, n_splits=5, seed=SEED)
        tr_i, va_i = folds_h[0]
        print(f"[chemprop:{name}] split: train={len(tr_i)} val={len(va_i)}", flush=True)
        ds_tr = make_dataset_single(tr_i, smi_arr, y_arr, w_arr)
        ds_va = make_dataset_single(va_i, smi_arr, y_arr, w_arr)
        loader_tr = cdata.build_dataloader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        loader_va = cdata.build_dataloader(ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        mpnn = cmodels.MPNN(
            message_passing=cnn.BondMessagePassing(depth=DEPTH, d_h=HIDDEN_DIM),
            agg=cnn.MeanAggregation(),
            predictor=cnn.RegressionFFN(n_tasks=1, n_layers=FFN_LAYERS, dropout=DROPOUT),
        )
        n_params = sum(p.numel() for p in mpnn.parameters())
        print(f"[chemprop:{name}] MPNN params: {n_params:,}", flush=True)
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
        trainer = L.Trainer(
            max_epochs=MAX_EPOCHS,
            callbacks=[es],
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
        )
        print(f"[chemprop:{name}] fitting... wall budget {WALL_BUDGET_S/60:.0f} min", flush=True)
        sys.stderr.write(f"[chemprop:{name}:stderr] about to call trainer.fit\n"); sys.stderr.flush()
        fit_start_h = time.time()
        try:
            trainer.fit(mpnn, loader_tr, loader_va)
        except Exception as exc:
            sys.stderr.write(f"[chemprop:{name}:stderr] trainer.fit EXCEPTION: {type(exc).__name__}: {exc}\n")
            sys.stderr.flush()
            traceback.print_exc(file=sys.stderr)
            raise
        fit_elapsed_h = time.time() - fit_start_h
        print(f"[chemprop:{name}] fit done in {fit_elapsed_h/60:.1f} min "
              f"(best epoch {trainer.current_epoch})", flush=True)
        return mpnn, trainer, fit_elapsed_h, n_params

    # --- MAIN head: all corpus rows, pec50 ---
    fit_start = time.time()
    mpnn_main, trainer_main, fit_main_s, n_params_main = fit_one_head(
        "main_pec50", smiles, y_pec50_sc, w, scaffolds, pec50_mean, pec50_std,
    )
    if time.time() - t_start > WALL_BUDGET_S:
        raise TimeoutError(f"Chemprop main head exceeded wall budget {WALL_BUDGET_S/60} min")

    # --- NULL head: filtered to non-NaN rows; trained separately ---
    null_mask = ~np.isnan(y_null)
    n_null = int(null_mask.sum())
    print(f"[chemprop:null] training on {n_null}/{len(corpus)} non-NaN rows", flush=True)
    use_null = n_null >= 500
    mpnn_null = trainer_null = None
    if use_null:
        null_idx = np.where(null_mask)[0]
        smi_n = [smiles[i] for i in null_idx]
        y_n = y_null[null_idx]
        w_n = w[null_idx]
        scaf_n = [scaffolds[i] for i in null_idx]
        null_mean = float(np.mean(y_n))
        null_std = float(np.std(y_n, ddof=1)) or 1.0
        y_n_sc = (y_n - null_mean) / null_std
        mpnn_null, trainer_null, fit_null_s, _ = fit_one_head(
            "null_pec50", smi_n, y_n_sc, w_n, scaf_n, null_mean, null_std,
        )
    else:
        null_mean, null_std, fit_null_s = pec50_mean, pec50_std, 0.0

    fit_elapsed = time.time() - fit_start
    print(f"[chemprop] both heads done in {fit_elapsed/60:.1f} min", flush=True)
    if time.time() - t_start > WALL_BUDGET_S:
        raise TimeoutError(f"Chemprop full fit exceeded wall budget {WALL_BUDGET_S/60} min")

    def predict_with(model, trainer_h, smi_list, mean, std):
        dpts = [cdata.MoleculeDatapoint.from_smi(s) for s in smi_list]
        ds = cdata.MoleculeDataset(dpts)
        loader = cdata.build_dataloader(ds, batch_size=128, shuffle=False, num_workers=0)
        raw = trainer_h.predict(model, loader)
        p_sc = torch.cat(raw).numpy().reshape(-1)
        return p_sc * std + mean

    te_main = predict_with(mpnn_main, trainer_main, te_df["std_smiles"].tolist(), pec50_mean, pec50_std)
    ph_main = predict_with(mpnn_main, trainer_main, ph_df["std_smiles"].tolist(), pec50_mean, pec50_std)
    if use_null:
        te_null = predict_with(mpnn_null, trainer_null, te_df["std_smiles"].tolist(), null_mean, null_std)
        ph_null = predict_with(mpnn_null, trainer_null, ph_df["std_smiles"].tolist(), null_mean, null_std)
        # Aux ensemble: 70/30 (main dominant; null head is the small biological gain)
        te_preds = 0.70 * te_main + 0.30 * te_null
        ph_preds = 0.70 * ph_main + 0.30 * ph_null
    else:
        te_preds, ph_preds = te_main, ph_main

    # Clip to training-pec50 range +/- 0.5
    lo = float(np.nanmin(y_pec50)) - 0.5
    hi = float(np.nanmax(y_pec50)) + 0.5
    te_preds = np.clip(te_preds, lo, hi)
    ph_preds = np.clip(ph_preds, lo, hi)

    return te_preds, ph_preds, {
        "fit_minutes": fit_elapsed / 60,
        "fit_main_minutes": fit_main_s / 60,
        "fit_null_minutes": fit_null_s / 60,
        "n_params_main": n_params_main,
        "best_epoch_main": trainer_main.current_epoch,
        "best_epoch_null": (trainer_null.current_epoch if use_null else None),
        "null_head_used": bool(use_null),
        "null_head_n_rows": int(n_null),
    }


# ---------------------------------------------------------------------------
# LGBM fallback
# ---------------------------------------------------------------------------

def fit_lgbm_fallback(corpus, te_df, ph_df):
    """Weighted LGBM(combined) on the augmented corpus."""
    from lightgbm import LGBMRegressor
    from pxr.featurize import combined, impute

    print("[lgbm] featurizing (combined Morgan + RDKit)...", flush=True)
    X_tr = impute(combined(corpus["std_smiles"].tolist()))
    X_te = impute(combined(te_df["std_smiles"].tolist()))
    X_ph = impute(combined(ph_df["std_smiles"].tolist()))
    y_tr = corpus["pec50"].to_numpy(dtype=np.float32)
    w_tr = corpus["weight"].to_numpy(dtype=np.float32)

    model = LGBMRegressor(
        n_estimators=500, num_leaves=64, learning_rate=0.05,
        n_jobs=-1, random_state=SEED, verbose=-1,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    te_preds = model.predict(X_te)
    ph_preds = model.predict(X_ph)
    # Same training-range clip as chemprop
    lo = float(np.min(y_tr)) - 0.5
    hi = float(np.max(y_tr)) + 0.5
    te_preds = np.clip(te_preds, lo, hi)
    ph_preds = np.clip(ph_preds, lo, hi)
    return te_preds, ph_preds


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 70, flush=True)
    print("nb950 -- chemprop_aux v2: retrain with augmented corpus", flush=True)
    print("=" * 70, flush=True)

    corpus, te_df, ph_df, drop_log = build_corpus()

    summary = {
        "script": "nb950_chemprop_aux_v2",
        "tier_weights": TIER_W,
        "cad_cv_penalty": CAD_CV_PENALTY,
        "corpus_total": int(len(corpus)),
        "corpus_by_source": corpus["source"].value_counts().to_dict(),
        "n_test_513": int(len(te_df)),
        "n_phase1_unblinded": int(len(ph_df)),
        "null_head_n_labels": int(corpus["pec50_null"].notna().sum()),
        "leaked_sp_dropped": 1,
        "dirty_rows_dropped": drop_log,
        "fallback_used": False,
        "model": "chemprop_aux_v2",
    }

    fallback = False
    try:
        te_preds, ph_preds, fit_info = fit_chemprop(corpus, te_df, ph_df, t_start)
        summary.update(fit_info)
    except Exception as exc:
        print(f"[chemprop] FAILED ({type(exc).__name__}): {exc}", flush=True)
        traceback.print_exc()
        print("[fallback] running LGBM(combined) on the same weighted corpus", flush=True)
        fallback = True
        te_preds, ph_preds = fit_lgbm_fallback(corpus, te_df, ph_df)
        summary["fallback_used"] = True
        summary["model"] = "nb950b_lgbm_v2"

    # Save predictions
    if fallback:
        te_out_name = "te_nb950b_lgbm_v2.npy"
        sub_out_name = "nb950b_lgbm_v2.csv"
    else:
        te_out_name = "te_chemprop_aux_v2.npy"
        sub_out_name = "nb950_chemprop_aux_v2.csv"

    np.save(DATA_PROCESSED / te_out_name, te_preds.astype(np.float32))
    print(f"[save] {te_out_name}: shape={te_preds.shape} mean={te_preds.mean():.3f} "
          f"std={te_preds.std():.3f}", flush=True)

    # Submission CSV (need SMILES + Molecule Name + pEC50, ordered to match test-513)
    raw_test = load_test()
    raw_test["std_smiles"] = raw_test["smiles"].apply(standardize_smiles)
    pred_map = dict(zip(te_df["std_smiles"].tolist(), te_preds.tolist()))
    raw_test["pEC50"] = raw_test["std_smiles"].map(pred_map)
    # If any std_smiles missed (shouldn't happen), use training mean
    if raw_test["pEC50"].isna().any():
        fill = float(corpus["pec50"].mean())
        n_miss = int(raw_test["pEC50"].isna().sum())
        print(f"[save] WARNING: {n_miss} test rows had no prediction; "
              f"filling with training mean {fill:.3f}", flush=True)
        raw_test["pEC50"] = raw_test["pEC50"].fillna(fill)
    sub = raw_test.rename(columns={"smiles": "SMILES", "name": "Molecule Name"})
    sub = sub[["SMILES", "Molecule Name", "pEC50"]]
    sub.to_csv(SUBMISSIONS / sub_out_name, index=False)
    print(f"[save] submission: {SUBMISSIONS / sub_out_name}  ({len(sub)} rows)", flush=True)

    # Honest validation on phase1_unblinded
    y_ph_true = ph_df["pec50"].to_numpy(dtype=np.float32)
    ph_rae = float(rae(y_ph_true, ph_preds))
    summary["phase1_unblinded_RAE"] = ph_rae
    summary["phase1_unblinded_n"] = int(len(ph_df))
    summary["phase1_unblinded_pred_mean"] = float(ph_preds.mean())
    summary["phase1_unblinded_pred_std"] = float(ph_preds.std())
    summary["phase1_unblinded_true_mean"] = float(y_ph_true.mean())
    summary["phase1_unblinded_true_std"] = float(y_ph_true.std())

    print("=" * 70, flush=True)
    print(f"phase1_unblinded RAE: {ph_rae:.4f} "
          f"(target <= 0.6216 to beat original chemprop_aux)", flush=True)
    print(f"  improvement vs 0.6216: {0.6216 - ph_rae:+.4f}", flush=True)
    print("=" * 70, flush=True)

    summary["wall_time_min"] = (time.time() - t_start) / 60
    summary_path = DATA_PROCESSED / "nb950_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[save] {summary_path}", flush=True)
    print(f"[done] wall time {summary['wall_time_min']:.1f} min "
          f"(fallback={fallback})", flush=True)
    return summary


if __name__ == "__main__":
    main()
