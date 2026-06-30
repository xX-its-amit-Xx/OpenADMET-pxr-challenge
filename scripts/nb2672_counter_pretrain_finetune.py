"""nb2672 -- Counter-assay pretrain -> PXR fine-tune via LGBM warm-start.

NEW PARADIGM:
    Cycle 134/136 paradigm exhaustion (no aux signal beats nb2103 on the
    chemprop_aux residual) prompted this transfer-learning attempt.  Instead
    of using counter pec50 as an auxiliary FEATURE (nb1202) or as a separate
    HEAD in a multitask model (chemprop_aux already does that), here we use
    it as a PRETRAINING TASK for the residual LGBM:

    Stage A (PRETRAIN): LGBM(max_depth=4, n_est=300, lr=0.03) on counter-
        assay data (2858 rows, more data than the 253 unblind), target =
        counter_pec50.  The booster learns counter-axis structure.
    Stage B (FINETUNE): warm-start the booster (lgb.train init_model=...)
        on PXR pec50 data (X_unb + 253 unblind anchor residual context).
        Additional 100 rounds at lr=0.01 (10x slower).

    Hypothesis: the counter axis pre-baked into stage A's trees provides
    a useful low-level inductive bias (compound-promiscuity priors) that
    survives the fine-tune step and reduces residual RAE vs cold-start.

PROTOCOL:
    1.  Load counter (2858) and unblind (253) data with truth.
    2.  Build combined features (Morgan-2048 + RDKit-217 = 2265 dims)
        for counter, train (4139), test (513) -- the only feature space
        we can compute uniformly across these three sets locally.
        (The 117-col matrix requires avalon/chemprop_embed for non-test
        rows, which aren't cached -- combined() is the X equivalent.)
    3.  Compute the residual on the 253 unblind:
            residual = y_unb - chemprop_aux_te[unb_idx]
    4.  For each kf_seed in {1001, 1002, 1003, 1004, 1005}:
          5-fold scaffold split on the 253 unblind.
          Per fold (tr_loc, va_loc):
            Stage A: LGBM(MSE, max_depth=4, num_leaves=15, n_est=300,
                lr=0.03) trained on counter rows  (X_co, y_co)
                -> save booster
            Stage B: lgb.train warm-start with init_model = stage A booster,
                fit on X_unb[tr_loc] residual[tr_loc], add 100 rounds
                at lr=0.01.
            -> predict residual on X_unb[va_loc]
          pred_oof[kf] = anchor + resid_oof  (n=253)
          per_kf_rae[kf] = rae(y_unb, pred_oof[kf])
    5.  Deploy te:
          Refit stage A on counter, stage B on FULL X_unb residual, predict
          on X_te (513) -> add anchor_te.
          One model per kf_seed; average over 5 kf_seeds (the te output is
          the same model class with the same data, so per-kf variation is
          tiny -- this just keeps the artifact symmetric with the OOF).
    6.  Mean / std over 5 kf_seeds.
    7.  GATE:
          mean_rae < 0.4570 -> "PROMOTE"
          mean_rae < 0.4598 -> "MARGINAL_BEAT"
          else              -> "FAIL"

Outputs:
    data/processed/nb2672_summary.json
    data/processed/nb2672_pred_oof.npy   (253,) float32
    data/processed/te_nb2672.npy         (513,) float32
    submissions/nb2672_counter_pretrain_finetune.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize, bemis_murcko
from pxr.data import load_test, load_counter
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2672"

# -- Protocol -------------------------------------------------------------
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
RESID_FOLDS = 5

# Stage A: pretrain on counter
STAGE_A_PARAMS = dict(
    objective="regression",
    max_depth=4,
    num_leaves=15,
    learning_rate=0.03,
    min_child_samples=5,
    reg_lambda=2.0,
    verbosity=-1,
)
STAGE_A_N_EST = 300

# Stage B: fine-tune on PXR residual, slower lr, fewer rounds
STAGE_B_PARAMS = dict(
    objective="regression",
    max_depth=4,
    num_leaves=15,
    learning_rate=0.01,
    min_child_samples=5,
    reg_lambda=2.0,
    verbosity=-1,
)
STAGE_B_N_EST = 100

# -- Reference / Gates ----------------------------------------------------
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216

GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"


def _build_counter_set():
    """Load counter, return (smiles_std, y_counter, scaffolds)."""
    co = load_counter()
    co = co[co["smiles"].notna() & co["pec50"].notna()].copy()
    smis = co["smiles"].astype(str).tolist()
    std = []
    for s in smis:
        m = standardize(s)
        std.append(Chem.MolToSmiles(m) if m is not None else None)
    keep_mask = np.array([s is not None and len(s) > 0 for s in std], dtype=bool)
    co_kept = co[keep_mask].reset_index(drop=True).copy()
    co_kept["std_smi"] = [std[i] for i in range(len(std)) if keep_mask[i]]
    # collapse to unique std_smi via median (handles dup batches in counter)
    g = co_kept.groupby("std_smi", as_index=False).agg(pec50=("pec50", "median"))
    smiles_std = g["std_smi"].tolist()
    y_counter = g["pec50"].to_numpy(dtype=np.float64)
    return smiles_std, y_counter


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Counter-assay PRETRAIN -> PXR FINETUNE (LGBM warm-start)")
    print(f"          Stage A: counter, n_est={STAGE_A_N_EST}, lr=0.03")
    print(f"          Stage B: PXR residual, +{STAGE_B_N_EST} rounds, lr=0.01")
    print(f"          {RESID_FOLDS}-fold scaffold CV on 253 unblind, "
          f"{len(KF_SEEDS)} kf_seeds")
    print(f"          GATE PROMOTE < {GATE_PROMOTE}   MARGINAL < {GATE_MARGINAL}")
    print("=" * 78)

    # ---- Load truth / anchor ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor))
    print(f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
          f"(ref {CHEMPROP_AUX_REF:.4f})")
    residual = y_unb - anchor
    print(f"[load] residual mean={residual.mean():+.3f}  std={residual.std():.3f}")

    # ---- Build counter (pretrain) features ----
    print("\n" + "-" * 78)
    print("STEP 1: counter-assay pretrain features  (combined: Morgan+RDKit)")
    print("-" * 78)
    co_smiles, y_counter = _build_counter_set()
    print(f"   n_counter (unique std_smi) = {len(co_smiles)}")
    X_counter = impute(combined(co_smiles)).astype(np.float32)
    print(f"   X_counter shape = {X_counter.shape}  "
          f"y_counter mean={y_counter.mean():.2f} std={y_counter.std():.2f}")

    # ---- Build unblind + test features ----
    print("\n" + "-" * 78)
    print("STEP 2: unblind (253) + test (513) features")
    print("-" * 78)
    unb_smiles_raw = [te_smiles[i] for i in unb_idx]
    unb_std = []
    for s in unb_smiles_raw:
        m = standardize(s)
        unb_std.append(Chem.MolToSmiles(m) if m is not None else s)
    X_unb = impute(combined(unb_std)).astype(np.float32)
    print(f"   X_unb shape = {X_unb.shape}")

    te_std = []
    for s in te_smiles:
        m = standardize(s)
        te_std.append(Chem.MolToSmiles(m) if m is not None else s)
    X_te = impute(combined(te_std)).astype(np.float32)
    print(f"   X_te shape  = {X_te.shape}")

    # ---- Scaffolds for 5-fold split on the 253 unblind ----
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles_raw]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   unique unblind scaffolds = {n_unique_scaf}")

    # Sanity: feature width must match
    assert X_counter.shape[1] == X_unb.shape[1] == X_te.shape[1], (
        f"feat-width mismatch counter/unb/te = "
        f"{X_counter.shape[1]}/{X_unb.shape[1]}/{X_te.shape[1]}"
    )
    feat_dim = int(X_counter.shape[1])
    print(f"   feat_dim (X) = {feat_dim}")

    # ---- 5 kf_seeds x 5-fold pretrain-finetune cross-fit ----
    print("\n" + "-" * 78)
    print(f"STEP 3: pretrain-finetune residual cross-fit  "
          f"{len(KF_SEEDS)} kf_seeds x {RESID_FOLDS} folds = "
          f"{len(KF_SEEDS)*RESID_FOLDS} fits")
    print("-" * 78)

    pred_oof_per_kf = np.zeros((len(KF_SEEDS), n_unb), dtype=np.float64)
    per_kf_rae = []

    with TemporaryDirectory(prefix="nb2672_warmstart_") as _tmpdir:
        tmpdir = Path(_tmpdir)
        # Train the Stage-A booster ONCE per kf_seed (counter target doesn't
        # depend on which unblind rows are held out).  We retrain per kf_seed
        # so the warm-start RNG differs in stage B (LGBM seed differs).
        for k_i, kf_seed in enumerate(KF_SEEDS):
            ts_kf = time.time()
            print(f"\n  --- kf_seed={kf_seed} -----------------------------")

            # Stage A: pretrain on counter
            paramsA = dict(STAGE_A_PARAMS, seed=kf_seed)
            dsA = lgb.Dataset(X_counter, label=y_counter)
            bstA = lgb.train(paramsA, dsA, num_boost_round=STAGE_A_N_EST)
            pathA = tmpdir / f"stage_A_kf{kf_seed}.txt"
            bstA.save_model(str(pathA))
            # quick in-sample sanity on counter
            r_self = float(rae(y_counter, bstA.predict(X_counter)))
            print(f"      stage A pretrained on counter (n={len(y_counter)})  "
                  f"in-sample counter RAE = {r_self:.4f}")

            # 5-fold scaffold CV on unblind
            splits = scaffold_kfold_indices(
                unb_scaffolds, n_splits=RESID_FOLDS, shuffle=True, seed=kf_seed,
            )
            oof_resid = np.full(n_unb, np.nan, dtype=np.float64)
            for fi, (tr_loc, va_loc) in enumerate(splits):
                paramsB = dict(STAGE_B_PARAMS, seed=kf_seed)
                dsB = lgb.Dataset(X_unb[tr_loc], label=residual[tr_loc])
                bstB = lgb.train(
                    paramsB, dsB,
                    num_boost_round=STAGE_B_N_EST,
                    init_model=str(pathA),
                )
                oof_resid[va_loc] = bstB.predict(X_unb[va_loc])
            assert not np.any(np.isnan(oof_resid)), "fold did not cover all rows"
            pred_oof_per_kf[k_i] = anchor + oof_resid
            kf_rae = float(rae(y_unb, pred_oof_per_kf[k_i]))
            per_kf_rae.append(kf_rae)
            print(f"      kf={kf_seed}  scaffold-CV pred_oof RAE = "
                  f"{kf_rae:.4f}  wall={time.time()-ts_kf:.1f}s")

        # ---- Deploy: per-kf full-fit te, then average ----
        print("\n" + "-" * 78)
        print("STEP 4: deploy refit (per-kf stage A + stage B on full 253)")
        print("-" * 78)
        pred_te_per_kf = np.zeros((len(KF_SEEDS), n_test), dtype=np.float64)
        for k_i, kf_seed in enumerate(KF_SEEDS):
            ts_d = time.time()
            paramsA = dict(STAGE_A_PARAMS, seed=kf_seed)
            dsA = lgb.Dataset(X_counter, label=y_counter)
            bstA = lgb.train(paramsA, dsA, num_boost_round=STAGE_A_N_EST)
            pathA = tmpdir / f"deploy_stage_A_kf{kf_seed}.txt"
            bstA.save_model(str(pathA))

            paramsB = dict(STAGE_B_PARAMS, seed=kf_seed)
            dsB = lgb.Dataset(X_unb, label=residual)
            bstB = lgb.train(
                paramsB, dsB,
                num_boost_round=STAGE_B_N_EST,
                init_model=str(pathA),
            )
            te_resid = bstB.predict(X_te)
            pred_te_per_kf[k_i] = te_anchor_513 + te_resid
            print(f"   deploy kf={kf_seed}  wall={time.time()-ts_d:.1f}s  "
                  f"te_mean={pred_te_per_kf[k_i].mean():.3f}  "
                  f"te_std={pred_te_per_kf[k_i].std():.3f}")

    # ---- Aggregate ----
    print("\n" + "-" * 78)
    print("STEP 5: aggregate")
    print("-" * 78)
    per_kf_arr = np.array(per_kf_rae, dtype=np.float64)
    mean_rae = float(per_kf_arr.mean())
    std_rae = float(per_kf_arr.std(ddof=1))
    min_rae = float(per_kf_arr.min())
    max_rae = float(per_kf_arr.max())
    median_rae = float(np.median(per_kf_arr))

    print(f"   per-kf RAE:")
    for kf, r in zip(KF_SEEDS, per_kf_rae):
        print(f"     kf={kf}  RAE={r:.4f}")
    print(f"   mean +/- std (5 kf)  = {mean_rae:.4f} +/- {std_rae:.5f}")
    print(f"   median               = {median_rae:.4f}")
    print(f"   min / max            = {min_rae:.4f} / {max_rae:.4f}")
    print(f"   delta vs nb2171 (0.4682) = {mean_rae - NB2171_REF:+.4f}")

    # Pooled OOF / pooled te (average across kfs)
    pred_oof_unb = pred_oof_per_kf.mean(axis=0)
    pred_te_513 = pred_te_per_kf.mean(axis=0)
    pooled_oof_rae = float(rae(y_unb, pred_oof_unb))
    te_unb_in_rae = float(rae(y_unb, pred_te_513[unb_idx]))
    print(f"\n   pooled (5kf mean) OOF RAE   = {pooled_oof_rae:.4f}")
    print(f"   te[unb_idx] in-sample RAE   = {te_unb_in_rae:.4f}")
    print(f"   pred_oof_unb std = {pred_oof_unb.std():.3f}  "
          f"(truth_std {y_unb.std():.3f})")

    # ---- GATE ----
    print("\n" + "-" * 78)
    print("STEP 6: GATE")
    print("-" * 78)
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"   mean_rae = {mean_rae:.4f}")
    print(f"     < {GATE_PROMOTE}  -> PROMOTE")
    print(f"     < {GATE_MARGINAL} -> MARGINAL_BEAT")
    print(f"     else            -> FAIL")
    print(f"   -> {verdict}")

    # ---- Save artifacts ----
    print("\n" + "-" * 78)
    print("STEP 7: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_counter_pretrain_finetune.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": pred_te_513.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    summary = {
        "tag": TAG,
        "method": "counter_pretrain_finetune_warmstart",
        "rationale": (
            "Pretrain LGBM on counter-assay (2858 rows), warm-start via "
            "init_model, fine-tune on PXR residual on 253 unblind."
        ),
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "feature_set": "combined_morgan_rdkit",
        "feat_dim": feat_dim,
        "stage_A_params": STAGE_A_PARAMS,
        "stage_A_n_est": STAGE_A_N_EST,
        "stage_B_params": STAGE_B_PARAMS,
        "stage_B_n_est": STAGE_B_N_EST,
        "kf_seeds": KF_SEEDS,
        "resid_folds": RESID_FOLDS,
        "n_counter_unique_smi": int(len(y_counter)),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds_unb": int(n_unique_scaf),
        "per_kf_rae": per_kf_rae,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "median_rae": median_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "pooled_oof_rae_5kf_mean": pooled_oof_rae,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "delta_vs_nb2171": mean_rae - NB2171_REF,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_rae +/- std (5 kf)  = "
          f"{mean_rae:.4f} +/- {std_rae:.5f}")
    print(f"   per-kf:  " +
          ", ".join(f"{r:.4f}" for r in per_kf_rae))
    print(f"   delta vs nb2171 (0.4682) = {mean_rae - NB2171_REF:+.4f}")
    print(f"   pooled OOF RAE           = {pooled_oof_rae:.4f}")
    print(f"   te[unb_idx] RAE          = {te_unb_in_rae:.4f}")
    print(f"   verdict                  = {verdict}")
    print(f"   wall                     = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae",
        "std_rae",
        "per_kf_rae",
        "pooled_oof_rae_5kf_mean",
        "te_unb_in_sample_rae",
        "delta_vs_nb2171",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
