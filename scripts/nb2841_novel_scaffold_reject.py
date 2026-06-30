"""nb2841 -- Hard-reject novel-scaffold test rows by returning train mean.

NEW PARADIGM (vs cycle-134 paradigm-exhaustion thesis):
    Per the rare-scaffold quantile-compression failure mode (memo entry
    `feedback_failure_mode_quantile_compression`): 50 worst nb503 errors are
    90% novel scaffolds (scaf_train_freq == 0). All predictors agree wrong on
    these rows -- the model has no inductive support for them, so any
    LGBM/anchor prediction is variance-compressed noise. The 1-parameter
    rank-stretch (nb562) extracts the last drops of decompression but cannot
    repair rows that have zero train support.

    ABSTENTION via the mean predictor: if the model has no signal, the
    Bayes-optimal estimate is the prior mean. RAE on novel-scaffold rows
    bounded above at ~1.0 (mean predictor) is better than the variance-
    compressed LGBM trying to predict the tail. Trade-off: we lose any
    correctness on the rows where the model would happen to be right
    (regression toward mean over-shrinks the few high-truth novel-scaffold
    rows).

PROTOCOL:
    1. Compute Bemis-Murcko scaffold frequency over the 4139 TRAIN set
       (NOT 253 unblind -- we want true novel-to-train).
    2. For each of 513 test rows:
        if scaf_train_freq == 0 (or scaffold is None / ringless):
            pred = train_mean
        else:
            pred = nb2240[i]  (te_nb2240.npy on 513; nb2240_pred_oof on 253)
    3. 5-fold scaffold CV on 253 unblind with kf_seed = 1001.
       Per fold: identify novel-scaf valid rows; gated_pred[va] =
       train_mean on novel rows, nb2240_oof[va] on rest.
       Pool across folds and compute RAE.
    4. Repeat across kf_seeds {1001..1005} for 5-seed mean.

GATE:
    mean_rae < 0.4570  -> "PROMOTE"
    mean_rae < 0.4598  -> "MARGINAL_BEAT"
    else                -> "FAIL"

OUTPUTS:
    scripts/nb2841_novel_scaffold_reject.py
    data/processed/nb2841_summary.json
    data/processed/nb2841_pred_oof.npy   (253,) float32 gated mean-bag
    data/processed/te_nb2841.npy         (513,) float32 deploy
    submissions/nb2841_novel_scaffold_reject.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import Counter

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2841"

# nb2240 inputs (PRIMARY-1 candidate, cycle-172 5-anchor pyramid w/ K=20 swap)
NB2240_OOF_PATH = DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy"  # (253,) honest K=20 mean-bag
NB2240_TE_PATH = DATA_PROCESSED / "te_nb2240.npy"                 # (513,) deploy pyramid

N_FOLDS = 5
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]

# Gate thresholds
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Cross-reference
NB2240_PYRAMID_REF = 0.4598  # pooled_rae_mean_seeds from nb2240_summary.json
NB2240_K20_OOF_REF = 0.4630  # mean-bag K=20 OOF RAE (the OOF we actually use as substrate)


def compute_train_scaffold_freq() -> dict[str, int]:
    """Bemis-Murcko scaffold frequency over the 4139 TRAIN compounds."""
    tr = load_train()
    print(f"[train] n={len(tr)}")
    scafs = [bemis_murcko(s) for s in tr["smiles"].values]
    freq: Counter[str] = Counter(s for s in scafs if s)
    n_none = sum(1 for s in scafs if not s)
    print(
        f"[train_scaf] unique={len(freq)}  none_scaffold_rows={n_none}  "
        f"total_with_scaffold={sum(freq.values())}"
    )
    return dict(freq), float(tr["pec50"].mean())


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Hard-reject novel-scaffold test rows by returning train mean")
    print(f"        scaf_train_freq == 0 -> pred = train_mean; else -> nb2240")
    print(f"        scaffold-CV {N_FOLDS}-fold  kf_seeds={KF_SEEDS}")
    print(f"        GATE: <{GATE_PROMOTE} PROMOTE / "
          f"<{GATE_MARGINAL} MARGINAL_BEAT / else FAIL")
    print("=" * 78)

    # ---- Load test set + truth ----
    te = load_test()
    n_te = len(te)
    te_names = te["name"].values if "name" in te.columns else te["Molecule Name"].values
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_scaffolds = [bemis_murcko(s) for s in te_smiles]
    print(f"[test] n={n_te}")

    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[unblind] n={n_unb}")

    # ---- Scaffold frequency from 4139 train + train_mean ----
    train_scaf_freq, train_mean = compute_train_scaffold_freq()
    print(f"[train_mean] pEC50 = {train_mean:.4f}")

    # ---- Decide novel vs seen for each of 513 ----
    novel_mask_te = np.zeros(n_te, dtype=bool)
    none_scaf_mask_te = np.zeros(n_te, dtype=bool)
    for i, sc in enumerate(te_scaffolds):
        if sc is None or sc == "":
            none_scaf_mask_te[i] = True
            novel_mask_te[i] = True
        elif train_scaf_freq.get(sc, 0) == 0:
            novel_mask_te[i] = True

    n_novel_te = int(novel_mask_te.sum())
    n_seen_te = int((~novel_mask_te).sum())
    n_none_te = int(none_scaf_mask_te.sum())
    print(
        f"[gate_te] novel={n_novel_te} ({n_novel_te/n_te:.1%})  "
        f"seen={n_seen_te} ({n_seen_te/n_te:.1%})  "
        f"none_scaffold={n_none_te}"
    )

    novel_mask_unb = novel_mask_te[unb_idx]
    n_novel_unb = int(novel_mask_unb.sum())
    n_seen_unb = int((~novel_mask_unb).sum())
    print(
        f"[gate_unb] novel={n_novel_unb} ({n_novel_unb/n_unb:.1%})  "
        f"seen={n_seen_unb} ({n_seen_unb/n_unb:.1%})"
    )

    # ---- Load nb2240 OOF (253) + te (513) ----
    nb2240_oof = np.load(NB2240_OOF_PATH).astype(np.float64)
    nb2240_te = np.load(NB2240_TE_PATH).astype(np.float64)
    assert nb2240_oof.shape == (n_unb,), nb2240_oof.shape
    assert nb2240_te.shape == (n_te,), nb2240_te.shape
    nb2240_oof_rae = float(rae(y_unb, nb2240_oof))
    nb2240_te_unb_rae = float(rae(y_unb, nb2240_te[unb_idx]))
    print(
        f"[load] nb2240_oof RAE (K=20 mean-bag) = {nb2240_oof_rae:.4f}  "
        f"(ref {NB2240_K20_OOF_REF:.4f})"
    )
    print(
        f"[load] nb2240_te[unb] RAE (deploy)    = {nb2240_te_unb_rae:.4f}  "
        f"(in-sample optimistic)"
    )

    # ---- Mean-only baseline RAE ----
    mean_only = np.full(n_unb, train_mean, dtype=np.float64)
    mean_only_rae = float(rae(y_unb, mean_only))
    print(f"[load] mean-only baseline (all 253)  = {mean_only_rae:.4f}")

    # ---- Per-partition RAE breakdown on 253 ----
    per_part = {}
    if n_novel_unb > 0:
        y_novel = y_unb[novel_mask_unb]
        r2240_novel = float(rae(y_novel, nb2240_oof[novel_mask_unb]))
        rmean_novel = float(rae(y_novel, np.full(n_novel_unb, train_mean)))
        per_part["novel"] = {
            "n": n_novel_unb,
            "nb2240_oof_rae": r2240_novel,
            "train_mean_rae": rmean_novel,
            "gated_choice": "train_mean",
            "y_mean": float(y_novel.mean()),
            "y_std": float(y_novel.std()),
        }
        print(
            f"\n[partition_novel] n={n_novel_unb}  "
            f"nb2240_RAE={r2240_novel:.4f}  train_mean_RAE={rmean_novel:.4f}  "
            f"(gate picks train_mean)"
        )
    if n_seen_unb > 0:
        y_seen = y_unb[~novel_mask_unb]
        r2240_seen = float(rae(y_seen, nb2240_oof[~novel_mask_unb]))
        rmean_seen = float(rae(y_seen, np.full(n_seen_unb, train_mean)))
        per_part["seen"] = {
            "n": n_seen_unb,
            "nb2240_oof_rae": r2240_seen,
            "train_mean_rae": rmean_seen,
            "gated_choice": "nb2240",
            "y_mean": float(y_seen.mean()),
            "y_std": float(y_seen.std()),
        }
        print(
            f"[partition_seen]  n={n_seen_unb}  "
            f"nb2240_RAE={r2240_seen:.4f}  train_mean_RAE={rmean_seen:.4f}  "
            f"(gate picks nb2240)"
        )

    # ---- 5-fold scaffold CV: gated OOF on 253 ----
    # On each fold: validation rows split into novel (use train_mean) and
    # seen (use nb2240_oof[i]). The OOF substrate (nb2240_oof) is already a
    # honest cross-fit from the K=20 LGBM mean-bag (nb2240 stage 1). We
    # don't refit per fold -- we *gate* the existing OOF.
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"\n[scaffold] unique unb scaffolds = {n_unique_scaf}")

    print("\n" + "-" * 78)
    print(f"5-SEED SCAFFOLD-CV  seeds={KF_SEEDS}  folds={N_FOLDS}")
    print("-" * 78)
    per_seed_rae: list[float] = []
    per_seed_diag: list[dict] = []
    per_seed_oofs: list[np.ndarray] = []
    for seed in KF_SEEDS:
        ts = time.time()
        splits = scaffold_kfold_indices(
            unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=seed,
        )
        oof = np.full(n_unb, np.nan, dtype=np.float64)
        fold_diags = []
        for fold_i, (tr_loc, va_loc) in enumerate(splits):
            # Gate the validation predictions:
            #   novel-scaffold rows -> train_mean
            #   seen-scaffold rows  -> nb2240_oof (substrate)
            va_novel = novel_mask_unb[va_loc]
            n_va_novel = int(va_novel.sum())
            n_va_seen = int((~va_novel).sum())
            gated_va = np.where(
                va_novel,
                np.full(len(va_loc), train_mean, dtype=np.float64),
                nb2240_oof[va_loc],
            )
            oof[va_loc] = gated_va
            fold_diags.append({
                "fold": int(fold_i),
                "n_va": int(len(va_loc)),
                "n_va_novel": n_va_novel,
                "n_va_seen": n_va_seen,
            })
        if np.isnan(oof).any():
            raise RuntimeError("scaffold split did not cover all rows")
        seed_rae = float(rae(y_unb, oof))
        per_seed_rae.append(seed_rae)
        per_seed_diag.append({
            "kf_seed": int(seed),
            "pooled_rae": seed_rae,
            "fold_diags": fold_diags,
        })
        per_seed_oofs.append(oof)
        n_novel_total = sum(d["n_va_novel"] for d in fold_diags)
        n_seen_total = sum(d["n_va_seen"] for d in fold_diags)
        print(f"   seed={seed}  pooled_RAE={seed_rae:.4f}  "
              f"novel/seen={n_novel_total}/{n_seen_total}  "
              f"wall={time.time()-ts:.1f}s")

    per_seed_mean = float(np.mean(per_seed_rae))
    per_seed_std = float(np.std(per_seed_rae))
    # All 5 seeds produce identical OOF because the gate is deterministic
    # in the test row (not in the fold) -- mean-of-seed OOF == single-seed OOF.
    mean_bag_oof = np.mean(np.column_stack(per_seed_oofs), axis=1)
    rae_mean_bag = float(rae(y_unb, mean_bag_oof))

    print("\n[cv] per_seed_mean RAE = "
          f"{per_seed_mean:.4f}  std={per_seed_std:.4f}")
    print(f"[cv] mean_bag RAE      = {rae_mean_bag:.4f}")
    print(f"[cv] nb2240 OOF (ref)  = {nb2240_oof_rae:.4f}  "
          f"(d = {rae_mean_bag - nb2240_oof_rae:+.4f})")
    print(f"[cv] nb2240 pyramid    = {NB2240_PYRAMID_REF:.4f}  "
          f"(d = {rae_mean_bag - NB2240_PYRAMID_REF:+.4f})")

    # ---- Deploy te (513): apply same gate to deploy substrate ----
    deploy_te = np.where(
        novel_mask_te,
        np.full(n_te, train_mean, dtype=np.float64),
        nb2240_te,
    ).astype(np.float32)
    te_unb_in_sample_rae = float(rae(y_unb, deploy_te[unb_idx]))
    print(f"\n[deploy] te(513) mean/std = "
          f"{deploy_te.mean():.3f}/{deploy_te.std():.3f}")
    print(f"[deploy] te[unb_idx] in-sample RAE = {te_unb_in_sample_rae:.4f}")

    # ---- Save artefacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, mean_bag_oof.astype(np.float32))
    np.save(te_path, deploy_te)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_novel_scaffold_reject.csv"
    pd.DataFrame({
        "SMILES": te_smiles,
        "Molecule Name": te_names,
        "pEC50": deploy_te.astype(np.float32),
    }).to_csv(sub_csv, index=False)
    print(f"[save] {sub_csv}")

    # ---- Gate ----
    if rae_mean_bag < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif rae_mean_bag < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print("\n" + "=" * 78)
    print("GATE EVALUATION")
    print("=" * 78)
    print(f"   mean_bag_rae        = {rae_mean_bag:.4f}")
    print(f"   < {GATE_PROMOTE:.4f}  (PROMOTE)       = "
          f"{rae_mean_bag < GATE_PROMOTE}")
    print(f"   < {GATE_MARGINAL:.4f}  (MARGINAL_BEAT) = "
          f"{rae_mean_bag < GATE_MARGINAL}")
    print(f"   VERDICT             = {verdict}")

    summary = {
        "tag": TAG,
        "method": "novel_scaffold_hard_reject_train_mean_else_nb2240",
        "rationale": (
            "Hard reject novel-scaffold test rows by substituting the train "
            "pEC50 mean (abstention via Bayes-optimal prior). For test rows "
            "with scaf_train_freq == 0 (or ringless scaffold), pred = "
            "train_mean. For all other rows, pred = nb2240. Targets the "
            "rare-scaffold quantile-compression failure mode: 90% of worst "
            "errors are novel scaffolds where all predictors agree wrong; "
            "mean predictor caps RAE at ~1.0 on those rows, dodging the "
            "variance-compressed tail penalty."
        ),
        "rule": "scaf_train_freq == 0 (or None) -> train_mean; else -> nb2240",
        "scaf_source": "Bemis-Murcko over 4139 TRAIN compounds (not 253 unblind)",
        "nb2240_oof_path": str(NB2240_OOF_PATH),
        "nb2240_te_path": str(NB2240_TE_PATH),
        "train_mean": train_mean,
        "n_te": n_te,
        "n_unb": n_unb,
        "n_novel_te": n_novel_te,
        "n_seen_te": n_seen_te,
        "n_none_scaffold_te": n_none_te,
        "n_novel_unb": n_novel_unb,
        "n_seen_unb": n_seen_unb,
        "frac_novel_te": float(n_novel_te / n_te),
        "frac_seen_te": float(n_seen_te / n_te),
        "frac_novel_unb": float(n_novel_unb / n_unb),
        "frac_seen_unb": float(n_seen_unb / n_unb),
        "n_unique_scaffolds_unb": n_unique_scaf,
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "nb2240_oof_rae": nb2240_oof_rae,
        "nb2240_te_unb_rae_in_sample": nb2240_te_unb_rae,
        "mean_only_rae": mean_only_rae,
        "per_partition": per_part,
        "per_seed_rae": [float(r) for r in per_seed_rae],
        "per_seed_mean_rae": per_seed_mean,
        "per_seed_std_rae": per_seed_std,
        "mean_bag_rae": rae_mean_bag,
        "mean_rae": rae_mean_bag,  # alias for gate consumers
        "per_seed_diag": per_seed_diag,
        "delta_vs_nb2240_oof": rae_mean_bag - nb2240_oof_rae,
        "delta_vs_nb2240_pyramid": rae_mean_bag - NB2240_PYRAMID_REF,
        "nb2240_pyramid_ref": NB2240_PYRAMID_REF,
        "nb2240_K20_oof_ref": NB2240_K20_OOF_REF,
        "te_unb_in_sample_rae": te_unb_in_sample_rae,
        "te_deploy_mean": float(deploy_te.mean()),
        "te_deploy_std": float(deploy_te.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   mean_bag RAE        = {rae_mean_bag:.4f}  ({verdict})")
    print(f"   per_seed mean       = {per_seed_mean:.4f} +/- {per_seed_std:.4f}")
    print(f"   delta vs nb2240_oof = {rae_mean_bag - nb2240_oof_rae:+.4f}")
    print(f"   delta vs pyramid    = {rae_mean_bag - NB2240_PYRAMID_REF:+.4f}")
    print(f"   wall                = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_novel_te",
        "n_seen_te",
        "n_novel_unb",
        "n_seen_unb",
        "train_mean",
        "nb2240_oof_rae",
        "mean_bag_rae",
        "per_seed_mean_rae",
        "per_seed_std_rae",
        "verdict",
        "delta_vs_nb2240_oof",
        "te_unb_in_sample_rae",
        "submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
