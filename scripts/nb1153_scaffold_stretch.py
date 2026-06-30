"""nb1153 -- Scaffold-CV rank-stretch sweep on nb2112 deploy / nb2103 K=28 OOF.

CONTEXT
-------
Per memory `feedback_rank_stretch_universal`: scalar rank-stretch
`pred' = mu + s * (pred - mu)` with s in [1.05, 1.10] gives -0.003 to -0.005
RAE on every variance-compressed predictor under RANDOM KFold cross-fit
(pred_std ~0.75 vs truth_std ~1.03 on novel-scaffold OOD).  This sweep
verifies whether the same universal s also fires under SCAFFOLD-aware CV.

DATA
----
nb2112 is a 25-fit DEPLOY on all 253 unblind (no honest OOF).  We use the
nb2103 K=28 mean-bag OOF (5-fold KFold cross-fit, 5-seed bag), pooled RAE
0.4737 -- the closest honest 253-row prediction available for the
nb2112/nb2103 path.  Spec quoted target 0.5057; the actual K=28 mean-bag
honest cross-fit is 0.4737 (see nb2103_summary.json).  Both numbers are
reported below.

PROTOCOL
--------
1. Load nb2103 K=28 mean-bag OOF on the 253 unblind rows.
   Truth = pxr-challenge_TEST_PHASE_1_UNBLINDED.csv aligned by Molecule Name.
2. Compute Murcko scaffolds for the 253 unblind SMILES.
3. scaffold_kfold_indices(scaffolds, n_splits=5, seed=42).
4. For each s in {0.95, 1.00, 1.05, 1.10, 1.15}:
   for each fold:
       mu_train     = mean(pred[fold_train])
       stretched_va = mu_train + s * (pred[fold_val] - mu_train)
   concatenate into 253-row OOS, compute pooled scaffold-CV RAE.
5. Baseline = s=1.00 (which by construction recovers ~0.4737).
6. Gate: best_s wins only if RAE(s*) < RAE(s=1.00) - 0.003.
7. If best beats 0.5027 (spec ceiling 0.5057 - 0.003 gate): build deploy CSV
   te = mu_513 + s * (te_nb2112 - mu_513) with mu_513 = mean(nb2103 OOF)
   (per nb562 convention: mu fit on the 253 training set).

OUTPUTS
-------
scripts/nb1153_scaffold_stretch.py
data/processed/nb1153_summary.json
submissions/nb1153_scaffold_stretch_s{S:.2f}.csv  (only if deploy gate passes)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

from pxr.chem import standardize
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, DATA_RAW, SUBMISSIONS

TAG = "nb1153"
NB2103_K28_OOF_PATH = DATA_PROCESSED / "nb2103_mean_bag_oof_K28.npy"
NB2112_TE_PATH = DATA_PROCESSED / "te_nb2112.npy"
TEST_BLINDED_CSV = DATA_RAW / "pxr-challenge_TEST_BLINDED.csv"
UNBLINDED_CSV = DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"
UNB_IDX_PATH = DATA_PROCESSED / "_audit_unblind_idx.npy"
UNB_Y_PATH = DATA_PROCESSED / "_audit_unblind_y.npy"

# Sweep grid + protocol params
S_GRID = [0.95, 1.00, 1.05, 1.10, 1.15]
N_FOLDS = 5
SCAFFOLD_SEED = 42
GATE = 0.003
SPEC_BASELINE = 0.5057       # value quoted in task spec
SPEC_DEPLOY_GATE = 0.5027    # task: "if best beats 0.5027 build deploy CSV"


def _murcko(smi: str) -> str | None:
    m = standardize(smi)
    if m is None:
        return None
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        if sc is None:
            return None
        s = Chem.MolToSmiles(sc)
        return s if s else None
    except Exception:
        return None


def _stretch(p: np.ndarray, mu: float, s: float) -> np.ndarray:
    return mu + s * (p - mu)


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- SCAFFOLD-CV RANK-STRETCH SWEEP")
    print(f"          predictor : nb2103 K=28 mean-bag OOF (253)")
    print(f"          anchor    : te_nb2112 (513)")
    print(f"          s grid    : {S_GRID}")
    print(f"          n_folds   : {N_FOLDS}  (scaffold-aware, seed={SCAFFOLD_SEED})")
    print(f"          gate      : -{GATE:.3f} RAE vs s=1.00 baseline")
    print("=" * 78)

    # ---- Load anchor truth aligned by Molecule Name ----
    te_df = pd.read_csv(TEST_BLINDED_CSV)
    n_test = len(te_df)
    name_to_idx = {n: i for i, n in enumerate(te_df["Molecule Name"].tolist())}

    unb_df = pd.read_csv(UNBLINDED_CSV)
    unb_df = unb_df[unb_df["Molecule Name"].isin(name_to_idx)].reset_index(drop=True)
    unb_idx = np.array(
        [name_to_idx[n] for n in unb_df["Molecule Name"]], dtype=int
    )
    unb_y = unb_df["pEC50"].astype(float).to_numpy(dtype=np.float64)
    unb_smiles = unb_df["SMILES"].astype(str).tolist()
    n_unb = len(unb_y)
    print(f"\n[load] n_test={n_test}  n_unb={n_unb}")

    # Cross-check vs cached unb_idx (positional alignment)
    cached_unb_idx = np.load(UNB_IDX_PATH)
    cached_unb_y = np.load(UNB_Y_PATH)
    assert cached_unb_idx.shape[0] == n_unb, "unb_idx length mismatch"
    if not np.array_equal(np.sort(unb_idx), np.sort(cached_unb_idx)):
        print("WARN: unb_idx ordering differs from cache; aligning by name only")

    # ---- Load nb2103 K=28 OOF (253) and te_nb2112 (513) ----
    if not NB2103_K28_OOF_PATH.exists():
        raise FileNotFoundError(NB2103_K28_OOF_PATH)
    if not NB2112_TE_PATH.exists():
        raise FileNotFoundError(NB2112_TE_PATH)
    pred_oof = np.load(NB2103_K28_OOF_PATH).astype(np.float64)
    te_nb2112 = np.load(NB2112_TE_PATH).astype(np.float64)
    assert pred_oof.shape == (n_unb,), f"OOF shape {pred_oof.shape}"
    assert te_nb2112.shape == (n_test,), f"te shape {te_nb2112.shape}"

    # The nb2103 OOF is positionally aligned to cached_unb_y (same 253 set).
    # Use cached_unb_y for the metric (matches what nb2103 was scored against).
    y = cached_unb_y.astype(np.float64)
    pred_std = float(pred_oof.std())
    truth_std = float(y.std())
    base_rae = float(rae(y, pred_oof))
    print(f"\n[base] nb2103 K=28 OOF pooled RAE      = {base_rae:.6f}")
    print(f"[base] pred mean/std                   = "
          f"{pred_oof.mean():.3f} / {pred_std:.3f}")
    print(f"[base] truth mean/std                  = "
          f"{y.mean():.3f} / {truth_std:.3f}")
    print(f"[base] variance-match ideal s          = "
          f"{truth_std / max(pred_std, 1e-9):.4f}")
    print(f"[ref ] task spec quoted baseline       = {SPEC_BASELINE:.4f}")
    print(f"[ref ] task spec deploy gate           = {SPEC_DEPLOY_GATE:.4f}")

    # ---- Murcko scaffolds for the 253 unblind ----
    print("\n[scaf] computing Murcko scaffolds for 253 unblind ...")
    scaffolds = [_murcko(s) for s in unb_smiles]
    n_none = sum(1 for s in scaffolds if not s)
    sc_counter: dict[str, int] = defaultdict(int)
    for s in scaffolds:
        if s:
            sc_counter[s] += 1
    print(f"[scaf] empty/none={n_none}  unique={len(sc_counter)}  "
          f"largest_group={max(sc_counter.values(), default=0)}")

    splits = scaffold_kfold_indices(
        scaffolds, n_splits=N_FOLDS, shuffle=True, seed=SCAFFOLD_SEED
    )
    fold_sizes = [len(va) for _, va in splits]
    print(f"[scaf] fold val sizes = {fold_sizes}  (sum={sum(fold_sizes)})")

    # ---- Sweep s in scaffold cross-fit ----
    print("\n" + "-" * 78)
    print("SCAFFOLD-CV STRETCH SWEEP")
    print("-" * 78)
    per_s = []
    for s in S_GRID:
        oof_s = np.full(n_unb, np.nan, dtype=np.float64)
        fold_raes = []
        fold_mus = []
        for fold, (tr_i, va_i) in enumerate(splits):
            mu_tr = float(pred_oof[tr_i].mean())
            oof_s[va_i] = _stretch(pred_oof[va_i], mu_tr, s)
            fold_raes.append(float(rae(y[va_i], oof_s[va_i])))
            fold_mus.append(mu_tr)
        pooled = float(rae(y, oof_s))
        per_s.append({
            "s": float(s),
            "pooled_scaffold_cv_rae": pooled,
            "fold_raes": [float(r) for r in fold_raes],
            "fold_mus": [float(m) for m in fold_mus],
            "fold_val_sizes": fold_sizes,
            "pred_std_after": float(oof_s.std()),
        })
        print(f"  s={s:.2f}:  pooled scaffold-CV RAE = {pooled:.6f}   "
              f"per-fold [{min(fold_raes):.4f}, {max(fold_raes):.4f}]   "
              f"std={oof_s.std():.3f}")

    # Identify s=1.00 baseline + best
    s1 = next(r for r in per_s if abs(r["s"] - 1.0) < 1e-9)
    baseline_rae = s1["pooled_scaffold_cv_rae"]
    best = min(per_s, key=lambda r: r["pooled_scaffold_cv_rae"])
    delta_vs_baseline = best["pooled_scaffold_cv_rae"] - baseline_rae
    beats_baseline_with_gate = (
        best["pooled_scaffold_cv_rae"] < baseline_rae - GATE
    )
    beats_spec_baseline_with_gate = (
        best["pooled_scaffold_cv_rae"] < SPEC_BASELINE - GATE
    )
    beats_spec_deploy_gate = (
        best["pooled_scaffold_cv_rae"] < SPEC_DEPLOY_GATE
    )

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  baseline s=1.00 scaffold-CV RAE  = {baseline_rae:.6f}  "
          f"(KFold pooled = {base_rae:.6f})")
    print(f"  best s                           = {best['s']:.2f}")
    print(f"  best scaffold-CV RAE             = "
          f"{best['pooled_scaffold_cv_rae']:.6f}")
    print(f"  delta vs s=1.00 (negative=win)   = {delta_vs_baseline:+.6f}")
    print(f"  beats s=1.00 by gate {GATE:.3f}      = "
          f"{beats_baseline_with_gate}")
    print(f"  beats spec base {SPEC_BASELINE:.4f} by gate = "
          f"{beats_spec_baseline_with_gate}")
    print(f"  beats spec deploy gate {SPEC_DEPLOY_GATE:.4f}  = "
          f"{beats_spec_deploy_gate}")

    deploy_path = None
    deploy_te_mean = None
    deploy_te_std = None
    deploy_method = "none"
    # Deploy iff:
    #   (a) winning s gives a real stretch (s != 1.00, else CSV == te_nb2112),
    #   (b) winning s beats s=1.00 by at least GATE, and
    #   (c) winning s also clears spec absolute deploy gate.
    nontrivial_s = abs(best["s"] - 1.0) > 1e-9
    if nontrivial_s and beats_baseline_with_gate and beats_spec_deploy_gate:
        s_star = best["s"]
        mu_full = float(pred_oof.mean())
        te_deploy = _stretch(te_nb2112, mu_full, s_star).astype(np.float32)
        out_csv = SUBMISSIONS / f"{TAG}_scaffold_stretch_s{s_star:.2f}.csv"
        df_sub = pd.DataFrame({
            "SMILES": te_df["SMILES"],
            "Molecule Name": te_df["Molecule Name"],
            "pEC50": te_deploy,
        })
        df_sub.to_csv(out_csv, index=False)
        deploy_path = str(out_csv)
        deploy_te_mean = float(te_deploy.mean())
        deploy_te_std = float(te_deploy.std())
        deploy_method = f"stretch_s{s_star:.2f}_mu{mu_full:.3f}"
        print(f"\n[deploy] s*={s_star:.2f}  mu_full={mu_full:.3f}  "
              f"te mean/std={deploy_te_mean:.3f}/{deploy_te_std:.3f}")
        print(f"[deploy] CSV: {out_csv}")
    else:
        why = []
        if not nontrivial_s:
            why.append("best s == 1.00 (no stretch -> identical to te_nb2112)")
        if not beats_baseline_with_gate:
            why.append(f"best does not beat s=1.00 by gate {GATE}")
        if not beats_spec_deploy_gate:
            why.append(f"best does not clear spec gate {SPEC_DEPLOY_GATE}")
        print(f"\n[deploy] gate NOT met  ->  no submission CSV written")
        print(f"[deploy] reason: {'; '.join(why)}")

    summary = {
        "tag": TAG,
        "method": "scaffold_cv_rank_stretch_sweep_on_nb2103_K28_oof",
        "predictor_oof_path": str(NB2103_K28_OOF_PATH),
        "anchor_te_path": str(NB2112_TE_PATH),
        "n_unb": int(n_unb),
        "n_test": int(n_test),
        "s_grid": [float(s) for s in S_GRID],
        "n_folds": int(N_FOLDS),
        "scaffold_seed": int(SCAFFOLD_SEED),
        "gate": float(GATE),
        "spec_baseline_quoted": float(SPEC_BASELINE),
        "spec_deploy_gate": float(SPEC_DEPLOY_GATE),
        "nb2103_K28_kfold_pooled_rae": float(base_rae),
        "scaffold_cv_baseline_s1_rae": float(baseline_rae),
        "pred_mean": float(pred_oof.mean()),
        "pred_std": float(pred_std),
        "truth_mean": float(y.mean()),
        "truth_std": float(truth_std),
        "variance_match_ideal_s": float(truth_std / max(pred_std, 1e-9)),
        "n_scaffolds_unique": int(len(sc_counter)),
        "n_scaffold_none": int(n_none),
        "largest_scaffold_group": int(max(sc_counter.values(), default=0)),
        "fold_val_sizes": [int(x) for x in fold_sizes],
        "per_s": per_s,
        "best_s": float(best["s"]),
        "best_scaffold_cv_rae": float(best["pooled_scaffold_cv_rae"]),
        "delta_best_vs_s1": float(delta_vs_baseline),
        "beats_s1_with_gate": bool(beats_baseline_with_gate),
        "beats_spec_baseline_with_gate": bool(beats_spec_baseline_with_gate),
        "beats_spec_deploy_gate": bool(beats_spec_deploy_gate),
        "deploy_method": deploy_method,
        "deploy_submission_csv": deploy_path,
        "deploy_te_mean": deploy_te_mean,
        "deploy_te_std": deploy_te_std,
        "wall_sec": round(time.time() - t0, 2),
    }

    out_json = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[save] summary: {out_json}")
    print(f"[done] wall = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_unb", "n_test", "n_scaffolds_unique", "largest_scaffold_group",
        "fold_val_sizes",
        "nb2103_K28_kfold_pooled_rae",
        "scaffold_cv_baseline_s1_rae",
        "best_s", "best_scaffold_cv_rae",
        "delta_best_vs_s1",
        "beats_s1_with_gate", "beats_spec_baseline_with_gate",
        "beats_spec_deploy_gate",
        "deploy_method", "deploy_submission_csv",
    ):
        print(f"  {k}: {res.get(k)}")
