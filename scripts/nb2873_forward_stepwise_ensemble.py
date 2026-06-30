"""nb2873 -- Per-fold forward-stepwise ensemble member selection.

NEW PARADIGM (vs SLSQP simplex / LASSO+simplex / static equal-weight):
    Every prior blend on the 3 PRE-clean anchors {nb2240_K20, chemprop_aux,
    counter_clean} either (a) optimizes a continuous weight vector on the
    simplex (SLSQP/LASSO+renorm), which is a Wahba-style projection on top
    of a single training fold and has K-1 degrees of freedom; or (b) uses
    a static recipe (equal weight, fixed simplex). Both decisions are made
    ONCE per fold and applied homogeneously.

    nb2873 swaps the *continuous* weight search for *discrete* greedy
    forward-stepwise SELECTION: per fold, start the ensemble at the single
    strongest anchor (nb2240_K20), then test whether ADDING chemprop_aux
    -- under equal-weighting of the currently-selected set -- decreases
    fold-train pooled RAE; if yes, accept; else reject. Repeat for
    counter_clean. Selected anchors are equal-weight-averaged on the
    validation fold. This is the smallest-capacity ensemble selector
    possible: per-fold 3-bit decision (which subset of {a, b, c} contains
    a) -> 4 possible outcomes per fold. K-1 free parameters of SLSQP
    collapse to K-1 BINARY decisions; the inductive bias is "anchor
    contributes if and only if it improves fold-train RAE under uniform
    weighting." This is robustly invariant to anchor scale/bias offsets
    and cannot overfit to noise the way unconstrained continuous weights
    can on a tiny K=3 anchor pool.

SUBSTRATE (PRE-clean only, 3 anchors -- same as nb2820/nb2844):
    - nb2240_K20      (K=20 residual stack on chemprop_aux)
    - chemprop_aux    (nb1133, 4139 PRE-unblind only)
    - counter_clean   (nb2490 counter-assay residual on chemprop_aux,
                       nb730-free)

    nb730/nb562/nb503 EXCLUDED (POST contamination chain / not PRE-clean).

PROTOCOL:
    - 5-fold scaffold CV on 253 unblind, kf_seed=1001
    - Per fold:
        1. selected = [nb2240_K20]  (anchor 0 always seeded)
        2. best_rae = rae(y_tr, equal_avg(selected) on tr)
        3. for cand in [chemprop_aux, counter_clean]:
               trial = selected + [cand]
               trial_rae = rae(y_tr, equal_avg(trial) on tr)
               if trial_rae < best_rae:
                   selected = trial
                   best_rae = trial_rae
        4. pred_va = equal_avg(selected on va)
    - Pooled cross-fit RAE on 253 = mean_rae.
    - Deploy: same greedy forward-stepwise on FULL 253 (not held-out),
      equal-weight-average selected anchors on te 513.

GATE:
    mean_rae < 0.4570  -> PROMOTE
    mean_rae < 0.4598  -> MARGINAL_BEAT
    else               -> FAIL

Outputs:
    scripts/nb2873_forward_stepwise_ensemble.py
    data/processed/nb2873_summary.json
    data/processed/nb2873_pred_oof.npy   (253,) float32
    data/processed/te_nb2873.npy         (513,) float32
    submissions/nb2873_forward_stepwise_ensemble.csv  (on non-FAIL)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2873"
N_FOLDS = 5
KF_SEED = 1001
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Anchor order matters: index 0 = seed anchor (always selected first),
# remaining = candidates tested in order for greedy addition.
CANDIDATE_ANCHORS = [
    ("nb2240_K20",   DATA_PROCESSED / "nb2240_mean_bag_oof_K20.npy",
                     DATA_PROCESSED / "te_nb2240_K20.npy",
                     "PRE-clean seed anchor (K=20 residual stack on chemprop_aux)"),
    ("chemprop_aux", DATA_PROCESSED / "nb1133_chemprop_aux_pred_oof.npy",
                     DATA_PROCESSED / "te_chemprop_aux.npy",
                     "PRE-clean (4139 PRE-unblind only)"),
    ("counter_clean",DATA_PROCESSED / "nb2490_pred_oof.npy",
                     DATA_PROCESSED / "te_nb2490.npy",
                     "PRE-clean counter-assay residual on chemprop_aux (nb730-free)"),
]


def forward_stepwise_select(P: np.ndarray, y: np.ndarray,
                            anchor_names: list[str]) -> tuple[list[int], list[dict]]:
    """Greedy forward-stepwise selection of anchors.

    Always seeds with anchor index 0. For each remaining candidate, accepts
    iff equal-weight average INCLUDING it has strictly lower RAE than the
    current selected set.

    Returns (selected_indices, decision_log).
    """
    K = P.shape[1]
    assert K == len(anchor_names)
    selected = [0]  # seed
    cur_blend = P[:, selected].mean(axis=1)
    cur_rae = float(rae(y, cur_blend))
    decision_log = [{
        "step": 0,
        "action": "seed",
        "candidate": anchor_names[0],
        "selected_after": [anchor_names[i] for i in selected],
        "rae_after": cur_rae,
    }]
    for cand_idx in range(1, K):
        trial = selected + [cand_idx]
        trial_blend = P[:, trial].mean(axis=1)
        trial_rae = float(rae(y, trial_blend))
        accepted = trial_rae < cur_rae
        decision_log.append({
            "step": cand_idx,
            "action": "accept" if accepted else "reject",
            "candidate": anchor_names[cand_idx],
            "trial_rae": trial_rae,
            "cur_rae_before": cur_rae,
            "selected_after": ([anchor_names[i] for i in trial] if accepted
                               else [anchor_names[i] for i in selected]),
            "rae_after": trial_rae if accepted else cur_rae,
        })
        if accepted:
            selected = trial
            cur_rae = trial_rae
    return selected, decision_log


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Per-fold forward-stepwise ensemble selection")
    print("=" * 78)

    # ---- Load test + unblind ----
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test} n_unb={n_unb}")

    # ---- Resolve anchors (strict: need all 3) ----
    anchor_names = []
    anchor_provenance = {}
    oof_cols, te_cols = [], []
    anchor_skipped = {}
    for name, oof_path, te_path, prov in CANDIDATE_ANCHORS:
        if not oof_path.exists():
            anchor_skipped[name] = f"pred_oof missing at {oof_path}"
            print(f"   SKIP {name}: pred_oof missing")
            continue
        if not te_path.exists():
            anchor_skipped[name] = f"te missing at {te_path}"
            print(f"   SKIP {name}: te missing")
            continue
        oof = np.load(oof_path).astype(np.float64)
        te_v = np.load(te_path).astype(np.float64)
        if oof.shape[0] != n_unb:
            anchor_skipped[name] = f"shape mismatch oof={oof.shape} expected ({n_unb},)"
            print(f"   SKIP {name}: shape mismatch")
            continue
        if te_v.shape[0] != n_test:
            anchor_skipped[name] = f"shape mismatch te={te_v.shape} expected ({n_test},)"
            print(f"   SKIP {name}: te shape mismatch")
            continue
        anchor_names.append(name)
        anchor_provenance[name] = prov
        oof_cols.append(oof)
        te_cols.append(te_v)
    K = len(anchor_names)
    if K < 3:
        raise RuntimeError(f"Need 3 PRE-clean anchors, got {K}: {anchor_names}")

    P_unb = np.column_stack(oof_cols)  # (253, K)
    P_te = np.column_stack(te_cols)    # (513, K)
    rae_anchors = {k: float(rae(y_unb, P_unb[:, i]))
                   for i, k in enumerate(anchor_names)}
    print(f"[anchors] K={K}  seed={anchor_names[0]}")
    for k in anchor_names:
        print(f"   {k:14s}  unb_RAE={rae_anchors[k]:.4f}  [{anchor_provenance[k]}]")
    if anchor_skipped:
        print(f"[skipped] {anchor_skipped}")

    # ---- Scaffold folds ----
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique={n_unique_scaf}  n_folds={N_FOLDS}  kf_seed={KF_SEED}")

    # ---- Per-fold forward-stepwise CV ----
    print("\n" + "-" * 78)
    print(f"FORWARD-STEPWISE CV ({N_FOLDS}-fold scaffold, kf_seed={KF_SEED})")
    print("-" * 78)
    splits = scaffold_kfold_indices(unb_scaffolds, n_splits=N_FOLDS,
                                    shuffle=True, seed=KF_SEED)
    oof = np.full(n_unb, np.nan, dtype=np.float64)
    fold_rae = []
    per_fold_selected = []
    per_fold_decisions = []
    for f_idx, (tr_loc, va_loc) in enumerate(splits):
        selected, decision_log = forward_stepwise_select(
            P_unb[tr_loc], y_unb[tr_loc], anchor_names
        )
        blend_va = P_unb[va_loc][:, selected].mean(axis=1)
        oof[va_loc] = blend_va
        r = float(rae(y_unb[va_loc], blend_va))
        fold_rae.append(r)
        sel_names = [anchor_names[i] for i in selected]
        per_fold_selected.append(sel_names)
        per_fold_decisions.append(decision_log)
        print(f"   fold {f_idx}  selected={sel_names}  va_RAE={r:.4f}  "
              f"n_tr={len(tr_loc)} n_va={len(va_loc)}")

    pooled = float(rae(y_unb, oof))
    mean_fold = float(np.mean(fold_rae))
    print(f"\n[cv] pooled RAE         = {pooled:.4f}")
    print(f"[cv] mean fold RAE      = {mean_fold:.4f}")

    # ---- Selection frequency across folds ----
    sel_freq = {k: 0 for k in anchor_names}
    for sel in per_fold_selected:
        for k in sel:
            sel_freq[k] += 1
    print(f"[cv] selection freq (out of {N_FOLDS} folds):")
    for k in anchor_names:
        print(f"   {k:14s}  {sel_freq[k]}/{N_FOLDS}")

    # ---- Gate ----
    mean_rae = pooled
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"\n[gate] mean_rae {mean_rae:.4f}  "
          f"(< {GATE_PROMOTE} PROMOTE / < {GATE_MARGINAL} MARGINAL)  ->  {verdict}")

    # ---- Deploy: greedy forward-stepwise on FULL 253, apply to te 513 ----
    print("\n" + "-" * 78)
    print("DEPLOY: forward-stepwise on full 253, equal-avg selected on 513")
    print("-" * 78)
    selected_deploy, decision_log_deploy = forward_stepwise_select(
        P_unb, y_unb, anchor_names
    )
    sel_names_deploy = [anchor_names[i] for i in selected_deploy]
    te_pred = P_te[:, selected_deploy].mean(axis=1).astype(np.float32)
    te_pred = np.clip(te_pred, 3.0, 9.0)
    te_unb_in = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   deploy selected = {sel_names_deploy}  (n={len(selected_deploy)})")
    for d in decision_log_deploy:
        print(f"   step {d['step']}  {d['action']:6s}  cand={d['candidate']:14s}  "
              f"rae_after={d['rae_after']:.4f}")
    print(f"   te mean={te_pred.mean():.3f} std={te_pred.std():.3f}")
    print(f"   te[unb_idx] in-sample RAE = {te_unb_in:.4f}  (expected << pooled)")

    # ---- Save artifacts ----
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof.astype(np.float32))
    np.save(te_path, te_pred)
    print(f"[save] {oof_path}")
    print(f"[save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_forward_stepwise_ensemble.csv"
    if verdict != "FAIL":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"[save] {sub_csv}")
    else:
        print(f"[skip] no submission CSV (FAIL gate)")

    summary = {
        "tag": TAG,
        "method": "per_fold_forward_stepwise_ensemble_equal_weight",
        "anchor_pool": anchor_names,
        "anchor_provenance": anchor_provenance,
        "anchor_skipped": anchor_skipped,
        "anchor_in_rae": rae_anchors,
        "seed_anchor": anchor_names[0],
        "n_folds": N_FOLDS,
        "kf_seed": KF_SEED,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "K_anchors": K,
        "fold_rae": fold_rae,
        "per_fold_selected": per_fold_selected,
        "per_fold_decisions": per_fold_decisions,
        "selection_freq": sel_freq,
        "pooled_rae": pooled,
        "mean_fold_rae": mean_fold,
        "mean_rae": mean_rae,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "deploy_selected": sel_names_deploy,
        "deploy_decision_log": decision_log_deploy,
        "te_unb_in_sample_rae": te_unb_in,
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "oof_npy_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict != "FAIL" else None,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   pooled RAE          = {pooled:.4f}  ({verdict})")
    print(f"   mean fold RAE       = {mean_fold:.4f}")
    print(f"   K anchors available = {K}  ({anchor_names})")
    print(f"   deploy selected     = {sel_names_deploy}")
    print(f"   selection freq      = {sel_freq}")
    print(f"   wall                = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("mean_rae", "pooled_rae", "mean_fold_rae", "verdict",
              "K_anchors", "deploy_selected", "selection_freq",
              "te_unb_in_sample_rae", "submission_csv"):
        print(f"  {k}: {res.get(k)}")
