"""nb3132 -- Per-fold LEARNED (q_cut, w_low, w_high) via fold-train grid search.

NEW PARADIGM (full adaptive vs nb3090 global grid, nb3104 per-fold sigmoid):
    nb3090 froze ONE (q_cut, w_low, w_high) globally and verified on val.
    nb3104 LEARNS sigmoid (center, slope, w_low, w_high) per fold.
    nb3132 LEARNS the hard-split (q_cut, w_low, w_high) PER FOLD on the
    fold-train slice, then applies the fold-best tuple to fold-val.
    Removes the global-mis-spec failure mode of nb3090: a single tuple
    cannot be simultaneously optimal across all 5 scaffold-CV splits.

    Per-fold blend (same hard-split form as nb3090):
        q_thr  = quantile(fold-train K18_pred, q_cut)
        row p_k18 <= q_thr -> blended = w_low  * K18 + (1 - w_low)  * K19
        row p_k18 >  q_thr -> blended = w_high * K18 + (1 - w_high) * K19

    Per-fold inner grid (fold-train held-in eval, pick min-RAE tuple):
        q_cut  in {0.30, 0.35, 0.40, 0.45}      (4)
        w_low  in {0.85, 0.90, 0.95}            (3)
        w_high in {0.40, 0.45, 0.50}            (3)
        => 4 * 3 * 3 = 36 combos per fold
    Stitch across 5 outer folds. Pooled RAE per kf_seed. 15 fresh kf_seeds
    {1141..1155}.

GATE (on 15-seed mean):
    mean < 0.4475 -> "BETTER"
    else          -> "FAIL"

References:
    nb2960 K18 deep-30 OOF         = 0.4536
    nb3000 K19 deep-30 OOF         = 0.4607
    nb3082 sigmoid fixed slope=10  = 0.4475 (15 seeds)
    nb3090 finer q_cut 15-combo    = 0.4472 (15 seeds)
    nb3104 per-fold sigmoid params = ? (parent paradigm)
    nb2171 prior post-hoc ceiling  = 0.4682

Inputs:
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/nb2960_K18_30seed_oof.npy
    data/processed/nb2960_K18_30seed_te.npy
    data/processed/nb3000_K19_30seed_oof.npy
    data/processed/te_nb3000_K19.npy

Outputs:
    data/processed/nb3132_summary.json
    data/processed/nb3132_pred_oof.npy  (253,) float32 median-seed OOF
    data/processed/te_nb3132.npy        (513,) float32 deploy te
    submissions/nb3132_per_fold_optimal_w_q.csv (only on BETTER verdict)
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

TAG = "nb3132"
PARENT_TAG = "nb3104"

# -- Inputs --------------------------------------------------------------------
K_LABELS = ["K18", "K19"]
OOF_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_oof.npy",
    "K19": DATA_PROCESSED / "nb3000_K19_30seed_oof.npy",
}
TE_PATHS = {
    "K18": DATA_PROCESSED / "nb2960_K18_30seed_te.npy",
    "K19": DATA_PROCESSED / "te_nb3000_K19.npy",
}
K_DEPTH = {"K18": "deep30", "K19": "deep30"}

# -- CV protocol ---------------------------------------------------------------
N_FOLDS = 5
KF_SEEDS = list(range(1141, 1156))  # 15 fresh seeds {1141..1155}

# -- Per-fold inner grid -------------------------------------------------------
Q_CUTS = [0.30, 0.35, 0.40, 0.45]
W_LOWS = [0.85, 0.90, 0.95]
W_HIGHS = [0.40, 0.45, 0.50]

# -- Gates ---------------------------------------------------------------------
GATE_BETTER = 0.4475

# -- References ----------------------------------------------------------------
REF_NB3090 = 0.4472
REF_NB3082 = 0.4475
REF_NB3080 = 0.4475
REF_K18 = 0.4536
REF_K19 = 0.4607
REF_NB2171 = 0.4682


def _blend_quantile_conditional(
    p_k18: np.ndarray,
    p_k19: np.ndarray,
    q_thr: float,
    w_low: float,
    w_high: float,
) -> np.ndarray:
    """Per-row hard-split blend (same form as nb3090).

    rows with p_k18 <= q_thr -> (w_low  * p_k18 + (1-w_low)  * p_k19)
    rows with p_k18 >  q_thr -> (w_high * p_k18 + (1-w_high) * p_k19)
    """
    low_mask = p_k18 <= q_thr
    out = np.empty_like(p_k18, dtype=np.float64)
    out[low_mask] = w_low * p_k18[low_mask] + (1.0 - w_low) * p_k19[low_mask]
    out[~low_mask] = w_high * p_k18[~low_mask] + (1.0 - w_high) * p_k19[~low_mask]
    return out


def _learn_fold_params(
    p_k18_tr: np.ndarray,
    p_k19_tr: np.ndarray,
    y_tr: np.ndarray,
) -> dict:
    """Inner grid search on fold-train: pick the (q_cut, w_low, w_high) tuple
    that minimizes RAE on the fold-train predictions.
    """
    best_tuple = None
    best_rae = float("inf")
    n_combos = 0
    for q_cut in Q_CUTS:
        q_thr = float(np.quantile(p_k18_tr, q_cut))
        for w_low in W_LOWS:
            for w_high in W_HIGHS:
                n_combos += 1
                pred_tr = _blend_quantile_conditional(
                    p_k18_tr, p_k19_tr, q_thr, w_low, w_high,
                )
                r = float(rae(y_tr, pred_tr))
                if r < best_rae:
                    best_rae = r
                    best_tuple = {
                        "q_cut": float(q_cut),
                        "q_thr": float(q_thr),
                        "w_low": float(w_low),
                        "w_high": float(w_high),
                        "train_rae": round(r, 4),
                    }
    return {"best": best_tuple, "n_combos": n_combos}


def _run_one_seed(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
    unb_scaffolds: list[str],
    kf_seed: int,
) -> dict:
    """Per-fold-learned (q_cut, w_low, w_high) pipeline at a single kf_seed."""
    splits = scaffold_kfold_indices(
        unb_scaffolds, n_splits=N_FOLDS, shuffle=True, seed=kf_seed,
    )
    n_unb = len(y_unb)
    oof_blend = np.full(n_unb, np.nan, dtype=np.float64)
    fold_picks = []
    fold_val_raes = []
    for fold_i, (tr_loc, va_loc) in enumerate(splits):
        p_k18_tr = P_unb[tr_loc, 0]
        p_k19_tr = P_unb[tr_loc, 1]
        y_tr = y_unb[tr_loc]

        learn = _learn_fold_params(p_k18_tr, p_k19_tr, y_tr)
        bp = learn["best"]

        val_pred = _blend_quantile_conditional(
            P_unb[va_loc, 0], P_unb[va_loc, 1],
            bp["q_thr"], bp["w_low"], bp["w_high"],
        )
        oof_blend[va_loc] = val_pred
        val_rae = float(rae(y_unb[va_loc], val_pred))
        fold_val_raes.append(val_rae)
        fold_picks.append({
            "fold": int(fold_i),
            "n_tr": int(len(tr_loc)),
            "n_va": int(len(va_loc)),
            "q_cut": bp["q_cut"],
            "q_thr": round(bp["q_thr"], 4),
            "w_low": bp["w_low"],
            "w_high": bp["w_high"],
            "train_rae": bp["train_rae"],
            "val_rae": round(val_rae, 4),
        })

    if np.isnan(oof_blend).any():
        raise RuntimeError(f"kf_seed={kf_seed}: splits did not cover all rows")
    pooled = float(rae(y_unb, oof_blend))
    return {
        "kf_seed": int(kf_seed),
        "pooled_rae": pooled,
        "per_fold_val_rae_mean": float(np.mean(fold_val_raes)),
        "fold_picks": fold_picks,
        "oof": oof_blend,
    }


def _deploy_params(
    P_unb: np.ndarray,
    y_unb: np.ndarray,
) -> dict:
    """Deploy-time: learn (q_cut, w_low, w_high) on full unblind (253)."""
    learn = _learn_fold_params(P_unb[:, 0], P_unb[:, 1], y_unb)
    return learn["best"]


def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PER-FOLD learned (q_cut, w_low, w_high) via fold-train grid")
    print(f"          grid: q_cut in {Q_CUTS}")
    print(f"                w_low in {W_LOWS} x w_high in {W_HIGHS}")
    print(f"          => {len(Q_CUTS)*len(W_LOWS)*len(W_HIGHS)} combos per fold")
    print(f"          kf_seeds = {{{KF_SEEDS[0]}..{KF_SEEDS[-1]}}} (n={len(KF_SEEDS)})")
    print(f"          gate: mean < {GATE_BETTER} -> BETTER")
    print("=" * 78)

    # -- Load test, truth ----------------------------------------------------
    te = load_test()
    n_test = len(te)
    te_smiles = (te["smiles"].astype(str).tolist()
                 if "smiles" in te.columns else te["SMILES"].astype(str).tolist())
    te_names = (te["name"].values
                if "name" in te.columns else te["Molecule Name"].values)
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    assert n_unb == 253, f"expected 253 unblind, got {n_unb}"
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    # -- Load K18, K19 deep-30 anchor OOFs + te arrays ------------------------
    print("\n" + "-" * 78)
    print("STEP 1: load K18, K19 deep-30 OOFs and te arrays")
    print("-" * 78)
    oof_cols, te_cols = [], []
    per_K_full_rae = {}
    for k in K_LABELS:
        oof = np.load(OOF_PATHS[k]).astype(np.float64)
        te_arr = np.load(TE_PATHS[k]).astype(np.float64)
        if oof.shape != (n_unb,):
            raise ValueError(f"{k} OOF shape {oof.shape} != ({n_unb},)")
        if te_arr.shape != (n_test,):
            raise ValueError(f"{k} te shape {te_arr.shape} != ({n_test},)")
        oof_cols.append(oof)
        te_cols.append(te_arr)
        r = float(rae(y_unb, oof))
        per_K_full_rae[k] = round(r, 4)
        print(f"   {k} ({K_DEPTH[k]:>6s}): oof_RAE = {r:.4f}  "
              f"te_mean={te_arr.mean():.3f}  te_std={te_arr.std():.3f}")

    P_unb = np.column_stack(oof_cols)  # (253, 2)
    P_te = np.column_stack(te_cols)    # (513, 2)

    # Leak sanity
    leak_flags = {}
    for i, k in enumerate(K_LABELS):
        frac = float(np.mean(np.isclose(P_unb[:, i], y_unb, atol=1e-6)))
        leak_flags[k] = round(frac, 4)
        if frac > 0.05:
            print(f"   WARN {k}: {frac:.1%} rows == truth -- possible leak")
    corr = float(np.corrcoef(P_unb.T)[0, 1])
    print(f"   pairwise corr({K_LABELS[0]}, {K_LABELS[1]}) = {corr:.4f}")

    # -- Scaffolds (kf_seed independent) -------------------------------------
    print("\n" + "-" * 78)
    print("STEP 2: scaffolds for outer CV")
    print("-" * 78)
    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) or "" for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"   n_unique_scaffolds = {n_unique_scaf}")

    # -- 15-seed run ---------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"RUN: per-fold inner-grid x {len(KF_SEEDS)} kf_seeds")
    print("-" * 78)
    pooled_raes = []
    seed_records = []
    oof_stack = []
    fold_pick_distribution = []  # per-fold pick tally across all (seed, fold)
    for s in KF_SEEDS:
        ts = time.time()
        res = _run_one_seed(P_unb, y_unb, unb_scaffolds, s)
        pooled_raes.append(res["pooled_rae"])
        oof_stack.append(res["oof"])
        seed_records.append({
            "kf_seed": res["kf_seed"],
            "pooled_rae": round(res["pooled_rae"], 4),
            "per_fold_val_rae_mean": round(res["per_fold_val_rae_mean"], 4),
            "fold_picks": res["fold_picks"],
        })
        fold_pick_distribution.extend(res["fold_picks"])
        print(f"   seed={s}: pooled={res['pooled_rae']:.4f}  "
              f"fold_val_mean={res['per_fold_val_rae_mean']:.4f}  "
              f"wall={time.time()-ts:.2f}s")

    arr = np.asarray(pooled_raes, dtype=np.float64)
    n_seeds = len(arr)
    mean_rae = float(arr.mean())
    std_rae = float(arr.std(ddof=1)) if n_seeds > 1 else 0.0
    sem = std_rae / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
    t_mult = 2.145  # t_{0.025, df=14}
    ci_low = mean_rae - t_mult * sem
    ci_high = mean_rae + t_mult * sem
    median_rae = float(np.median(arr))
    med_seed_idx = int(np.argsort(arr)[n_seeds // 2])
    median_seed = KF_SEEDS[med_seed_idx]
    oof_for_save = oof_stack[med_seed_idx].astype(np.float32)

    print("\n" + "-" * 78)
    print("AGGREGATE")
    print("-" * 78)
    print(f"   mean       = {mean_rae:.4f}")
    print(f"   std        = {std_rae:.4f}")
    print(f"   sem        = {sem:.4f}")
    print(f"   95% CI     = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   median     = {median_rae:.4f}  (seed {median_seed})")
    print(f"   min/max    = [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"   delta vs nb3090 ceiling {REF_NB3090} = {mean_rae - REF_NB3090:+.4f}")
    print(f"   delta vs gate          {GATE_BETTER} = {mean_rae - GATE_BETTER:+.4f}")

    # -- Fold pick tally -----------------------------------------------------
    pick_tally = {}
    for fp in fold_pick_distribution:
        key = (fp["q_cut"], fp["w_low"], fp["w_high"])
        pick_tally[key] = pick_tally.get(key, 0) + 1
    pick_sorted = sorted(pick_tally.items(), key=lambda kv: -kv[1])
    print("\n   top picks (q_cut, w_low, w_high) -> count "
          f"out of {len(fold_pick_distribution)} fold-picks:")
    for (qc, wl, wh), cnt in pick_sorted[:10]:
        print(f"     ({qc}, {wl}, {wh}) -> {cnt}")

    # -- Deploy on full unblind ----------------------------------------------
    print("\n" + "-" * 78)
    print("DEPLOY (learn params on full 253)")
    print("-" * 78)
    bp = _deploy_params(P_unb, y_unb)
    print(f"   deploy params: q_cut={bp['q_cut']}  q_thr={bp['q_thr']:.4f}")
    print(f"                  w_low={bp['w_low']}  w_high={bp['w_high']}")
    te_pred_raw = _blend_quantile_conditional(
        P_te[:, 0], P_te[:, 1],
        bp["q_thr"], bp["w_low"], bp["w_high"],
    )
    te_pred = np.clip(te_pred_raw.astype(np.float32), 3.0, 9.0)
    te_unb_in_rae = float(rae(y_unb, te_pred[unb_idx]))
    print(f"   te(513) mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    print(f"   te[unb] in-sample RAE  = {te_unb_in_rae:.4f}")

    # -- Gate -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("GATE")
    print("-" * 78)
    if mean_rae < GATE_BETTER:
        verdict = "BETTER"
        ladder_action = (
            f"PROMOTE candidate. nb3132 per-fold learned (q_cut, w_low, w_high) "
            f"15-seed mean {mean_rae:.4f} beats gate {GATE_BETTER:.4f} "
            f"({mean_rae - GATE_BETTER:+.4f}). Full per-fold adaptive grid "
            "extracts gain over the global frozen tuple of nb3090. Re-verify "
            "with deep-30 seeds before deploy."
        )
    else:
        verdict = "FAIL"
        ladder_action = (
            f"REJECT. nb3132 per-fold learned (q_cut, w_low, w_high) 15-seed mean "
            f"{mean_rae:.4f} not below gate {GATE_BETTER:.4f} "
            f"({mean_rae - GATE_BETTER:+.4f}). Per-fold inner-grid learning "
            "on (q_cut, w_low, w_high) is in-sample optimistic on fold-train "
            "and adds variance without translating to val; closes the per-fold "
            "fully-adaptive hard-split axis on this anchor pair. Keep nb3090 / "
            "prior PRIMARY-1."
        )
    print(f"   verdict       = {verdict}")
    print(f"   ladder action = {ladder_action}")

    # -- Save artifacts -------------------------------------------------------
    print("\n" + "-" * 78)
    print("SAVE")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, oof_for_save)
    np.save(te_path, te_pred)
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_per_fold_optimal_w_q.csv"
    if verdict == "BETTER":
        pd.DataFrame({
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": te_pred,
        }).to_csv(sub_csv, index=False)
        print(f"   [save] {sub_csv}")
    else:
        print(f"   [skip] verdict={verdict}; no submission CSV written")

    # JSON-serializable pick tally
    pick_tally_json = [
        {"q_cut": k[0], "w_low": k[1], "w_high": k[2], "count": int(v)}
        for k, v in pick_sorted
    ]

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "per_fold_learned_qcut_wlow_whigh_inner_grid_K18_K19_deep30",
        "anchor_pool": K_LABELS,
        "anchor_depth": K_DEPTH,
        "anchor_pre_unblind": True,
        "per_K_full_oof_rae": per_K_full_rae,
        "anchor_leak_eq_truth_frac": leak_flags,
        "oof_pairwise_corr": round(corr, 4),
        "grid": {
            "q_cuts": Q_CUTS,
            "w_lows": W_LOWS,
            "w_highs": W_HIGHS,
            "n_combos_per_fold": len(Q_CUTS) * len(W_LOWS) * len(W_HIGHS),
        },
        "blend_form": (
            "p_k18 <= q_thr -> w_low*K18 + (1-w_low)*K19; "
            "p_k18 >  q_thr -> w_high*K18 + (1-w_high)*K19"
        ),
        "n_folds": N_FOLDS,
        "kf_seeds": KF_SEEDS,
        "n_seeds": len(KF_SEEDS),
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "seed_records": seed_records,
        "pick_tally_top": pick_tally_json[:25],
        "n_unique_fold_picks": int(len(pick_sorted)),
        "mean_rae": round(mean_rae, 4),
        "std_rae": round(std_rae, 4),
        "sem_rae": round(sem, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "median_rae": round(median_rae, 4),
        "min_rae": round(float(arr.min()), 4),
        "max_rae": round(float(arr.max()), 4),
        "median_seed": int(median_seed),
        "pooled_rae_array": [round(float(v), 4) for v in pooled_raes],
        "ref_nb3090": REF_NB3090,
        "ref_nb3082": REF_NB3082,
        "ref_nb3080": REF_NB3080,
        "ref_K18_deep30": REF_K18,
        "ref_K19_deep30": REF_K19,
        "ref_nb2171": REF_NB2171,
        "delta_vs_nb3090": round(mean_rae - REF_NB3090, 4),
        "delta_vs_gate": round(mean_rae - GATE_BETTER, 4),
        "deploy_params": {
            "q_cut": bp["q_cut"],
            "q_thr": round(bp["q_thr"], 4),
            "w_low": bp["w_low"],
            "w_high": bp["w_high"],
            "train_rae_on_full_unblind": bp["train_rae"],
        },
        "te_mean": float(te_pred.mean()),
        "te_std": float(te_pred.std()),
        "te_unb_in_sample_rae": round(te_unb_in_rae, 4),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv) if verdict == "BETTER" else None,
        "gate_better": GATE_BETTER,
        "verdict": verdict,
        "ladder_action": ladder_action,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   15-seed mean    = {mean_rae:.4f} +/- {std_rae:.4f}")
    print(f"   95% CI          = [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"   delta vs nb3090 = {mean_rae - REF_NB3090:+.4f}")
    print(f"   delta vs gate   = {mean_rae - GATE_BETTER:+.4f}")
    print(f"   verdict         = {verdict}")
    print(f"   wall            = {time.time()-t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "mean_rae", "std_rae", "ci95_low", "ci95_high",
        "delta_vs_nb3090", "delta_vs_gate", "verdict", "ladder_action",
    ):
        print(f"  {k}: {res.get(k)}")
    print(f"  per_K_full_oof_rae: {res.get('per_K_full_oof_rae')}")
    print(f"  deploy_params: {res.get('deploy_params')}")
    print(f"  pick_tally_top (first 5): {res.get('pick_tally_top')[:5]}")
