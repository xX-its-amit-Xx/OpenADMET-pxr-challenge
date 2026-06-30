"""nb2701 -- PC-algorithm causal feature selection on K=20 + y_unb.

NEW PARADIGM (Spirtes-Glymour-Scheines, 1991):
    PC discovers a CPDAG (Markov-equivalence class of DAGs) from observational
    data via conditional independence tests on the empirical covariance.  Run
    on the joint matrix (X_K20 || y_unb) it identifies a subset of K features
    that are directed CAUSAL PARENTS of y (i.e. C-faithful Markov neighbours
    not d-separated by any conditioning set), as opposed to mere correlational
    neighbours.

    Hypothesis: causal-parent features carry the signal that is invariant
    under intervention -- if PXR mechanism is the same on 4139-train and 513-
    test, the causal subgraph is stationary across the train/test scaffold
    shift while spurious correlational features may not be.  Train LGBM on
    the directed-parent subset only and compare to nb2171 0.4682 deep-30
    ceiling on the same chemprop_aux anchor.

    This is a substrate change in the cycle-134 sense: anchor unchanged
    (chemprop_aux), feature set CHANGED (raw K=20 -> causal-parent subset of
    K=20).  Cross-paradigm-feature-ranker memo (feedback_cycle139) flagged
    LASSO/Boruta/MI/PermImp as paradigm-mismatched against LGBM; PC is
    distribution-free (Fisher-Z on Pearson partial corr) but selects on the
    CAUSAL DAG criterion, not on linear / mutual-info magnitude -- different
    inductive bias, may or may not overlap with SHAP-LGBM-native.

PROTOCOL:
    1. Build K=20 substrate by slicing X_117 (data/processed/pyramid/) with
       nb2231 K=20 surviving idx (the verified RFE snapshot).
    2. Combine X_K20 (253, 20) with y_unb (253,) into a (253, 21) data
       matrix where col 20 is y.
    3. Run causallearn PC with alpha=0.05 fisherz CI test.
    4. Extract causal parents of y (col 20) in the resulting CPDAG: any
       feature i with a directed edge i -> y, encoded as
       graph[i, y_col] == -1 AND graph[y_col, i] == 1.
    5. Train LGBM on the causal-parent subset (residual on chemprop_aux):
       5-fold scaffold CV, 5 kf_seeds {1001..1005}, 5 lgbm bag seeds {0..4}.
    6. Report mean_rae (per-kf 5-bag-mean) across 5 kf_seeds.
    7. Gate:
         mean_rae < 0.4570 -> PROMOTE
         mean_rae < 0.4598 -> MARGINAL_BEAT
         else              -> FAIL

INSTALL:
    If `import causallearn` fails, save string "INSTALL_FAILED" to summary
    and exit cleanly.

Inputs:
    data/processed/pyramid/X_117_unb.npy      (253, 117)
    data/processed/pyramid/X_117_te.npy       (513, 117)
    data/processed/_audit_unblind_idx.npy
    data/processed/_audit_unblind_y.npy
    data/processed/te_chemprop_aux.npy
    data/processed/nb2231_summary.json        (K=20 surviving idx in 117)

Outputs:
    data/processed/nb2701_summary.json
    data/processed/nb2701_pred_oof.npy   (253,) float32
    data/processed/te_nb2701.npy         (513,) float32
    submissions/nb2701_pc_causal_selection.csv  (always written)
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
import lightgbm as lgb

from pxr.chem import bemis_murcko
from pxr.data import load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb2701"
PARENT_TAG = "nb2171"

# Protocol parameters
KF_SEEDS = [1001, 1002, 1003, 1004, 1005]
BAG_SEEDS = [0, 1, 2, 3, 4]
RESID_FOLDS = 5
PC_ALPHA = 0.05
PC_INDEP_TEST = "fisherz"

# Reference numbers
NB2171_REF = 0.4682
CHEMPROP_AUX_REF = 0.6216

# Gates
GATE_PROMOTE = 0.4570
GATE_MARGINAL = 0.4598

# Input artefact paths
X117_UNB_PATH = DATA_PROCESSED / "pyramid" / "X_117_unb.npy"
X117_TE_PATH = DATA_PROCESSED / "pyramid" / "X_117_te.npy"
ANCHOR_TE_PATH = DATA_PROCESSED / "te_chemprop_aux.npy"
NB2231_SUMMARY = DATA_PROCESSED / "nb2231_summary.json"


def _save_install_failed():
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "tag": TAG,
                "verdict": "INSTALL_FAILED",
                "method": "pc_causal_selection",
                "note": "causallearn import failed",
            },
            f,
            indent=2,
        )
    print(f"[save] {out_path}")
    print("INSTALL_FAILED")


# Install gate
try:
    from causallearn.search.ConstraintBased.PC import pc as causal_pc  # noqa: F401
except Exception as e:  # pragma: no cover
    print(f"causallearn import failed: {e}")
    _save_install_failed()
    sys.exit(0)


def _lgbm_params(seed: int) -> dict:
    return dict(
        objective="regression",
        max_depth=4,
        num_leaves=15,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=2,
        verbosity=-1,
    )


def _residual_cross_fit_scaffold(X, residual, scaffolds, kf_seed, lgbm_seed):
    """5-fold scaffold-CV residual prediction (one LGBM bag entry)."""
    n = len(residual)
    splits = scaffold_kfold_indices(
        scaffolds, n_splits=RESID_FOLDS, shuffle=True, seed=kf_seed,
    )
    oof = np.full(n, np.nan, dtype=np.float64)
    for tr_loc, va_loc in splits:
        mdl = lgb.LGBMRegressor(**_lgbm_params(lgbm_seed))
        mdl.fit(X[tr_loc], residual[tr_loc])
        oof[va_loc] = mdl.predict(X[va_loc])
    if np.isnan(oof).any():
        raise RuntimeError("scaffold split did not cover all rows")
    return oof


def _train_full_then_predict_te(X_unb, residual, X_te, seed):
    mdl = lgb.LGBMRegressor(**_lgbm_params(seed))
    mdl.fit(X_unb, residual)
    return mdl.predict(X_te).astype(np.float32)


def _extract_causal_parents(graph: np.ndarray, y_col: int) -> tuple[list[int], list[int]]:
    """Return (directed_parents, undirected_neighbours) of y_col.

    causallearn encoding:
        graph[i, j] == -1  -> tail at i on (i, j) edge
        graph[i, j] == +1  -> arrowhead at i on (i, j) edge
        Directed edge i -> j  iff  graph[i, j] == -1 AND graph[j, i] == 1
        Undirected edge i -- j iff graph[i, j] == -1 AND graph[j, i] == -1
    """
    n = graph.shape[0]
    directed_parents = []
    undirected = []
    for i in range(n):
        if i == y_col:
            continue
        if graph[i, y_col] == -1 and graph[y_col, i] == 1:
            directed_parents.append(i)
        elif graph[i, y_col] == -1 and graph[y_col, i] == -1:
            undirected.append(i)
    return directed_parents, undirected


def main():
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- PC algorithm causal feature selection on K=20 + y_unb")
    print(f"        alpha={PC_ALPHA}  indep_test={PC_INDEP_TEST}")
    print(f"        ref nb2171 deep-30 ceiling = {NB2171_REF:.4f}")
    print(f"        gate PROMOTE < {GATE_PROMOTE} / MARGINAL_BEAT < {GATE_MARGINAL}")
    print("=" * 78)

    # ---- load truth, anchor, substrate ----
    te = load_test()
    n_test = len(te)
    te_smiles = (
        te["smiles"].astype(str).tolist()
        if "smiles" in te.columns
        else te["SMILES"].astype(str).tolist()
    )
    te_names = (
        te["name"].values if "name" in te.columns else te["Molecule Name"].values
    )
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    n_unb = len(y_unb)
    print(f"[load] n_test={n_test}  n_unb={n_unb}")

    unb_smiles = [te_smiles[i] for i in unb_idx]
    unb_scaffolds = [bemis_murcko(s) for s in unb_smiles]
    n_unique_scaf = len({s for s in unb_scaffolds if s})
    print(f"[scaffold] unique scaffolds in 253 = {n_unique_scaf}")

    te_anchor_513 = np.load(ANCHOR_TE_PATH).astype(np.float64)
    anchor_unb = te_anchor_513[unb_idx]
    rae_anchor = float(rae(y_unb, anchor_unb))
    residual = y_unb - anchor_unb
    print(
        f"[load] chemprop_aux te[unb_idx] RAE = {rae_anchor:.4f} "
        f"(ref {CHEMPROP_AUX_REF:.4f})  "
        f"resid mean={residual.mean():+.4f} std={residual.std():.4f}"
    )

    # ---- build K=20 from nb2231 RFE surviving idx on 117-col substrate ----
    if not NB2231_SUMMARY.exists():
        raise FileNotFoundError(f"missing {NB2231_SUMMARY}")
    with open(NB2231_SUMMARY) as f:
        nb2231 = json.load(f)
    snap20 = nb2231["snapshots"]["20"]
    k20_idx = list(snap20["surviving_idx_in_117"])
    k20_names = list(snap20["surviving_names"])
    if len(k20_idx) != 20:
        raise ValueError(f"K=20 subset has {len(k20_idx)} indices, expected 20")
    print(f"[load] K=20 surviving idx (20 of 117): {k20_idx[:5]}...{k20_idx[-3:]}")

    if not X117_UNB_PATH.exists() or not X117_TE_PATH.exists():
        raise FileNotFoundError(
            f"missing pyramid substrate: {X117_UNB_PATH} / {X117_TE_PATH}"
        )
    X_unb_117 = np.load(X117_UNB_PATH).astype(np.float32)
    X_te_117 = np.load(X117_TE_PATH).astype(np.float32)
    if X_unb_117.shape != (n_unb, 117) or X_te_117.shape != (n_test, 117):
        raise ValueError(
            f"shape mismatch: X_117_unb {X_unb_117.shape} / X_117_te {X_te_117.shape}"
        )
    X_k20_unb = X_unb_117[:, k20_idx].astype(np.float32)
    X_k20_te = X_te_117[:, k20_idx].astype(np.float32)
    print(f"[substrate] X_K20_unb = {X_k20_unb.shape}  X_K20_te = {X_k20_te.shape}")

    # ---- combine X_K20 + y_unb into (253, 21) ----
    print("\n" + "-" * 78)
    print(f"STEP 1: build joint matrix (253, 21) = K=20 features || y_unb")
    print("-" * 78)
    Z = np.column_stack([X_k20_unb.astype(np.float64), y_unb.astype(np.float64)])
    y_col = 20
    print(f"   Z shape = {Z.shape}  y_col = {y_col}")
    print(
        f"   Z col stats:  feat_mean={Z[:, :20].mean():.3f}  "
        f"feat_std={Z[:, :20].std():.3f}  "
        f"y_mean={Z[:, y_col].mean():.3f}  y_std={Z[:, y_col].std():.3f}"
    )

    # ---- run PC ----
    print("\n" + "-" * 78)
    print(
        f"STEP 2: PC algorithm  alpha={PC_ALPHA}  CI test={PC_INDEP_TEST}"
    )
    print("-" * 78)
    node_names = [f"K{i}_{k20_names[i]}" for i in range(20)] + ["y_unb"]
    t_pc = time.time()
    cg = causal_pc(
        Z,
        alpha=PC_ALPHA,
        indep_test=PC_INDEP_TEST,
        stable=True,
        show_progress=False,
        verbose=False,
        node_names=node_names,
    )
    pc_wall = time.time() - t_pc
    graph = np.asarray(cg.G.graph)
    print(f"   PC finished in {pc_wall:.2f}s  graph shape = {graph.shape}")

    # ---- extract directed parents of y ----
    print("\n" + "-" * 78)
    print(f"STEP 3: extract directed causal parents of y_unb (col {y_col})")
    print("-" * 78)
    directed_parents, undirected = _extract_causal_parents(graph, y_col)
    print(f"   directed parents (i -> y)    : {directed_parents}")
    print(f"   undirected neighbours (i--y) : {undirected}")
    parent_names = [k20_names[i] for i in directed_parents]
    for i, nm in zip(directed_parents, parent_names):
        print(f"      K{i}  ({nm})")

    causal_subset = list(directed_parents)
    used_undirected = False
    if len(causal_subset) == 0:
        # PC found no directed parents; per faithfulness, fall back to all
        # undirected neighbours (Markov blanket subset) so the LGBM is not
        # trained on a 0-column matrix.  If still empty, train on the full
        # K=20 (degenerates to nb2241 substrate).
        if len(undirected) > 0:
            print(
                "   [fallback] no directed parents; using undirected "
                f"neighbours: {undirected}"
            )
            causal_subset = list(undirected)
            used_undirected = True
        else:
            print("   [fallback] no directed AND no undirected; using all K=20")
            causal_subset = list(range(20))

    X_sel_unb = X_k20_unb[:, causal_subset].astype(np.float32)
    X_sel_te = X_k20_te[:, causal_subset].astype(np.float32)
    n_sel = X_sel_unb.shape[1]
    print(f"\n   selected K_causal = {n_sel}  X_sel_unb = {X_sel_unb.shape}")

    # ---- LGBM residual cross-fit on causal subset ----
    print("\n" + "-" * 78)
    print(
        f"STEP 4: LGBM residual  {len(KF_SEEDS)} kf_seeds x {len(BAG_SEEDS)} "
        f"bag_seeds = {len(KF_SEEDS) * len(BAG_SEEDS)} cross-fits"
    )
    print("-" * 78)

    pred_unb_all = np.zeros(
        (len(KF_SEEDS), len(BAG_SEEDS), n_unb), dtype=np.float64
    )
    pred_te_per_bag = np.zeros((len(BAG_SEEDS), n_test), dtype=np.float64)
    per_seed_te_done = np.zeros(len(BAG_SEEDS), dtype=bool)
    per_kf_bagmean_rae = []
    total = len(KF_SEEDS) * len(BAG_SEEDS)
    done = 0
    for k_i, kf_seed in enumerate(KF_SEEDS):
        print(f"\n  --- kf_seed={kf_seed} ----------")
        per_seed_corr_rae = []
        for b_i, b_seed in enumerate(BAG_SEEDS):
            ts = time.time()
            resid_oof = _residual_cross_fit_scaffold(
                X_sel_unb,
                residual,
                unb_scaffolds,
                kf_seed=kf_seed,
                lgbm_seed=b_seed,
            )
            pred_unb_all[k_i, b_i] = anchor_unb + resid_oof
            per_seed_corr_rae.append(
                float(rae(y_unb, anchor_unb + resid_oof))
            )
            if not per_seed_te_done[b_i]:
                te_resid = _train_full_then_predict_te(
                    X_sel_unb, residual, X_sel_te, seed=b_seed,
                )
                pred_te_per_bag[b_i] = te_anchor_513 + te_resid
                per_seed_te_done[b_i] = True
            done += 1
            print(
                f"      kf={kf_seed} bag={b_seed}  "
                f"rae={per_seed_corr_rae[-1]:.4f}  "
                f"wall={time.time() - ts:.2f}s  ({done}/{total})"
            )
        bagmean_pred = pred_unb_all[k_i].mean(axis=0)
        bagmean_rae = float(rae(y_unb, bagmean_pred))
        per_kf_bagmean_rae.append(bagmean_rae)
        print(
            f"   kf_seed={kf_seed}  5-bag-mean POOLED RAE = {bagmean_rae:.4f}"
        )

    per_kf_arr = np.array(per_kf_bagmean_rae, dtype=np.float64)
    mean_rae = float(per_kf_arr.mean())
    std_rae = float(per_kf_arr.std(ddof=1))
    median_rae = float(np.median(per_kf_arr))
    min_rae = float(per_kf_arr.min())
    max_rae = float(per_kf_arr.max())

    print("\n" + "-" * 78)
    print("STEP 5: multi-kf statistics")
    print("-" * 78)
    print("   per-kf 5-bag-mean RAE:")
    for kf, r in zip(KF_SEEDS, per_kf_bagmean_rae):
        print(f"     kf={kf}  RAE={r:.4f}")
    print(f"\n   mean +/- std  = {mean_rae:.4f} +/- {std_rae:.5f}")
    print(f"   median        = {median_rae:.4f}")
    print(f"   min / max     = {min_rae:.4f} / {max_rae:.4f}")
    print(f"   anchor RAE                = {rae_anchor:.4f}")
    print(f"   delta vs anchor           = {mean_rae - rae_anchor:+.4f}")
    print(f"   delta vs nb2171 (0.4682)  = {mean_rae - NB2171_REF:+.4f}")

    # ---- deploy ----
    print("\n" + "-" * 78)
    print("STEP 6: deploy artifacts")
    print("-" * 78)
    pred_oof_unb = pred_unb_all.reshape(-1, n_unb).mean(axis=0)
    pred_te_513 = pred_te_per_bag.mean(axis=0)
    deploy_pooled_rae = float(rae(y_unb, pred_oof_unb))
    te_unb_in_rae = float(rae(y_unb, pred_te_513[unb_idx]))
    print(
        f"   deploy bag-mean (5kf x 5bag) OOF pooled RAE = "
        f"{deploy_pooled_rae:.4f}"
    )
    print(
        f"   te[unb_idx] in-sample RAE                   = "
        f"{te_unb_in_rae:.4f}"
    )
    print(
        f"   pred_oof_unb std = {pred_oof_unb.std():.3f} "
        f"(truth_std {y_unb.std():.3f})"
    )
    print(
        f"   pred_te_513 mean/std = {pred_te_513.mean():.3f}/"
        f"{pred_te_513.std():.3f}"
    )

    # ---- gate ----
    print("\n" + "-" * 78)
    print("STEP 7: GATE")
    print("-" * 78)
    if mean_rae < GATE_PROMOTE:
        verdict = "PROMOTE"
    elif mean_rae < GATE_MARGINAL:
        verdict = "MARGINAL_BEAT"
    else:
        verdict = "FAIL"
    print(f"   mean_rae = {mean_rae:.4f}")
    print(f"     <{GATE_PROMOTE} -> PROMOTE")
    print(f"     <{GATE_MARGINAL} -> MARGINAL_BEAT")
    print(f"     else            -> FAIL")
    print(f"   -> {verdict}")

    # ---- save ----
    print("\n" + "-" * 78)
    print("STEP 8: save artifacts")
    print("-" * 78)
    oof_path = DATA_PROCESSED / f"{TAG}_pred_oof.npy"
    te_path = DATA_PROCESSED / f"te_{TAG}.npy"
    np.save(oof_path, pred_oof_unb.astype(np.float32))
    np.save(te_path, pred_te_513.astype(np.float32))
    print(f"   [save] {oof_path}")
    print(f"   [save] {te_path}")

    sub_csv = SUBMISSIONS / f"{TAG}_pc_causal_selection.csv"
    pd.DataFrame(
        {
            "SMILES": te_smiles,
            "Molecule Name": te_names,
            "pEC50": pred_te_513.astype(np.float32),
        }
    ).to_csv(sub_csv, index=False)
    print(f"   [save] {sub_csv}")

    summary = {
        "tag": TAG,
        "parent_tag": PARENT_TAG,
        "method": "pc_causal_parent_subset_then_lgbm_residual_on_chemprop_aux",
        "rationale": (
            "PC CPDAG isolates directed-parent features from spurious "
            "correlational neighbours; compare against K=20 raw substrate"
        ),
        "anchor_base": "chemprop_aux",
        "anchor_in_rae": rae_anchor,
        "anchor_pre_unblind": True,
        "pc_alpha": PC_ALPHA,
        "pc_indep_test": PC_INDEP_TEST,
        "pc_wall_sec": round(pc_wall, 3),
        "k20_idx_in_117": [int(j) for j in k20_idx],
        "k20_names": k20_names,
        "directed_parents_idx_in_K20": [int(j) for j in directed_parents],
        "directed_parents_names": parent_names,
        "undirected_neighbours_idx_in_K20": [int(j) for j in undirected],
        "used_undirected_fallback": bool(used_undirected),
        "n_selected_causal": int(n_sel),
        "selected_idx_in_K20": [int(j) for j in causal_subset],
        "kf_seeds": KF_SEEDS,
        "bag_seeds": BAG_SEEDS,
        "n_kf": len(KF_SEEDS),
        "n_bag": len(BAG_SEEDS),
        "n_total_fits": len(KF_SEEDS) * len(BAG_SEEDS),
        "resid_folds": RESID_FOLDS,
        "n_unb": int(n_unb),
        "n_te": int(n_test),
        "n_unique_scaffolds": int(n_unique_scaf),
        "per_kf_bagmean_rae": per_kf_bagmean_rae,
        "mean_rae": mean_rae,
        "std_rae": std_rae,
        "median_rae": median_rae,
        "min_rae": min_rae,
        "max_rae": max_rae,
        "deploy_pooled_rae": deploy_pooled_rae,
        "te_unb_in_sample_rae": te_unb_in_rae,
        "te_mean": float(pred_te_513.mean()),
        "te_std": float(pred_te_513.std()),
        "pred_oof_path": str(oof_path),
        "te_npy_path": str(te_path),
        "submission_csv": str(sub_csv),
        "nb2171_ref": NB2171_REF,
        "chemprop_aux_ref": CHEMPROP_AUX_REF,
        "delta_vs_nb2171": mean_rae - NB2171_REF,
        "delta_vs_anchor": mean_rae - rae_anchor,
        "gate_promote": GATE_PROMOTE,
        "gate_marginal": GATE_MARGINAL,
        "verdict": verdict,
        "wall_sec": round(time.time() - t0, 2),
    }
    out_path = DATA_PROCESSED / f"{TAG}_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   [save] {out_path}")

    print("\n" + "=" * 78)
    print(f"=== {TAG} SUMMARY ===")
    print(f"   PC alpha = {PC_ALPHA}  CI = {PC_INDEP_TEST}")
    print(f"   directed parents (K_causal) = {n_sel}  "
          f"(of K=20)  idx_in_K20 = {causal_subset}")
    print(
        f"   per-kf 5-bag-mean = "
        + ", ".join(f"{r:.4f}" for r in per_kf_bagmean_rae)
    )
    print(f"   mean +/- std      = {mean_rae:.4f} +/- {std_rae:.5f}")
    print(f"   delta vs nb2171   = {mean_rae - NB2171_REF:+.4f}")
    print(f"   verdict           = {verdict}")
    print(f"   wall              = {time.time() - t0:.1f}s")
    print("=" * 78)
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in (
        "n_selected_causal",
        "mean_rae",
        "std_rae",
        "min_rae",
        "max_rae",
        "deploy_pooled_rae",
        "te_unb_in_sample_rae",
        "delta_vs_nb2171",
        "verdict",
    ):
        print(f"  {k}: {res.get(k)}")
