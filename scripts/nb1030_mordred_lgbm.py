"""nb1030 -- Mordred 1800-descriptor LightGBM Huber baseline.

Hypothesis: Mordred descriptors (~1613 numeric, 2D-only) span a feature axis
that is orthogonal to Morgan FP + RDKit Descriptors used by nb972. If
Pearson(te_nb1030, te_nb972) < 0.95, the two predictors carry distinct
information and a multi-seed bag of (chemprop_aux, nb1030) may beat the
nb1014 (chemprop_aux, nb972) bag.

Recipe:
  - Mordred Calculator (ignore_3D=True) -> ~1613 raw descriptors on
    4139 CRC train + 513 test. Coerce object/bool/inf -> NaN, median impute
    across the full 4652 panel (no train/test leak: median, not mean
    embedding).
  - LightGBM Huber alpha=2.0, n_estimators=2000, num_leaves=128, lr=0.025,
    scaffold 5-fold CV with early stopping (patience 200) on the held-out
    fold; mean best_iter rolled into a full-data final fit.
  - Predict 513 test. in_RAE on 253 Phase-1 unblind. Pearson vs te_nb972.
  - If Pearson < 0.95: re-run the nb1014 multi-seed bag with the pool
    (chemprop_aux, nb1030) and report cross-fit RAE + deploy submission.

Outputs:
  C:/pxr_artifacts/nb1030/X_mordred_train.npy
  C:/pxr_artifacts/nb1030/X_mordred_test.npy
  C:/pxr_artifacts/nb1030/feature_names.json
  data/processed/oof_nb1030.npy
  data/processed/te_nb1030.npy
  data/processed/nb1030_summary.json
  submissions/nb1030_mordred_lgbm.csv
  (conditional) submissions/nb1030_nb1014bag_chemprop_aux.csv
                data/processed/te_nb1030_nb1014bag.npy
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

from pathlib import Path

import numpy as np
import pandas as pd

# numpy 2.x compat patch for mordred 0.6 / mordredcommunity
if not hasattr(np, "product"):
    np.product = np.prod  # type: ignore[attr-defined]

import lightgbm as lgb
from rdkit import Chem
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from pxr.chem import bemis_murcko
from pxr.data import load_test, load_train
from pxr.eval import rae, scaffold_kfold_indices
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

TAG = "nb1030"
ARTIFACT_DIR = Path("C:/pxr_artifacts/nb1030")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5

LGB_PARAMS = dict(
    objective="huber",
    alpha=2.0,
    n_estimators=2000,
    learning_rate=0.025,
    num_leaves=128,
    min_child_samples=20,
    reg_lambda=0.2,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
    n_jobs=4,
)
EARLY_STOP = 200

# nb1014 protocol constants
STRETCH_GRID = np.round(np.arange(1.00, 2.001, 0.05), 2).tolist()
SEEDS = [0, 1, 7, 42, 137]
NB1014_REF_RAE = 0.5994  # nb1001 seed-42 pooled, nb1014 ref


# ---------- Mordred backend ----------

def compute_mordred(smiles: list[str], n_proc: int = 4) -> tuple[np.ndarray, list[str], dict]:
    """Compute mordredcommunity descriptors. Returns (X, names, meta)."""
    notes: dict = {"backend": "mordred", "n_proc": n_proc}
    try:
        from mordred import Calculator, descriptors
    except Exception as e:
        notes["backend"] = "rdkit_fallback"
        notes["mordred_import_error"] = repr(e)
        return _rdkit_fallback(smiles, notes)

    try:
        calc = Calculator(descriptors, ignore_3D=True)
        notes["mordred_n_desc"] = len(calc.descriptors)
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        # mol may be None for unparseable -- replace with placeholder
        valid_mask = np.array([m is not None for m in mols])
        for i, m in enumerate(mols):
            if m is None:
                mols[i] = Chem.MolFromSmiles("C")  # placeholder; row will be NaN'd
        df = calc.pandas(mols, nproc=n_proc, quiet=True)
        names = list(df.columns)
        # Coerce all columns to numeric float; mordred returns object cols
        # (Missing/Error wrappers) and bool cols
        X = np.full(df.shape, np.nan, dtype=np.float64)
        for j, col in enumerate(names):
            col_vals = df[col].values
            # vectorize: try float cast, NaN on failure
            arr = np.empty(len(col_vals), dtype=np.float64)
            for i, v in enumerate(col_vals):
                try:
                    fv = float(v)
                    if not np.isfinite(fv):
                        fv = np.nan
                    arr[i] = fv
                except Exception:
                    arr[i] = np.nan
            X[:, j] = arr
        # Mark unparseable rows as fully NaN (will pick up training median)
        if not valid_mask.all():
            X[~valid_mask, :] = np.nan
        notes["valid_smiles"] = int(valid_mask.sum())
        notes["raw_shape"] = list(X.shape)
        return X, names, notes
    except Exception as e:
        notes["backend"] = "rdkit_fallback"
        notes["mordred_runtime_error"] = repr(e)
        return _rdkit_fallback(smiles, notes)


def _rdkit_fallback(smiles: list[str], notes: dict) -> tuple[np.ndarray, list[str], dict]:
    """RDKit descriptors (~217) + ring fusion + E-state (~300 dim)."""
    from rdkit.Chem import Descriptors, EState, GraphDescriptors, rdMolDescriptors

    names = [n for n, _ in Descriptors._descList]
    fns = [f for _, f in Descriptors._descList]
    extras = [
        "n_arom_rings", "n_sat_rings", "n_aliph_rings", "n_hetero_rings",
        "n_spiro", "n_bridge", "n_fused_pairs", "n_macrocycles",
        "balaban_j", "bertz_ct",
    ]
    estate_idx_count = 79  # EState fingerprint length
    estate_names = [f"estate_{i}" for i in range(estate_idx_count)]
    estate_sum_names = [f"estate_sum_{i}" for i in range(estate_idx_count)]
    all_names = names + extras + estate_names + estate_sum_names

    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is None:
            rows.append([np.nan] * len(all_names))
            continue
        try:
            base = [fn(mol) for fn in fns]
        except Exception:
            base = [np.nan] * len(fns)
        # ring fusion stats
        try:
            ri = mol.GetRingInfo()
            bond_rings = ri.BondRings()
            n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
            n_sat = rdMolDescriptors.CalcNumSaturatedRings(mol)
            n_aliph = rdMolDescriptors.CalcNumAliphaticRings(mol)
            n_het = rdMolDescriptors.CalcNumHeteroatoms(mol)
            n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
            n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
            # fused = shared bond rings
            n_fused = 0
            for i in range(len(bond_rings)):
                for j in range(i + 1, len(bond_rings)):
                    if set(bond_rings[i]) & set(bond_rings[j]):
                        n_fused += 1
            n_macro = sum(1 for r in ri.AtomRings() if len(r) >= 12)
            bj = GraphDescriptors.BalabanJ(mol)
            bct = GraphDescriptors.BertzCT(mol)
            extras_v = [n_arom, n_sat, n_aliph, n_het, n_spiro,
                        n_bridge, n_fused, n_macro, bj, bct]
        except Exception:
            extras_v = [np.nan] * len(extras)
        try:
            es_idx, es_sum = EState.Fingerprinter.FingerprintMol(mol)
            es_idx = list(es_idx) + [0.0] * (estate_idx_count - len(es_idx))
            es_sum = list(es_sum) + [0.0] * (estate_idx_count - len(es_sum))
            es_idx = es_idx[:estate_idx_count]
            es_sum = es_sum[:estate_idx_count]
        except Exception:
            es_idx = [0.0] * estate_idx_count
            es_sum = [0.0] * estate_idx_count
        rows.append(base + extras_v + list(es_idx) + list(es_sum))

    X = np.array(rows, dtype=np.float64)
    notes["fallback_n_features"] = len(all_names)
    notes["raw_shape"] = list(X.shape)
    return X, all_names, notes


def median_impute_inf(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replace +-inf with NaN, drop fully-NaN columns, median-impute remainder."""
    X = X.astype(np.float64)
    X[~np.isfinite(X)] = np.nan
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep]
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isnan(col_med), 0.0, col_med)
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            col[nan_mask] = col_med[j]
            X[:, j] = col
    return X.astype(np.float32), keep


# ---------- nb1014 bag protocol ----------

def slsqp_w0(p0: np.ndarray, p1: np.ndarray, y: np.ndarray) -> float:
    P = np.column_stack([p0, p1])
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0.0, 1.0), (0.0, 1.0)]
    res = minimize(
        lambda w: float(np.sum((P @ w - y) ** 2)),
        np.array([0.5, 0.5]),
        method="SLSQP", bounds=bnds, constraints=cons,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    return float(res.x[0])


def best_stretch_on(blend_train: np.ndarray, y_train: np.ndarray,
                    mu: float) -> tuple[float, float]:
    best_s, best_r = 1.0, float("inf")
    for s in STRETCH_GRID:
        stretched = mu + s * (blend_train - mu)
        r = float(rae(y_train, stretched))
        if r < best_r:
            best_r = r
            best_s = float(s)
    return best_s, best_r


def run_one_seed(P_unb: np.ndarray, y_unb: np.ndarray, seed: int) -> dict:
    n_unb = len(y_unb)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.full(n_unb, np.nan)
    folds = []
    for k, (tr_loc, va_loc) in enumerate(kf.split(np.arange(n_unb))):
        w0_f = slsqp_w0(P_unb[tr_loc, 0], P_unb[tr_loc, 1], y_unb[tr_loc])
        blend_tr = w0_f * P_unb[tr_loc, 0] + (1.0 - w0_f) * P_unb[tr_loc, 1]
        mu_tr = float(blend_tr.mean())
        s_f, rae_tr = best_stretch_on(blend_tr, y_unb[tr_loc], mu_tr)
        blend_va = w0_f * P_unb[va_loc, 0] + (1.0 - w0_f) * P_unb[va_loc, 1]
        oof[va_loc] = mu_tr + s_f * (blend_va - mu_tr)
        folds.append({"fold": k, "w0": w0_f, "s": s_f, "mu_tr": mu_tr,
                      "n_va": int(len(va_loc))})
    return {"seed": seed, "folds": folds,
            "pooled_rae": float(rae(y_unb, oof)), "oof": oof}


def run_nb1014_bag(te_nb1030: np.ndarray, te_names: np.ndarray,
                    te_smiles: np.ndarray, unb_idx: np.ndarray,
                    y_unb: np.ndarray) -> dict:
    """Mirror of nb1014 protocol with pool = (chemprop_aux, nb1030)."""
    print("\n" + "-" * 78)
    print("NB1014-STYLE BAG  pool = (chemprop_aux, nb1030)")
    print("-" * 78)
    te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
    preds_513 = np.column_stack([te_chemprop, te_nb1030.astype(np.float64)])
    P_unb = preds_513[unb_idx]
    per_seed_rae, all_w0, all_s, seed_results = [], [], [], []
    for seed in SEEDS:
        res = run_one_seed(P_unb, y_unb, seed)
        per_seed_rae.append(res["pooled_rae"])
        for f in res["folds"]:
            all_w0.append(f["w0"]); all_s.append(f["s"])
        seed_results.append({"seed": seed, "pooled_rae": res["pooled_rae"]})
        print(f"   seed {seed:>3d}: pooled_RAE = {res['pooled_rae']:.4f}")
    mean_rae = float(np.mean(per_seed_rae))
    std_rae = float(np.std(per_seed_rae))
    mean_w0 = float(np.mean(all_w0))
    mean_s = float(np.mean(all_s))
    blend_unb_all = mean_w0 * P_unb[:, 0] + (1.0 - mean_w0) * P_unb[:, 1]
    mu_deploy = float(blend_unb_all.mean())
    in_rae_bag = float(rae(y_unb, mu_deploy + mean_s * (blend_unb_all - mu_deploy)))
    blend_513 = mean_w0 * preds_513[:, 0] + (1.0 - mean_w0) * preds_513[:, 1]
    deploy_513 = (mu_deploy + mean_s * (blend_513 - mu_deploy)).astype(np.float32)
    print(f"\n[bag] mean pooled CV RAE = {mean_rae:.4f}  (std {std_rae:.4f})")
    print(f"[bag] deploy w0={mean_w0:.4f}  s={mean_s:.4f}  mu={mu_deploy:.4f}")
    print(f"[bag] in-sample 253 RAE  = {in_rae_bag:.4f}")
    np.save(DATA_PROCESSED / f"te_{TAG}_nb1014bag.npy", deploy_513)
    sub_path = SUBMISSIONS / f"{TAG}_nb1014bag_chemprop_aux.csv"
    pd.DataFrame({"SMILES": te_smiles, "Molecule Name": te_names,
                  "pEC50": deploy_513}).to_csv(sub_path, index=False)
    print(f"[save] {sub_path}")
    return {
        "ran": True,
        "per_seed_pooled_rae": per_seed_rae,
        "mean_pooled_rae": mean_rae,
        "std_pooled_rae": std_rae,
        "mean_w0_chemprop_aux": mean_w0,
        "mean_w1_nb1030": float(1.0 - mean_w0),
        "mean_s": mean_s,
        "deploy_mu_blend": mu_deploy,
        "in_sample_rae_overfit_bound": in_rae_bag,
        "submission": str(sub_path),
        "delta_vs_nb1014_ref": mean_rae - NB1014_REF_RAE,
        "beats_nb1014": mean_rae < NB1014_REF_RAE - 0.005,
    }


# ---------- main ----------

def main() -> dict:
    t0 = time.time()
    print("=" * 78)
    print(f"{TAG} -- Mordred 1800-dim LGBM Huber + nb1014-style bag")
    print("=" * 78)

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr); n_te = len(te)
    print(f"[load] n_train={n_tr}  n_test={n_te}")

    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, SEED)

    # ---- Mordred (train + test in single pass) ----
    smis_all = tr["smiles"].tolist() + te["smiles"].tolist()
    print(f"\n[mordred] computing descriptors on {len(smis_all)} mols "
          f"(nproc=4)...")
    t_md = time.time()
    X_all, names, mord_notes = compute_mordred(smis_all, n_proc=4)
    print(f"[mordred] backend={mord_notes['backend']} raw_shape={X_all.shape} "
          f"elapsed={time.time()-t_md:.1f}s")

    # Drop all-NaN cols, median impute, replace inf
    X_all, keep = median_impute_inf(X_all)
    kept_names = [n for n, k in zip(names, keep) if k]
    print(f"[impute] post-impute shape={X_all.shape} (dropped {sum(~keep)} all-NaN cols)")

    X_tr = X_all[:n_tr]
    X_te = X_all[n_tr:]
    np.save(ARTIFACT_DIR / "X_mordred_train.npy", X_tr)
    np.save(ARTIFACT_DIR / "X_mordred_test.npy", X_te)
    with open(ARTIFACT_DIR / "feature_names.json", "w") as f:
        json.dump({"backend": mord_notes["backend"],
                    "n_features": int(X_tr.shape[1]),
                    "names": kept_names, "notes": mord_notes}, f, indent=2)

    # ---- Scaffold CV ----
    oof = np.full(n_tr, np.nan)
    best_iters, fold_raes = [], []
    print(f"\n[cv] LGBM Huber a=2.0 n_est=2000 nl=128 lr=0.025  "
          f"({N_FOLDS}-fold scaffold)")
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.train(
            LGB_PARAMS,
            lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx]),
            valid_sets=[lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx])],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(-1)],
        )
        oof[va_idx] = m.predict(X_tr[va_idx], num_iteration=m.best_iteration)
        fr = rae(y_tr[va_idx], oof[va_idx])
        fold_raes.append(fr)
        best_iters.append(int(m.best_iteration or LGB_PARAMS["n_estimators"]))
        print(f"   fold {fold+1}  best_iter={best_iters[-1]:5d}  RAE={fr:.4f}  "
              f"elapsed={time.time()-t0:6.1f}s", flush=True)
    oof_rae = float(rae(y_tr, oof))
    mean_best = int(np.mean(best_iters))
    print(f"[cv] OOF RAE = {oof_rae:.4f}  mean_best_iter = {mean_best}")

    # ---- Final fit on all train ----
    print(f"\n[final] full-train fit, n_est={mean_best}...")
    final_params = dict(LGB_PARAMS, n_estimators=mean_best)
    m_final = lgb.train(final_params, lgb.Dataset(X_tr, label=y_tr),
                        callbacks=[lgb.log_evaluation(-1)])
    te_preds = np.clip(m_final.predict(X_te),
                        y_tr.min() - 0.5, y_tr.max() + 0.5).astype(np.float32)

    # ---- in_RAE on 253 ----
    unb_idx = np.load(DATA_PROCESSED / "_audit_unblind_idx.npy")
    y_unb = np.load(DATA_PROCESSED / "_audit_unblind_y.npy").astype(np.float64)
    in_r = float(rae(y_unb, te_preds[unb_idx].astype(np.float64)))
    print(f"[deploy] te mean/std = {te_preds.mean():.3f}/{te_preds.std():.3f}  "
          f"in_RAE(253) = {in_r:.4f}")

    # ---- Pearson vs nb972 ----
    te_nb972 = np.load(DATA_PROCESSED / "te_nb972_long_train.npy").astype(np.float64)
    pearson_972 = float(np.corrcoef(te_preds.astype(np.float64), te_nb972)[0, 1])
    print(f"[corr] Pearson(te_nb1030, te_nb972) = {pearson_972:.4f}")
    try:
        te_chemprop = np.load(DATA_PROCESSED / "te_chemprop_aux.npy").astype(np.float64)
        pearson_cp = float(np.corrcoef(te_preds.astype(np.float64), te_chemprop)[0, 1])
        print(f"[corr] Pearson(te_nb1030, te_chemprop_aux) = {pearson_cp:.4f}")
    except FileNotFoundError:
        pearson_cp = None

    # ---- Save base submission ----
    np.save(DATA_PROCESSED / f"oof_{TAG}.npy", oof)
    np.save(DATA_PROCESSED / f"te_{TAG}.npy", te_preds)
    base_sub = SUBMISSIONS / f"{TAG}_mordred_lgbm.csv"
    pd.DataFrame({"SMILES": te["smiles"].values,
                  "Molecule Name": te["name"].values,
                  "pEC50": te_preds}).to_csv(base_sub, index=False)
    print(f"[save] te_{TAG}.npy, oof_{TAG}.npy, {base_sub}")

    # ---- Conditional bag ----
    bag_summary = {"ran": False, "reason": f"pearson_{pearson_972:.4f}_>=_0.95"}
    if pearson_972 < 0.95:
        bag_summary = run_nb1014_bag(te_preds, te["name"].values,
                                      te["smiles"].values, unb_idx, y_unb)
    else:
        print(f"\n[skip] Pearson {pearson_972:.4f} >= 0.95 -- skip nb1014 bag")

    summary = {
        "tag": TAG,
        "mordred_notes": mord_notes,
        "n_features_kept": int(X_tr.shape[1]),
        "lgb_params": {k: v for k, v in LGB_PARAMS.items() if k != "verbose"},
        "fold_best_iters": best_iters,
        "fold_raes": [float(x) for x in fold_raes],
        "mean_best_iter": mean_best,
        "oof_rae": oof_rae,
        "in_rae_253": in_r,
        "test_mean": float(te_preds.mean()),
        "test_std": float(te_preds.std()),
        "pearson_nb972": pearson_972,
        "pearson_chemprop_aux": pearson_cp,
        "base_submission": str(base_sub),
        "nb1014_bag": bag_summary,
        "wall_sec": round(time.time() - t0, 2),
    }
    with open(DATA_PROCESSED / f"{TAG}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {DATA_PROCESSED / f'{TAG}_summary.json'}")
    print(f"\n=== {TAG} done in {time.time()-t0:.1f}s ===")
    return summary


if __name__ == "__main__":
    res = main()
    print("\n==== SUMMARY ====")
    for k in ("n_features_kept", "oof_rae", "in_rae_253",
              "pearson_nb972", "pearson_chemprop_aux",
              "base_submission"):
        print(f"  {k}: {res.get(k)}")
    bag = res.get("nb1014_bag", {})
    if bag.get("ran"):
        print("  bag.mean_pooled_rae:", bag.get("mean_pooled_rae"))
        print("  bag.mean_w0_chemprop_aux:", bag.get("mean_w0_chemprop_aux"))
        print("  bag.mean_s:", bag.get("mean_s"))
        print("  bag.beats_nb1014:", bag.get("beats_nb1014"))
        print("  bag.submission:", bag.get("submission"))
    else:
        print("  bag.ran: False  (reason:", bag.get("reason"), ")")
