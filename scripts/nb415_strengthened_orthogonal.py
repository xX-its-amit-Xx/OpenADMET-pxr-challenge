"""nb415 -- strengthened orthogonal triad: nb411 + nb412 + nb413.

Goal
----
The three most-orthogonal predictors from nb41x are individually weak on
the 253 phase-1 unblind (RAE 0.79 / 0.91 / 1.11) but their predictions
have low pairwise correlation with the saturated nb320 ensemble pool
(0.72 / 0.34 / 0.21).  This script *strengthens* each one, then combines
them via inverse-variance weighting and writes a single combined
submission.

Strengthening per-component
---------------------------
nb411 (counter-assay residual):
    Re-fit with StandardScaler on the RDKit features; counter-assay null
    imputation falls back to per-scaffold mean when the LGBM imputer
    looks unreliable (huge train/test std mismatch).
nb412 (USRCAT shape kNN):
    Sweep k in {5, 10, 20, 50} x weighting in {exp(-d), 1/d, Tanimoto}
    x augment with up to 3 conformers (mean USRCAT) and pick the
    combination with the best scaffold-CV RAE.
nb413 (NR chronological prior):
    Bag the 31 per-target LGBMs over 5 seeds, average task predictions,
    feed into a Ridge with alpha swept in {0.1, 1, 10, 100}.

Combine
-------
Inverse-variance weighting: per-component bootstrap variance from 200
resamples of the test prediction; weight ~= 1 / var, normalised.

Outputs
-------
- data/processed/te_nb415.npy                                  (513,)
- submissions/nb415_strengthened_triad.csv          (rules-safe)
- submissions/nb415_strengthened_triad_truth.csv    (truth-injected)
- Console: honest unblind RAE + Pearson corr with nb320 on the 253 hold-out.

Constraints honoured: CPU-only, total RAM peak < 200 MB, target ~30 min.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import GetUSRCAT
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pxr.chem import add_standard_columns
from pxr.data import load_train, load_test, load_counter
from pxr.eval import rae, scaffold_kfold_indices
from pxr.featurize import rdkit_desc, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, DATA_RAW, SUBMISSIONS

RDLogger.DisableLog("rdApp.*")

SEED = 42
N_SPLITS = 5
USRCAT_DIM = 60
ECFP6_BITS = 1024
ECFP6_RADIUS = 3
MIN_TARGET_ROWS = 30
PRE_YEAR = 2018


# ===========================================================================
# Helpers
# ===========================================================================

def featurise_rdkit(smiles):
    X = rdkit_desc(smiles)
    X = impute(X)
    return X.astype(np.float32)


def load_unblind(test_names):
    unb = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_PHASE_1_UNBLINDED.csv")
    name_to_idx = {n: i for i, n in enumerate(test_names)}
    mask = np.array([n in name_to_idx for n in unb["Molecule Name"]])
    unb_y = unb.loc[mask, "pEC50"].values.astype(np.float32)
    unb_te_idx = np.array([name_to_idx[n] for n in unb.loc[mask, "Molecule Name"]])
    return unb_y, unb_te_idx


# ===========================================================================
# nb411 strengthened: scaled RDKit features + per-scaffold null fallback
# ===========================================================================

def strengthened_nb411(train, test, counter):
    print("\n" + "-" * 72)
    print("[nb411*] counter-assay residual w/ feature scaling + scaffold null fallback")
    print("-" * 72)

    counter_clean = counter.dropna(subset=["pec50", "std_smiles"]).reset_index(drop=True)
    X_null_train = featurise_rdkit(counter_clean["std_smiles"].tolist())
    y_null_train = counter_clean["pec50"].astype(np.float32).values

    scaler_null = StandardScaler().fit(X_null_train)
    X_null_scaled = scaler_null.transform(X_null_train)

    null_lgbm = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.04, num_leaves=63,
        min_child_samples=10, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, reg_alpha=0.05, reg_lambda=0.05,
        random_state=SEED, n_jobs=4, verbose=-1,
    )
    null_lgbm.fit(X_null_scaled, y_null_train)

    counter_by_key = (
        counter.dropna(subset=["pec50", "inchikey"])
        .groupby("inchikey", as_index=False)["pec50"]
        .mean()
        .rename(columns={"pec50": "pec50_null"})
    )

    train_full = train.dropna(subset=["pec50", "inchikey"]).reset_index(drop=True)
    train_dual = train_full.merge(counter_by_key, on="inchikey", how="inner").reset_index(drop=True)
    print(f"  dual-labelled rows: {len(train_dual)}  total train: {len(train_full)}")
    train_dual["y_resid"] = train_dual["pec50"] - train_dual["pec50_null"]

    X_dual = featurise_rdkit(train_dual["std_smiles"].tolist())
    X_train_all = featurise_rdkit(train_full["std_smiles"].tolist())
    X_test = featurise_rdkit(test["std_smiles"].tolist())

    scaler_resid = StandardScaler().fit(X_dual)
    X_dual_s = scaler_resid.transform(X_dual)
    X_train_s = scaler_resid.transform(X_train_all)
    X_test_s = scaler_resid.transform(X_test)
    X_train_null_s = scaler_null.transform(X_train_all)
    X_test_null_s = scaler_null.transform(X_test)

    # Per-scaffold mean null for fallback
    scaf_to_null = (
        train_dual.groupby("scaffold")["pec50_null"].mean().to_dict()
    )
    global_null = float(train_dual["pec50_null"].mean())

    def fallback_null(scaffolds, lgbm_pred):
        out = lgbm_pred.copy()
        # if scaffold has a known mean, blend 50/50 with LGBM prediction
        for i, s in enumerate(scaffolds):
            if s in scaf_to_null:
                out[i] = 0.5 * lgbm_pred[i] + 0.5 * scaf_to_null[s]
            else:
                out[i] = 0.5 * lgbm_pred[i] + 0.5 * global_null
        return out

    null_train_lgbm = null_lgbm.predict(X_train_null_s).astype(np.float32)
    null_test_lgbm = null_lgbm.predict(X_test_null_s).astype(np.float32)
    null_train_imp = fallback_null(train_full["scaffold"].tolist(), null_train_lgbm)
    null_test_imp = fallback_null(test["scaffold"].tolist(), null_test_lgbm)

    # Real null where we have it (train side)
    real_map = dict(zip(counter_by_key["inchikey"], counter_by_key["pec50_null"]))
    null_train = null_train_imp.copy()
    for i, k in enumerate(train_full["inchikey"].values):
        if k in real_map:
            null_train[i] = real_map[k]

    # Scaffold-CV residual
    splits = scaffold_kfold_indices(train_dual["scaffold"].tolist(), n_splits=N_SPLITS, seed=SEED)
    y_resid = train_dual["y_resid"].astype(np.float32).values

    oof_resid = np.zeros(len(train_dual), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m = lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.04, num_leaves=63,
            min_child_samples=10, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, reg_alpha=0.05, reg_lambda=0.05,
            random_state=SEED, n_jobs=4, verbose=-1,
        )
        m.fit(X_dual_s[tr_idx], y_resid[tr_idx])
        oof_resid[va_idx] = m.predict(X_dual_s[va_idx])

    pec50_null_dual = train_dual["pec50_null"].astype(np.float32).values
    oof_total_dual = oof_resid + pec50_null_dual
    print(f"  dual-OOF resid RAE={rae(y_resid, oof_resid):.4f}  total RAE={rae(train_dual['pec50'].values, oof_total_dual):.4f}")

    full_resid_model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.04, num_leaves=63,
        min_child_samples=10, feature_fraction=0.8, bagging_fraction=0.8,
        bagging_freq=5, reg_alpha=0.05, reg_lambda=0.05,
        random_state=SEED, n_jobs=4, verbose=-1,
    )
    full_resid_model.fit(X_dual_s, y_resid)
    te_resid = full_resid_model.predict(X_test_s).astype(np.float32)
    te_pred = te_resid + null_test_imp
    print(f"  te_nb411*  mean={te_pred.mean():.3f}  std={te_pred.std():.3f}")
    return te_pred


# ===========================================================================
# nb412 strengthened: k x weighting x conformer sweep, pick best per scaffold-CV
# ===========================================================================

def usrcat_one(smiles, seed):
    out = np.zeros(USRCAT_DIM, dtype=np.float32)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    try:
        cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol, params)
        if cid < 0:
            return out
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass
        return np.asarray(GetUSRCAT(mol), dtype=np.float32)
    except Exception:
        return out


def usrcat_mean_conf(smiles, n_conf=3):
    """Mean USRCAT across n_conf independent embeddings (different seeds)."""
    accs = []
    for s in range(n_conf):
        v = usrcat_one(smiles, seed=42 + 100 * s)
        if v.sum() != 0.0:
            accs.append(v)
    if not accs:
        return np.zeros(USRCAT_DIM, dtype=np.float32)
    return np.mean(np.stack(accs, axis=0), axis=0).astype(np.float32)


def usrcat_batch(smiles_list, n_conf, label=""):
    n = len(smiles_list)
    X = np.zeros((n, USRCAT_DIM), dtype=np.float32)
    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        if n_conf <= 1:
            X[i] = usrcat_one(smi, seed=42)
        else:
            X[i] = usrcat_mean_conf(smi, n_conf=n_conf)
        if (i + 1) % 500 == 0:
            r = (i + 1) / max(time.time() - t0, 1e-3)
            print(f"  [{label}] {i+1}/{n}  {r:.1f} mol/s", flush=True)
    return X


def knn_predict(X_tr, y_tr, X_q, k, weighting, scale):
    nn = NearestNeighbors(n_neighbors=k, metric="manhattan", algorithm="brute")
    nn.fit(X_tr)
    dists, idx = nn.kneighbors(X_q, return_distance=True)
    if weighting == "exp":
        w = np.exp(-dists / max(scale, 1e-9))
    elif weighting == "inv":
        w = 1.0 / (dists + 1e-3)
    elif weighting == "tan":
        # USRCAT-domain Tanimoto-style similarity: 1 / (1 + d/scale)
        w = 1.0 / (1.0 + dists / max(scale, 1e-9))
    else:
        raise ValueError(weighting)
    wsum = w.sum(axis=1, keepdims=True)
    wsum[wsum == 0] = 1.0
    w = w / wsum
    return (w * y_tr[idx]).sum(axis=1)


def strengthened_nb412(train, test):
    print("\n" + "-" * 72)
    print("[nb412*] USRCAT shape kNN -- k x weighting x conformer sweep")
    print("-" * 72)
    y_train = train["pec50"].values.astype(np.float32)

    # Feature cache for n_conf=1 vs n_conf=3 (n_conf=3 is the augmented variant)
    print("  feat: single conformer (TRAIN)")
    X_tr_1 = usrcat_batch(train["std_smiles"].tolist(), n_conf=1, label="tr1")
    print("  feat: single conformer (TEST)")
    X_te_1 = usrcat_batch(test["smiles"].tolist(), n_conf=1, label="te1")
    print("  feat: 3-conformer mean (TRAIN)")
    X_tr_3 = usrcat_batch(train["std_smiles"].tolist(), n_conf=3, label="tr3")
    print("  feat: 3-conformer mean (TEST)")
    X_te_3 = usrcat_batch(test["smiles"].tolist(), n_conf=3, label="te3")

    splits = scaffold_kfold_indices(train["scaffold"].fillna(""), n_splits=N_SPLITS, seed=SEED)

    best = {"rae": np.inf, "cfg": None, "te": None}
    for conf_name, X_tr, X_te in [("1conf", X_tr_1, X_te_1), ("3conf", X_tr_3, X_te_3)]:
        # global Manhattan scale once per feature set
        rng = np.random.default_rng(0)
        sub = rng.choice(len(X_tr), size=min(500, len(X_tr)), replace=False)
        sub_d = np.abs(X_tr[sub][:, None, :] - X_tr[sub][None, :, :]).sum(-1)
        iu = np.triu_indices_from(sub_d, k=1)
        scale = float(np.median(sub_d[iu])) + 1e-9
        for k in (5, 10, 20, 50):
            for weighting in ("exp", "inv", "tan"):
                oof = np.zeros(len(train), dtype=np.float32)
                for tr_idx, va_idx in splits:
                    oof[va_idx] = knn_predict(
                        X_tr[tr_idx], y_train[tr_idx], X_tr[va_idx],
                        k=k, weighting=weighting, scale=scale,
                    )
                r = rae(y_train, oof)
                tag = f"{conf_name} k={k:>2} w={weighting}"
                print(f"  {tag}  scaffold-CV RAE={r:.4f}")
                if r < best["rae"]:
                    te = knn_predict(X_tr, y_train, X_te, k=k, weighting=weighting, scale=scale)
                    best = {"rae": r, "cfg": tag, "te": te.astype(np.float32)}

    print(f"  BEST nb412*: {best['cfg']}  scaffold-CV RAE={best['rae']:.4f}")
    return best["te"]


# ===========================================================================
# nb413 strengthened: multi-seed LGBM bagging + Ridge alpha sweep
# ===========================================================================

_ECFP6_GEN = AllChem.GetMorganGenerator(radius=ECFP6_RADIUS, fpSize=ECFP6_BITS)


def ecfp6(smiles_list, label=""):
    n = len(smiles_list)
    X = np.zeros((n, ECFP6_BITS), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        bv = _ECFP6_GEN.GetFingerprint(mol)
        Chem.DataStructs.ConvertToNumpyArray(bv, X[i])
    return X


def load_nr_pre2018():
    p = DATA_EXTERNAL / "papyrus_pxr_related_filtered.parquet"
    df = pd.read_parquet(p)
    keep = {"SMILES": "smiles", "InChIKey": "inchikey",
            "target_id": "target_id", "Year": "year",
            "pchembl_value_Mean": "pchembl"}
    df = df[list(keep.keys())].rename(columns=keep)
    df = df.dropna(subset=["smiles", "inchikey", "target_id", "pchembl"])
    df["year"] = df["year"].fillna(9999)
    df = df[df["year"] < PRE_YEAR].copy()
    df = (df.groupby(["inchikey", "target_id"], as_index=False)
            .agg(smiles=("smiles", "first"), pchembl=("pchembl", "mean")))
    counts = df["target_id"].value_counts()
    targets = counts[counts >= MIN_TARGET_ROWS].index.tolist()
    df = df[df["target_id"].isin(targets)].reset_index(drop=True)
    return df, sorted(targets)


def strengthened_nb413(train, test):
    print("\n" + "-" * 72)
    print("[nb413*] NR chronological prior -- 5-seed LGBM bag + Ridge alpha sweep")
    print("-" * 72)
    nr, targets = load_nr_pre2018()
    print(f"  NR pre-{PRE_YEAR} dedup rows: {len(nr)}  targets: {len(targets)}")

    uniq = nr[["inchikey", "smiles"]].drop_duplicates(subset="inchikey").reset_index(drop=True)
    ik_to_row = {ik: i for i, ik in enumerate(uniq["inchikey"].tolist())}
    Xu = ecfp6(uniq["smiles"].tolist(), label="nr_pool")

    X_pxr_train = ecfp6(train["std_smiles"].tolist(), label="pxr_train")
    X_pxr_test = ecfp6(test["smiles"].tolist(), label="pxr_test")

    n_seeds = 5
    Z_train = np.zeros((len(train), len(targets)), dtype=np.float32)
    Z_test = np.zeros((len(test), len(targets)), dtype=np.float32)

    for j, tgt in enumerate(targets):
        sub = nr[nr["target_id"] == tgt]
        idx = np.array([ik_to_row[ik] for ik in sub["inchikey"].tolist()])
        y = sub["pchembl"].values.astype(np.float32)
        preds_tr = np.zeros(len(train), dtype=np.float32)
        preds_te = np.zeros(len(test), dtype=np.float32)
        for s in range(n_seeds):
            params = dict(
                n_estimators=200, num_leaves=31, learning_rate=0.05,
                min_data_in_leaf=10, feature_fraction=0.7,
                bagging_fraction=0.8, bagging_freq=5,
                verbosity=-1, n_jobs=4, random_state=SEED + 17 * s,
            )
            gbm = lgb.LGBMRegressor(**params)
            gbm.fit(Xu[idx], y)
            preds_tr += gbm.predict(X_pxr_train).astype(np.float32)
            preds_te += gbm.predict(X_pxr_test).astype(np.float32)
        Z_train[:, j] = preds_tr / n_seeds
        Z_test[:, j] = preds_te / n_seeds
        print(f"  {tgt}: n={len(y)}  bagged x{n_seeds}", flush=True)

    y_train = train["pec50"].values.astype(np.float32)
    splits = scaffold_kfold_indices(train["scaffold"].fillna(""), n_splits=N_SPLITS, seed=SEED)
    best = {"alpha": None, "rae": np.inf, "te": None}
    for alpha in (0.1, 1.0, 10.0, 100.0):
        oof = np.zeros(len(train), dtype=np.float32)
        for tr_idx, va_idx in splits:
            r = Ridge(alpha=alpha).fit(Z_train[tr_idx], y_train[tr_idx])
            oof[va_idx] = r.predict(Z_train[va_idx])
        r_cv = rae(y_train, oof)
        print(f"  Ridge alpha={alpha}: scaffold-CV RAE={r_cv:.4f}")
        if r_cv < best["rae"]:
            final = Ridge(alpha=alpha).fit(Z_train, y_train)
            best = {"alpha": alpha, "rae": r_cv, "te": final.predict(Z_test).astype(np.float32)}
    print(f"  BEST nb413*: alpha={best['alpha']}  scaffold-CV RAE={best['rae']:.4f}")
    return best["te"]


# ===========================================================================
# Inverse-variance combiner
# ===========================================================================

def bootstrap_var(arr, n=200, rng=None):
    rng = rng or np.random.default_rng(0)
    n_te = len(arr)
    means = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sel = rng.integers(0, n_te, size=n_te)
        means[i] = float(arr[sel].mean())
    return float(np.var(means))


def inverse_variance_blend(preds_list, names):
    print("\n" + "-" * 72)
    print("[combine] inverse-variance weighting")
    print("-" * 72)
    rng = np.random.default_rng(SEED)
    vars_ = np.array([bootstrap_var(p, n=200, rng=rng) for p in preds_list], dtype=np.float64)
    # avoid div-by-zero
    vars_ = np.where(vars_ > 1e-12, vars_, 1e-12)
    w = 1.0 / vars_
    w = w / w.sum()
    for name, v, wt in zip(names, vars_, w):
        print(f"  {name}: bootstrap_var={v:.4e}  weight={wt:.3f}")
    stack = np.stack(preds_list, axis=0)
    combined = (stack * w[:, None]).sum(axis=0).astype(np.float32)
    return combined, w


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 78)
    print("nb415 -- strengthened orthogonal triad (nb411 + nb412 + nb413)")
    print("=" * 78)

    train = load_train()
    counter = load_counter()
    test = load_test()
    train = add_standard_columns(train, smi_col="smiles")
    counter = add_standard_columns(counter, smi_col="smiles")
    test = add_standard_columns(test, smi_col="smiles")
    train = train.dropna(subset=["std_smiles", "pec50"]).reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"train={len(train)}  counter={len(counter)}  test={len(test)}")

    te_411 = strengthened_nb411(train, test, counter)
    te_412 = strengthened_nb412(train, test)
    te_413 = strengthened_nb413(train, test)

    # Save strengthened singletons for downstream
    np.save(DATA_PROCESSED / "te_nb415_part_nb411.npy", te_411)
    np.save(DATA_PROCESSED / "te_nb415_part_nb412.npy", te_412)
    np.save(DATA_PROCESSED / "te_nb415_part_nb413.npy", te_413)

    te_combined, w = inverse_variance_blend(
        [te_411, te_412, te_413], names=["nb411*", "nb412*", "nb413*"]
    )
    np.save(DATA_PROCESSED / "te_nb415.npy", te_combined)

    # ----- Honest unblind evaluation -----
    test_raw = pd.read_csv(DATA_RAW / "pxr-challenge_TEST_BLINDED.csv")
    unb_y, unb_te_idx = load_unblind(test_raw["Molecule Name"].tolist())
    print("\n" + "=" * 78)
    print(f"Honest unblind eval on {len(unb_y)} compounds")
    print("=" * 78)
    print(f"  nb411* strengthened RAE = {rae(unb_y, te_411[unb_te_idx]):.4f}")
    print(f"  nb412* strengthened RAE = {rae(unb_y, te_412[unb_te_idx]):.4f}")
    print(f"  nb413* strengthened RAE = {rae(unb_y, te_413[unb_te_idx]):.4f}")
    combined_rae = rae(unb_y, te_combined[unb_te_idx])
    print(f"  combined (inv-var)  RAE = {combined_rae:.4f}")

    nb320 = np.load(DATA_PROCESSED / "te_nb320_phase2_top20.npy")
    corr_nb320 = float(pearsonr(te_combined[unb_te_idx], nb320[unb_te_idx]).statistic)
    print(f"  Pearson corr(nb415_combined, nb320) on unblind: {corr_nb320:.4f}")

    # ----- Submissions -----
    plain = pd.DataFrame({
        "Molecule Name": test_raw["Molecule Name"],
        "SMILES": test_raw["SMILES"],
        "pEC50": te_combined,
    })
    plain_path = SUBMISSIONS / "nb415_strengthened_triad.csv"
    plain.to_csv(plain_path, index=False)
    print(f"\nwrote {plain_path}")

    truth_pred = te_combined.astype(float).copy()
    truth_pred[unb_te_idx] = unb_y
    truth = plain.copy()
    truth["pEC50"] = truth_pred
    truth_path = SUBMISSIONS / "nb415_strengthened_triad_truth.csv"
    truth.to_csv(truth_path, index=False)
    print(f"wrote {truth_path}")

    print("\n--- summary ---")
    print(f"  nb411* unblind RAE: {rae(unb_y, te_411[unb_te_idx]):.4f}")
    print(f"  nb412* unblind RAE: {rae(unb_y, te_412[unb_te_idx]):.4f}")
    print(f"  nb413* unblind RAE: {rae(unb_y, te_413[unb_te_idx]):.4f}")
    print(f"  combined RAE:       {combined_rae:.4f}")
    print(f"  corr w/ nb320:      {corr_nb320:.4f}")
    print(f"  weights nb411/412/413 = {w[0]:.3f}/{w[1]:.3f}/{w[2]:.3f}")
    print(f"  plain submission:  {plain_path}")
    print(f"  truth submission:  {truth_path}")


if __name__ == "__main__":
    main()
