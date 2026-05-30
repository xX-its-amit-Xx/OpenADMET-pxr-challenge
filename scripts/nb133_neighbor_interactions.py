"""nb133 -- Tanimoto-expanded NEIGHBOR INTERACTION KNOWLEDGE.

nb132 gives 41-dim per-target activity profile of top-k Tanimoto neighbors.
nb133 extends with INTERACTION-quality features that medicinal chemists
actually use when reasoning about a compound's mechanism:

  1. Per-target engagement frequency: of top-k neighbors, what fraction
     have measured activity at each target? (binary 'engaged' signal vs continuous)

  2. Multi-target promiscuity: how many distinct targets do the top-k
     neighbors collectively engage? (promiscuous = pan-NR; specific = clean target)

  3. Activity diversity: std of neighbor pchembl values per target
     (high std = uncertain SAR neighborhood; low std = predictable)

  4. Measurement confidence proxy: avg pchembl_value_N for the neighbors
     (more replications = more reliable Papyrus entry → more confident features)

  5. Tox21 NR engagement: for each query, does any Tanimoto neighbor in Tox21
     (NR-AhR, NR-PPAR-gamma, NR-AR, NR-ER) test positive? Binary panel.

These features encode the medicinal-chemistry-relevant 'what kind of binder
profile does the neighborhood look like'.
"""
import os, sys, warnings, time
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import lightgbm as lgb
from scipy.stats import spearmanr

from pxr.data import load_train, load_test
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import add_standard_columns
from pxr.featurize import combined, impute
from pxr.paths import DATA_PROCESSED, DATA_EXTERNAL, SUBMISSIONS

LGBM_BASE = dict(
    n_estimators=1500, num_leaves=63, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
    objective="mae", n_jobs=4, random_state=42, verbose=-1,
)


def morgan_fp(smi, radius=2, nbits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nbits)


def std_smi(s):
    try:
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m) if m is not None else None
    except Exception:
        return None


def main():
    print("=== nb133: Neighbor interaction knowledge ===\n")

    tr = load_train(); te_df = load_test()
    tr = add_standard_columns(tr)
    y_tr = tr["pec50"].values.astype(np.float64)
    smiles_tr = tr["std_smiles"].tolist()
    smiles_te = te_df["smiles"].tolist()

    oof_base = np.load(DATA_PROCESSED / "oof_nb197_dense_grid.npy").astype(np.float64)
    te_base  = np.load(DATA_PROCESSED / "te_nb197_dense_grid.npy").astype(np.float64)
    base_rae = rae(y_tr, oof_base)
    print(f"Base nb197: OOF RAE={base_rae:.4f}")

    # ── Load Papyrus (with pchembl_value_N for confidence) ────────────────────
    papy = pd.read_parquet(DATA_EXTERNAL / "papyrus_pxr_related_filtered.parquet")
    smi_col = "SMILES" if "SMILES" in papy.columns else "SMILES_Stripped"
    val_col = "pchembl_value_Mean"
    print(f"Papyrus: {len(papy):,} records  {papy['accession'].nunique()} targets")

    # Per (compound, target) aggregation with measurement count
    papy_agg = papy.groupby([smi_col, "accession"]).agg(
        pchembl=(val_col, "median"),
        n_meas=("pchembl_value_N", "sum") if "pchembl_value_N" in papy.columns else (val_col, "count"),
    ).reset_index()

    # Wide pivots
    wide_activity = papy_agg.pivot(index=smi_col, columns="accession", values="pchembl")
    wide_n_meas = papy_agg.pivot(index=smi_col, columns="accession", values="n_meas").fillna(0)
    targets = list(wide_activity.columns)
    print(f"Wide pivot: {wide_activity.shape}  targets={len(targets)}")

    # Per-compound total measurements across all targets (promiscuity quality)
    compound_total_meas = wide_n_meas.sum(axis=1)
    compound_n_targets = (wide_n_meas > 0).sum(axis=1)

    # ── Tox21 binary NR panel for additional interaction knowledge ────────────
    tox_path = DATA_EXTERNAL / "tox21_moleculenet.csv.gz"
    tox_lookup = {}  # std_smi -> dict of NR binary labels
    if not tox_path.exists():
        print("Downloading Tox21 (MoleculeNet)...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
            tox_path)
    try:
        tox = pd.read_csv(tox_path)
        nr_cols = [c for c in tox.columns if c.startswith("NR-")]
        print(f"Tox21 NR labels: {nr_cols}")
        tox["std_smi"] = tox["smiles"].apply(std_smi)
        tox = tox.dropna(subset=["std_smi"]).set_index("std_smi")
        for s, row in tox.iterrows():
            tox_lookup[s] = {nr: row[nr] for nr in nr_cols}
        print(f"Tox21 mapping built: {len(tox_lookup):,} compounds")
    except Exception as e:
        print(f"Tox21 load failed: {e}")
        nr_cols = []

    # ── Build reference fingerprints ──────────────────────────────────────────
    print("\nBuilding reference fingerprints...")
    ref_smiles = wide_activity.index.tolist()
    ref_fps, ref_valid = [], []
    for s in ref_smiles:
        fp = morgan_fp(s)
        ref_fps.append(fp)
        ref_valid.append(fp is not None)
    print(f"  Valid: {sum(ref_valid):,}/{len(ref_smiles):,}")
    activity_matrix = wide_activity.values
    n_meas_matrix = wide_n_meas.values
    total_meas_arr = compound_total_meas.values
    n_targets_arr = compound_n_targets.values

    # Pre-build Tox21 lookup matrix for ref compounds
    tox_per_ref = []
    for s in ref_smiles:
        if s in tox_lookup:
            tox_per_ref.append({nr: tox_lookup[s].get(nr) for nr in nr_cols})
        else:
            tox_per_ref.append({nr: np.nan for nr in nr_cols})

    # ── Per-query interaction features ────────────────────────────────────────
    def expand(query_smiles_list, label):
        print(f"\nExpanding {len(query_smiles_list)} {label} compounds...")
        n = len(query_smiles_list)
        n_t = len(targets)
        K = 8  # top-k neighbors

        # Output blocks:
        f_activity = np.full((n, n_t), np.nan)             # sim-weighted avg activity per target
        f_engagement = np.zeros((n, n_t))                  # % neighbors with measured target
        f_activity_std = np.zeros((n, n_t))                # std of neighbor activities per target
        f_neighbor_meas = np.zeros(n)                      # avg total measurements of neighbors
        f_neighbor_promiscuity = np.zeros(n)               # avg # targets engaged by neighbors
        f_tox_nr = np.full((n, len(nr_cols)), np.nan)      # tox21 NR positive frac among neighbors
        f_max_neighbor_pchembl = np.zeros(n)               # best activity any neighbor has shown
        f_max_sim = np.zeros(n)
        f_n_neighbors_05 = np.zeros(n)                     # # neighbors with sim >= 0.5

        t0 = time.time()
        for i, qsmi in enumerate(query_smiles_list):
            qfp = morgan_fp(qsmi)
            if qfp is None: continue
            sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, ref_fps))
            f_max_sim[i] = sims.max()
            f_n_neighbors_05[i] = (sims >= 0.5).sum()

            top_idx = np.argsort(sims)[::-1][:K]
            top_sims = sims[top_idx]

            # Per-target features
            for t in range(n_t):
                acts = activity_matrix[top_idx, t]
                valid = np.isfinite(acts)
                if valid.sum() > 0:
                    w = top_sims[valid]
                    f_activity[i, t] = np.dot(w, acts[valid]) / w.sum()
                    f_activity_std[i, t] = np.std(acts[valid])
                f_engagement[i, t] = valid.sum() / K

            # Aggregate quality features
            f_neighbor_meas[i] = total_meas_arr[top_idx].mean()
            f_neighbor_promiscuity[i] = n_targets_arr[top_idx].mean()

            # Best activity any neighbor has shown across any target
            max_per_neighbor = np.nanmax(activity_matrix[top_idx, :], axis=1)
            max_per_neighbor = max_per_neighbor[np.isfinite(max_per_neighbor)]
            if len(max_per_neighbor) > 0:
                f_max_neighbor_pchembl[i] = max_per_neighbor.max()

            # Tox21 NR positive fraction among neighbors
            for k, nr in enumerate(nr_cols):
                vals = []
                for tidx in top_idx:
                    v = tox_per_ref[tidx].get(nr)
                    if pd.notna(v):
                        vals.append(v)
                if vals:
                    f_tox_nr[i, k] = np.mean(vals)

            if (i+1) % 500 == 0:
                print(f"  {i+1}/{n}  ({time.time()-t0:.0f}s)")

        # Fill NaN: activity median per column; tox to 0.5; engagement is already filled
        for t in range(n_t):
            med = np.nanmedian(f_activity[:, t])
            f_activity[:, t] = np.where(np.isfinite(f_activity[:, t]),
                                         f_activity[:, t],
                                         med if np.isfinite(med) else 5.0)
        for k in range(len(nr_cols)):
            f_tox_nr[:, k] = np.where(np.isfinite(f_tox_nr[:, k]), f_tox_nr[:, k], 0.5)

        feats = np.column_stack([
            f_activity, f_engagement, f_activity_std,
            f_neighbor_meas.reshape(-1, 1),
            f_neighbor_promiscuity.reshape(-1, 1),
            f_tox_nr,
            f_max_neighbor_pchembl.reshape(-1, 1),
            f_max_sim.reshape(-1, 1),
            f_n_neighbors_05.reshape(-1, 1),
        ])
        print(f"  {label} feature shape: {feats.shape}")
        return feats

    expand_tr = expand(smiles_tr, "train")
    expand_te = expand(smiles_te, "test")

    # ── Sanity correlations ───────────────────────────────────────────────────
    print("\n[Correlations of interaction-block features with PXR pEC50]")
    # Just the summary features
    n_t = len(targets)
    summary_idx = {
        "neighbor_avg_meas": 3*n_t + 0,
        "neighbor_promiscuity": 3*n_t + 1,
        "max_neighbor_pchembl": 3*n_t + 1 + len(nr_cols) + 1,
        "max_sim": 3*n_t + 1 + len(nr_cols) + 2,
        "n_neighbors_05": 3*n_t + 1 + len(nr_cols) + 3,
    }
    for name, idx in summary_idx.items():
        if idx < expand_tr.shape[1]:
            col = expand_tr[:, idx]
            if np.unique(col).size > 1:
                rho, _ = spearmanr(col, y_tr)
                print(f"  {name:25s}: ρ={rho:+.3f}")

    # ── Augmented LGBM ────────────────────────────────────────────────────────
    print("\n── Augmented LGBM scaffold CV ──")
    X_tr_base = combined(smiles_tr); X_tr_base = impute(X_tr_base)
    X_te_base = combined(smiles_te); X_te_base = impute(X_te_base)
    X_tr_aug = np.hstack([X_tr_base, expand_tr])
    X_te_aug = np.hstack([X_te_base, expand_te])
    print(f"  Aug shape: train={X_tr_aug.shape}  test={X_te_aug.shape}")

    scaffolds = tr["scaffold"].tolist()
    folds = scaffold_kfold_indices(scaffolds, n_splits=5)

    for name, Xt, Xe in [
        ("base_only", X_tr_base, X_te_base),
        ("interactions", X_tr_aug, X_te_aug),
    ]:
        oof = np.zeros(len(y_tr))
        te_preds = []
        for tr_idx, va_idx in folds:
            m = lgb.LGBMRegressor(**LGBM_BASE)
            m.fit(Xt[tr_idx], y_tr[tr_idx],
                  eval_set=[(Xt[va_idx], y_tr[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
            oof[va_idx] = m.predict(Xt[va_idx])
            te_preds.append(m.predict(Xe))
        te_pred = np.mean(te_preds, axis=0)
        r = rae(y_tr, oof); ratio = te_pred.std() / oof.std()
        print(f"  {name:15s}: OOF RAE={r:.4f}  ratio={ratio:.3f}  te_std={te_pred.std():.4f}")
        if name == "interactions":
            np.save(DATA_PROCESSED / "oof_nb133_interactions.npy", oof)
            np.save(DATA_PROCESSED / "te_nb133_interactions.npy", te_pred)
            print(f"  Saved oof_nb133_interactions.npy + te_nb133_interactions.npy")


if __name__ == "__main__":
    main()
