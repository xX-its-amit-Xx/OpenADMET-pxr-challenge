"""nb284 -- Multi-metric + biomechanistic-aligned model selection and blend.

User insight (verbatim, paraphrased): RAE-only ranking overfits to noise.
Models with OOF RAE in the 0.30-0.60 band that have strong Spearman / Kendall /
R-squared, or that align with biomechanistic priors (Boltz2 ipTM, docking
affinity, counter-assay selectivity), may generalise BETTER to the
analog-expansion test set than the over-tuned RAE-0.28 stack. Current LB
is 0.7487; we need to find candidates that potentially beat that.

Pipeline:
  1. Load every oof_*.npy (paired with te_*.npy where available)
  2. Score each on RAE, Spearman, Kendall, R^2, MAE, te_std
  3. Compute biomech alignment for test predictions:
     - Boltz2 ipTM correlation (high binding-mode confidence -> high pec50)
     - Docking affinity correlation (lower kcal/mol -> higher pec50)
     - Counter-assay selectivity gap
  4. Identify the "mid-RAE high-rank biomech-aligned" pool
  5. Build a multi-metric SLSQP blend over THAT pool (not just lowest-RAE)
  6. Compare to nb239 baseline + write submission candidate
"""
import os, sys, warnings, json
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from scipy.optimize import minimize
from sklearn.metrics import r2_score

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def load_pair(name):
    """Load oof_<name>.npy and te_<name>.npy if both exist."""
    oof = DATA_PROCESSED / f"oof_{name}.npy"
    te  = DATA_PROCESSED / f"te_{name}.npy"
    if oof.exists() and te.exists():
        try:
            o = np.load(oof); t = np.load(te)
            return o, t
        except Exception:
            return None, None
    return None, None


def main():
    print("=== nb284: Multi-metric + biomech-aligned model selection ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    n_tr = len(y); n_te = 513
    print(f"train n={n_tr}, test n={n_te}, train pec50: mean={y.mean():.3f} std={y.std():.3f}")

    # ============================
    # Step 1: enumerate all model pairs
    # ============================
    print("\n--- Step 1: enumerate oof/te pairs ---")
    pairs = []
    for f in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        name = f.stem.replace("oof_", "")
        o, t = load_pair(name)
        if o is None or t is None: continue
        if o.shape != (n_tr,) or t.shape != (n_te,): continue
        if not np.isfinite(o).all() or not np.isfinite(t).all(): continue
        pairs.append((name, o, t))
    print(f"Valid (oof, te) pairs: {len(pairs)}")

    # ============================
    # Step 2: multi-metric scoring
    # ============================
    print("\n--- Step 2: multi-metric scoring ---")
    rows = []
    for name, o, t in pairs:
        try:
            r = rae(y, o)
            sp, _ = spearmanr(y, o)
            kd, _ = kendalltau(y, o)
            r2 = r2_score(y, o)
            mae = np.abs(y - o).mean()
            te_std = float(t.std())
            te_mean = float(t.mean())
            rows.append(dict(name=name, rae=r, spearman=sp, kendall=kd, r2=r2, mae=mae,
                             te_mean=te_mean, te_std=te_std))
        except Exception:
            continue
    sc = pd.DataFrame(rows)
    sc = sc.sort_values('rae').reset_index(drop=True)
    print(f"Scored {len(sc)} models")
    print(f"\nTop 15 by RAE:")
    print(sc.head(15).to_string(index=False))

    print(f"\nTop 15 by Spearman:")
    print(sc.nlargest(15, 'spearman').to_string(index=False))

    print(f"\nTop 15 by Kendall:")
    print(sc.nlargest(15, 'kendall').to_string(index=False))

    # ============================
    # Step 3: biomech alignment for TEST predictions
    # ============================
    print("\n--- Step 3: biomech alignment (test predictions) ---")
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    te_names = te_df['Molecule Name'].tolist()

    # 3a. Boltz2 ipTM
    boltz = pd.read_parquet("data/processed/boltz_dargason_features_test.parquet")
    boltz_idx = {n: i for i, n in enumerate(boltz['name'])}
    iptm = np.array([boltz.iloc[boltz_idx[n]]['ligand_iptm_best'] if n in boltz_idx else np.nan
                     for n in te_names])
    print(f"  Boltz ligand_iptm_best coverage: {(~np.isnan(iptm)).sum()}/{n_te}")

    # 3b. Docking — take best affinity across receptors per compound
    dock = pd.read_parquet("data/external/docking_batch0_complete.parquet")
    dock_best = dock.groupby('name')['affinity'].min().to_dict()  # min = most negative = best binding
    aff = np.array([dock_best.get(n, np.nan) for n in te_names])
    print(f"  Docking affinity coverage: {(~np.isnan(aff)).sum()}/{n_te}")

    # 3c. Counter-assay selectivity proxy — use a counter-assay test prediction if available
    counter_te = None
    for cname in ['oof_lgbm_counter_soft', 'oof_counter_delta', 'nb153_counter_multi']:
        p = DATA_PROCESSED / f"te_{cname}.npy"
        if p.exists():
            counter_te = np.load(p); break
    if counter_te is None or counter_te.shape != (n_te,):
        counter_te = None
    print(f"  Counter-assay test pred available: {counter_te is not None}")

    # Score each model's test predictions against biomech priors
    biomech_rows = []
    for name, o, t in pairs:
        scores = {'name': name}
        # Boltz: higher iptm -> higher pec50 (positive correlation)
        mask = ~np.isnan(iptm)
        if mask.sum() > 30:
            sp_bz, _ = spearmanr(iptm[mask], t[mask])
            scores['boltz_sp'] = sp_bz
        # Docking: lower affinity -> higher pec50 (negative correlation; flip sign)
        mask = ~np.isnan(aff)
        if mask.sum() > 30:
            sp_dk, _ = spearmanr(-aff[mask], t[mask])  # -aff so positive = better binding
            scores['docking_sp'] = sp_dk
        # Counter-assay selectivity: pred_pxr - pred_null should correlate with binding affinity
        if counter_te is not None:
            sel = t - counter_te
            # selectivity should correlate with t itself (active PXR binders are selective)
            sp_sel, _ = spearmanr(sel, t)
            scores['selectivity_sp'] = sp_sel
        biomech_rows.append(scores)
    bm = pd.DataFrame(biomech_rows)
    sc = sc.merge(bm, on='name', how='left')

    # Composite biomech alignment score: mean of available signed correlations
    bm_cols = [c for c in ['boltz_sp', 'docking_sp', 'selectivity_sp'] if c in sc.columns]
    sc['biomech_score'] = sc[bm_cols].mean(axis=1)
    print(f"\nBiomech alignment columns: {bm_cols}")

    # ============================
    # Step 4: identify mid-RAE high-rank biomech-aligned pool
    # ============================
    print("\n--- Step 4: mid-RAE filtering ---")
    # User said accept 0.4-0.6 RAE; widen slightly to 0.30-0.65 to include the existing top stack
    # AND give weight to mid-RAE candidates
    mid = sc[(sc['rae'] >= 0.30) & (sc['rae'] <= 0.65)].copy()
    print(f"Mid-RAE (0.30-0.65) candidates: {len(mid)}")

    # Composite quality: high rank correlation + reasonable RAE + biomech alignment
    mid['composite'] = (
        0.4 * mid['spearman'].fillna(0) +
        0.3 * mid['kendall'].fillna(0) +
        0.1 * mid['r2'].fillna(0).clip(lower=0) +
        0.2 * mid['biomech_score'].fillna(0).clip(lower=0)
    )
    mid_top = mid.nlargest(25, 'composite')
    print(f"\nTop 25 mid-RAE by composite (Spearman+Kendall+R²+biomech):")
    show_cols = ['name', 'rae', 'spearman', 'kendall', 'r2', 'te_std', 'biomech_score', 'composite']
    print(mid_top[show_cols].to_string(index=False))

    # ============================
    # Step 5: multi-metric SLSQP blend over the composite pool
    # ============================
    print("\n--- Step 5: SLSQP over composite pool (also include nb239 base) ---")
    base_names = ['nb224_pool_plus_2', 'nb179_stack', 'multi_template_delta', 'delta_loso']
    pool_names = list(dict.fromkeys(base_names + mid_top['name'].tolist()))
    # Keep only those we can load
    pool_data = []
    for nm in pool_names:
        o, t = load_pair(nm)
        if o is None or o.shape != (n_tr,): continue
        pool_data.append((nm, o, t))
    print(f"Pool size: {len(pool_data)}")

    M_oof = np.column_stack([o for _, o, _ in pool_data])
    M_te  = np.column_stack([t for _, _, t in pool_data])
    names = [nm for nm, _, _ in pool_data]

    # Loss: weighted combination — minimise RAE but reward Spearman of blend
    def loss(w):
        pred = M_oof @ w
        r = rae(y, pred)
        sp, _ = spearmanr(y, pred)
        # Penalise low test-std (collapse)
        te_pred = M_te @ w
        te_std = te_pred.std()
        # train_std ~ 0.78
        std_penalty = max(0, 0.5 - te_std) * 0.5
        return r - 0.05 * sp + std_penalty

    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * len(names)
    best = None
    for seed in range(200):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(len(names)))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    blend_oof = M_oof @ best.x
    blend_te  = M_te @ best.x
    r = rae(y, blend_oof)
    sp, _ = spearmanr(y, blend_oof)
    kd, _ = kendalltau(y, blend_oof)
    r2 = r2_score(y, blend_oof)
    mae = np.abs(y - blend_oof).mean()
    print(f"\nBlend OOF: RAE={r:.4f}  Spearman={sp:.4f}  Kendall={kd:.4f}  R²={r2:.4f}  MAE={mae:.4f}")
    print(f"Blend test: mean={blend_te.mean():.3f}  std={blend_te.std():.3f}  range=[{blend_te.min():.2f}, {blend_te.max():.2f}]")
    print("\nActive weights (>0.01):")
    for nm, w in sorted(zip(names, best.x), key=lambda x: -x[1]):
        if w > 0.01:
            print(f"  {w:.4f}  {nm}")

    # Save blend
    sub = pd.DataFrame({
        'Molecule Name': te_names,
        'SMILES': te_df['SMILES'],
        'pEC50': blend_te,
    })
    out_csv = SUBMISSIONS / "nb284_multimetric_biomech_blend.csv"
    sub.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    # Save scoring table for reference
    sc.to_csv(DATA_PROCESSED / "nb284_all_models_scored.csv", index=False)
    print(f"Wrote scoring table -> data/processed/nb284_all_models_scored.csv")

    # Also save just the blend OOF/TE for downstream stacking
    np.save(DATA_PROCESSED / "oof_nb284_multimetric_biomech.npy", blend_oof)
    np.save(DATA_PROCESSED / "te_nb284_multimetric_biomech.npy", blend_te)

    # ============================
    # Step 6: also build a Spearman-first variant (less RAE-greedy)
    # ============================
    print("\n--- Step 6: Spearman-first blend (RAE secondary) ---")
    def loss_sp(w):
        pred = M_oof @ w
        sp, _ = spearmanr(y, pred)
        r = rae(y, pred)
        te_std = (M_te @ w).std()
        std_pen = max(0, 0.5 - te_std) * 0.5
        return -sp + 0.3 * r + std_pen
    best_sp = None
    for seed in range(200):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(len(names)))
        res = minimize(loss_sp, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-9})
        if best_sp is None or res.fun < best_sp.fun: best_sp = res
    sp_oof = M_oof @ best_sp.x
    sp_te = M_te @ best_sp.x
    sp_sp, _ = spearmanr(y, sp_oof)
    print(f"Spearman-first blend: RAE={rae(y, sp_oof):.4f}  Spearman={sp_sp:.4f}  te_std={sp_te.std():.3f}")
    print("Active weights (>0.01):")
    for nm, w in sorted(zip(names, best_sp.x), key=lambda x: -x[1]):
        if w > 0.01:
            print(f"  {w:.4f}  {nm}")

    sub2 = pd.DataFrame({
        'Molecule Name': te_names, 'SMILES': te_df['SMILES'], 'pEC50': sp_te,
    })
    out_csv2 = SUBMISSIONS / "nb284_spearman_first_blend.csv"
    sub2.to_csv(out_csv2, index=False)
    print(f"Wrote {out_csv2}")


if __name__ == "__main__":
    main()
