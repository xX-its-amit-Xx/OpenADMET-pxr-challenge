"""nb302 -- Wide-pool SLSQP combining nb239 base + all nb285-301 candidates.

Goal: see if multiple new candidates jointly add value when blended with the
4-base nb239 stack. nb290 alone added 0.0195 weight; perhaps several together
fill different gaps.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb302: wide-pool SLSQP over nb239 base + nb285-301 candidates ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    n_tr = len(y)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    n_te = len(te_df)

    # Base 4
    base = {
        'nb224': ('oof_nb224_pool_plus_2', 'te_nb224_pool_plus_2'),
        'nb179s': ('oof_nb179_stack', 'te_nb179_stack'),
        'mtd':   ('oof_multi_template_delta', 'te_oof_multi_template_delta'),
        'loso':  ('oof_delta_loso', 'te_oof_delta_loso'),
    }
    # nb285-301 candidates
    new_cands = {
        'nb286_v2':    ('oof_nb286_clean_v2', 'te_nb286_clean_v2'),
        'nb288_gp':    ('oof_nb288_gp_corrected', 'te_nb288_gp_corrected'),
        'nb289_ttf':   ('oof_nb289_test_time_finetune', 'te_nb289_test_time_finetune'),
        'nb290_mmp':   ('oof_nb290_mmp_transform', 'te_nb290_mmp_transform'),
        'nb291_bio':   ('oof_nb291', 'te_nb291'),
        'nb292_rule':  ('oof_nb292_molrule', 'te_nb292_molrule'),
        'nb293_conf':  ('oof_nb293_conformal', 'te_nb293_conformal'),
        'nb294_hetn':  ('oof_nb294_hetnll', 'te_nb294_hetnll'),
        'nb295_rag':   ('oof_nb295_rag', 'te_nb295_rag'),
        'nb296_tda':   ('oof_nb296_tda', 'te_nb296_tda'),
        'nb298_pose':  ('oof_nb298_pose_ifp', 'te_nb298_pose_ifp'),
        'nb299_clip':  ('oof_nb299_nrclip', 'te_nb299_nrclip'),
        'nb300_diff':  ('oof_nb300_diffusion_aug', 'te_nb300_diffusion_aug'),
        'nb297_pysr':  ('oof_nb297_pysr', 'te_nb297_pysr'),
        'nb303_dann':  ('oof_nb303_dann', 'te_nb303_dann'),
        'nb305_mope':  ('oof_nb305_mope', 'te_nb305_mope'),
        'nb306_mil':   ('oof_nb306_cepsmim', 'te_nb306_cepsmim'),
        'nb307_soci':  ('oof_nb307_soci', 'te_nb307_soci'),
        'nb311_stable':('oof_nb311_stable', 'te_nb311_stable'),
        'nb312_relab': ('oof_nb312_relabeled', 'te_nb312_relabeled'),
        'nb313_admet': ('oof_nb313_admet', 'te_nb313_admet'),
        'nb314_tdc':   ('oof_nb314_tdc', 'te_nb314_tdc'),
        'nb315_v1':    ('oof_nb315_v1_nb107_enh', 'te_nb315_v1_nb107_enh'),
        'nb315_v2':    ('oof_nb315_v2_nb145_enh', 'te_nb315_v2_nb145_enh'),
        'nb315_v3':    ('oof_nb315_v3_nb117_enh', 'te_nb315_v3_nb117_enh'),
        'nb317_mt':    ('oof_nb317_external_mt', 'te_nb317_external_mt'),
        'nb309_nov':   ('oof_nb309_novartis', 'te_nb309_novartis'),
        'nb310_az':    ('oof_nb310_az', 'te_nb310_az'),
    }
    pool = {**base, **new_cands}

    names = []
    oofs = []; tes = []
    for nm, (op, tp) in pool.items():
        of = DATA_PROCESSED / f"{op}.npy"
        tf = DATA_PROCESSED / f"{tp}.npy"
        if not of.exists() or not tf.exists():
            print(f"  SKIP {nm}: missing {of.name if not of.exists() else tf.name}")
            continue
        o = np.load(of); t = np.load(tf)
        if o.shape != (n_tr,) or t.shape != (n_te,):
            print(f"  SKIP {nm}: shape {o.shape}/{t.shape}")
            continue
        if not np.isfinite(o).all() or not np.isfinite(t).all():
            print(f"  SKIP {nm}: non-finite values")
            continue
        names.append(nm); oofs.append(o); tes.append(t)
    print(f"\nValid pool: {len(names)} models: {names}")

    M_oof = np.column_stack(oofs); M_te = np.column_stack(tes)

    # ============================
    # SLSQP — pure RAE objective
    # ============================
    print("\n--- SLSQP (RAE objective) ---")
    def loss(w): return rae(y, M_oof @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * len(names)
    best = None
    for seed in range(80):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(len(names)))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-10})
        if best is None or res.fun < best.fun: best = res
    pred_oof = M_oof @ best.x
    pred_te = M_te @ best.x
    r = rae(y, pred_oof)
    sp, _ = spearmanr(y, pred_oof)
    print(f"OOF RAE={r:.4f}  Spearman={sp:.4f}  te_std={pred_te.std():.3f}  te_mean={pred_te.mean():.3f}")
    print("Active weights (>=0.005):")
    for nm, w in sorted(zip(names, best.x), key=lambda x: -x[1]):
        if w >= 0.005:
            print(f"  {w:.4f}  {nm}")

    np.save(DATA_PROCESSED / "oof_nb302_full_pool.npy", pred_oof)
    np.save(DATA_PROCESSED / "te_nb302_full_pool.npy", pred_te)

    # Submission
    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': pred_te,
    })
    out = SUBMISSIONS / "nb302_full_pool_blend.csv"
    sub.to_csv(out, index=False)
    print(f"\nWrote {out}")

    # ============================
    # SLSQP with Spearman bonus + te_std collapse penalty
    # ============================
    print("\n--- SLSQP (RAE - 0.05*Spearman + collapse-penalty) ---")
    def loss2(w):
        p = M_oof @ w
        r2 = rae(y, p)
        sp2, _ = spearmanr(y, p)
        te_p = M_te @ w
        col_pen = max(0, 0.55 - te_p.std()) * 0.3
        return r2 - 0.05 * sp2 + col_pen
    best2 = None
    for seed in range(80):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(len(names)))
        res = minimize(loss2, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol': 1e-10})
        if best2 is None or res.fun < best2.fun: best2 = res
    pred_oof2 = M_oof @ best2.x; pred_te2 = M_te @ best2.x
    r2 = rae(y, pred_oof2)
    sp2, _ = spearmanr(y, pred_oof2)
    print(f"OOF RAE={r2:.4f}  Spearman={sp2:.4f}  te_std={pred_te2.std():.3f}  te_mean={pred_te2.mean():.3f}")
    print("Active weights (>=0.005):")
    for nm, w in sorted(zip(names, best2.x), key=lambda x: -x[1]):
        if w >= 0.005:
            print(f"  {w:.4f}  {nm}")

    sub2 = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': pred_te2,
    })
    out2 = SUBMISSIONS / "nb302_full_pool_multimetric.csv"
    sub2.to_csv(out2, index=False)
    print(f"Wrote {out2}")


if __name__ == "__main__":
    main()
