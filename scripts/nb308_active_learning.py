"""nb308 -- Active-learning-style adaptive routing by model disagreement.

Idea: per-test-compound, compute std across N existing model predictions.
Route each compound to the pool/model whose strengths fit its disagreement:
- low-disagreement (consensus): use the tightest pool (nb239 SLSQP)
- mid-disagreement: use the wider nb302 full-pool blend
- high-disagreement (need exploration): multi-modal weighted mean

OOF for nb308 = nb302 OOF (proxy, since we cannot route in CV without
re-deriving disagreement per fold). Then standard 5-way SLSQP appended to
the nb239 base.
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize

from pxr.data import load_train
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS


def main():
    print("=== nb308: Active-learning routing by disagreement ===\n")
    tr = load_train(); tr = add_standard_columns(tr)
    y = tr['pec50'].values.astype(np.float64)
    n_tr = len(y)
    te_df = pd.read_csv("data/raw/pxr-challenge_TEST_BLINDED.csv")
    n_te = len(te_df)

    # Pool of test predictions to compute disagreement over.
    pool_files = [
        ('nb239', 'te_nb239_full_slsqp.npy'),
        ('nb302', 'te_nb302_full_pool.npy'),
        ('nb290_mmp', 'te_nb290_mmp_transform.npy'),
        ('nb293_conf', 'te_nb293_conformal.npy'),
        ('nb288_gp',  'te_nb288_gp_corrected.npy'),
        ('nb286_v2',  'te_nb286_clean_v2.npy'),
        ('nb289_ttf', 'te_nb289_test_time_finetune.npy'),
        ('nb294_het', 'te_nb294_hetnll.npy'),
        ('nb295_rag', 'te_nb295_rag.npy'),
    ]
    loaded = []
    names = []
    for nm, f in pool_files:
        p = DATA_PROCESSED / f
        if not p.exists():
            print(f"  skip {nm}: {f} missing"); continue
        v = np.load(p)
        if v.shape != (n_te,) or not np.isfinite(v).all():
            print(f"  skip {nm}: shape {v.shape}, finite={np.isfinite(v).all()}"); continue
        loaded.append(v); names.append(nm)
    print(f"Loaded {len(loaded)} predictions: {names}")
    M = np.column_stack(loaded)  # (n_te, K)

    # Disagreement = std across the K models per row.
    disag = M.std(axis=1)
    print(f"\nDisagreement: min={disag.min():.3f}  med={np.median(disag):.3f}"
          f"  q75={np.quantile(disag, 0.75):.3f}  max={disag.max():.3f}")

    # 3 buckets: low (bottom 1/3), mid (middle), high (top 1/3).
    q33, q66 = np.quantile(disag, [1/3, 2/3])
    low = disag <= q33
    high = disag >= q66
    mid = ~low & ~high
    print(f"Buckets: low={low.sum()}, mid={mid.sum()}, high={high.sum()}")

    # Identify routing predictions
    nb239 = np.load(DATA_PROCESSED / "te_nb239_full_slsqp.npy")
    nb302 = np.load(DATA_PROCESSED / "te_nb302_full_pool.npy")
    # multi-modal weighted mean: weight inversely by abs deviation from row median.
    med = np.median(M, axis=1, keepdims=True)
    dev = np.abs(M - med)
    w_mm = 1.0 / (dev + 0.1)
    w_mm = w_mm / w_mm.sum(axis=1, keepdims=True)
    multi_modal = (M * w_mm).sum(axis=1)

    te_active = np.empty(n_te)
    te_active[low] = nb239[low]
    te_active[mid] = nb302[mid]
    te_active[high] = multi_modal[high]
    print(f"\nte_active: mean={te_active.mean():.3f}  std={te_active.std():.3f}")

    # OOF: copy nb302 OOF (proxy; routing is test-only).
    oof_nb302 = np.load(DATA_PROCESSED / "oof_nb302_full_pool.npy")
    np.save(DATA_PROCESSED / "oof_nb308_active.npy", oof_nb302)
    np.save(DATA_PROCESSED / "te_nb308_active.npy", te_active)
    r_self = rae(y, oof_nb302)
    sp_self, _ = spearmanr(y, oof_nb302)
    print(f"OOF (proxy=nb302): RAE={r_self:.4f}  Spearman={sp_self:.4f}")
    print(f"te_std={te_active.std():.3f}  te_mean={te_active.mean():.3f}")

    # 5-way SLSQP with nb239 base
    print("\n=== 5-way SLSQP with nb308 ===")
    nb224 = np.load(DATA_PROCESSED / "oof_nb224_pool_plus_2.npy")
    nb179s = np.load(DATA_PROCESSED / "oof_nb179_stack.npy")
    mtd  = np.load(DATA_PROCESSED / "oof_multi_template_delta.npy")
    loso = np.load(DATA_PROCESSED / "oof_delta_loso.npy")
    M_oof = np.column_stack([nb224, nb179s, mtd, loso, oof_nb302])

    te224 = np.load(DATA_PROCESSED / "te_nb224_pool_plus_2.npy")
    te179s = np.load(DATA_PROCESSED / "te_nb179_stack.npy")
    temtd  = np.load(DATA_PROCESSED / "te_oof_multi_template_delta.npy")
    teloso = np.load(DATA_PROCESSED / "te_oof_delta_loso.npy")
    M_te = np.column_stack([te224, te179s, temtd, teloso, te_active])

    nms = ['nb224', 'nb179s', 'mtd', 'loso', 'nb308']

    def loss(w): return rae(y, M_oof @ w)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0, 1.0)] * 5
    best = None
    for seed in range(150):
        rng = np.random.default_rng(seed)
        w0 = rng.dirichlet(np.ones(5))
        res = minimize(loss, w0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-10, 'maxiter': 300})
        if best is None or res.fun < best.fun: best = res
    pred_oof = M_oof @ best.x
    pred_te  = M_te  @ best.x
    r = rae(y, pred_oof); sp, _ = spearmanr(y, pred_oof)
    print(f"\n5-way SLSQP: OOF RAE={r:.4f}  Spearman={sp:.4f}  "
          f"te_std={pred_te.std():.3f}  te_mean={pred_te.mean():.3f}")
    for nm, w in zip(nms, best.x):
        print(f"  w[{nm}] = {w:.4f}")

    sub = pd.DataFrame({
        'Molecule Name': te_df['Molecule Name'],
        'SMILES': te_df['SMILES'],
        'pEC50': pred_te,
    })
    out = SUBMISSIONS / "nb308_active_learning.csv"
    sub.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
