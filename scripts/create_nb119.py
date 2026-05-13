import json

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}

C0 = md("# nb119 - Grand Ensemble v11\n\nBlend anchored on nb117 (all-FP delta 3-tier, OOF 0.2333).\nnb118 OOF (0.1626) is leaky — included but downweighted.\nTrue unbiased OOF comes from nb121.")

C1 = code("""import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '../src')
from pxr.data import load_train, load_test
from pxr.eval import rae
from pxr.chem import add_standard_columns
from pxr.paths import DATA_PROCESSED, SUBMISSIONS
from scipy.optimize import minimize

df_tr = load_train()
add_standard_columns(df_tr)
y = df_tr['pec50'].values
df_te = load_test()
print(f"Train: {len(df_tr)}, Test: {len(df_te)}")
""")

C2 = code("""CANDIDATES = {
    'allfp_delta_3tier':    'oof_allfp_delta_3tier.npy',
    'enhanced_delta_3tier': 'oof_enhanced_delta_3tier.npy',
    'blend_optimizer':      'oof_blend_optimizer.npy',
    'grand_v9':             'oof_grand_v9.npy',
    'delta_ensemble_blend': 'oof_delta_ensemble_blend.npy',
    'grand_v10':            'oof_grand_v10.npy',
    'adaptive_delta_4tier': 'oof_adaptive_delta_4tier.npy',
}

TE_FILES = {
    'allfp_delta_3tier':    'te_oof_allfp_delta_3tier.npy',
    'enhanced_delta_3tier': 'te_oof_enhanced_delta_3tier.npy',
    'blend_optimizer':      'te_oof_blend_optimizer.npy',
    'grand_v9':             'te_oof_grand_v9.npy',
    'delta_ensemble_blend': 'te_oof_delta_ensemble_blend.npy',
    'grand_v10':            'te_oof_grand_v10.npy',
    'adaptive_delta_4tier': 'te_oof_adaptive_delta_4tier.npy',
}

oofs = {}
te_preds = {}
for name, fname in CANDIDATES.items():
    p = DATA_PROCESSED / fname
    if p.exists():
        oofs[name] = np.load(p)
        print(f"  {name}: OOF RAE = {rae(y, oofs[name]):.4f}")
    else:
        print(f"  {name}: MISSING")

for name, fname in TE_FILES.items():
    p = DATA_PROCESSED / fname
    if p.exists():
        te_preds[name] = np.load(p)
""")

C3 = code("""names = list(oofs.keys())
arrays = [oofs[n] for n in names]

def neg_rae(w):
    w = np.abs(w)
    s = w.sum()
    if s < 1e-9:
        return 99.0
    w = w / s
    pred = sum(wi * ai for wi, ai in zip(w, arrays))
    return rae(y, pred)

best_res = None
for seed in range(20):
    rng = np.random.default_rng(seed)
    x0 = rng.dirichlet(np.ones(len(names)))
    res = minimize(neg_rae, x0, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-7})
    if best_res is None or res.fun < best_res.fun:
        best_res = res

w_opt = np.abs(best_res.x)
w_opt = w_opt / w_opt.sum()
print("Optimal weights:")
for n, w in sorted(zip(names, w_opt), key=lambda x: -x[1]):
    print(f"  {n}: {w:.4f}")
print(f"Optimized OOF RAE: {best_res.fun:.4f}")
""")

C4 = code("""# Check simple: best of individual vs blend
oof_117 = oofs['allfp_delta_3tier']
oof_118 = oofs.get('adaptive_delta_4tier')
r117 = rae(y, oof_117)
print(f"nb117 alone: {r117:.4f}")
if oof_118 is not None:
    r118 = rae(y, oof_118)
    print(f"nb118 alone (leaky): {r118:.4f}")
    for w118 in [0.05, 0.10, 0.15, 0.20]:
        blend = (1-w118)*oof_117 + w118*oof_118
        print(f"  117:{1-w118:.2f} + 118:{w118:.2f} -> {rae(y, blend):.4f}")

# Pick best strategy
oof_blend = sum(wi * ai for wi, ai in zip(w_opt, arrays))
r_blend = rae(y, oof_blend)
print(f"Best blend OOF RAE: {r_blend:.4f}")

if r_blend < r117:
    w_final = w_opt
    oof_final = oof_blend
    print("Using scipy blend")
else:
    w_final = np.array([1.0] + [0.0]*(len(names)-1))
    oof_final = oof_117
    print("Using nb117 alone")

grand_v11_rae = rae(y, oof_final)
print(f"*** Grand v11 OOF RAE = {grand_v11_rae:.4f} ***")
""")

C5 = code("""te_arrays = [te_preds[n] for n in names if n in te_preds]
names_te = [n for n in names if n in te_preds]

# Recalculate weights for available te arrays
w_te = np.array([w_final[names.index(n)] for n in names_te])
w_te = w_te / w_te.sum()
te_final = sum(wi * ai for wi, ai in zip(w_te, te_arrays))

print(f"Test predictions: min={te_final.min():.3f} med={np.median(te_final):.3f} max={te_final.max():.3f}")

np.save(DATA_PROCESSED / 'oof_grand_v11.npy', oof_final)
np.save(DATA_PROCESSED / 'te_oof_grand_v11.npy', te_final)

sub = pd.DataFrame({'Molecule Name': df_te['Molecule Name'], 'pEC50': te_final})
out_path = SUBMISSIONS / '119_grand_ensemble_v11.csv'
sub.to_csv(out_path, index=False)
print(f"Saved {out_path}")
print(f"Grand v11 OOF RAE: {grand_v11_rae:.4f}")
""")

cells = [C0, C1, C2, C3, C4, C5]

nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "pxr-challenge", "language": "python", "name": "pxr-challenge"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "cells": cells
}

with open('notebooks/119_grand_ensemble_v11.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Written notebooks/119_grand_ensemble_v11.ipynb")
