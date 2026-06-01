"""Phase-1 post-mortem: build the shared analysis substrate.

Run ONCE. Produces compact, verified artifacts under data/processed/postmortem/
that every post-mortem notebook reads read-only. No notebook recomputes these.

Sources (all verified before writing):
  - data/raw/pxr-challenge_TEST_BLINDED.csv           (513 test, SMILES order)
  - data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv (253 now-revealed truths)
  - data/processed/_audit_oof_pairs.csv               (344 model -> oof/te .npy)
  - data/processed/_audit_unblind_idx.npy / _y.npy    (index into 513, truth)
  - src/pxr (load_train, chem)
"""
import os, re, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, 'src')
import pxr.data as D
import pxr.chem as C
from pxr.eval import rae

BS = chr(92)
OUT = 'data/processed/postmortem'
os.makedirs(OUT, exist_ok=True)

def norm(p): return p.replace(BS, '/')

# ---------------------------------------------------------------- inputs
test = pd.read_csv('data/raw/pxr-challenge_TEST_BLINDED.csv')          # 513
unb  = pd.read_csv('data/raw/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv')# 253
idx  = np.load('data/processed/_audit_unblind_idx.npy')               # 253 -> [0,513)
yth  = np.load('data/processed/_audit_unblind_y.npy')                 # 253 truth
au   = pd.read_csv('data/processed/_audit_oof_pairs.csv')             # 344 models
tr   = D.load_train()                                                  # 4139
assert len(test) == 513 and len(idx) == 253 and len(tr) == 4139

# verify mapping: idx/y consistent with the unblinded CSV (defensive)
name2pos = {n: i for i, n in enumerate(test['Molecule Name'])}
unb_pos  = unb['Molecule Name'].map(name2pos).values
yy = unb.set_index('Molecule Name')['pEC50']
y_csv = np.array([yy[test['Molecule Name'][i]] for i in idx])
assert np.array_equal(np.sort(unb_pos), np.sort(idx)), "unblind idx mismatch"
assert np.allclose(y_csv, yth), "unblind y mismatch"
print(f"[ok] unblind mapping verified: 253 truths, mean {yth.mean():.3f} std {yth.std():.3f}")

# verify OOF row order aligns to load_train order via a known model
y_train = tr['pec50'].values.astype(float)
chk = au[au['name'] == 'oof_lgbm_base'].iloc[0]
oof_chk = np.load(norm(chk['train_oof_path']))
assert abs(rae(y_train, oof_chk) - chk['train_oof_rae']) < 1e-4, "OOF order != load_train order"
print("[ok] OOF arrays align to load_train() row order")

# ---------------------------------------------------------------- chemistry (all 513)
smi_test = test['SMILES'].tolist()
smi_train = tr['smiles'].tolist()
fp_test = C.morgan_fp_batch(smi_test).astype(np.float32)   # (513,2048) 0/1
fp_train = C.morgan_fp_batch(smi_train).astype(np.float32) # (4139,2048)

# Tanimoto novelty: test vs train
inter = fp_test @ fp_train.T                  # (513,4139)
a = fp_test.sum(1); b = fp_train.sum(1)
tan = inter / (a[:, None] + b[None, :] - inter + 1e-9)
nn_sim = tan.max(1)
nn_arg = tan.argmax(1)
nn_name = [tr['name'].iloc[j] for j in nn_arg]
n_analog_50 = (tan >= 0.50).sum(1)
n_analog_70 = (tan >= 0.70).sum(1)

# scaffolds
scaf_test = [C.bemis_murcko(s) for s in smi_test]
gscaf_test = [C.bemis_murcko(s, generic=True) for s in smi_test]
scaf_train = [C.bemis_murcko(s) for s in smi_train]
from collections import Counter
scaf_freq = Counter([s for s in scaf_train if s])
scaf_train_freq = np.array([scaf_freq.get(s, 0) for s in scaf_test])

# physchem
phys = pd.DataFrame([C.compute_physchem(s) or {} for s in smi_test])

chem = pd.DataFrame({
    'name': test['Molecule Name'].values,
    'smiles': smi_test,
    'pos': np.arange(513),
    'scaffold': scaf_test,
    'generic_scaffold': gscaf_test,
    'nn_sim_train': nn_sim,
    'nn_train_name': nn_name,
    'n_analog_sim50': n_analog_50,
    'n_analog_sim70': n_analog_70,
    'scaf_train_freq': scaf_train_freq,
    'scaf_novel': scaf_train_freq == 0,
})
chem = pd.concat([chem, phys], axis=1)
chem['is_unblind'] = chem['pos'].isin(set(idx.tolist()))
chem.to_parquet(f'{OUT}/pm_test_chem_all513.parquet')
print(f"[ok] chemistry for 513 test compounds  (unblind={chem.is_unblind.sum()})")

# ---------------------------------------------------------------- prediction matrices
names, te_cols, oof_cols, rows = [], [], [], []
LOW, HIGH = 3.5, 5.5   # truth activity bins: low / mid / high
bin_lo = yth < LOW; bin_hi = yth >= HIGH; bin_mid = ~bin_lo & ~bin_hi

def fam_of(n):
    s = n.lower()
    pairs = [
        ('boltz|struct|pocket|dock|pose|lddt|pcs', 'structure3D'),
        ('chemberta|grover|esm|protbert|bert|foundation|molformer|unimol', 'foundation_emb'),
        ('chemprop|mpnn|gnn|hetgnn|dmpnn|graph', 'gnn'),
        ('dann|mope|soci|karpathy|cep|adversar', 'domain_adapt'),
        ('delta|loso|anchor', 'delta_ml'),
        ('mmp|free_wilson|freewilson|matched', 'mmp_fragment'),
        ('knn|neighbor|analog|tanimoto', 'knn_analogy'),
        ('grand|slsqp|blend|stack|ensemble|fusion|meta|bma|huber', 'blend_stack'),
        ('chembl|pubchem|bindingdb|external|papyrus|aug|pseudo|transfer|nr_|multi_nr', 'external_aug'),
        ('catboost|xgb|xgboost|lgbm|gbm|rf|forest', 'gbdt'),
        ('emax|counter|selectivity|assay|aux|free|hardness|quantile|stochastic|width|recap|residual', 'aux_signal'),
    ]
    for pat, fam in pairs:
        if re.search(pat, s): return fam
    return 'other'

def safe_slope(p, y):
    try:
        if p.std() < 1e-6: return np.nan
        return float(np.polyfit(p, y, 1)[0])
    except Exception:
        return np.nan

def safe_corr(p, y):
    try:
        if p.std() < 1e-6: return np.nan
        return float(np.corrcoef(p, y)[0, 1])
    except Exception:
        return np.nan

for _, r in au.iterrows():
    name = r['name']
    te = np.load(norm(r['te_path']))
    oof = np.load(norm(r['train_oof_path']))
    if te.ndim != 1 or te.shape[0] != 513 or oof.shape[0] != 4139:
        print(f"  [skip-shape] {name}: te{te.shape} oof{oof.shape}")
        continue
    p = te[idx]                       # 253 unblind preds
    res = p - yth                     # residual (pred - truth)
    finite = bool(np.all(np.isfinite(p)) and np.all(np.isfinite(oof)))
    te_rae = float(rae(yth, p)) if finite else np.nan
    # in_matrix: well-behaved enough to enter consensus / blends / matrices
    in_matrix = bool(finite and p.std() > 1e-6 and np.isfinite(te_rae) and te_rae < 5.0)
    nbm = re.search(r'nb(\d+)', name.lower())
    rows.append(dict(
        name=name, family=fam_of(name),
        nb=int(nbm.group(1)) if nbm else -1,
        finite=finite, in_matrix=in_matrix,
        train_oof_rae=float(r['train_oof_rae']),
        te_unblind_rae=te_rae,
        pred_std=float(p.std()) if finite else np.nan,
        pred_mean=float(p.mean()) if finite else np.nan,
        bias=float(res.mean()) if finite else np.nan,
        mae=float(np.abs(res).mean()) if finite else np.nan,
        pearson=safe_corr(p, yth),
        slope=safe_slope(p, yth),   # truth ~ a + b*pred ; b<1 => compression
        bias_low=float(res[bin_lo].mean()) if finite else np.nan,   # >0 => over-predict inactives
        bias_mid=float(res[bin_mid].mean()) if finite else np.nan,
        bias_high=float(res[bin_hi].mean()) if finite else np.nan,  # <0 => under-predict actives
        mae_low=float(np.abs(res[bin_lo]).mean()) if finite else np.nan,
        mae_mid=float(np.abs(res[bin_mid]).mean()) if finite else np.nan,
        mae_high=float(np.abs(res[bin_hi]).mean()) if finite else np.nan,
    ))
    if in_matrix:
        names.append(name); te_cols.append(p); oof_cols.append(oof)

meta = pd.DataFrame(rows)
pred_unblind = np.array(te_cols).T          # (253, K)
oof_train = np.array(oof_cols).T            # (4139, K)
resid_unblind = pred_unblind - yth[:, None]
K = len(names)
print(f"[ok] built matrices: pred_unblind {pred_unblind.shape}, oof_train {oof_train.shape}")
print(f"     truth-std={yth.std():.3f}  |  median pred_std={meta.pred_std.median():.3f}  median slope={meta.slope.median():.3f}")

# consensus over the legitimate predictors (te_unblind_rae < 0.72) among matrix cols
name_to_col = {n: i for i, n in enumerate(names)}
meta_m = meta[meta['name'].isin(names)].copy()
meta_m['col'] = meta_m['name'].map(name_to_col)
good_cols = meta_m.loc[meta_m['te_unblind_rae'] < 0.72, 'col'].values
n_good = len(good_cols)
chem_unb = chem[chem.is_unblind].copy().sort_values('pos').reset_index(drop=True)
# align chem_unb rows to the idx order used by yth/pred matrices
order = {p: k for k, p in enumerate(idx)}
chem_unb['k'] = chem_unb['pos'].map(order)
chem_unb = chem_unb.sort_values('k').reset_index(drop=True)
chem_unb['truth'] = yth
chem_unb['consensus_pred'] = pred_unblind[:, good_cols].mean(1)
chem_unb['consensus_std'] = pred_unblind[:, good_cols].std(1)        # model disagreement
chem_unb['consensus_resid'] = chem_unb['consensus_pred'] - chem_unb['truth']
chem_unb['abs_resid'] = chem_unb['consensus_resid'].abs()
chem_unb.to_parquet(f'{OUT}/pm_compounds.parquet')
print(f"[ok] consensus over {n_good} legit models (rae<0.72): "
      f"RAE={rae(yth, chem_unb.consensus_pred.values):.4f}")

# representative model per family = best te_unblind_rae among finite models
mfin = meta[meta['finite'] & meta['te_unblind_rae'].notna()]
reps = mfin.loc[mfin.groupby('family')['te_unblind_rae'].idxmin()].sort_values('te_unblind_rae')
reps[['family', 'name', 'nb', 'train_oof_rae', 'te_unblind_rae']].to_csv(f'{OUT}/pm_family_reps.csv', index=False)

meta.to_parquet(f'{OUT}/pm_model_meta.parquet')
np.save(f'{OUT}/pm_pred_unblind.npy', pred_unblind)
np.save(f'{OUT}/pm_resid_unblind.npy', resid_unblind)
np.save(f'{OUT}/pm_oof_train.npy', oof_train)
np.save(f'{OUT}/pm_unblind_y.npy', yth)
np.save(f'{OUT}/pm_train_y.npy', y_train)
with open(f'{OUT}/pm_model_names.txt', 'w') as f:
    f.write('\n'.join(names))
tr[['name', 'smiles', 'pec50']].to_parquet(f'{OUT}/pm_train_min.parquet')
pd.DataFrame({'scaffold': scaf_train}).to_parquet(f'{OUT}/pm_train_scaffolds.parquet')

with open(f'{OUT}/pm_meta.json', 'w') as f:
    json.dump(dict(K=K, n_unblind=253, n_test=513, n_train=4139,
                   truth_std=float(yth.std()), truth_mean=float(yth.mean()),
                   bins=dict(low=f'<{LOW}', mid=f'{LOW}-{HIGH}', high=f'>={HIGH}',
                             n_low=int(bin_lo.sum()), n_mid=int(bin_mid.sum()), n_high=int(bin_hi.sum())),
                   n_good=int(n_good)), f, indent=2)

print("\n=== FAMILY SUMMARY (best model per family) ===")
print(reps[['family', 'name', 'nb', 'te_unblind_rae']].to_string(index=False))
print("\n=== FAMILY-LEVEL unblind RAE (median over members) ===")
print(meta.groupby('family')['te_unblind_rae'].agg(['count', 'median', 'min']).sort_values('median').to_string())
print(f"\nbins: low(<{LOW})={int(bin_lo.sum())} mid={int(bin_mid.sum())} high(>={HIGH})={int(bin_hi.sum())}")
print("ALL ARTIFACTS WRITTEN TO", OUT)
