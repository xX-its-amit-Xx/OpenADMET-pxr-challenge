"""
nb1311_autogluon.py
AutoGluon TabularPredictor AutoML sweep on 4139 train combined features (2265-dim).
Evaluates on 253 unblinded test compounds. Falls back to H2O then sklearn if needed.
"""

import sys, os, json, time, subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, 'D:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge')

from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae

# ── Output dirs ──────────────────────────────────────────────────────────────
OUT_DIR = "C:/pxr_work/meta_stacking"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading train/test data...")
tr = load_train()
te = load_test()

train_smiles = tr['smiles'].tolist()
test_smiles  = te['smiles'].tolist()
y_train      = tr['pec50'].values.astype(np.float32)

print(f"Train: {len(tr)} rows | Test: {len(te)} rows")

# ── Featurize ─────────────────────────────────────────────────────────────────
print("Computing combined features (2265-dim)...")
t0 = time.time()
Xtr = combined(train_smiles).astype(np.float32)
Xte = combined(test_smiles).astype(np.float32)
Xtr = impute(Xtr).astype(np.float32)
Xte = impute(Xte).astype(np.float32)
print(f"Features computed in {time.time()-t0:.1f}s | shape: {Xtr.shape}")

# ── Load 253 unblinded labels ──────────────────────────────────────────────────
unblind_idx = np.load('data/processed/_audit_unblind_idx.npy')   # (253,) index into test (513)
unblind_y   = np.load('data/processed/_audit_unblind_y.npy')     # (253,) true pEC50

Xte_253 = Xte[unblind_idx]
y_253   = unblind_y

feat_cols = [f'f{i}' for i in range(Xtr.shape[1])]

# ── Build DataFrames ──────────────────────────────────────────────────────────
train_df = pd.DataFrame(Xtr, columns=feat_cols)
train_df['pec50'] = y_train

test253_df = pd.DataFrame(Xte_253, columns=feat_cols)
test513_df = pd.DataFrame(Xte, columns=feat_cols)

results = {}
pred_te513 = None

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — AutoGluon TabularPredictor
# ═══════════════════════════════════════════════════════════════════════════════
HAS_AG = False
try:
    from autogluon.tabular import TabularPredictor
    HAS_AG = True
    print("\n[AutoGluon] Available — running TabularPredictor (10 min budget)...")
except ImportError:
    print("[AutoGluon] Not available — attempting pip install...")
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 'autogluon.tabular', '--quiet'],
        capture_output=True, text=True
    )
    print(r.stdout[-500:] if r.stdout else "")
    print(r.stderr[-500:] if r.stderr else "")
    try:
        from autogluon.tabular import TabularPredictor
        HAS_AG = True
        print("[AutoGluon] Installed successfully.")
    except ImportError:
        print("[AutoGluon] Install FAILED — will try fallbacks.")

if HAS_AG:
    ag_path = os.path.join(OUT_DIR, "ag_predictor")
    t0 = time.time()
    try:
        predictor = TabularPredictor(
            label='pec50',
            eval_metric='mean_absolute_error',
            path=ag_path,
            verbosity=2
        ).fit(
            train_df,
            time_limit=600,          # 10 min
            presets='medium_quality',
            num_cpus=4,
        )

        # Leaderboard
        lb = predictor.leaderboard(silent=True)
        print("\n[AutoGluon] Leaderboard:")
        print(lb.to_string())

        # Predict 513
        pred_te513 = predictor.predict(test513_df).values.astype(np.float64)

        # Evaluate on 253
        pred_253 = pred_te513[unblind_idx]
        ag_rae = float(rae(y_253, pred_253))
        elapsed = time.time() - t0
        print(f"\n[AutoGluon] 253-unblind RAE: {ag_rae:.4f}  (elapsed {elapsed:.0f}s)")

        # Best model name
        best_model = lb.iloc[0]['model'] if len(lb) else "unknown"
        results['autogluon'] = {
            'status': 'success',
            'rae_253': ag_rae,
            'best_model': best_model,
            'leaderboard': lb[['model','score_val']].head(10).to_dict('records'),
            'elapsed_s': elapsed,
        }
    except Exception as e:
        print(f"[AutoGluon] ERROR: {e}")
        results['autogluon'] = {'status': 'error', 'error': str(e)}
        HAS_AG = False   # allow fallback below

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — H2O AutoML (fallback if AutoGluon unavailable/failed)
# ═══════════════════════════════════════════════════════════════════════════════
if not HAS_AG:
    HAS_H2O = False
    try:
        import h2o
        from h2o.automl import H2OAutoML
        HAS_H2O = True
    except ImportError:
        print("[H2O] Not available — skipping H2O AutoML.")

    if HAS_H2O:
        print("\n[H2O] Running H2O AutoML (10 min budget)...")
        try:
            h2o.init(max_mem_size='4G', nthreads=4)
            h2o_train = h2o.H2OFrame(train_df)
            h2o_test  = h2o.H2OFrame(test513_df)

            aml = H2OAutoML(max_runtime_secs=600, seed=42, verbosity='warn')
            aml.train(x=feat_cols, y='pec50', training_frame=h2o_train)

            lb_h2o = aml.leaderboard.as_data_frame()
            print("\n[H2O] Leaderboard (top 5):")
            print(lb_h2o.head(5).to_string())

            pred_te513 = aml.predict(h2o_test).as_data_frame()['predict'].values.astype(np.float64)
            pred_253   = pred_te513[unblind_idx]
            h2o_rae    = float(rae(y_253, pred_253))
            print(f"[H2O] 253-unblind RAE: {h2o_rae:.4f}")

            results['h2o'] = {
                'status': 'success',
                'rae_253': h2o_rae,
                'best_model': lb_h2o.iloc[0]['model_id'] if len(lb_h2o) else 'unknown',
            }
            h2o.shutdown(prompt=False)
        except Exception as e:
            print(f"[H2O] ERROR: {e}")
            results['h2o'] = {'status': 'error', 'error': str(e)}

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Sklearn comprehensive pipeline (always runs as reference / fallback)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[sklearn] Running comprehensive sklearn pipeline for comparison...")
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

sk_results = {}

# MLP
print("  Fitting MLPRegressor(256,128,64)...")
mlp = Pipeline([
    ('sc', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(256, 128, 64), max_iter=500,
                          learning_rate_init=0.001, random_state=42, early_stopping=True,
                          validation_fraction=0.1, n_iter_no_change=20))
])
mlp.fit(Xtr, y_train)
p_mlp = mlp.predict(Xte_253)
r_mlp = float(rae(y_253, p_mlp))
print(f"    MLP RAE: {r_mlp:.4f}")
sk_results['mlp'] = r_mlp

# Bagging of DTs
print("  Fitting BaggingRegressor(DT depth=8, n=200)...")
bag = BaggingRegressor(
    estimator=DecisionTreeRegressor(max_depth=8),
    n_estimators=200, random_state=42, n_jobs=4
)
bag.fit(Xtr, y_train)
p_bag = bag.predict(Xte_253)
r_bag = float(rae(y_253, p_bag))
print(f"    Bagging RAE: {r_bag:.4f}")
sk_results['bagging_dt'] = r_bag

# Random Forest
print("  Fitting RandomForestRegressor(n=300)...")
rf = RandomForestRegressor(n_estimators=300, max_features='sqrt', random_state=42, n_jobs=4)
rf.fit(Xtr, y_train)
p_rf = rf.predict(Xte_253)
r_rf = float(rae(y_253, p_rf))
print(f"    RF RAE: {r_rf:.4f}")
sk_results['random_forest'] = r_rf

# VotingRegressor with available estimators
voters = [('rf', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=2))]
if HAS_LGB:
    voters.append(('lgb', lgb.LGBMRegressor(n_estimators=500, num_leaves=64,
                                              learning_rate=0.05, random_state=42,
                                              verbosity=-1)))
if HAS_XGB:
    voters.append(('xgb', xgb.XGBRegressor(n_estimators=400, max_depth=6,
                                             learning_rate=0.05, random_state=42,
                                             verbosity=0)))
voters.append(('mlp', Pipeline([
    ('sc', StandardScaler()),
    ('mlp', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42))
])))

if len(voters) >= 2:
    print(f"  Fitting VotingRegressor ({[v[0] for v in voters]})...")
    voter = VotingRegressor(voters, n_jobs=1)
    voter.fit(Xtr, y_train)
    p_vote = voter.predict(Xte_253)
    r_vote = float(rae(y_253, p_vote))
    print(f"    Voting RAE: {r_vote:.4f}")
    sk_results['voting'] = r_vote

best_sk_name = min(sk_results, key=sk_results.get)
best_sk_rae  = sk_results[best_sk_name]
print(f"\n[sklearn] Best: {best_sk_name} RAE={best_sk_rae:.4f}")
results['sklearn'] = {'all': sk_results, 'best_name': best_sk_name, 'best_rae': best_sk_rae}

# ═══════════════════════════════════════════════════════════════════════════════
# Save predictions & results
# ═══════════════════════════════════════════════════════════════════════════════
if pred_te513 is not None:
    out_npy = os.path.join(OUT_DIR, 'autogluon_te_513.npy')
    np.save(out_npy, pred_te513)
    print(f"\nSaved 513 predictions: {out_npy}")
else:
    # fall back to best sklearn on full 513
    best_sk_map = {
        'mlp': mlp, 'bagging_dt': bag, 'random_forest': rf
    }
    best_sk_model = best_sk_map.get(best_sk_name, rf)
    pred_te513 = best_sk_model.predict(Xte).astype(np.float64)
    out_npy = os.path.join(OUT_DIR, 'autogluon_te_513.npy')
    np.save(out_npy, pred_te513)
    print(f"\nSaved 513 sklearn fallback predictions: {out_npy}")

# Summary JSON
out_json = os.path.join(OUT_DIR, 'autogluon_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"Saved results JSON: {out_json}")

# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
if 'autogluon' in results and results['autogluon'].get('status') == 'success':
    ag = results['autogluon']
    print(f"AutoGluon best model : {ag['best_model']}")
    print(f"AutoGluon RAE (253)  : {ag['rae_253']:.4f}")
if 'h2o' in results and results['h2o'].get('status') == 'success':
    print(f"H2O best model       : {results['h2o']['best_model']}")
    print(f"H2O RAE (253)        : {results['h2o']['rae_253']:.4f}")
print(f"Sklearn best         : {best_sk_name}  RAE={best_sk_rae:.4f}")
print(f"Sklearn all          : { {k: f'{v:.4f}' for k, v in sk_results.items()} }")
print("="*60)
