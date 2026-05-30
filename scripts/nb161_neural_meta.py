"""nb161 — Neural Network Meta-Learner on Clean Base OOF.

GBDT meta-learners (LGBM, CatBoost, XGB) with base-only OOFs all achieve
~0.30-0.31 OOF RAE but collapse on test (ratio=0.57). Hypothesis:
GBDT selects specific OOF feature combinations that work on training
but don't generalize test variance.

Neural network with dropout may solve this because:
- Dropout forces diverse feature combinations (prevents reliance on correlated OOFs)
- BatchNorm normalizes activations, reducing sensitivity to specific feature values
- L2 weight reg prevents any single OOF from dominating

Architecture: OOF+assay → Linear(512) → BN → ReLU → Dropout(0.3)
             → Linear(256) → BN → ReLU → Dropout(0.2)
             → Linear(128) → ReLU → Linear(1)

Loss: MAE (directly optimizes competition metric)
"""
import os, sys, warnings
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pxr.data import load_train, load_test
from pxr.featurize import combined, impute
from pxr.eval import rae, scaffold_kfold_indices
from pxr.chem import bemis_murcko
from pxr.paths import DATA_PROCESSED, SUBMISSIONS

N_FOLDS = 5
COLLAPSE_THRESH = 0.58
DEVICE = "cpu"

META_STEMS = {
    "nb136_xgb_meta", "nb138_elnet_blend", "nb139_adaptive_blend",
    "nb140_xgb_lgbm_meta", "nb141_xgb_ablation", "nb142_xgb_calibrated",
    "nb143_oofassay_meta", "nb144_grand_v10", "nb145_level3_meta",
    "nb146_pca_oof_meta", "nb147_oofrdkit_meta", "nb148_meta_disagreement",
    "nb149_meta_maeloss", "nb150_residual_ensemble", "nb151_grand_v11",
    "nb152_lgbm_mae_tuned", "nb153_grand_v12", "nb154_lgbm_mae_filtered",
    "nb155_grand_v13", "nb156_catboost_mae", "nb157_optuna_lgbm_mae",
    "nb158_collapse_fix", "nb159_variance_scaled_ensemble", "nb160_pca_meta",
    "nb161_neural_meta",
    "nb112_grand_v3", "nb125_2way_blend", "nb127_grand_v5",
    "nb129_post_hoc_blend", "nb134_grand_v9",
}

LGBM_AUX = dict(
    n_estimators=400, num_leaves=32, learning_rate=0.05, min_child_samples=8,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4
)


class MetaNet(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), dropout=(0.3, 0.2, 0.1)):
        super().__init__()
        layers = []
        prev = in_dim
        for h, d in zip(hidden, dropout):
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(d)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_neural_meta(X_tr, y_tr, X_va, y_va, X_te, n_epochs=200, lr=3e-4, batch_size=256,
                      weight_decay=1e-4, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardize features using training statistics
    mu = X_tr.mean(axis=0); sigma = X_tr.std(axis=0) + 1e-8
    X_tr_s = (X_tr - mu) / sigma
    X_va_s = (X_va - mu) / sigma
    X_te_s = (X_te - mu) / sigma

    Xt = torch.tensor(X_tr_s, dtype=torch.float32)
    yt = torch.tensor(y_tr,   dtype=torch.float32)
    Xv = torch.tensor(X_va_s, dtype=torch.float32)
    yv = torch.tensor(y_va,   dtype=torch.float32)
    Xe = torch.tensor(X_te_s, dtype=torch.float32)

    model = MetaNet(in_dim=X_tr.shape[1])
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr/10)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    best_val_mae = 1e9
    best_state = None
    patience = 30; patience_ctr = 0

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = torch.mean(torch.abs(model(xb) - yb))
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv).numpy()
        val_mae = np.mean(np.abs(y_va - val_pred))
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        va_preds = model(Xv).numpy()
        te_preds = model(Xe).numpy()
    return va_preds, te_preds


def load_base_oofs(n_tr, y_tr, thresh=COLLAPSE_THRESH, exclude_stems=None):
    exclude_stems = exclude_stems or set()
    results = []
    for p in sorted(DATA_PROCESSED.glob("oof_*.npy")):
        stem = p.stem.replace("oof_", "")
        if stem in exclude_stems:
            continue
        for te_pref in ("te_", "te_oof_"):
            te_p = DATA_PROCESSED / f"{te_pref}{stem}.npy"
            if te_p.exists(): break
        else:
            continue
        try:
            oof = np.load(p).astype(np.float64)
            te  = np.load(te_p).astype(np.float64)
            if oof.ndim == 2: oof = oof[:, 0]
            if te.ndim == 2:  te  = te[:, 0]
            if len(oof) != n_tr: continue
            oof = np.where(np.isfinite(oof), oof, np.nanmean(oof))
            te  = np.where(np.isfinite(te),  te,  np.nanmean(te))
            ratio = te.std() / oof.std() if oof.std() > 0 else 0
            if ratio < thresh: continue
            r = rae(y_tr, oof)
            results.append(dict(stem=stem, oof=oof, te=te, ratio=ratio, rae=r))
        except Exception:
            pass
    results.sort(key=lambda x: x["rae"])
    return results


def build_assay_features(tr, te, splits, y_tr, n_tr):
    raw_train   = pd.read_csv("data/raw/pxr-challenge_TRAIN.csv")
    raw_counter = pd.read_csv("data/raw/pxr-challenge_counter-assay_TRAIN.csv")
    emax_col = "Emax.vs.pos.ctrl_estimate (dimensionless)"
    emax_raw    = raw_train[emax_col].values.astype(np.float64)
    emax_log    = np.log10(np.clip(emax_raw, 0.05, 10.0))
    counter_map = raw_counter.set_index("Molecule Name")["pEC50"].to_dict()
    mol_names   = raw_train["Molecule Name"].values
    pec50_null  = np.array([counter_map.get(n, np.nan) for n in mol_names], dtype=np.float64)
    null_imputed = np.where(np.isnan(pec50_null), np.nanmedian(pec50_null), pec50_null)
    selectivity  = y_tr - null_imputed
    has_null     = (~np.isnan(pec50_null)).astype(np.float32)
    X_str = impute(combined(tr["smiles"].tolist()))
    X_str_te = impute(combined(te["smiles"].tolist()))
    oof_emax = np.full(n_tr, np.nan); oof_null = np.full(n_tr, np.nan); oof_sel = np.full(n_tr, np.nan)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        m_em = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=emax_log[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_emax[va_idx] = 10.0 ** m_em.predict(X_str[va_idx])
        m_nl = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=null_imputed[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_null[va_idx] = m_nl.predict(X_str[va_idx])
        m_sl = lgb.train(LGBM_AUX, lgb.Dataset(X_str[tr_idx], label=selectivity[tr_idx]),
                         callbacks=[lgb.log_evaluation(-1)])
        oof_sel[va_idx] = m_sl.predict(X_str[va_idx])
    m_em_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=emax_log), callbacks=[lgb.log_evaluation(-1)])
    m_nl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=null_imputed), callbacks=[lgb.log_evaluation(-1)])
    m_sl_f = lgb.train(LGBM_AUX, lgb.Dataset(X_str, label=selectivity), callbacks=[lgb.log_evaluation(-1)])
    te_emax = 10.0 ** m_em_f.predict(X_str_te)
    te_null = m_nl_f.predict(X_str_te)
    te_sel  = m_sl_f.predict(X_str_te)
    assay_oof = np.column_stack([oof_emax, oof_null, oof_sel, has_null, np.log1p(np.clip(oof_emax, 0, None))])
    assay_te  = np.column_stack([te_emax, te_null, te_sel, np.zeros(len(X_str_te)), np.log1p(np.clip(te_emax, 0, None))])
    return assay_oof, assay_te


def main():
    print("=== nb161: Neural Network Meta-Learner on Clean Base OOF ===\n")

    tr = load_train(); te = load_test()
    y_tr = tr["pec50"].values.astype(np.float64)
    n_tr = len(y_tr)
    scaffolds = tr["smiles"].map(bemis_murcko).tolist()
    splits = scaffold_kfold_indices(scaffolds, N_FOLDS, 42)

    print("Building assay features...")
    assay_oof, assay_te = build_assay_features(tr, te, splits, y_tr, n_tr)

    print("Loading base model OOFs (exclude meta-stacks)...")
    mods = load_base_oofs(n_tr, y_tr, exclude_stems=META_STEMS)
    print(f"  {len(mods)} base models loaded")
    oof_mat = np.column_stack([m["oof"] for m in mods])
    te_mat  = np.column_stack([m["te"]  for m in mods])
    X_tr_all = np.hstack([oof_mat, assay_oof])
    X_te_all  = np.hstack([te_mat, assay_te])
    print(f"  Meta features: {X_tr_all.shape}")

    results = {}

    # A: Single neural network config
    print("\n--- A: Neural meta (512→256→128, dropout 0.3/0.2/0.1) ---")
    oof_nn = np.full(n_tr, np.nan)
    te_nn_folds = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        va_pred, te_pred = train_neural_meta(
            X_tr_all[tr_idx].astype(np.float32),
            y_tr[tr_idx].astype(np.float32),
            X_tr_all[va_idx].astype(np.float32),
            y_tr[va_idx].astype(np.float32),
            X_te_all.astype(np.float32),
            n_epochs=200, lr=3e-4, batch_size=128, weight_decay=1e-4, seed=fold
        )
        oof_nn[va_idx] = va_pred
        te_nn_folds.append(te_pred)
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], va_pred):.4f}", flush=True)
    te_nn_a = np.mean(te_nn_folds, axis=0)
    r_a = rae(y_tr, oof_nn)
    ratio_a = te_nn_a.std() / oof_nn.std()
    flag = "PASS" if ratio_a >= COLLAPSE_THRESH else "FAIL"
    print(f"  [A: neural meta] RAE={r_a:.4f}  ratio={ratio_a:.2f}  [{flag}]", flush=True)
    results["A_neural"] = (r_a, oof_nn.copy(), te_nn_a.copy(), ratio_a)

    # B: Larger network (1024→512→256→128)
    print("\n--- B: Larger neural meta (1024→512→256) ---")
    class BigMetaNet(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    def train_big(X_tr, y_tr, X_va, y_va, X_te, seed=0):
        torch.manual_seed(seed)
        mu = X_tr.mean(0); sigma = X_tr.std(0) + 1e-8
        Xt = torch.tensor(((X_tr - mu)/sigma).astype(np.float32))
        yt = torch.tensor(y_tr.astype(np.float32))
        Xv = torch.tensor(((X_va - mu)/sigma).astype(np.float32))
        yv_arr = y_va
        Xe = torch.tensor(((X_te - mu)/sigma).astype(np.float32))
        model = BigMetaNet(X_tr.shape[1])
        opt   = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-4)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=128, shuffle=True)
        best_val = 1e9; best_state = None; patience_ctr = 0
        for epoch in range(200):
            model.train()
            for xb, yb in loader:
                opt.zero_grad()
                torch.mean(torch.abs(model(xb) - yb)).backward()
                opt.step()
            sched.step()
            model.eval()
            with torch.no_grad():
                vp = model(Xv).numpy()
            v_mae = np.mean(np.abs(yv_arr - vp))
            if v_mae < best_val: best_val = v_mae; best_state = {k:v.clone() for k,v in model.state_dict().items()}; patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= 30: break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            return model(Xv).numpy(), model(Xe).numpy()

    oof_b = np.full(n_tr, np.nan); te_b_folds = []
    for fold, (tr_idx, va_idx) in enumerate(splits):
        vp, tp = train_big(X_tr_all[tr_idx], y_tr[tr_idx], X_tr_all[va_idx], y_tr[va_idx], X_te_all, seed=fold)
        oof_b[va_idx] = vp; te_b_folds.append(tp)
        print(f"    fold {fold+1}  RAE={rae(y_tr[va_idx], vp):.4f}", flush=True)
    te_b = np.mean(te_b_folds, axis=0)
    r_b = rae(y_tr, oof_b); ratio_b = te_b.std() / oof_b.std()
    flag = "PASS" if ratio_b >= COLLAPSE_THRESH else "FAIL"
    print(f"  [B: big neural] RAE={r_b:.4f}  ratio={ratio_b:.2f}  [{flag}]")
    results["B_big_neural"] = (r_b, oof_b.copy(), te_b.copy(), ratio_b)

    # C: Ensemble A+B
    oof_c = (oof_nn + oof_b) / 2; te_c = (te_nn_a + te_b) / 2
    r_c = rae(y_tr, oof_c); ratio_c = te_c.std() / oof_c.std()
    flag = "PASS" if ratio_c >= COLLAPSE_THRESH else "FAIL"
    print(f"  [C: neural ensemble A+B] RAE={r_c:.4f}  ratio={ratio_c:.2f}  [{flag}]")
    results["C_neural_ens"] = (r_c, oof_c, te_c, ratio_c)

    # Summary
    print(f"\n=== Summary ===")
    for k, (r, _, _, ratio) in sorted(results.items(), key=lambda x: x[1][0]):
        flag = "PASS" if ratio >= COLLAPSE_THRESH else "FAIL"
        print(f"  {k:55s}  RAE={r:.4f}  ratio={ratio:.2f}  [{flag}]")

    valid = {k: v for k, v in results.items() if v[3] >= COLLAPSE_THRESH}
    if not valid: valid = results
    best_label = min(valid, key=lambda k: valid[k][0])
    best_r, best_oof, best_te, best_ratio = valid[best_label]
    print(f"\nBest: {best_label}  OOF RAE={best_r:.4f}  ratio={best_ratio:.2f}")
    print(f"(nb149: 0.3069, nb155: 0.3044)")

    best_te_out = np.clip(best_te, y_tr.min() - 0.5, y_tr.max() + 0.5)
    np.save(DATA_PROCESSED / "oof_nb161_neural_meta.npy", best_oof)
    np.save(DATA_PROCESSED / "te_nb161_neural_meta.npy",  best_te_out)
    sub = pd.DataFrame({"Molecule Name": te["name"].values, "pEC50": best_te_out})
    sub.to_csv(SUBMISSIONS / "161_neural_meta.csv", index=False)
    print(f"\nSaved: submissions/161_neural_meta.csv  OOF RAE={best_r:.4f}")


if __name__ == "__main__":
    main()
