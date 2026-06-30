"""nb1174 — HONEST gate for AIMNet2 QM/NNP physics scalars (model-queen physics axis).

Validate like rich-z / xTB (nb1172): MARGINAL-over-anchor, NOT standalone RAE. Mirrors the
nb1130 clean-holdout convention (combined-only leakage-free GBMs + ChemProp GNN clean OOF,
3 seeds, ~250-scaffold holdouts). MATCHED comparison on the SAME folds:
  control   = GBMs on 'combined'                  + GNN clean OOF   (deployed-style anchor)
  treatment = GBMs on 'combined' + 9 AIMNet2 QM   + GNN clean OOF
Deploy only if matched < -0.001. Reports corr(AIMNet2 corrector, anchor error) (rich-z diag).

AIMNet2 scalars (DFT-trained NNP single-point on ETKDGv3+MMFF conformer):
  energy, qmin, qmax, qabs_mean, qstd, qsum_abs, dipole(|sum q_i r_i|), fmax, frms.
(Pooled per-atom embeddings deliberately NOT gated — chempropembed-sink risk per memory.)
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from nb1126_combinatorial_search import make_model, feature_matrix, CACHE
from src.pxr.data import load_train
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from sklearn.preprocessing import StandardScaler

P = "data/processed"; LOG = "C:/pxr_work/search/results.jsonl"
AIM = "C:/pxr_work/aimnet2/aimnet_features.csv"
ACOLS = ["aimnet_energy", "aimnet_qmin", "aimnet_qmax", "aimnet_qabs_mean", "aimnet_qstd",
         "aimnet_qsum_abs", "aimnet_dipole", "aimnet_fmax", "aimnet_frms"]


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def fit_pred(c, d, trn_idx, Xfull, te_rows, ytr, Xqm_full=None, qm_te=None):
    se = d["se"]; mask = np.ones(len(ytr), bool)
    if c["prep"] == "noisy20": mask = se <= np.quantile(se, 0.8)
    elif c["prep"] == "noisy30": mask = se <= np.quantile(se, 0.7)
    A = Xfull; B = Xfull[te_rows]
    if Xqm_full is not None:
        A = np.hstack([Xfull, Xqm_full]); B = np.hstack([Xfull[te_rows], qm_te])
    m = make_model(c["arch"], c["hp"]); use = trn_idx[mask[trn_idx]]
    if c["arch"] in ("ridge", "enet"):
        sc = StandardScaler().fit(A[use]); m.fit(sc.transform(A[use]), ytr[use]); return m.predict(sc.transform(B))
    m.fit(A[use], ytr[use]); return m.predict(B)


def main():
    recs = [json.loads(l) for l in open(LOG) if l.strip()] if os.path.exists(LOG) else []
    valid = [r for r in recs if "ps_rae" in r and "error" not in r
             and r["arch"] in ("lgbm", "xgb", "cat", "histgb", "mlp")]
    valid.sort(key=lambda r: r["ps_rae"])
    bestper = {}
    for r in valid:
        bestper.setdefault(r["arch"], r)
    topK = list(bestper.values())
    print(f"diverse ensemble ({len(topK)} archs): {[(r['arch'], round(r['ps_rae'],3)) for r in topK]}")

    d = np.load(CACHE); ytr = d["ytr"]
    gnn_oof = np.load(f"{P}/oof_chemprop_aux.npy")
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    scaf = np.array([murcko(s) for s in tr["smiles"]])

    # AIMNet2 features aligned to the 4139 train order (by name); median-impute errored rows
    adf = pd.read_csv(AIM)
    atr_df = adf[adf.src == "train"].set_index("name").reindex(tr["name"])
    Xqm_full = atr_df[ACOLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    nan_rows = int(np.isnan(Xqm_full).any(axis=1).sum())
    med = np.nanmedian(Xqm_full, axis=0)
    inds = np.where(np.isnan(Xqm_full))
    Xqm_full[inds] = np.take(med, inds[1])
    print(f"AIMNet2 train rows={len(atr_df)}  errored/imputed={nan_rows}")
    Xcomb, _ = feature_matrix(d, "combined")

    c_raes, t_raes, corrs = [], [], []
    for seed in range(3):
        folds = scaffold_kfold_indices(scaf.tolist(), n_splits=round(len(ytr) / 250), seed=100 + seed)
        ho = min((f[1] for f in folds), key=lambda ix: abs(len(ix) - 250))
        ho = np.array(ho); trn = np.array([i for i in range(len(ytr)) if i not in set(ho.tolist())])
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)

        sc = StandardScaler().fit(Xqm_full[trn])
        Xqm_std = sc.transform(Xqm_full)
        xho = Xqm_std[ho]

        ctrl_ps, treat_ps = [], []
        for c in topK:
            ctrl_ps.append(fit_pred(c, d, trn, Xcomb, ho, ytr))
            treat_ps.append(fit_pred(c, d, trn, Xcomb, ho, ytr, Xqm_full=Xqm_std, qm_te=xho))
        ctrl_ps.append(gnn_oof[ho]); treat_ps.append(gnn_oof[ho])
        ctrl = np.clip(np.mean(ctrl_ps, 0), lo, hi)
        treat = np.clip(np.mean(treat_ps, 0), lo, hi)
        c_raes.append(rae(ytr[ho], ctrl)); t_raes.append(rae(ytr[ho], treat))

        err = ytr[ho] - ctrl
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
        anchor_trn = np.clip(np.mean(
            [fit_pred(c, d, trn, Xcomb, trn, ytr) for c in topK] + [gnn_oof[trn]], 0), lo, hi)
        corr_m = HGB(max_iter=200, max_depth=3, learning_rate=0.05).fit(Xqm_std[trn], ytr[trn] - anchor_trn)
        pred_corr = corr_m.predict(xho)
        corrs.append(float(np.corrcoef(pred_corr, err)[0, 1]))

    cm, tm = float(np.mean(c_raes)), float(np.mean(t_raes))
    delta = tm - cm
    print(f"\ncontrol  (combined+GNN)          RAE = {cm:.4f} +/- {np.std(c_raes):.4f}")
    print(f"treatment(combined+AIMNet2+GNN)  RAE = {tm:.4f} +/- {np.std(t_raes):.4f}")
    print(f"MATCHED delta = {delta:+.4f}  (deploy if < -0.0010)")
    print(f"corr(AIMNet2-corrector, anchor error) per seed = {[round(c,3) for c in corrs]}  mean {np.mean(corrs):+.3f}")
    out = {"control_rae": cm, "treatment_rae": tm, "matched_delta": delta,
           "corr_with_anchor_error": float(np.mean(corrs)), "per_seed_corr": corrs,
           "c_raes": c_raes, "t_raes": t_raes, "errored_imputed": nan_rows,
           "deploy": bool(delta < -0.001)}
    json.dump(out, open(f"{P}/nb1174_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1174_summary.json  ->  DEPLOY={out['deploy']}")


if __name__ == "__main__":
    main()
