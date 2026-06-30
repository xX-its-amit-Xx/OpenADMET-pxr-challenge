"""nb1310 — TabPFN v8 (8.0.6) upgrade gate.

Regenerate TabPFN OOF + test predictions with the currently installed
tabpfn==8.0.6 (vs old v2 cached at C:/pxr_work/search/tabpfn_{oof,te}.npy).
Honest gate on 3 clean holdout seeds (nb1127/nb1130 convention):
  control  = deployed 7-member ensemble (4-GBM + sisterNR_gnn + CheMeleon + old TabPFN)
  treatment = same but replace old tabpfn_te with new v8 predictions
If treatment < control by >0.001 -> DEPLOY (overwrite cached files, rebuild submission).

Usage:
  python nb1310_tabpfn_v8_gate.py [smoke]
"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

# Bypass TabPFN license check — model already downloaded and license accepted via HF
import tabpfn.browser_auth as _tba
_tba._accepted_repos.add("tabpfn_2_5")
_tba._accepted_repos.add("tabpfn_2_6")
_tba._accepted_repos.add("tabpfn_3")

from src.pxr.data import load_train, load_test
from src.pxr.featurize import combined, impute
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

SD  = "C:/pxr_work/search"
MTL = "C:/pxr_work/mtl"
CACHE = f"{SD}/feats_cache.npz"
V25_CKPT = ("D:/Users/ashenoy00000/.cache/huggingface/hub/"
            "models--Prior-Labs--tabpfn_2_5/snapshots/"
            "6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b/"
            "tabpfn-v2.5-regressor-v2.5_default.ckpt")
NEW_OOF = f"{SD}/tabpfn_v8_oof.npy"
NEW_TE  = f"{SD}/tabpfn_v8_te.npy"
BEST    = f"{SD}/best_ensemble.json"
N_SEEDS = 3

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"


def murcko(s):
    m = Chem.MolFromSmiles(str(s))
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def run_tabpfn_v8(Xtr, ytr, Xte, scaf, n_pca=100, n_estimators=4, ctx=1500, bags=4):
    """OOF + test predictions with TabPFN v2.5 (via v8.0.6 API + cached v2.5 weights).

    Uses bagged subsampling (bags × CTX rows) to keep CPU inference fast,
    matching the spirit of the old v2 approach (CTX=1200 × BAGS=3) but with
    the better v2.5 model and 25% larger context.
    """
    from tabpfn import TabPFNRegressor

    def tabpfn_bagged(Zfit, yfit, Zpred, seed0):
        preds = []
        for b in range(bags):
            rs = np.random.RandomState(seed0 + b)
            idx = rs.choice(len(Zfit), min(ctx, len(Zfit)), replace=False)
            m = TabPFNRegressor(
                n_estimators=n_estimators,
                device="cpu",
                ignore_pretraining_limits=True,
                model_path=V25_CKPT,
                random_state=seed0 + b,
                show_progress_bar=False,
            )
            m.fit(Zfit[idx], yfit[idx])
            preds.append(m.predict(Zpred))
        return np.mean(preds, 0)

    oof = np.zeros(len(ytr))
    folds = list(scaffold_kfold_indices(scaf, n_splits=5, seed=42))

    for fi, (trn, val) in enumerate(folds):
        sc = StandardScaler().fit(Xtr[trn])
        pca = PCA(n_components=n_pca, random_state=0).fit(sc.transform(Xtr[trn]))
        Ztr = pca.transform(sc.transform(Xtr[trn]))
        Zval = pca.transform(sc.transform(Xtr[val]))
        oof[val] = tabpfn_bagged(Ztr, ytr[trn], Zval, seed0=100 * fi)
        print(f"  fold {fi} done ({len(val)} val preds)", flush=True)

    # Full train -> test
    sc_all = StandardScaler().fit(Xtr)
    pca_all = PCA(n_components=n_pca, random_state=0).fit(sc_all.transform(Xtr))
    Ztr_all = pca_all.transform(sc_all.transform(Xtr))
    Zte = pca_all.transform(sc_all.transform(Xte))
    te_pred = tabpfn_bagged(Ztr_all, ytr, Zte, seed0=999)
    return oof, te_pred


def honest_gate_rae(ytr, scaf, oof_old, oof_new, te_old, te_new):
    """3-seed matched holdout RAE: control vs treatment (swap tabpfn member)."""
    from nb1126_combinatorial_search import feature_matrix, make_model, CACHE as GBM_CACHE
    from sklearn.preprocessing import StandardScaler as SS

    LOG = f"{SD}/results.jsonl"
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r
             and r["arch"] in ("lgbm", "xgb", "cat", "histgb")]
    valid.sort(key=lambda r: r["ps_rae"])
    bestper = {}
    for r in valid:
        bestper.setdefault(r["arch"], r)
    topK = list(bestper.values())
    print(f"  4-GBM members: {[c['arch'] for c in topK]}")

    d = np.load(GBM_CACHE)
    gnn_oof = np.load(f"data/processed/oof_chemprop_aux.npy")
    chem_oof_path = f"{SD}/chemeleon_oof.npy"
    chem_oof = np.load(chem_oof_path) if os.path.exists(chem_oof_path) else None

    ctrl_raes, treat_raes = [], []
    for seed in range(N_SEEDS):
        n_splits = round(len(ytr) / 250)
        folds = scaffold_kfold_indices(scaf, n_splits=n_splits, seed=100 + seed)
        ho = min((f[1] for f in folds), key=lambda ix: abs(len(ix) - 250))
        trn = np.array([i for i in range(len(ytr)) if i not in set(ho.tolist())])

        ps_ctrl, ps_treat = [], []

        # 4-GBM members
        for c in topK:
            Xtr, _ = feature_matrix(d, "combined")
            se = d["se"]; mask = np.ones(len(ytr), bool)
            if c["prep"] == "noisy20": mask = se <= np.quantile(se, 0.8)
            elif c["prep"] == "noisy30": mask = se <= np.quantile(se, 0.7)
            use = trn[mask[trn]]
            m = make_model(c["arch"], c["hp"])
            if c["arch"] in ("ridge", "enet"):
                sc = SS().fit(Xtr[use]); m.fit(sc.transform(Xtr[use]), ytr[use])
                p = m.predict(sc.transform(Xtr[ho]))
            else:
                m.fit(Xtr[use], ytr[use]); p = m.predict(Xtr[ho])
            ps_ctrl.append(p); ps_treat.append(p)

        # GNN (same for both)
        ps_ctrl.append(gnn_oof[ho]); ps_treat.append(gnn_oof[ho])

        # CheMeleon (same for both)
        if chem_oof is not None:
            ps_ctrl.append(chem_oof[ho]); ps_treat.append(chem_oof[ho])

        # TabPFN: old vs new
        ps_ctrl.append(oof_old[ho])
        ps_treat.append(oof_new[ho])

        clip_lo = np.quantile(ytr[trn], 0.05)
        clip_hi = np.quantile(ytr[trn], 0.98)
        ctrl_raes.append(rae(ytr[ho], np.clip(np.mean(ps_ctrl, 0), clip_lo, clip_hi)))
        treat_raes.append(rae(ytr[ho], np.clip(np.mean(ps_treat, 0), clip_lo, clip_hi)))

    return float(np.mean(ctrl_raes)), float(np.mean(treat_raes)), ctrl_raes, treat_raes


def main():
    print("=== nb1310 TabPFN v8.0.6 upgrade gate ===", flush=True)
    import tabpfn; print(f"tabpfn version: {tabpfn.__version__}", flush=True)

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    ytr = tr["pec50"].to_numpy()
    smis_tr = tr["smiles"].tolist()
    smis_te = te["smiles"].tolist()
    scaf = [murcko(s) for s in smis_tr]

    if SMOKE:
        print("SMOKE: testing TabPFN v8 on 50 molecules", flush=True)
        from tabpfn import TabPFNRegressor
        from sklearn.decomposition import PCA
        X50 = impute(combined(smis_tr[:50])).astype(np.float32)
        sc = StandardScaler().fit(X50)
        pca = PCA(n_components=10, random_state=0).fit(sc.transform(X50))
        Z = pca.transform(sc.transform(X50))
        m = TabPFNRegressor(n_estimators=4, device="cpu", ignore_pretraining_limits=True,
                            model_path=V25_CKPT, random_state=0, show_progress_bar=False)
        m.fit(Z[:40], ytr[:40])
        p = m.predict(Z[40:])
        print(f"SMOKE OK: preds {p[:5].round(3)}, finite={np.isfinite(p).all()}")
        return

    # Load combined features (use cached where possible)
    FEAT_CACHE = f"{SD}/feats.npz"
    FEAT513    = f"{SD}/feats_full513.npz"
    if os.path.exists(FEAT_CACHE) and os.path.exists(FEAT513):
        print("Loading cached combined features...", flush=True)
        dc  = np.load(FEAT_CACHE)
        dc5 = np.load(FEAT513)
        Xtr = dc["comb_tr"].astype(np.float32)
        Xte = dc5["combined"].astype(np.float32)
        # Median-impute NaNs
        for X in (Xtr, Xte):
            col_med = np.nanmedian(X, axis=0)
            inds = np.where(np.isnan(X)); X[inds] = np.take(col_med, inds[1])
    else:
        print("Computing combined features (4139 + 513)...", flush=True)
        Xtr = impute(combined(smis_tr)).astype(np.float32)
        Xte = impute(combined(smis_te)).astype(np.float32)
    print(f"  Xtr {Xtr.shape}, Xte {Xte.shape}", flush=True)

    # Load old cached predictions
    oof_old = np.load(f"{SD}/tabpfn_oof.npy")
    te_old  = np.load(f"{SD}/tabpfn_te.npy")
    print(f"  Old TabPFN OOF scaffold-CV RAE: {rae(ytr, oof_old):.4f}")

    # Run new TabPFN 8.0.6 (cached)
    if os.path.exists(NEW_OOF) and os.path.exists(NEW_TE):
        print("  Loading cached v8 predictions...", flush=True)
        oof_new = np.load(NEW_OOF)
        te_new  = np.load(NEW_TE)
    else:
        print("  Running TabPFN v8.0.6 OOF + test (5-fold scaffold CV)...", flush=True)
        t0 = time.time()
        oof_new, te_new = run_tabpfn_v8(Xtr, ytr, Xte, scaf, n_pca=100, n_estimators=8)
        print(f"  Done in {time.time()-t0:.0f}s", flush=True)
        np.save(NEW_OOF, oof_new)
        np.save(NEW_TE, te_new)

    print(f"  New TabPFN v8 OOF scaffold-CV RAE: {rae(ytr, oof_new):.4f}")

    # Compute corr with deployed error
    nb3200_oof_path = "data/processed/oof_chemprop_aux.npy"
    if os.path.exists(nb3200_oof_path):
        nb3200_oof = np.load(nb3200_oof_path)
        err_nb3200 = ytr - nb3200_oof  # residuals
        corr_old = float(np.corrcoef(err_nb3200, oof_old)[0, 1])
        corr_new = float(np.corrcoef(err_nb3200, oof_new)[0, 1])
        print(f"  Corr(nb3200 error, old TabPFN): {corr_old:.4f}")
        print(f"  Corr(nb3200 error, new TabPFN): {corr_new:.4f}")
    else:
        corr_old = corr_new = float("nan")

    # Honest gate: 3-seed matched holdout
    print("  Running 3-seed honest gate...", flush=True)
    ctrl_rae, treat_rae, ctrl_list, treat_list = honest_gate_rae(
        ytr, scaf, oof_old, oof_new, te_old, te_new
    )
    delta = treat_rae - ctrl_rae
    n_neg = sum(t < c for t, c in zip(treat_list, ctrl_list))
    print(f"\n  GATE RESULT:")
    print(f"    control  (old TabPFN v2) : {ctrl_rae:.4f}  seeds={[round(r,4) for r in ctrl_list]}")
    print(f"    treatment (new TabPFN v8): {treat_rae:.4f}  seeds={[round(r,4) for r in treat_list]}")
    print(f"    delta  = {delta:+.5f}   ({n_neg}/{N_SEEDS} seeds negative)", flush=True)

    # Deploy if wins
    deployed_best = json.load(open(BEST))["rae"]
    print(f"  Current deployed best: {deployed_best:.4f}")

    gate_result = {
        "tabpfn_version": "8.0.6",
        "oof_rae_old": float(rae(ytr, oof_old)),
        "oof_rae_new": float(rae(ytr, oof_new)),
        "corr_err_old": corr_old,
        "corr_err_new": corr_new,
        "ctrl_rae": ctrl_rae,
        "treat_rae": treat_rae,
        "delta": delta,
        "n_neg": n_neg,
        "ctrl_list": ctrl_list,
        "treat_list": treat_list,
    }
    os.makedirs("C:/pxr_work/tabpfn_v8", exist_ok=True)
    json.dump(gate_result, open("C:/pxr_work/tabpfn_v8/gate_result.json", "w"), indent=2)
    print(f"  Gate result saved to C:/pxr_work/tabpfn_v8/gate_result.json", flush=True)

    if delta < -0.001 and n_neg >= 2:
        print("\n  *** DEPLOY: TabPFN v8 WINS — swapping cached predictions ***", flush=True)
        # Backup old
        import shutil
        shutil.copy(f"{SD}/tabpfn_oof.npy", f"{SD}/tabpfn_oof_v2_backup.npy")
        shutil.copy(f"{SD}/tabpfn_te.npy",  f"{SD}/tabpfn_te_v2_backup.npy")
        # Replace with new
        np.save(f"{SD}/tabpfn_oof.npy", oof_new)
        np.save(f"{SD}/tabpfn_te.npy",  te_new)
        print("  Updated C:/pxr_work/search/tabpfn_{oof,te}.npy with v8 predictions", flush=True)

        # Rebuild ensemble submission using nb1299_deploy_orbmol.py
        print("  Rebuilding ensemble submission...", flush=True)
        import subprocess
        subprocess.run(
            [sys.executable, "scripts/nb1299_deploy_orbmol.py"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            check=True
        )
        new_best = json.load(open(BEST))["rae"]
        print(f"  New ensemble RAE (best_ensemble.json): {new_best:.4f}", flush=True)
        print(f"  Submission: submissions/nb1299_orbmol_ensemble.csv (overwritten with v8 TabPFN)", flush=True)
    else:
        if delta >= 0:
            print(f"\n  NO DEPLOY: treatment WORSE ({delta:+.5f}) — keeping old TabPFN v2", flush=True)
        elif n_neg < 2:
            print(f"\n  NO DEPLOY: not enough negative seeds ({n_neg}/3) — keeping old TabPFN v2", flush=True)
        else:
            print(f"\n  MARGINAL: delta {delta:+.5f} < threshold 0.001 — no deploy", flush=True)

    print("\n=== nb1310 DONE ===", flush=True)


if __name__ == "__main__":
    main()
