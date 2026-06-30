"""nb1311_mitra_gate.py — Mitra (Amazon AutoGluon NeurIPS-2025 tabular FM, MSE head)
as an ADD-MEMBER on the deployed 4-GBM+CheMeleon+TabPFN+sn_oof ensemble.

Approach:
- Features: CheMeleon 2048-d train embeddings -> PCA-100 (per-fold, scaffold-aware)
- OOF: scaffold 5-fold on 4139 train -> cache C:/pxr_work/mitra/mitra_oof.npy
- Test: predict all 513 test compounds -> mitra_te.npy
- Gate: add-member to deployed config (4-GBM+QM+CheMeleon+TabPFN+sn_oof), 3 clean seeds
  Deploy if matched delta < -0.001 AND treatment < best_ensemble.json.rae

Usage: python scripts/nb1311_mitra_gate.py [smoke]
"""
import os, sys, json, time, shutil, tempfile
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"

SD   = "C:/pxr_work/search"
MTL  = "C:/pxr_work/mtl"
MDIR = "C:/pxr_work/mitra"
BEST = f"{SD}/best_ensemble.json"
LOG  = f"{SD}/results.jsonl"
OOF_OUT = f"{MDIR}/mitra_oof.npy"
TE_OUT  = f"{MDIR}/mitra_te.npy"
N_SEEDS = 3
N_PCA   = 50    # lower PCA dims to speed up Mitra attention
CTX     = 40    # subsample this many train rows per bag — timing shows ~88s/bag at CTX=40
BAGS    = 8     # bags to average per fold (compensates for tiny CTX)

os.makedirs(MDIR, exist_ok=True)


# ---- Feature preparation ----

def murcko(s):
    m = Chem.MolFromSmiles(str(s))
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else ""


def load_chemeleon_embeddings():
    """Load CheMeleon 2048-d embeddings for train (4139) and test (513)."""
    tr_path = f"{SD}/chemeleon_tr.npy"
    te_path = f"{SD}/chemeleon_te.npy"
    if os.path.exists(tr_path) and os.path.exists(te_path):
        Xtr = np.load(tr_path).astype(np.float32)
        Xte = np.load(te_path).astype(np.float32)
        print(f"  Loaded CheMeleon embs: Xtr={Xtr.shape}, Xte={Xte.shape}", flush=True)
        return Xtr, Xte
    raise FileNotFoundError(f"CheMeleon embeddings missing: {tr_path} / {te_path}")


# ---- Mitra runner ----

def run_mitra_single(Zfit, yfit, Zpred, seed):
    """Fit Mitra on (Zfit, yfit), predict Zpred. Single bag."""
    from autogluon.tabular import TabularPredictor

    cols = [f"f{i}" for i in range(Zfit.shape[1])]
    df_fit = pd.DataFrame(Zfit, columns=cols)
    df_fit["target"] = yfit.astype(float)
    df_pred = pd.DataFrame(Zpred, columns=cols)

    tmpdir = tempfile.mkdtemp(prefix=f"mitra_s{seed}_")
    try:
        pred = TabularPredictor(
            label="target", verbosity=0,
            problem_type="regression",
            path=tmpdir,
        ).fit(
            df_fit,
            hyperparameters={"MITRA": {"n_estimators": 1}},
            num_bag_folds=0,
            num_stack_levels=0,
            time_limit=3600,
        ).predict(df_pred).to_numpy()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return pred.astype(np.float32)


def run_mitra_bagged(Ztr, ytr, Zpred, fold_seed, ctx=CTX, bags=BAGS):
    """CTX-subsampled bagging: subsample ctx rows, run BAGS times, average."""
    bag_preds = []
    for b in range(bags):
        rs = np.random.RandomState(fold_seed * 100 + b)
        idx = rs.choice(len(Ztr), min(ctx, len(Ztr)), replace=False)
        p = run_mitra_single(Ztr[idx], ytr[idx], Zpred, seed=fold_seed * 100 + b)
        bag_preds.append(p)
        print(f"    bag {b}/{bags} done (ctx={len(idx)})", flush=True)
    return np.mean(bag_preds, 0)


# ---- Gate helpers (nb1301 convention) ----

def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"])
    bp = {}
    for r in valid:
        bp.setdefault(r["arch"], r)
    return list(bp.values())


def load_qm_blocks(tr_names):
    """Load AIMNet2+strain+D4+DBSTEP+OrbMol feature blocks, impute NaNs."""
    def aligned_block(csv_path, cols):
        df = pd.read_csv(csv_path)
        if "src" in df.columns:
            df = df[df.src == "train"]
        df = df.drop_duplicates(subset="name", keep="first")
        sub = df.set_index("name").reindex(tr_names)
        X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(med, inds[1])
        return X.astype(np.float32)

    ACOLS = ["aimnet_energy", "aimnet_qmin", "aimnet_qmax", "aimnet_qabs_mean",
             "aimnet_qstd", "aimnet_qsum_abs", "aimnet_dipole", "aimnet_fmax", "aimnet_frms"]
    SCOLS = ["strain_relax_mean", "strain_relax_max", "conf_espread", "conf_erange",
             "conf_n", "rmsd_mean", "rmsd_max", "e_per_heavy"]
    DCOLS = ["d4_alpha_sum", "d4_alpha_mean", "d4_alpha_std", "d4_alpha_max",
             "d4_c6diag_mean", "d4_c6diag_std", "d4_c6_total", "d4_edisp",
             "d4_edisp_per_atom", "d4_cn_mean", "d4_cn_max", "d4_qeeq_min",
             "d4_qeeq_max", "d4_qeeq_std", "d4_qeeq_absum"]
    DBCOLS = ["vbur_r25", "vbur_r35", "vbur_r45", "vbur_r55", "vbur_r65",
              "ster_L", "ster_Bmin", "ster_Bmax", "ster_aniso",
              "npr1", "npr2", "asphericity", "spherocity", "eccentricity",
              "radgyr", "inertial_sf"]
    ORBCOLS = ["orb_energy", "orb_energy_per_ha", "orb_fmax", "orb_frms", "orb_fstd",
               "orb_conf_mean", "orb_conf_std", "orb_conf_node_mean", "orb_conf_node_std",
               "orb_conf_node_min", "orb_node_emb_mean", "orb_node_emb_std", "orb_node_emb_norm"]

    Xqm  = aligned_block("C:/pxr_work/aimnet2/aimnet_features.csv", ACOLS)
    Xst  = aligned_block("C:/pxr_work/strain/strain_features.csv",  SCOLS)
    Xd4  = aligned_block("C:/pxr_work/d4/d4_features.csv",          DCOLS)
    Xdb  = aligned_block("C:/pxr_work/dbstep/dbstep_features.csv",  DBCOLS)
    Xorb = aligned_block("C:/pxr_work/orbmol/orbmol_features.csv",  ORBCOLS)
    return Xqm, Xst, Xd4, Xdb, Xorb


# ---- Main ----

def main():
    print("=== nb1311 Mitra add-member gate ===", flush=True)

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test().reset_index(drop=True)
    ytr = tr["pec50"].to_numpy(float)
    n_tr = len(ytr)
    scaf = [murcko(s) for s in tr["smiles"].tolist()]

    if SMOKE:
        print("SMOKE: mini run (CTX=40, BAGS=2)")
        Xtr_c, Xte_c = load_chemeleon_embeddings()
        sc = StandardScaler().fit(Xtr_c[:60])
        pca = PCA(n_components=10, random_state=0).fit(sc.transform(Xtr_c[:60]))
        Z60 = pca.transform(sc.transform(Xtr_c[:60]))
        Z10 = pca.transform(sc.transform(Xtr_c[60:70]))
        t0 = time.time()
        p = run_mitra_bagged(Z60, ytr[:60], Z10, fold_seed=0, ctx=40, bags=2)
        print(f"SMOKE OK pred={p.shape} finite={np.isfinite(p).all()} t={time.time()-t0:.1f}s")
        return

    # Load CheMeleon embeddings
    Xtr_c, Xte_c = load_chemeleon_embeddings()

    # --- Step 1: Scaffold 5-fold OOF ---
    if os.path.exists(OOF_OUT):
        print(f"  Loaded cached OOF from {OOF_OUT}", flush=True)
        oof = np.load(OOF_OUT)
    else:
        print(f"  Running scaffold 5-fold OOF (PCA-{N_PCA}, CTX={CTX}×{BAGS}bags)...", flush=True)
        oof = np.zeros(n_tr, dtype=np.float32)
        folds = list(scaffold_kfold_indices(scaf, n_splits=5, seed=42))
        for fi, (trn, val) in enumerate(folds):
            t0 = time.time()
            # PCA fit on train fold only (no leakage)
            sc = StandardScaler().fit(Xtr_c[trn])
            pca = PCA(n_components=N_PCA, random_state=0).fit(sc.transform(Xtr_c[trn]))
            Ztr = pca.transform(sc.transform(Xtr_c[trn]))
            Zval = pca.transform(sc.transform(Xtr_c[val]))
            oof[val] = run_mitra_bagged(Ztr, ytr[trn], Zval, fold_seed=fi)
            # Checkpoint after each fold
            np.save(OOF_OUT, oof)
            print(f"  fold {fi} done ({len(val)} val preds) t={time.time()-t0:.0f}s  "
                  f"fold RAE={rae(ytr[val], oof[val]):.4f}", flush=True)

        print(f"  Full OOF scaffold-CV RAE: {rae(ytr, oof):.4f}", flush=True)
        np.save(OOF_OUT, oof)

    print(f"  OOF shape={oof.shape} scaffold-CV RAE={rae(ytr, oof):.4f}", flush=True)

    # Corr with nb3200 error
    nb3200_path = "data/processed/oof_chemprop_aux.npy"
    if os.path.exists(nb3200_path):
        nb3200_oof = np.load(nb3200_path)
        err = ytr - nb3200_oof
        corr = float(np.corrcoef(err, oof)[0, 1])
        print(f"  Corr(nb3200 error, mitra_oof) = {corr:+.4f}", flush=True)

    # --- Step 2: Full train -> test predictions ---
    if os.path.exists(TE_OUT):
        print(f"  Loaded cached test preds from {TE_OUT}", flush=True)
        te_pred = np.load(TE_OUT)
    else:
        print(f"  Predicting 513 test compounds (CTX={CTX}×{BAGS}bags)...", flush=True)
        sc_all = StandardScaler().fit(Xtr_c)
        pca_all = PCA(n_components=N_PCA, random_state=0).fit(sc_all.transform(Xtr_c))
        Ztr_all = pca_all.transform(sc_all.transform(Xtr_c))
        Zte = pca_all.transform(sc_all.transform(Xte_c))
        te_pred = run_mitra_bagged(Ztr_all, ytr, Zte, fold_seed=999)
        np.save(TE_OUT, te_pred)
        print(f"  Test preds done: shape={te_pred.shape} finite={np.isfinite(te_pred).all()}", flush=True)

    # --- Step 3: Honest add-member gate (OrbMol deployed config) ---
    print(f"\n  Running honest add-member gate (3 clean seeds)...", flush=True)

    from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
    d = np.load(CACHE); se = d["se"]
    Xfull, _ = feature_matrix(d, "combined")
    chem_oof = np.load(f"{SD}/chemeleon_oof.npy")
    tab_oof  = np.load(f"{SD}/tabpfn_oof.npy")
    tr_names = tr["name"].tolist()

    Xqm, Xst, Xd4, Xdb, Xorb = load_qm_blocks(tr_names)
    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"  4-GBM: {[c['arch'] for c in topK]}", flush=True)

    ctrl_raes, treat_raes = [], []
    for seed in range(N_SEEDS):
        ho  = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        noisy20 = se <= np.quantile(se, 0.8)
        noisy30 = se <= np.quantile(se, 0.7)
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)

        # Standardize QM blocks per-seed on train
        Xqm_s  = StandardScaler().fit(Xqm[trn]).transform(Xqm)
        Xst_s  = StandardScaler().fit(Xst[trn]).transform(Xst)
        Xd4_s  = StandardScaler().fit(Xd4[trn]).transform(Xd4)
        Xdb_s  = StandardScaler().fit(Xdb[trn]).transform(Xdb)
        Xorb_s = StandardScaler().fit(Xorb[trn]).transform(Xorb)
        Xqm_block = np.hstack([Xqm_s, Xst_s, Xd4_s, Xdb_s, Xorb_s])

        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()

        gbm_preds = []
        for c in topK:
            use = trn
            if c["prep"] == "noisy20": use = trn[noisy20[trn]]
            elif c["prep"] == "noisy30": use = trn[noisy30[trn]]
            A = np.hstack([Xfull, Xqm_block])
            m = make_model(c["arch"], c["hp"])
            m.fit(A[use], ytr[use])
            gbm_preds.append(m.predict(A[ho]))

        base_members = gbm_preds + [chem_oof[ho], tab_oof[ho], gnn]
        ctrl_ens  = np.clip(np.mean(base_members, 0), lo, hi)
        treat_ens = np.clip(np.mean(base_members + [oof[ho]], 0), lo, hi)

        c_rae = rae(ytr[ho], ctrl_ens)
        t_rae = rae(ytr[ho], treat_ens)
        ctrl_raes.append(c_rae)
        treat_raes.append(t_rae)
        print(f"  seed{seed}  ctrl={c_rae:.4f}  treat={t_rae:.4f}  d={t_rae-c_rae:+.4f}", flush=True)

    cm = float(np.mean(ctrl_raes))
    tm = float(np.mean(treat_raes))
    delta = tm - cm
    n_neg = sum(t < c for t, c in zip(treat_raes, ctrl_raes))
    prev_best = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4231

    print(f"\n  Control  (deployed) RAE = {cm:.4f}  seeds={[round(r,4) for r in ctrl_raes]}")
    print(f"  Treatment(+Mitra)   RAE = {tm:.4f}  seeds={[round(r,4) for r in treat_raes]}")
    print(f"  Matched delta = {delta:+.5f}  ({n_neg}/{N_SEEDS} seeds negative)")
    print(f"  Deployed best = {prev_best:.4f}")

    deploy = bool(delta < -0.001 and tm < prev_best and n_neg >= 2)
    out = {
        "model": "Mitra (AutoGluon 1.5, NeurIPS-2025)",
        "features": f"CheMeleon-2048 -> PCA-{N_PCA}",
        "oof_rae": float(rae(ytr, oof)),
        "ctrl_rae": cm, "treat_rae": tm, "delta": delta,
        "ctrl_raes": ctrl_raes, "treat_raes": treat_raes, "n_neg": n_neg,
        "prev_best": prev_best, "deploy": deploy,
    }
    os.makedirs("data/processed", exist_ok=True)
    json.dump(out, open("data/processed/nb1311_mitra_summary.json", "w"), indent=2)
    print(f"\n  Result saved -> data/processed/nb1311_mitra_summary.json  DEPLOY={deploy}")

    if deploy:
        print("\n*** DEPLOY: Mitra WINS — building 513 submission ***", flush=True)
        # Build submission
        from src.pxr.data import load_test as lt
        test_df = lt().reset_index(drop=True)
        sc_all = StandardScaler().fit(Xtr_c)
        pca_all = PCA(n_components=N_PCA, random_state=0).fit(sc_all.transform(Xtr_c))
        Xte_c2 = np.load(f"{SD}/chemeleon_te.npy").astype(np.float32)
        Zte = pca_all.transform(sc_all.transform(Xte_c2))
        # Reload test preds
        te_pred2 = np.load(TE_OUT)
        # Merge with deployed test preds
        deployed_sub = pd.read_csv("submissions/nb1299_orbmol_ensemble.csv")
        cur_preds = deployed_sub.set_index("Molecule Name")["pEC50"].values
        # Add Mitra as extra member (equal weight)
        n_members_ctrl = len(topK) + 3  # GBMs + CheMeleon + TabPFN + GNN
        new_ensemble = (cur_preds * n_members_ctrl + te_pred2) / (n_members_ctrl + 1)
        sub = pd.DataFrame({
            "Molecule Name": test_df["name"].values,
            "pEC50": new_ensemble.clip(ytr.min() - 0.1, ytr.max() + 0.1)
        })
        sub.to_csv("submissions/nb1311_mitra_ensemble.csv", index=False)
        print(f"  Saved: submissions/nb1311_mitra_ensemble.csv", flush=True)
        # Update best_ensemble.json
        best_data = json.load(open(BEST))
        best_data["rae"] = tm
        best_data["via"] = f"+Mitra (AutoGluon-1.5, MSE-head TFM, PCA-{N_PCA} CheMeleon) as 8th ensemble member, delta={delta:.5f}"
        best_data["submission"] = "submissions/nb1311_mitra_ensemble.csv"
        best_data["members"] = best_data.get("members", []) + ["mitra"]
        json.dump(best_data, open(BEST, "w"), indent=2)
        print(f"  Updated best_ensemble.json -> {tm:.4f}", flush=True)
    else:
        reason = "not enough neg seeds" if n_neg < 2 else (
            f"delta {delta:+.5f} >= -0.001" if delta >= -0.001 else
            f"treatment {tm:.4f} >= best {prev_best:.4f}"
        )
        print(f"\n  NO DEPLOY: {reason}", flush=True)

    print("\n=== nb1311 DONE ===", flush=True)


if __name__ == "__main__":
    main()
