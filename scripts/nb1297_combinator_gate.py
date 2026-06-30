"""nb1297 — COMBINATOR TICK: 2 non-standard compositions on deployed config (best 0.4242).

COMP-G: QM-ONLY SECOND-STAGE RESIDUAL GBM
  Stage-A = deployed 4-GBM(combined+QM) + CheMeleon + TabPFN + sn_oof.
  Stage-B = small LGBM on *only* the 48 QM scalars (AIMNet2+strain+D4+DBSTEP)
            trained to predict the STAGE-A TRAINING RESIDUAL.
  Hypothesis: the 4-GBMs dilute QM signal across 2265+48 features; a QM-only
  specialist can re-amplify nonlinear QM patterns that explain remaining error.
  Final = clip(stage-A + w * B_correction, lo, hi), w in {0.2, 0.3, 0.4}.

COMP-H: TANIMOTO-WEIGHTED k-NN RESIDUAL CORRECTION
  For each holdout compound, find k=3 nearest training neighbors (Tanimoto ECFP4).
  Stage-A training residuals propagate to holdout via Tanimoto-weighted mean.
  Hypothesis: activity cliff / local SAR errors are correlated within Tanimoto radius.
  Final = clip(stage-A + w * knn_correction, lo, hi), w in {0.1, 0.2, 0.3}.

Gate: 3 seeds from MTL/ho_idx_seed{0,1,2}.npy.  Matched delta vs control (same seed).
Deploy only if matched delta < -0.001 AND treatment < 0.4242.
"""
import os, sys, json
import numpy as np, pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"; N_SEEDS = 3

AIM = "C:/pxr_work/aimnet2/aimnet_features.csv"
ACOLS = ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean","aimnet_qstd",
         "aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]
STR = "C:/pxr_work/strain/strain_features.csv"
SCOLS = ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange",
         "conf_n","rmsd_mean","rmsd_max","e_per_heavy"]
D4 = "C:/pxr_work/d4/d4_features.csv"
DCOLS = ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max",
         "d4_c6diag_mean","d4_c6diag_std","d4_c6_total","d4_edisp",
         "d4_edisp_per_atom","d4_cn_mean","d4_cn_max","d4_qeeq_min",
         "d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]
DB = "C:/pxr_work/dbstep/dbstep_features.csv"
DBCOLS = ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65",
          "ster_L","ster_Bmin","ster_Bmax","ster_aniso",
          "npr1","npr2","asphericity","spherocity","eccentricity",
          "radgyr","inertial_sf"]


def topK_configs():
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r
             and r["arch"] in ("lgbm","xgb","cat","histgb")]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_pred_qm(c, Xtr_base, Xqm, ytr, use_idx, te_rows):
    Xfull = np.hstack([Xtr_base, Xqm])
    m = make_model(c["arch"], c["hp"])
    d = np.load(CACHE); se = d["se"]
    if c["prep"] == "noisy20": use_idx = use_idx[se[use_idx] <= np.quantile(se, 0.8)]
    elif c["prep"] == "noisy30": use_idx = use_idx[se[use_idx] <= np.quantile(se, 0.7)]
    m.fit(Xfull[use_idx], ytr[use_idx])
    return m.predict(Xfull[te_rows])


def aligned_block(csv_path, cols, tr_names):
    df = pd.read_csv(csv_path)
    df = df[df.src == "train"].drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(tr_names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X


def morgan_fp_matrix(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(2048, dtype=np.uint8))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(np.array(fp, dtype=np.uint8))
    return np.array(fps, dtype=np.float32)


def tanimoto_knn_correction(fp_trn, fp_ho, residuals_trn, k=3, eps=1e-8):
    """For each holdout compound, weighted-mean of k-NN training residuals by Tanimoto."""
    # Tanimoto = |A & B| / |A | B| for binary vectors
    # dot(A,B) / (|A|^2 + |B|^2 - dot(A,B))
    a_sum = fp_trn.sum(1)  # (n_trn,)
    b_sum = fp_ho.sum(1)   # (n_ho,)
    dot = fp_ho @ fp_trn.T  # (n_ho, n_trn)
    tan = dot / (b_sum[:, None] + a_sum[None, :] - dot + eps)  # (n_ho, n_trn)
    n_ho = len(fp_ho)
    corr = np.zeros(n_ho)
    for i in range(n_ho):
        sims = tan[i]
        top_k = np.argsort(sims)[-k:]
        wts = sims[top_k]; wt_sum = wts.sum()
        if wt_sum < eps:
            corr[i] = 0.0
        else:
            corr[i] = (wts * residuals_trn[top_k]).sum() / wt_sum
    return corr


def main():
    from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    ytr_full = tr["pec50"].to_numpy()
    n_tr = len(ytr_full)

    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]
    assert len(ytr) == n_tr == 4139, f"train size mismatch: {len(ytr)}"

    Xbase, _ = feature_matrix(d, "combined")  # (4139, 2265)
    Xqm_raw = aligned_block(AIM, ACOLS, tr["name"])
    Xst_raw = aligned_block(STR, SCOLS, tr["name"])
    Xd4_raw = aligned_block(D4, DCOLS, tr["name"])
    Xdb_raw = aligned_block(DB, DBCOLS, tr["name"])
    print(f"QM blocks: aim={Xqm_raw.shape[1]} str={Xst_raw.shape[1]} d4={Xd4_raw.shape[1]} db={Xdb_raw.shape[1]}")

    chem = np.load(f"{SD}/chemeleon_oof.npy")
    tab  = np.load(f"{SD}/tabpfn_oof.npy")
    # sisterNR GNN full-train OOF for training-residual computation
    # sn_oof_seed{s}.npy is holdout-only (size=244); use gnn_oof.npy for full-train proxy
    gnn_full = np.load(f"{SD}/gnn_oof.npy")
    topK = topK_configs()
    print(f"topK: {[c['arch'] for c in topK]}")

    # Compute Morgan FPs for COMP-H (expensive: only once)
    print("Computing Morgan FPs for k-NN correction...")
    fps_full = morgan_fp_matrix(tr["smiles"].tolist())  # (4139, 2048)
    print(f"  FPs: {fps_full.shape}")

    # Gate per seed
    g_weights = [0.2, 0.3, 0.4]   # COMP-G blend weights
    h_weights = [0.1, 0.2, 0.3]   # COMP-H blend weights
    g_results = {f"w{int(w*10)}": [] for w in g_weights}
    h_results = {f"w{int(w*10)}": [] for w in h_weights}
    ctrl_raes = []

    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        ho_set = set(ho.tolist())
        trn = np.array([i for i in range(n_tr) if i not in ho_set])
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        # sn_oof_seed is holdout-only (shape=244); used directly for holdout ensemble
        sn_ho = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()

        # Scale QM on training, apply to all (no leakage: scaler fit on trn only)
        sc_aim = StandardScaler().fit(Xqm_raw[trn])
        sc_str = StandardScaler().fit(Xst_raw[trn])
        sc_d4  = StandardScaler().fit(Xd4_raw[trn])
        sc_db  = StandardScaler().fit(Xdb_raw[trn])

        Xqm_full = np.hstack([sc_aim.transform(Xqm_raw),
                               sc_str.transform(Xst_raw),
                               sc_d4.transform(Xd4_raw),
                               sc_db.transform(Xdb_raw)])  # (4139, 48)

        # Stage-A: 4 GBMs on combined+QM, + chemeleon + tabpfn + sn_oof
        gbm_preds_ho = []
        gbm_preds_trn = []
        for c in topK:
            p_ho  = fit_pred_qm(c, Xbase, Xqm_full, ytr, trn, ho)
            p_trn = fit_pred_qm(c, Xbase, Xqm_full, ytr, trn, trn)
            gbm_preds_ho.append(p_ho)
            gbm_preds_trn.append(p_trn)

        # Stage-A ensemble on holdout (7 members: 4GBM + chem + tab + sn_ho)
        # Stage-A on training uses gnn_full as sn proxy (sn_oof_seed is holdout-only)
        stage_a_ho  = np.mean(gbm_preds_ho + [chem[ho], tab[ho], sn_ho], 0)
        stage_a_trn = np.mean(gbm_preds_trn + [chem[trn], tab[trn], gnn_full[trn]], 0)

        ctrl_pred = np.clip(stage_a_ho, lo, hi)
        ctrl_raes.append(rae(ytr[ho], ctrl_pred))
        r_trn = ytr[trn] - stage_a_trn  # training residuals

        # ---- COMP-G: QM-only stage-2 LGBM on residuals ----
        qm_model = LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=16, max_depth=4,
            min_child_samples=20, reg_alpha=1.0, reg_lambda=5.0,
            n_jobs=4, verbose=-1, random_state=seed)
        qm_model.fit(Xqm_full[trn], r_trn)
        stage_b_ho = qm_model.predict(Xqm_full[ho])
        for w in g_weights:
            pred = np.clip(stage_a_ho + w * stage_b_ho, lo, hi)
            g_results[f"w{int(w*10)}"].append(rae(ytr[ho], pred))

        # ---- COMP-H: Tanimoto k-NN residual correction ----
        knn_corr = tanimoto_knn_correction(fps_full[trn], fps_full[ho], r_trn, k=3)
        for w in h_weights:
            pred = np.clip(stage_a_ho + w * knn_corr, lo, hi)
            h_results[f"w{int(w*10)}"].append(rae(ytr[ho], pred))

        g_best_this_seed = min(rae(ytr[ho], np.clip(stage_a_ho + w*stage_b_ho, lo, hi)) for w in g_weights)
        h_best_this_seed = min(rae(ytr[ho], np.clip(stage_a_ho + w*knn_corr, lo, hi)) for w in h_weights)
        print(f"seed{seed}  ctrl={ctrl_raes[-1]:.4f}"
              f"  G-best={g_best_this_seed:.4f}"
              f"  H-best={h_best_this_seed:.4f}")

    cm = float(np.mean(ctrl_raes))
    prev = json.load(open(BEST))["rae"] if os.path.exists(BEST) else 0.4242
    print(f"\ncontrol (deployed) mean RAE = {cm:.4f}  (best_ensemble={prev})")

    print("\n=== COMP-G: QM-only residual stage-2 ===")
    g_deltas = {}
    best_g = None; best_g_rae = 9.9
    for k, raes in g_results.items():
        mn = float(np.mean(raes)); delta = mn - cm
        g_deltas[k] = {"mean_rae": mn, "delta": delta, "raes": raes}
        print(f"  {k}: {mn:.4f} (delta {delta:+.4f})")
        if mn < best_g_rae: best_g_rae = mn; best_g = k

    print("\n=== COMP-H: k-NN residual correction ===")
    h_deltas = {}
    best_h = None; best_h_rae = 9.9
    for k, raes in h_results.items():
        mn = float(np.mean(raes)); delta = mn - cm
        h_deltas[k] = {"mean_rae": mn, "delta": delta, "raes": raes}
        print(f"  {k}: {mn:.4f} (delta {delta:+.4f})")
        if mn < best_h_rae: best_h_rae = mn; best_h = k

    out = {
        "control_mean_rae": cm, "ctrl_raes": ctrl_raes,
        "deployed_best": prev,
        "comp_g": g_deltas, "comp_g_best_key": best_g, "comp_g_best_rae": best_g_rae,
        "comp_h": h_deltas, "comp_h_best_key": best_h, "comp_h_best_rae": best_h_rae,
    }
    json.dump(out, open(f"{P}/nb1297_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1297_summary.json")

    # Deploy check: a-priori best-w only (NOT argmax to avoid selection inflation)
    for name, best_rae, best_key, deltas in [
        ("COMP-G", best_g_rae, best_g, g_deltas),
        ("COMP-H", best_h_rae, best_h, h_deltas),
    ]:
        # Use the a-priori w=0.2 for the gate decision (lowest w = most conservative)
        apriori_key = list(deltas.keys())[0]
        apriori_rae = deltas[apriori_key]["mean_rae"]
        apriori_delta = deltas[apriori_key]["delta"]
        print(f"\n{name} a-priori (w={apriori_key}): RAE={apriori_rae:.4f} delta={apriori_delta:+.4f}")
        if apriori_delta < -0.001 and apriori_rae < prev:
            print(f"  -> BEATS gate (a-priori w): CANDIDATE FOR DEPLOY")
        else:
            print(f"  -> no deploy (a-priori gate not met)")


if __name__ == "__main__":
    main()
