"""nb1210 — COMBINATOR: two NON-STANDARD compositions of the DEPLOYED members,
honest-gated against the deployed FLAT-MEAN (best 0.4252, AIMNet2+strain+D4).

Members per seed on holdout (deployed config, AIMNet2+strain+D4 in the GBMs):
  4 GBM(combined+QM) preds, CheMeleon OOF, TabPFN OOF, sisterNR GNN OOF.
Control = deployed flat mean of all 7, clipped.

COMP-A  Coverage-tilted late fusion (CONDITIONAL composition, not flat mean):
  split members into ANALOG-memorizers {4 GBM + GNN} and FOUNDATION-generalizers
  {CheMeleon, TabPFN}. Per compound, tilt weight toward foundation when the
  top-1 Tanimoto coverage to the training fold is LOW (novel compounds where GBM
  analog-memorization is weakest). Ramp is a-priori (NOT tuned on the holdout):
  a_i = base + slope*(1 - cov_i), two fixed (base,slope) settings reported.

COMP-B  Cross-seed NNLS optimal static weights (META-LEARNER, not flat mean):
  learn non-negative member weights on the OTHER seeds' holdouts, apply to the
  held-out seed (weights NEVER fit on the eval seed). Tests if flat mean is
  suboptimal in a leak-free way.

Deploy a composition iff matched delta < -0.001 vs the deployed flat mean.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.chem import morgan_fp_batch
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
BEST = f"{SD}/best_ensemble.json"; LOG = f"{SD}/results.jsonl"; N_SEEDS = 3
AIM = "C:/pxr_work/aimnet2/aimnet_features.csv"
ACOLS = ["aimnet_energy", "aimnet_qmin", "aimnet_qmax", "aimnet_qabs_mean", "aimnet_qstd",
         "aimnet_qsum_abs", "aimnet_dipole", "aimnet_fmax", "aimnet_frms"]
STR = "C:/pxr_work/strain/strain_features.csv"
SCOLS = ["strain_relax_mean", "strain_relax_max", "conf_espread", "conf_erange",
         "conf_n", "rmsd_mean", "rmsd_max", "e_per_heavy"]
D4 = "C:/pxr_work/d4/d4_features.csv"
DCOLS = ["d4_alpha_sum", "d4_alpha_mean", "d4_alpha_std", "d4_alpha_max",
         "d4_c6diag_mean", "d4_c6diag_std", "d4_c6_total", "d4_edisp",
         "d4_edisp_per_atom", "d4_cn_mean", "d4_cn_max", "d4_qeeq_min",
         "d4_qeeq_max", "d4_qeeq_std", "d4_qeeq_absum"]


def topK_configs(archs):
    recs = [json.loads(l) for l in open(LOG) if l.strip()]
    valid = [r for r in recs if "ps_rae" in r and "error" not in r and r["arch"] in archs]
    valid.sort(key=lambda r: r["ps_rae"]); bp = {}
    for r in valid: bp.setdefault(r["arch"], r)
    return list(bp.values())


def fit_pred(c, Xfull, ytr, use_idx, te_rows, Xextra):
    A = np.hstack([Xfull, Xextra]); B = np.hstack([Xfull[te_rows], Xextra[te_rows]])
    m = make_model(c["arch"], c["hp"])
    m.fit(A[use_idx], ytr[use_idx]); return m.predict(B)


def aligned_block(csv_path, cols, tr_names):
    df = pd.read_csv(csv_path)
    df = df[df.src == "train"].drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(tr_names)
    X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0); inds = np.where(np.isnan(X)); X[inds] = np.take(med, inds[1])
    return X


def top1_tanimoto(fp_ho, fp_trn):
    """top-1 Tanimoto of each ho fp (M x 2048 uint8) to the trn set (N x 2048)."""
    A = fp_ho.astype(np.float32); B = fp_trn.astype(np.float32)
    inter = A @ B.T                                   # M x N intersection counts
    a = A.sum(1)[:, None]; b = B.sum(1)[None, :]
    tan = inter / (a + b - inter + 1e-9)
    return tan.max(1)


def main():
    d = np.load(CACHE); ytr = d["ytr"]; se = d["se"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    Xqm = aligned_block(AIM, ACOLS, tr["name"])
    Xst = aligned_block(STR, SCOLS, tr["name"])
    Xd4 = aligned_block(D4, DCOLS, tr["name"])
    fp = morgan_fp_batch(tr["smiles"].tolist())       # 4139 x 2048 uint8

    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"deployed members: {[r['arch'] for r in topK]} + CheMeleon + TabPFN + sisterNR-GNN")

    # Per-seed member predictions on the holdout (M x 7) + truth + coverage.
    seed_M, seed_y, seed_cov = [], [], []
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        trn = np.array([i for i in range(n_tr) if i not in set(ho.tolist())])
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        scq = StandardScaler().fit(Xqm[trn]); scs = StandardScaler().fit(Xst[trn]); scd = StandardScaler().fit(Xd4[trn])
        Xex = np.hstack([scq.transform(Xqm), scs.transform(Xst), scd.transform(Xd4)])
        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()
        gbm = [fit_pred(c, Xtr, ytr, trn, ho, Xex) for c in topK]
        M = np.column_stack(gbm + [chem[ho], tab[ho], gnn])   # M x 7
        cov = top1_tanimoto(fp[ho], fp[trn])
        seed_M.append((M, lo, hi)); seed_y.append(ytr[ho]); seed_cov.append(cov)

    GBM_N = len(topK)  # indices 0..GBM_N-1 = GBM, GBM_N..GBM_N+1 = chem,tab, last = gnn

    # ---- Control: deployed flat mean ----
    ctrl_raes = []
    for (M, lo, hi), y in zip(seed_M, seed_y):
        ctrl_raes.append(rae(y, np.clip(M.mean(1), lo, hi)))
    ctrl_m = float(np.mean(ctrl_raes))

    # ---- COMP-A: coverage-tilted late fusion ----
    # analog part = mean(4 GBM + GNN); foundation part = mean(chem,tab)
    # final = (1-a)*analog + a*foundation, a = clip(base + slope*(1-cov), 0, 1)
    rampsA = [("base.30/slope.30", 0.30, 0.30), ("base.286/slope.40", 0.286, 0.40)]
    compA = {}
    for tag, base, slope in rampsA:
        raes = []
        for (M, lo, hi), y, cov in zip(seed_M, seed_y, seed_cov):
            analog = M[:, list(range(GBM_N)) + [GBM_N + 2]].mean(1)
            found = M[:, [GBM_N, GBM_N + 1]].mean(1)
            a = np.clip(base + slope * (1.0 - cov), 0.0, 1.0)
            raes.append(rae(y, np.clip((1 - a) * analog + a * found, lo, hi)))
        compA[tag] = (float(np.mean(raes)), raes)

    # ---- COMP-B: cross-seed NNLS optimal static weights ----
    compB_raes = []
    for i in range(N_SEEDS):
        # fit weights on the OTHER seeds (concatenated), apply to seed i
        Xfit = np.vstack([seed_M[j][0] for j in range(N_SEEDS) if j != i])
        yfit = np.concatenate([seed_y[j] for j in range(N_SEEDS) if j != i])
        w, _ = nnls(Xfit, yfit)
        if w.sum() == 0: w = np.ones(GBM_N + 3)
        w = w / w.sum()
        M, lo, hi = seed_M[i]
        compB_raes.append(rae(seed_y[i], np.clip(M @ w, lo, hi)))
    compB_m = float(np.mean(compB_raes))

    print(f"\nCONTROL deployed flat mean        RAE = {ctrl_m:.4f}  ({[round(r,4) for r in ctrl_raes]})")
    for tag, (m, raes) in compA.items():
        print(f"COMP-A cov-tilt {tag:18s}  RAE = {m:.4f}  d={m-ctrl_m:+.4f}  ({[round(r,4) for r in raes]})")
    print(f"COMP-B cross-seed NNLS weights    RAE = {compB_m:.4f}  d={compB_m-ctrl_m:+.4f}  ({[round(r,4) for r in compB_raes]})")

    best_A = min(compA.values(), key=lambda x: x[0])[0]
    best_A_tag = min(compA.items(), key=lambda x: x[1][0])[0]
    out = {"control_rae": ctrl_m, "control_raes": ctrl_raes,
           "compA_best_tag": best_A_tag, "compA_best_rae": best_A, "compA_delta": best_A - ctrl_m,
           "compA_all": {k: v[0] for k, v in compA.items()},
           "compB_rae": compB_m, "compB_raes": compB_raes, "compB_delta": compB_m - ctrl_m,
           "deploy_A": bool(best_A - ctrl_m < -0.001), "deploy_B": bool(compB_m - ctrl_m < -0.001)}
    json.dump(out, open(f"{P}/nb1210_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1210_summary.json  deployA={out['deploy_A']} deployB={out['deploy_B']}")


if __name__ == "__main__":
    main()
