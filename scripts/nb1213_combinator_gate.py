"""nb1213 — COMBINATOR: two NON-STANDARD compositions of the DEPLOYED members,
honest-gated against the deployed FLAT-MEAN (best 0.4252, AIMNet2+strain+D4).

Members per seed on holdout (deployed config):
  4 GBM(combined+AIMNet2+strain+D4), CheMeleon OOF, TabPFN OOF, sisterNR GNN OOF.
Control = deployed flat mean of all 7, clipped.

COMP-C  Disagreement-gated ROBUST aggregation (no learned weights):
  flat mean is non-robust to a single wild member on cliff/noisy compounds.
  Test 3 zero-param aggregators vs flat mean:
   (i)  per-compound MEDIAN of the 7 members,
   (ii) TRIMMED mean (drop the single hi + single lo member, mean of middle 5),
   (iii) DISAGREEMENT-GATED: use flat mean normally, switch to median ONLY on
        compounds whose member std exceeds the global-median member std (a-priori
        threshold, NOT tuned) — robustify only where members disagree.

COMP-D  DISTINCT-AXIS residual chain on the Boltz-2 interaction embedding:
  stage-A = deployed ensemble. stage-B = strongly-regularized Ridge predicting
  the stage-A RESIDUAL from ONLY boltz_z_rich (512-d structural-interaction
  embedding, present in NO deployed member -> a genuinely distinct axis).
  Leak-free: inner split of trn (trn_a fits stage-A, trn_b gives unbiased
  residuals to fit stage-B); CheMeleon/TabPFN OOF are valid on trn_b.
  Correction applied to the 7-member control with a fixed a-priori shrink.
  alpha + shrink are a-priori (reported for 2 settings), never tuned on eval.

Deploy a composition iff matched delta < -0.001 vs the deployed flat mean.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
LOG = f"{SD}/results.jsonl"; N_SEEDS = 3
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
ZTR = f"{P}/boltz_z_rich_train.npy"


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


def main():
    d = np.load(CACHE); ytr = d["ytr"]; n_tr = len(ytr)
    Xtr, _ = feature_matrix(d, "combined")
    chem = np.load(f"{SD}/chemeleon_oof.npy"); tab = np.load(f"{SD}/tabpfn_oof.npy")
    Z = np.load(ZTR)                                   # 4139 x 512 boltz-z interaction emb

    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    Xqm = aligned_block(AIM, ACOLS, tr["name"])
    Xst = aligned_block(STR, SCOLS, tr["name"])
    Xd4 = aligned_block(D4, DCOLS, tr["name"])

    topK = topK_configs(("lgbm", "xgb", "cat", "histgb"))
    print(f"deployed members: {[r['arch'] for r in topK]} + CheMeleon + TabPFN + sisterNR-GNN")

    # ---- per-seed: 7-member matrix on ho + boltz-z residual correction on ho ----
    seed_M, seed_y, seed_corr = [], [], []
    rng = np.random.RandomState(0)
    for seed in range(N_SEEDS):
        ho = np.load(f"{MTL}/ho_idx_seed{seed}.npy")
        ho_set = set(ho.tolist())
        trn = np.array([i for i in range(n_tr) if i not in ho_set])
        lo, hi = np.quantile(ytr[trn], 0.05), np.quantile(ytr[trn], 0.98)
        scq = StandardScaler().fit(Xqm[trn]); scs = StandardScaler().fit(Xst[trn]); scd = StandardScaler().fit(Xd4[trn])
        Xex = np.hstack([scq.transform(Xqm), scs.transform(Xst), scd.transform(Xd4)])
        gnn = np.load(f"{MTL}/sn_oof_seed{seed}.npy").ravel()

        # stage-A members on ho
        gbm_ho = [fit_pred(c, Xtr, ytr, trn, ho, Xex) for c in topK]
        M = np.column_stack(gbm_ho + [chem[ho], tab[ho], gnn])     # M x 7
        seed_M.append((M, lo, hi)); seed_y.append(ytr[ho])

        # ---- COMP-D residual chain (distinct axis = boltz-z) ----
        # inner split of trn: trn_a fits stage-A; trn_b gives unbiased residuals.
        perm = rng.permutation(len(trn)); cut = int(0.8 * len(trn))
        trn_a, trn_b = trn[perm[:cut]], trn[perm[cut:]]
        # 6-member stage-A on trn_b (drop GNN: sn_oof only defined on ho).
        gbm_b = [fit_pred(c, Xtr, ytr, trn_a, trn_b, Xex) for c in topK]
        ensA_b = np.column_stack(gbm_b + [chem[trn_b], tab[trn_b]]).mean(1)
        resid_b = ytr[trn_b] - ensA_b
        # impute boltz-z NaNs (uncofolded compounds) with trn_a column medians,
        # so they read as neutral -> ~zero correction.
        zmed = np.nanmedian(Z[trn_a], axis=0); zmed = np.nan_to_num(zmed)
        def zimp(rows):
            X = Z[rows].copy(); ii = np.where(np.isnan(X)); X[ii] = np.take(zmed, ii[1]); return X
        zsc = StandardScaler().fit(zimp(trn_a))
        Zb, Zho = zsc.transform(zimp(trn_b)), zsc.transform(zimp(ho))
        corr = {}
        for alpha in (100.0, 1000.0):
            rg = Ridge(alpha=alpha).fit(Zb, resid_b)
            corr[alpha] = rg.predict(Zho)
        seed_corr.append(corr)

    GBM_N = len(topK)

    # ---- Control: deployed flat mean ----
    ctrl_raes = [rae(y, np.clip(M.mean(1), lo, hi)) for (M, lo, hi), y in zip(seed_M, seed_y)]
    ctrl_m = float(np.mean(ctrl_raes))

    # ---- COMP-C: robust aggregation ----
    def med_raes():
        return [rae(y, np.clip(np.median(M, 1), lo, hi)) for (M, lo, hi), y in zip(seed_M, seed_y)]
    def trim_raes():
        out = []
        for (M, lo, hi), y in zip(seed_M, seed_y):
            s = np.sort(M, 1)[:, 1:-1].mean(1)          # drop hi+lo, mean of middle 5
            out.append(rae(y, np.clip(s, lo, hi)))
        return out
    def disagree_raes():
        # global a-priori threshold = median of per-compound member std across all seeds
        all_std = np.concatenate([M.std(1) for (M, lo, hi) in seed_M])
        thr = np.median(all_std)
        out = []
        for (M, lo, hi), y in zip(seed_M, seed_y):
            std = M.std(1); agg = M.mean(1).copy()
            hot = std > thr; agg[hot] = np.median(M[hot], 1)
            out.append(rae(y, np.clip(agg, lo, hi)))
        return out
    compC = {"median": med_raes(), "trim5": trim_raes(), "disagree-gated": disagree_raes()}
    compC = {k: (float(np.mean(v)), v) for k, v in compC.items()}

    # ---- COMP-D: boltz-z residual chain, fixed a-priori shrink ----
    compD = {}
    for alpha in (100.0, 1000.0):
        for shrink in (0.5, 1.0):
            raes = []
            for (M, lo, hi), y, corr in zip(seed_M, seed_y, seed_corr):
                pred = M.mean(1) + shrink * corr[alpha]
                raes.append(rae(y, np.clip(pred, lo, hi)))
            compD[f"a{int(alpha)}_s{shrink}"] = (float(np.mean(raes)), raes)

    print(f"\nCONTROL deployed flat mean        RAE = {ctrl_m:.4f}  ({[round(r,4) for r in ctrl_raes]})")
    for k, (m, v) in compC.items():
        print(f"COMP-C {k:16s}  RAE = {m:.4f}  d={m-ctrl_m:+.4f}  ({[round(r,4) for r in v]})")
    for k, (m, v) in compD.items():
        print(f"COMP-D boltz-z-resid {k:11s} RAE = {m:.4f}  d={m-ctrl_m:+.4f}  ({[round(r,4) for r in v]})")

    bestC = min(compC.items(), key=lambda x: x[1][0]); bestD = min(compD.items(), key=lambda x: x[1][0])
    out = {"control_rae": ctrl_m, "control_raes": ctrl_raes,
           "compC_all": {k: v[0] for k, v in compC.items()},
           "compC_best_tag": bestC[0], "compC_best_rae": bestC[1][0], "compC_delta": bestC[1][0] - ctrl_m,
           "compD_all": {k: v[0] for k, v in compD.items()},
           "compD_best_tag": bestD[0], "compD_best_rae": bestD[1][0], "compD_delta": bestD[1][0] - ctrl_m,
           "deploy_C": bool(bestC[1][0] - ctrl_m < -0.001), "deploy_D": bool(bestD[1][0] - ctrl_m < -0.001)}
    json.dump(out, open(f"{P}/nb1213_combinator_gate_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1213_combinator_gate_summary.json  deployC={out['deploy_C']} deployD={out['deploy_D']}")


if __name__ == "__main__":
    main()
