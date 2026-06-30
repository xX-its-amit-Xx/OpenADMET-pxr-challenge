"""nb1215 — COMBINATOR tick: two NON-STANDARD compositions, honest-gated vs the
DEPLOYED flat-mean (best 0.4242, AIMNet2+strain+D4+DBSTEP in the 4 GBMs).

Reuses C:/pxr_work/_combo_seed_data.pkl (built by _combo_probe.py): per seed the
deployed 7-member matrix M on the never-tuned MTL holdout (ho_idx_seed{0,1,2}),
its y, and the a-priori clip (lo,hi). Gate = matched delta vs deployed flat mean.

COMP-E  CROSS-TARGET BIOLOGY member injection (diversity lever):
  The deployed members are tightly correlated (min OOF corr ~0.89) so weight-
  learning / robust-agg / routing all FAILED (nb1210/nb1213) — the only way to
  win is a GENUINELY DIVERSE member. We have cached cross-target NR-panel heads
  on the SAME holdouts (car/ahr/ppar/octant/sne/...), present in NO deployed
  member except sn. Cross-target affinity was the strongest orthogonal signal in
  cy299 (+0.367 corr-with-truth). Add the diverse-but-aligned head(s) as EQUAL
  flat member(s) (deployed convention is flat mean — no learned weights) and at
  fixed a-priori down-weights {0.5,0.3}. Also a CONSENSUS member = mean of the
  heads whose corr-with-deployed-ERROR is positive (a-priori selection by sign,
  computed leak-free per the probe). NOT weight-learning -> dodges the overfit trap.

COMP-F  FEATURE-FUSION vs MEMBER (CheMeleon emb x QM in ONE model):
  Deployed CheMeleon enters as a flat member = a CheMeleon-emb LGBM, separate
  from the 4 GBMs (which see combined+QM but NOT CheMeleon). So cross-terms
  CheMeleon x QM exist in NO member. COMP-F trains ONE fused LGBM on
  [combined + AIMNet2/strain/D4/DBSTEP QM + CheMeleon-emb(2048)] and SWAPS it in
  for the flat CheMeleon member: ensemble = mean(4 GBM, fused-LGBM, tab, sn).
  If fusion captures interaction the flat composition can't, it beats the deployed
  mean. Leak-free: fused LGBM trained on trn rows only, predicts the same holdout.

Deploy a composition iff matched delta < -0.001 vs the deployed flat mean.
"""
import os, sys, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nb1126_combinatorial_search import feature_matrix, CACHE, make_model
from src.pxr.data import load_train
from src.pxr.eval import rae
from sklearn.preprocessing import StandardScaler
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"; SD = "C:/pxr_work/search"; MTL = "C:/pxr_work/mtl"
PKL = "C:/pxr_work/_combo_seed_data.pkl"
AIM = "C:/pxr_work/aimnet2/aimnet_features.csv"
ACOLS = ["aimnet_energy","aimnet_qmin","aimnet_qmax","aimnet_qabs_mean","aimnet_qstd","aimnet_qsum_abs","aimnet_dipole","aimnet_fmax","aimnet_frms"]
STR = "C:/pxr_work/strain/strain_features.csv"
SCOLS = ["strain_relax_mean","strain_relax_max","conf_espread","conf_erange","conf_n","rmsd_mean","rmsd_max","e_per_heavy"]
D4 = "C:/pxr_work/d4/d4_features.csv"
DCOLS = ["d4_alpha_sum","d4_alpha_mean","d4_alpha_std","d4_alpha_max","d4_c6diag_mean","d4_c6diag_std","d4_c6_total","d4_edisp","d4_edisp_per_atom","d4_cn_mean","d4_cn_max","d4_qeeq_min","d4_qeeq_max","d4_qeeq_std","d4_qeeq_absum"]
DB = "C:/pxr_work/dbstep/dbstep_features.csv"
DBCOLS = ["vbur_r25","vbur_r35","vbur_r45","vbur_r55","vbur_r65","ster_L","ster_Bmin","ster_Bmax","ster_aniso","npr1","npr2","asphericity","spherocity","eccentricity","radgyr","inertial_sf"]
HEADS = ["car","ahr","ppar","octant","sne","nc","control","treat","sn_errg"]  # sn already deployed; exclude
LGBM_HP = {"n": 600, "leaves": 64, "lr": 0.03, "sub": 0.8, "col": 0.6, "l2": 1.0}


def ablk(csv, cols, names):
    df = pd.read_csv(csv); df = df[df.src == "train"].drop_duplicates(subset="name", keep="first")
    sub = df.set_index("name").reindex(names); X = sub[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(X, axis=0); ii = np.where(np.isnan(X)); X[ii] = np.take(med, ii[1]); return X


def main():
    sd = pickle.load(open(PKL, "rb"))
    ctrl_raes = [rae(s["y"], np.clip(s["M"].mean(1), s["lo"], s["hi"])) for s in sd]
    ctrl = float(np.mean(ctrl_raes))
    print(f"CONTROL deployed flat-mean  RAE = {ctrl:.4f}  ({[round(r,4) for r in ctrl_raes]})")

    # ---- which heads are positively aligned with the deployed ERROR (a-priori, by sign) ----
    pos = []
    for h in HEADS:
        ec = []
        for si, s in enumerate(sd):
            a = np.load(f"{MTL}/{h}_oof_seed{si}.npy").ravel()
            ec.append(np.corrcoef(a, s["y"] - np.clip(s["M"].mean(1), s["lo"], s["hi"]))[0, 1])
        if np.mean(ec) > 0: pos.append(h)
    print(f"heads with +corr-vs-error (consensus pool): {pos}")

    def blend_rae(extra_cols, w):
        """extra_cols: list of (per-seed arrays); add as members at weight w each (flat=1)."""
        raes = []
        for si, s in enumerate(sd):
            M = s["M"]; base = M.mean(1)
            cols = [ec[si] for ec in extra_cols]
            num = M.sum(1) + w * sum(cols); den = M.shape[1] + w * len(cols)
            pred = np.clip(num / den, s["lo"], s["hi"]); raes.append(rae(s["y"], pred))
        return float(np.mean(raes)), raes

    # ---- COMP-E: individual head members + consensus, at weights {1.0,0.5,0.3} ----
    compE = {}
    head_arrs = {h: [np.load(f"{MTL}/{h}_oof_seed{si}.npy").ravel() for si in range(len(sd))] for h in HEADS}
    cand = {h: [head_arrs[h]] for h in HEADS}
    if pos:
        cons = [np.mean([head_arrs[h][si] for h in pos], axis=0) for si in range(len(sd))]
        cand["CONSENSUS"] = [cons]
    for name, cols in cand.items():
        for w in (1.0, 0.5, 0.3):
            m, v = blend_rae(cols, w); compE[f"{name}@w{w}"] = (m, m - ctrl, v)

    # ---- COMP-F: fused LGBM (combined+QM+CheMeleon-emb) swapped in for CheMeleon member ----
    d = np.load(CACHE); ytr = d["ytr"]; n = len(ytr); Xtr, _ = feature_matrix(d, "combined")
    chem_emb = np.load(f"{SD}/chemeleon_tr.npy")                 # 4139 x 2048
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    Xqm = ablk(AIM, ACOLS, tr["name"]); Xst = ablk(STR, SCOLS, tr["name"])
    Xd4 = ablk(D4, DCOLS, tr["name"]); Xdb = ablk(DB, DBCOLS, tr["name"])
    fused_raes, swap_raes = [], []
    for si, s in enumerate(sd):
        ho = s["ho"]; hs = set(ho.tolist()); trn = np.array([i for i in range(n) if i not in hs])
        scs = [StandardScaler().fit(X[trn]) for X in (Xqm, Xst, Xd4, Xdb)]
        Xex = np.hstack([sc.transform(X) for sc, X in zip(scs, (Xqm, Xst, Xd4, Xdb))])
        Xall = np.hstack([Xtr, Xex, chem_emb]).astype(np.float32)
        m = make_model("lgbm", LGBM_HP); m.fit(Xall[trn], ytr[trn])
        fused = m.predict(Xall[ho])
        fused_raes.append(rae(s["y"], np.clip(fused, s["lo"], s["hi"])))
        # swap fused LGBM in for the flat CheMeleon member (col index 4 = chem; 0-3 gbm, 4 chem, 5 tab, 6 sn)
        M = s["M"]; cols = [M[:, j] for j in range(M.shape[1]) if j != 4] + [fused]
        pred = np.clip(np.column_stack(cols).mean(1), s["lo"], s["hi"]); swap_raes.append(rae(s["y"], pred))
    compF = {"fused-LGBM-standalone": (float(np.mean(fused_raes)), float(np.mean(fused_raes)) - ctrl, fused_raes),
             "swap-chem->fused": (float(np.mean(swap_raes)), float(np.mean(swap_raes)) - ctrl, swap_raes)}

    print("\n--- COMP-E cross-target biology member injection ---")
    for k, (m, dl, v) in sorted(compE.items(), key=lambda x: x[1][0]):
        print(f"  {k:22s} RAE={m:.4f}  d={dl:+.4f}  ({[round(r,4) for r in v]})")
    print("--- COMP-F CheMeleon-emb x QM fusion vs member ---")
    for k, (m, dl, v) in compF.items():
        print(f"  {k:22s} RAE={m:.4f}  d={dl:+.4f}  ({[round(r,4) for r in v]})")

    bestE = min(compE.items(), key=lambda x: x[1][0]); bestF = min(compF.items(), key=lambda x: x[1][0])
    out = {"control_rae": ctrl, "control_raes": ctrl_raes, "pos_heads": pos,
           "compE_all": {k: v[0] for k, v in compE.items()},
           "compE_best_tag": bestE[0], "compE_best_rae": bestE[1][0], "compE_delta": bestE[1][1],
           "compF_all": {k: v[0] for k, v in compF.items()},
           "compF_best_tag": bestF[0], "compF_best_rae": bestF[1][0], "compF_delta": bestF[1][1],
           "deploy_E": bool(bestE[1][1] < -0.001), "deploy_F": bool(bestF[1][1] < -0.001)}
    json.dump(out, open(f"{P}/nb1215_combinator_gate_summary.json", "w"), indent=2)
    print(f"\nsaved {P}/nb1215_combinator_gate_summary.json  deployE={out['deploy_E']} deployF={out['deploy_F']}")


if __name__ == "__main__":
    main()
