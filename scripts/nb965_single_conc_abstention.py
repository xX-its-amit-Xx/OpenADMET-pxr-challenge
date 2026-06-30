"""nb965 — A5: single-conc-neighbor-gated abstention/shrink for the F2 failure mode
(greasy-novel-inactive over-prediction, +1.23 per phase1 post-mortem).

Distinct from A1 (which added SC activation as a FEATURE -> absorbed/no help): here SC neighbor
activation is used as an EXTERNAL CALIBRATION signal to SHRINK over-confident predictions on novel
compounds whose SC neighbors are inactive. Calibration, not a feature -> sidesteps chempropembed
absorption. Distinct from prior confidence-shrink (which overfit at n=253) by using the external SC
signal, not the model's own confidence.

Shrink: pred_new = pred - lambda * gate_novel * gate_inactive * relu(pred - floor)
  gate_novel    = sigmoid((tau_sim - sim_to_train)/s)   high when novel (far from train)
  gate_inactive = sigmoid((tau_act - nbr_act)/s)        high when SC neighbors inactive
Single cross-fit lambda; multi-seed verify on the chemprop_aux anchor (253).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import inchi
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

D = "data/processed"
SEEDS = [42, 101, 202, 303, 404, 505, 606]
K = 5


def murcko(s):
    try:
        m = Chem.MolFromSmiles(s); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None
    except Exception: return None


def load_sc():
    try:
        from src.pxr.data import load_single_conc; return load_single_conc()
    except Exception:
        import pandas as pd
        return pd.read_csv("data/raw/pxr-challenge_single_concentration_TRAIN.csv")


def knn_signal(fp_q, fp_ref, vals, exclude_self=True, bs=150):
    """top-K sim-weighted mean of vals, + max-sim. Returns (wmean_val, max_sim)."""
    B = fp_ref.astype(np.float32); bsum = B.sum(1)[None, :]
    wm = np.zeros(len(fp_q)); ms = np.zeros(len(fp_q))
    for i in range(0, len(fp_q), bs):
        Q = fp_q[i:i+bs].astype(np.float32)
        inter = Q @ B.T; u = Q.sum(1)[:, None] + bsum - inter; u[u == 0] = 1.0
        sim = inter / u
        if exclude_self: sim = np.where(sim > 0.999, -1.0, sim)
        for j in range(sim.shape[0]):
            idx = np.argpartition(sim[j], -K)[-K:]; s = sim[j, idx]; ok = s > 0
            if ok.any():
                w = s[ok]/s[ok].sum(); wm[i+j] = np.sum(w*vals[idx][ok]); ms[i+j] = s[ok].max()
    return wm, ms


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test()
    unb_idx = np.load(f"{D}/_audit_unblind_idx.npy")
    y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb_idx].tolist()
    scaf = [murcko(s) for s in smiles]
    anchor = np.load(f"{D}/te_chemprop_aux.npy")[unb_idx]
    floor = float(np.quantile(tr["pec50"], 0.25))
    print(f"anchor RAE={rae(y,anchor):.4f}  inactive floor(q25 train)={floor:.3f}")

    sc = load_sc().dropna(subset=["smiles", "log2_fc_estimate"])
    sc_act = np.nan_to_num(sc["log2_fc_estimate"].to_numpy(float), posinf=10, neginf=-10)
    fp_sc = morgan_fp_batch(sc["smiles"].tolist())
    fp_te = morgan_fp_batch(smiles)
    fp_tr = morgan_fp_batch(tr["smiles"].tolist())

    nbr_act, _ = knn_signal(fp_te, fp_sc, sc_act, exclude_self=True)
    _, sim_train = knn_signal(fp_te, fp_tr, np.zeros(len(fp_tr)), exclude_self=True)
    print(f"253: nbr_act median={np.median(nbr_act):.3f}  sim_to_train median={np.median(sim_train):.3f}")

    # F2 diagnostic: on novel + inactive-neighbor compounds, does anchor over-predict?
    tau_sim, tau_act = np.median(sim_train), np.median(nbr_act)
    f2 = (sim_train < tau_sim) & (nbr_act < tau_act)
    over = anchor[f2] - y[f2]
    print(f"F2 cohort (novel & inactive-nbr) n={int(f2.sum())}: mean(anchor-y)={over.mean():+.3f} "
          f"(positive = over-prediction, the F2 signature)")

    s = 0.15
    g_nov = 1/(1+np.exp((sim_train - tau_sim)/s))
    g_ina = 1/(1+np.exp((nbr_act - tau_act)/s))
    gate = g_nov * g_ina

    def shrink(pred, lam):
        return pred - lam * gate * np.maximum(pred - floor, 0)

    rows = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        pred = anchor.copy()
        for tri, vai in folds:
            best_lam, best_r = 0.0, rae(y[tri], anchor[tri])
            for lam in np.linspace(0, 1.0, 21):
                r = rae(y[tri], shrink(anchor, lam)[tri])
                if r < best_r: best_r, best_lam = r, lam
            pred[vai] = shrink(anchor, best_lam)[vai]
        r_base = rae(y, anchor); r_shr = rae(y, pred)
        rows.append({"seed": seed, "base": round(r_base, 4), "shrunk": round(r_shr, 4),
                     "delta": round(r_shr - r_base, 5)})
        print(f"  seed {seed}: base={r_base:.4f} shrunk={r_shr:.4f} delta={r_shr-r_base:+.5f}")

    d = np.array([r["delta"] for r in rows])
    stable = d.mean() < 0 and abs(d.mean()) > d.std()
    print("\n" + "=" * 58)
    print(f"SC-abstention delta vs chemprop_aux: mean={d.mean():+.5f} std={d.std():.5f} "
          f"wins={int((d<0).sum())}/{len(SEEDS)} stable={stable}")
    print(">>> A5 REAL -> targets F2" if stable else ">>> A5 noise/no help -> F2 not fixed by SC-neighbor shrink")
    print("=" * 58)
    json.dump({"floor": floor, "f2_n": int(f2.sum()), "f2_overpred": float(over.mean()),
               "rows": rows, "delta_mean": float(d.mean()), "delta_std": float(d.std()),
               "stable": bool(stable)}, open(f"{D}/nb965_single_conc_abstention.json", "w"), indent=2)
    print(f"saved -> {D}/nb965_single_conc_abstention.json")


if __name__ == "__main__":
    main()
