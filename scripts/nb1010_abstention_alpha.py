"""nb1010 — ABSTENTION-alpha: the one positive-direction signal from the featurizer sweep.
Signaturizer applicability-alpha = per-compound off-manifold measure (low alpha = novel/off-manifold).
F2 failure (pm06): novel-scaffold inactives OVER-predicted (+1.23). Hypothesis: low-alpha -> anchor over-predicts.
STEP 1 (decisive diagnostic): does alpha predict signed error (over-prediction) on the HONEST nb3200 cross-fit?
STEP 2: if yes, alpha-gated shrinkage toward a conservative prior, cross-fit lambda. Test on nb3200 + chemprop_aux.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
from scipy.stats import pearsonr, spearmanr

D = "data/processed"; U = "C:/pxr_struct/signaturizer"
SEEDS = [42, 101, 202, 303, 404, 505, 606]


def murcko(s):
    m = Chem.MolFromSmiles(str(s)); return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else None


def gated_shrink_cv(anchor, y, scaf, w_off, priors_lo):
    """alpha-gated shrink: for off-manifold (high w_off) compounds pull toward train low-quantile prior.
    cross-fit lambda over grid per fold (prior from train fold only)."""
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    out = []
    for seed in SEEDS:
        folds = scaffold_kfold_indices(scaf, n_splits=5, seed=seed)
        pred = anchor.copy()
        for tri, vai in folds:
            prior = np.quantile(y[tri], 0.25)   # conservative target for off-manifold rows
            best_l, best_r = 0.0, 1e9
            for lam in grid:                      # pick lambda on TRAIN fold only
                pt = anchor[tri] - lam * w_off[tri] * (anchor[tri] - prior)
                r = rae(y[tri], pt)
                if r < best_r:
                    best_r, best_l = r, lam
            pred[vai] = anchor[vai] - best_l * w_off[vai] * (anchor[vai] - prior)
        out.append(rae(y, pred))
    return np.array(out)


def main():
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y = np.load(f"{D}/_audit_unblind_y.npy")
    smiles = te["smiles"].to_numpy()[unb].tolist(); scaf = [murcko(s) for s in smiles]
    alpha = np.nan_to_num(np.load(f"{U}/sign_applicability_513.npy").astype(np.float32))[unb]  # (253,7)
    a_mean = alpha.mean(1)
    w_off = 1.0 - (a_mean - a_mean.min()) / (a_mean.max() - a_mean.min() + 1e-9)  # high for off-manifold

    anchors = {"nb3200_oof": np.load(f"{D}/nb3200_pred_oof.npy"),
               "chemprop_aux": np.load(f"{D}/te_chemprop_aux.npy")[unb]}
    print(f"alpha_mean range [{a_mean.min():.3f},{a_mean.max():.3f}]; n={len(y)}\n")
    report = {}
    for name, anchor in anchors.items():
        anchor = np.asarray(anchor, np.float32)
        se = anchor - y                 # signed error: + = over-prediction (F2 direction)
        ae = np.abs(se)
        rc_se = pearsonr(a_mean, se); rc_ae = pearsonr(a_mean, ae); rs_ae = spearmanr(a_mean, ae)
        # low-alpha (off-manifold) quartile vs high-alpha
        q1 = a_mean <= np.quantile(a_mean, 0.25); q4 = a_mean >= np.quantile(a_mean, 0.75)
        print(f"--- anchor {name} (RAE {rae(y,anchor):.4f}) ---")
        print(f"  corr(alpha, signed_err)={rc_se[0]:+.3f} (p={rc_se[1]:.3f})  [F2 expects NEGATIVE: low-alpha->over-predict]")
        print(f"  corr(alpha, |err|)={rc_ae[0]:+.3f} (p={rc_ae[1]:.3f})  spearman={rs_ae[0]:+.3f}")
        print(f"  low-alpha Q1: mean_signed_err={se[q1].mean():+.3f} mean|err|={ae[q1].mean():.3f}  |  high-alpha Q4: {se[q4].mean():+.3f} / {ae[q4].mean():.3f}")
        rr = gated_shrink_cv(anchor, y, scaf, w_off, None)
        base = rae(y, anchor); d = rr.mean() - base
        stable = d < 0 and abs(d) > rr.std()
        print(f"  alpha-gated shrink: {rr.mean():.4f} +/- {rr.std():.4f}  delta={d:+.5f} stable={stable}\n")
        report[name] = {"rae": round(base, 4), "corr_signed": round(rc_se[0], 4), "p_signed": round(rc_se[1], 4),
                        "corr_abs": round(rc_ae[0], 4), "q1_signed": round(float(se[q1].mean()), 4),
                        "q4_signed": round(float(se[q4].mean()), 4), "gated_mean": round(float(rr.mean()), 4),
                        "gated_delta": round(float(d), 5), "gated_stable": bool(stable)}
    verdict = any(v["gated_stable"] for v in report.values())
    print("=" * 62)
    print(">>> ABSTENTION-ALPHA REAL -> off-manifold gating fixes the F2 tail" if verdict
          else ">>> abstention-alpha dead too -> alpha does not predict the error tail on honest cross-fit")
    print("=" * 62)
    json.dump(report, open(f"{U}/nb1010_abstention_alpha.json", "w"), indent=2)
    print(f"saved -> {U}/nb1010_abstention_alpha.json")


if __name__ == "__main__":
    main()
