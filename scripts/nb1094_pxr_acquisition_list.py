"""nb1094 — PXR Phase-2 ACQUISITION LIST (the deployable deliverable, validated by nb1093).

nb1093 proved coverage-greedy (facility-location) acquisition robustly beats random/maxsim and gives up to ~16x
data efficiency. Apply it to the REAL PXR challenge: rank a purchasable/measurable library by how much each compound
improves COVERAGE of the 513 test analogs, MARGINAL over the existing 4139 train compounds (so we only credit
compounds that cover test regions train covers POORLY = the novel-scaffold F2 blind spots).

  cov[t] = max Tanimoto(train, test_t)              # current coverage of each test compound by existing train
  greedy: repeatedly pick candidate c maximizing  sum_t max(sim(c,t) - cov[t], 0)  ; then cov |= sim(c,.)
  -> the ranked picks are 'measure these N compounds next' to most reduce test blind-spots.

Candidate library = data/processed/bfd_cofold_candidates.csv (minus compounds already in train CRC), tagged by source:
  single_conc = we already have the compound + single-point data (cheap CRC follow-up)
  ext_*       = external chemical space (purchase). Output ranked CSV + coverage report.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test, load_train
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

P = "data/processed"
N_SELECT = 500


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return Chem.MolToInchiKey(m) if m else None


def fp_float(smiles):
    return (morgan_fp_batch(smiles).astype(np.float32) > 0).astype(np.float32)


def tani_cross(A, B):
    """Tanimoto between rows of A (nA,bits) and B (nB,bits) -> (nA,nB)."""
    inter = A @ B.T
    return inter / np.clip(A.sum(1)[:, None] + B.sum(1)[None, :] - inter, 1, None)


def main():
    te = load_test().reset_index(drop=True)
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    cand = pd.read_csv(f"{P}/bfd_cofold_candidates.csv")

    tr_ik = set(tr["smiles"].map(ik).dropna())
    cand = cand[~cand["ik"].isin(tr_ik)].drop_duplicates("ik").reset_index(drop=True)
    print(f"test={len(te)} train={len(tr)} candidates(not-in-train)={len(cand)}", flush=True)

    print("fingerprinting...", flush=True)
    Fte = fp_float(te["smiles"].tolist())
    Ftr = fp_float(tr["smiles"].tolist())
    Fca = fp_float(cand["smiles"].tolist())

    # current coverage of each test compound by existing train
    cov = tani_cross(Ftr, Fte).max(0)                       # (513,)
    print(f"current test coverage by train: median {np.median(cov):.3f}  worst-decile {np.percentile(cov,10):.3f}", flush=True)
    blind = np.argsort(cov)[:30]
    print(f"30 worst-covered test compounds (blind spots): max-train-sim ranges "
          f"{cov[blind].min():.2f}..{cov[blind].max():.2f}", flush=True)

    Sct = tani_cross(Fca, Fte)                              # (n_cand, 513) candidate->test sim
    cov0 = cov.copy()
    picks, gains = [], []
    available = np.ones(len(cand), bool)
    cov_cur = cov.copy()
    for step in range(N_SELECT):
        marg = np.where(available[:, None], np.maximum(Sct - cov_cur[None, :], 0), 0).sum(1)
        j = int(np.argmax(marg))
        if marg[j] <= 1e-6: break
        picks.append(j); gains.append(float(marg[j])); available[j] = False
        cov_cur = np.maximum(cov_cur, Sct[j])
        if (step + 1) % 100 == 0:
            print(f"  picked {step+1}: median cov {np.median(cov_cur):.3f}  worst-decile {np.percentile(cov_cur,10):.3f}", flush=True)

    sel = cand.iloc[picks].copy()
    sel["rank"] = np.arange(1, len(picks) + 1)
    sel["coverage_gain"] = gains
    sel["n_test_newly_covered"] = [(Sct[j] > cov0).sum() for j in picks]   # vs ORIGINAL train coverage
    out_cols = ["rank", "smiles", "ik", "scaffold", "source", "coverage_gain", "n_test_newly_covered"]
    sel[out_cols].to_csv(f"{P}/nb1094_pxr_acquisition_list.csv", index=False)

    # coverage report at budgets
    rep = {}
    for n in [100, 250, 500]:
        if n > len(picks): n = len(picks)
        c = cov0.copy()
        for j in picks[:n]:
            c = np.maximum(c, Sct[j])
        rep[n] = dict(median_cov=float(np.median(c)), worst_decile=float(np.percentile(c, 10)),
                      frac_test_below_0p4=float((c < 0.4).mean()))
    print("\n=== PXR ACQUISITION COVERAGE REPORT ===")
    print(f"before (train only): median {np.median(cov0):.3f}  worst-decile {np.percentile(cov0,10):.3f}  "
          f"frac<0.4 {(cov0<0.4).mean():.3f}")
    for n, r in rep.items():
        print(f"+{n:4d} compounds: median {r['median_cov']:.3f}  worst-decile {r['worst_decile']:.3f}  "
              f"frac<0.4 {r['frac_test_below_0p4']:.3f}")
    print("\nsource breakdown of top-250 picks:")
    print(sel.head(250)["source"].value_counts().to_string())
    json.dump({"before": {"median": float(np.median(cov0)), "worst_decile": float(np.percentile(cov0, 10)),
                          "frac_below_0p4": float((cov0 < 0.4).mean())}, "after": rep,
               "n_picks": len(picks), "source_top250": sel.head(250)["source"].value_counts().to_dict()},
              open(f"{P}/nb1094_acquisition_report.json", "w"), indent=2)
    print(f"\nwrote {P}/nb1094_pxr_acquisition_list.csv ({len(picks)} compounds) + report json")


if __name__ == "__main__":
    main()
