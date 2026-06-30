"""nb950 — OOD-subset baseline: where does the activity error actually live?

Decomposes RAE on the 253 unblinded test compounds into:
  - SEEN-scaffold subset   (Bemis-Murcko scaffold appears in the 4139 train)
  - NOVEL-scaffold subset  (scaffold absent from train)

This sets the TARGET for the foundation-model bet. The hypothesis (feedback
memory: 90.5% of test scaffolds are novel; failure mode F2 = novel-scaffold
inactives over-predicted) says our error concentrates on the novel subset.
If our best stack (nb3200, pooled 0.4416) already crushed the novel tail,
a pretrained foundation model has little room. If nb3200's win is carried by
the seen subset while the novel tail is still ~0.7, the foundation bet is
well-motivated and THIS is the number it must beat.

Pure diagnostic on existing artifacts — no training, no GPU.
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception as e:
    print("RDKit import failed:", e); sys.exit(1)

D = "data/processed"


def murcko(smi: str) -> str:
    """Bemis-Murcko generic scaffold SMILES; '' on parse failure."""
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc) if sc is not None else ""
    except Exception:
        return ""


def main():
    # ---- substrate ----
    tr = load_train()
    te = load_test()
    unb_idx = np.load(f"{D}/_audit_unblind_idx.npy")           # 253 positions in the 513
    y = np.load(f"{D}/_audit_unblind_y.npy")                   # 253 truth
    te_smiles = te["smiles"].to_numpy()[unb_idx]              # 253 test SMILES

    # ---- scaffold vocabulary from train ----
    train_scaffolds = set()
    for smi in tr["smiles"]:
        s = murcko(smi)
        if s:
            train_scaffolds.add(s)
    test_scaf = np.array([murcko(s) for s in te_smiles])
    is_novel = np.array([s == "" or s not in train_scaffolds for s in test_scaf])
    n_novel, n_seen = int(is_novel.sum()), int((~is_novel).sum())
    print(f"train unique Murcko scaffolds: {len(train_scaffolds)}")
    print(f"253 unblind: novel-scaffold={n_novel} ({100*n_novel/253:.1f}%)  "
          f"seen-scaffold={n_seen} ({100*n_seen/253:.1f}%)")
    print()

    # ---- predictors to decompose ----
    preds = {}
    cp = np.load(f"{D}/te_chemprop_aux.npy")                  # 513 deploy
    preds["chemprop_aux (PRE-clean anchor)"] = cp[unb_idx]
    for name, fn in [("nb3200 (best stack, pred_oof)", f"{D}/nb3200_pred_oof.npy"),
                     ("nb3090 (blend, pred_oof)",      f"{D}/nb3090_pred_oof.npy")]:
        if os.path.exists(fn):
            v = np.load(fn)
            if len(v) == 253:
                preds[name] = v

    # ---- decompose RAE (pooled = LB-faithful, single denominator per subset) ----
    print(f"{'predictor':34s} {'overall':>9s} {'SEEN':>9s} {'NOVEL':>9s}  "
          f"{'novel-seen gap':>14s}")
    print("-" * 82)
    for name, p in preds.items():
        r_all = rae(y, p)
        r_seen = rae(y[~is_novel], p[~is_novel]) if n_seen else float("nan")
        r_novel = rae(y[is_novel], p[is_novel]) if n_novel else float("nan")
        print(f"{name:34s} {r_all:9.4f} {r_seen:9.4f} {r_novel:9.4f}  {r_novel-r_seen:14.4f}")

    # ---- mean-predictor reference per subset (RAE is relative to abs error of mean) ----
    print()
    print("Interpretation guide:")
    print("  RAE<1 beats the subset's mean predictor. A large NOVEL>SEEN gap means")
    print("  the error concentrates on novel chemistry -> the foundation-model target.")
    print("  If nb3200 NOVEL ~= chemprop_aux NOVEL, our stack did NOT fix the tail")
    print("  (the -0.0175 trajectory gain came from the seen subset) -> bet is live.")

    # ---- save for the foundation harness to target ----
    out = {
        "n_novel": n_novel, "n_seen": n_seen,
        "is_novel": is_novel.tolist(),
    }
    for name, p in preds.items():
        key = name.split()[0]
        out[f"{key}_novel_rae"] = float(rae(y[is_novel], p[is_novel])) if n_novel else None
        out[f"{key}_seen_rae"] = float(rae(y[~is_novel], p[~is_novel])) if n_seen else None
    import json
    json.dump(out, open(f"{D}/nb950_ood_subset_baseline.json", "w"), indent=2)
    np.save(f"{D}/nb950_is_novel_253.npy", is_novel)
    print(f"\nsaved -> {D}/nb950_ood_subset_baseline.json  +  nb950_is_novel_253.npy")


if __name__ == "__main__":
    main()
