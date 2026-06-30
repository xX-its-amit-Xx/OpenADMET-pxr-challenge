"""nb974 — PHASE 4: classify train+test compounds by PXR binding MODE, calibrate a per-mode
pEC50 range from train, and TEST the premise: does binding-mode membership predict pEC50?

Modes (from nb970/971 taxonomy): A_tripod, B_blade, C_skewer, D_blob, E_reach.
Features: heavy atoms, HBA/HBD, aromatic rings, fsp3 (2D, RDKit) + NPR1/NPR2/Rg (3D shape).
Train shape reused from nb954_desc3d_ckpt.npy (cols: NPR1=3, NPR2=4, Rg=8); test shape fresh.

VALIDATION (the crux): scaffold-CV where each val compound is predicted as its mode's fold-train
median pEC50. If that beats the global-mean predictor (RAE<1) the premise holds; how close it gets
to LGBM (0.57) measures how much pEC50 is mechanistically pose-determined.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae, scaffold_kfold_indices
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Descriptors3D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

D = "data/processed"
OUT = "C:/pxr_struct"
MODES = ["A_tripod", "B_blade", "C_skewer", "D_blob", "E_reach"]


def feats2d(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return dict(heavy=Descriptors.HeavyAtomCount(m), hba=Descriptors.NumHAcceptors(m),
                hbd=Descriptors.NumHDonors(m), arom=Descriptors.NumAromaticRings(m),
                fsp3=Descriptors.FractionCSP3(m), mw=Descriptors.MolWt(m),
                rotb=Descriptors.NumRotatableBonds(m))


def shape_from_conf(smi):
    try:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3(); p.randomSeed = 42
        if AllChem.EmbedMolecule(m, p) != 0:
            p.useRandomCoords = True
            if AllChem.EmbedMolecule(m, p) != 0:
                return None
        AllChem.MMFFOptimizeMolecule(m, maxIters=100)
        return (Descriptors3D.NPR1(m), Descriptors3D.NPR2(m), Descriptors3D.RadiusOfGyration(m))
    except Exception:
        return None


def shape_class(npr1, npr2):
    drod = np.hypot(npr1, npr2 - 1); ddisc = np.hypot(npr1 - 0.5, npr2 - 0.5); dsph = np.hypot(npr1 - 1, npr2 - 1)
    return ["rod", "disc", "sphere"][int(np.argmin([drod, ddisc, dsph]))]


def assign_mode(f, npr1, npr2):
    """Rule-based mode from taxonomy thresholds (transparent, validatable)."""
    sh = shape_class(npr1, npr2)
    h, hba, arom = f["heavy"], f["hba"], f["arom"]
    if h <= 20:
        return "A_tripod"
    if h >= 40 and hba >= 4:
        return "D_blob"
    if sh == "sphere" and h >= 30:
        return "D_blob"
    if sh == "disc" and arom >= 3:
        return "B_blade"
    if sh == "rod" and hba <= 2:
        return "E_reach"
    if sh == "rod":
        return "C_skewer"
    return "B_blade" if arom >= 2 else "A_tripod"


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test()
    y = tr["pec50"].to_numpy(float)

    # train shape from nb954 cache (NPR1=col3, NPR2=col4)
    d3 = np.load(f"{D}/nb954_desc3d_ckpt.npy")
    assert len(d3) == len(tr), "nb954 cache length mismatch"
    tr_npr1, tr_npr2 = d3[:, 3], d3[:, 4]

    print("classifying train ...", flush=True)
    tr_mode = []
    for i, smi in enumerate(tr["smiles"]):
        f = feats2d(smi)
        n1, n2 = tr_npr1[i], tr_npr2[i]
        if f is None or not np.isfinite(n1):
            tr_mode.append(None); continue
        tr_mode.append(assign_mode(f, n1, n2))
    tr_mode = np.array(tr_mode)

    print("classifying test (fresh conformers) ...", flush=True)
    te_mode, te_feat = [], []
    for smi in te["smiles"]:
        f = feats2d(smi); sh = shape_from_conf(smi)
        if f is None or sh is None:
            te_mode.append(None); te_feat.append(None); continue
        te_mode.append(assign_mode(f, sh[0], sh[1])); te_feat.append((f, sh))
    te_mode = np.array(te_mode)

    # per-mode pEC50 from train
    print("\n=== per-mode pEC50 (train) — the calibrated ranges ===")
    print(f"{'mode':10s} {'n_train':>8s} {'median':>7s} {'q25':>6s} {'q75':>6s} {'mean':>6s}   {'n_test':>6s}")
    mode_stats = {}
    for mode in MODES:
        m = tr_mode == mode
        if m.sum() == 0:
            print(f"{mode:10s} {'0':>8s}"); continue
        yy = y[m]
        ntest = int((te_mode == mode).sum())
        mode_stats[mode] = dict(n=int(m.sum()), median=float(np.median(yy)),
                                q25=float(np.percentile(yy, 25)), q75=float(np.percentile(yy, 75)),
                                mean=float(yy.mean()), n_test=ntest)
        print(f"{mode:10s} {m.sum():8d} {np.median(yy):7.2f} {np.percentile(yy,25):6.2f} "
              f"{np.percentile(yy,75):6.2f} {yy.mean():6.2f}   {ntest:6d}")

    # VALIDATION: does mode predict pEC50? effect size + scaffold-CV mode-median predictor
    valid = tr_mode != None  # noqa
    from scipy.stats import kruskal
    groups = [y[tr_mode == mo] for mo in MODES if (tr_mode == mo).sum() > 1]
    H, pval = kruskal(*groups)
    grand = y[valid].mean()
    ss_tot = np.sum((y[valid] - grand) ** 2)
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    eta2 = ss_between / ss_tot
    print(f"\nKruskal-Wallis H={H:.1f} p={pval:.2e}; eta^2 (variance explained by mode) = {eta2:.3f}")

    scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in tr["smiles"]]
    folds = scaffold_kfold_indices(scaf, n_splits=5, seed=42)
    pred = np.full(len(y), np.nan)
    for tri, vai in folds:
        mode_med = {mo: np.median(y[tri][tr_mode[tri] == mo]) for mo in MODES if (tr_mode[tri] == mo).sum() > 0}
        fallback = np.median(y[tri])
        for i in vai:
            pred[i] = mode_med.get(tr_mode[i], fallback)
    fin = np.isfinite(pred)
    rae_mode = rae(y[fin], pred[fin])
    rae_mean = rae(y[fin], np.full(fin.sum(), grand))
    print(f"\nscaffold-CV RAE:  mode-median predictor = {rae_mode:.4f}   |   global-mean = {rae_mean:.4f}")
    print(f"  (LGBM-combined ref = 0.57; mode-median between mean(1.0) and LGBM shows pose carries signal)")
    verdict = "PREMISE HOLDS: binding mode predicts pEC50" if rae_mode < rae_mean - 0.02 else "premise WEAK: mode barely beats mean"
    print(f"  >>> {verdict}")

    # test predictions: mode -> median pEC50 (point) + range
    te_pred = np.array([mode_stats.get(mo, {}).get("median", grand) if mo else grand for mo in te_mode])
    import pandas as pd
    out = pd.DataFrame({"name": te["name"] if "name" in te else range(len(te)),
                        "smiles": te["smiles"], "binding_mode": te_mode,
                        "pec50_mode_median": np.round(te_pred, 3)})
    out.to_csv(f"{OUT}/nb974_test_mode_pec50.csv", index=False)
    json.dump({"mode_stats": mode_stats, "eta2": float(eta2), "kruskal_p": float(pval),
               "rae_mode_median_cv": float(rae_mode), "rae_mean": float(rae_mean),
               "verdict": verdict,
               "test_mode_counts": {mo: int((te_mode == mo).sum()) for mo in MODES}},
              open(f"{OUT}/nb974_phase4_summary.json", "w"), indent=2)
    np.save(f"{D}/nb974_train_mode.npy", tr_mode.astype(object), allow_pickle=True)
    print(f"\ntest mode distribution: {dict((mo, int((te_mode==mo).sum())) for mo in MODES)}")
    print(f"saved -> {OUT}/nb974_test_mode_pec50.csv + nb974_phase4_summary.json")


if __name__ == "__main__":
    main()
