"""nb993 — VIZ Phase A: interactive dashboard for the test compounds.
mols2grid explorer (structure + pred/actual/error/novelty/mode/WHY-SHAP/nearest-neighbor, hover+filter+sort)
+ Plotly predicted-vs-actual scatter. Built for the 253 (have truth) and the 260 (blind) separately.
Interpretable LGBM(combined)->pEC50 model gives SHAP 'why high/low'; nb3200 OOF overlaid for the 253.
All outputs -> C:/pxr_struct/dash/.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.eval import rae
from src.pxr.featurize import combined, impute
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import lightgbm as lgb

D = "data/processed"; OUT = "C:/pxr_struct/dash"; os.makedirs(OUT, exist_ok=True)
N_MORGAN = 2048
RDKIT_NAMES = [n for n, _ in Descriptors._descList]
# friendly names for the most common drivers
FRIENDLY = {"PEOE_VSA4": "neg-charge surface", "PEOE_VSA12": "pos-charge surface", "PEOE_VSA6": "charge surface",
            "EState_VSA2": "electroneg surface", "VSA_EState3": "polar EState", "NumAromaticRings": "aromatic rings",
            "NumSaturatedRings": "saturated rings", "ExactMolWt": "molecular weight", "MolWt": "molecular weight",
            "Chi0n": "size/connectivity", "NumSaturatedHeterocycles": "sat. heterocycles", "SlogP_VSA8": "lipophilic surface",
            "fr_NH0": "tertiary amine", "NumHAcceptors": "H-bond acceptors", "NumHDonors": "H-bond donors",
            "TPSA": "polar surface area", "NumRotatableBonds": "flexibility"}


def feat_name(i):
    if i < N_MORGAN: return f"Morgan_bit_{i}"
    j = i - N_MORGAN
    return RDKIT_NAMES[j] if j < len(RDKIT_NAMES) else f"RDKit_{j}"


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test()
    unb = np.load(f"{D}/_audit_unblind_idx.npy"); y253 = np.load(f"{D}/_audit_unblind_y.npy")
    te_smiles = te["smiles"].astype(str).to_numpy()
    te_names = (te["name"].astype(str).to_numpy() if "name" in te else np.array([f"test_{i}" for i in range(len(te))]))
    print("featurizing ...", flush=True)
    Xtr = impute(combined(tr["smiles"].tolist())).astype(np.float32)
    Xte = impute(combined(list(te_smiles))).astype(np.float32)
    ytr = tr["pec50"].to_numpy(float)

    # interpretable model + SHAP (pred_contrib)
    print("training interpretable LGBM + SHAP ...", flush=True)
    mdl = lgb.LGBMRegressor(n_estimators=600, num_leaves=64, learning_rate=0.04, n_jobs=4, verbose=-1).fit(Xtr, ytr)
    pred_te = mdl.predict(Xte)
    contrib = mdl.predict(Xte, pred_contrib=True)  # (n, n_feat+1); last col = base
    base_val = contrib[0, -1]
    contrib = contrib[:, :-1]

    def why(i):
        c = contrib[i]; top = np.argsort(-np.abs(c))[:3]
        parts = []
        for j in top:
            nm = feat_name(j); fr = FRIENDLY.get(nm, nm)
            parts.append(f"{'↑' if c[j] > 0 else '↓'}{fr} ({c[j]:+.2f})")
        return " ; ".join(parts)

    # novelty + nearest train neighbor
    print("nearest neighbors ...", flush=True)
    fp_te = morgan_fp_batch(list(te_smiles)); fp_tr = morgan_fp_batch(tr["smiles"].tolist())
    A = fp_te.astype(np.float32); B = fp_tr.astype(np.float32)
    inter = A @ B.T; u = A.sum(1)[:, None] + B.sum(1)[None, :] - inter; u[u == 0] = 1
    sim = inter / u
    nn_idx = sim.argmax(1); nn_sim = sim.max(1)
    tr_scaf = set(MurckoScaffold.MurckoScaffoldSmiles(s) for s in tr["smiles"] if Chem.MolFromSmiles(s))
    te_scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in te_smiles]
    novel = np.array([s is None or s not in tr_scaf for s in te_scaf])

    # binding mode (nb974) if available
    mode_map = {}
    if os.path.exists(f"{OUT}/../nb974_test_mode_pec50.csv") or os.path.exists("C:/pxr_struct/nb974_test_mode_pec50.csv"):
        p = "C:/pxr_struct/nb974_test_mode_pec50.csv"
        if os.path.exists(p):
            md = pd.read_csv(p); mode_map = dict(zip(md["smiles"].astype(str), md["binding_mode"].astype(str)))
    modes = np.array([mode_map.get(s, "?") for s in te_smiles])

    # nb3200 OOF for the 253
    nb3200 = np.load(f"{D}/nb3200_pred_oof.npy")
    pos_in_253 = {int(p): k for k, p in enumerate(unb)}

    # build the master table
    rows = []
    for i in range(len(te_smiles)):
        in253 = i in pos_in_253
        k = pos_in_253.get(i)
        rows.append({
            "SMILES": te_smiles[i], "name": te_names[i],
            "set": "released(253)" if in253 else "blind(260)",
            "actual": round(float(y253[k]), 2) if in253 else None,
            "model_pred": round(float(pred_te[i]), 2),
            "nb3200_pred": round(float(nb3200[k]), 2) if in253 else None,
            "error": round(float(nb3200[k] - y253[k]), 2) if in253 else None,
            "novelty(sim2train)": round(float(nn_sim[i]), 2),
            "novel_scaffold": bool(novel[i]),
            "binding_mode": modes[i].replace("A_", "").replace("B_", "").replace("C_", "").replace("D_", "").replace("E_", ""),
            "WHY (top features)": why(i),
            "nearest_train": tr["name"].iloc[nn_idx[i]] if "name" in tr else f"train_{nn_idx[i]}",
            "neighbor_pEC50": round(float(ytr[nn_idx[i]]), 2),
            "neighbor_sim": round(float(nn_sim[i]), 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/nb993_dashboard_table.csv", index=False)
    print(f"253 nb3200 RAE check: {rae(y253, nb3200):.4f}; table {df.shape}")

    # ---- mols2grid interactive explorers ----
    import mols2grid
    common_tt = ["name", "set", "actual", "nb3200_pred", "model_pred", "error",
                 "novelty(sim2train)", "binding_mode", "WHY (top features)", "nearest_train", "neighbor_pEC50"]
    for tag, sub in [("released253", df[df["set"] == "released(253)"]), ("blind260", df[df["set"] == "blind(260)"]),
                     ("all513", df)]:
        sub = sub.reset_index(drop=True)
        subset = ["img", "name", "actual", "nb3200_pred", "error"] if tag != "blind260" else ["img", "name", "model_pred", "binding_mode"]
        g = mols2grid.MolGrid(sub, smiles_col="SMILES", size=(230, 180))
        g.save(f"{OUT}/explorer_{tag}.html",
               subset=subset, tooltip=common_tt)
        print(f"  explorer_{tag}.html  (n={len(sub)})")

    # ---- Plotly pred-vs-actual scatter (253) ----
    import plotly.express as px
    d253 = df[df["set"] == "released(253)"].copy()
    d253["abs_error"] = d253["error"].abs()
    fig = px.scatter(d253, x="actual", y="nb3200_pred", color="novelty(sim2train)",
                     size="abs_error", hover_data=["name", "error", "binding_mode", "WHY (top features)", "nearest_train", "neighbor_pEC50"],
                     color_continuous_scale="Viridis", title="nb3200 predicted vs actual pEC50 (253) — hover for why")
    lim = [d253["actual"].min() - 0.3, d253["actual"].max() + 0.3]
    fig.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1], line=dict(dash="dash"))
    fig.update_layout(width=950, height=750)
    fig.write_html(f"{OUT}/scatter_253.html")
    print(f"  scatter_253.html")
    json.dump({"n_253": int((df['set'] == 'released(253)').sum()), "n_260": int((df['set'] == 'blind(260)').sum()),
               "nb3200_rae_253": round(float(rae(y253, nb3200)), 4)}, open(f"{OUT}/nb993_summary.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/ (explorer_*.html, scatter_253.html, table.csv)")


if __name__ == "__main__":
    main()
