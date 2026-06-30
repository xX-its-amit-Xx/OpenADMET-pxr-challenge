"""nb995 — VIZ Phase B: chemical-similarity map (UMAP) of train vs test.
Embeds 4139 train + 513 test (Morgan ECFP4, Jaccard metric) into 2D so you can SEE the chemical
space: where train covers, where the test compounds sit, and which test clusters are novel (no train
neighbor). Interactive Plotly with hover. Annotates WHY the test is novel. Output -> C:/pxr_struct/dash/.
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_train, load_test
from src.pxr.chem import morgan_fp_batch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")

D = "data/processed"; OUT = "C:/pxr_struct/dash"; os.makedirs(OUT, exist_ok=True)


def main():
    tr = load_train().dropna(subset=["pec50"]).reset_index(drop=True)
    te = load_test()
    unb = np.load(f"{D}/_audit_unblind_idx.npy")
    pos253 = set(int(p) for p in unb)
    ytr = tr["pec50"].to_numpy(float)

    print("morgan fps ...", flush=True)
    fp_tr = morgan_fp_batch(tr["smiles"].tolist()).astype(np.float32)
    fp_te = morgan_fp_batch(te["smiles"].astype(str).tolist()).astype(np.float32)
    allfp = np.vstack([fp_tr, fp_te])

    # novelty: each test compound's max Tanimoto to train
    inter = fp_te @ fp_tr.T; u = fp_te.sum(1)[:, None] + fp_tr.sum(1)[None, :] - inter; u[u == 0] = 1
    te_maxsim = (inter / u).max(1)
    tr_scaf = set(MurckoScaffold.MurckoScaffoldSmiles(s) for s in tr["smiles"] if Chem.MolFromSmiles(s))
    te_scaf = [MurckoScaffold.MurckoScaffoldSmiles(s) if Chem.MolFromSmiles(s) else None for s in te["smiles"]]
    novel = np.array([s is None or s not in tr_scaf for s in te_scaf])

    print("UMAP (jaccard) on 4652 cpds ...", flush=True)
    import umap
    emb = umap.UMAP(n_neighbors=20, min_dist=0.25, metric="jaccard", random_state=42).fit_transform(allfp)
    e_tr, e_te = emb[:len(fp_tr)], emb[len(fp_tr):]

    # build dataframe
    rows = []
    for i in range(len(fp_tr)):
        rows.append({"x": float(e_tr[i, 0]), "y": float(e_tr[i, 1]), "set": "train",
                     "label": tr["name"].iloc[i] if "name" in tr else f"train_{i}",
                     "pEC50": round(float(ytr[i]), 2), "novelty": None})
    nb3200_te = None  # predictions for hover (use the interpretable preds from nb993 if present)
    if os.path.exists(f"{OUT}/nb993_dashboard_table.csv"):
        t = pd.read_csv(f"{OUT}/nb993_dashboard_table.csv"); pmap = dict(zip(t["name"].astype(str), t["model_pred"]))
    else:
        pmap = {}
    te_names = (te["name"].astype(str).to_numpy() if "name" in te else np.array([f"test_{i}" for i in range(len(te))]))
    for i in range(len(fp_te)):
        s = "released(253)" if i in pos253 else "blind(260)"
        rows.append({"x": float(e_te[i, 0]), "y": float(e_te[i, 1]), "set": s,
                     "label": te_names[i], "pEC50": pmap.get(te_names[i]),
                     "novelty": round(float(te_maxsim[i]), 2)})
    df = pd.DataFrame(rows)

    # stats for the annotation
    far = (te_maxsim < 0.4).mean(); vfar = (te_maxsim < 0.3).mean()
    print(f"test novelty: {100*novel.mean():.0f}% novel-scaffold; {100*far:.0f}% have NO train neighbor >=0.4 sim; "
          f"{100*vfar:.0f}% <0.3 sim")

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=df[df.set == "train"].x, y=df[df.set == "train"].y, mode="markers",
                  marker=dict(size=4, color="#cccccc", opacity=0.5), name=f"train (n={len(fp_tr)})",
                  text=[f"{r.label}<br>train pEC50 {r.pEC50}" for r in df[df.set == "train"].itertuples()], hoverinfo="text"))
    for s, col in [("released(253)", "#1f77b4"), ("blind(260)", "#d62728")]:
        d = df[df.set == s]
        fig.add_trace(go.Scattergl(x=d.x, y=d.y, mode="markers",
                      marker=dict(size=7, color=col, line=dict(width=0.5, color="black")), name=f"{s} (n={len(d)})",
                      text=[f"{r.label}<br>{s}<br>pred pEC50 {r.pEC50}<br>novelty(sim2train) {r.novelty}" for r in d.itertuples()],
                      hoverinfo="text"))
    fig.update_layout(width=1100, height=850, title=("PXR chemical space — train (grey) vs test (blue=released, red=blind)<br>"
                      f"<sub>{100*novel.mean():.0f}% of test = novel scaffold · {100*far:.0f}% have NO training neighbor at Tanimoto&ge;0.4 "
                      "— the test sits at/beyond the edge of our training chemistry</sub>"),
                      xaxis_title="UMAP-1", yaxis_title="UMAP-2", legend=dict(itemsizing="constant"))
    fig.write_html(f"{OUT}/similarity_map_umap.html")
    json.dump({"pct_novel_scaffold": round(100 * float(novel.mean()), 1),
               "pct_no_neighbor_0.4": round(100 * float(far), 1), "pct_no_neighbor_0.3": round(100 * float(vfar), 1),
               "median_test_maxsim": round(float(np.median(te_maxsim)), 3)}, open(f"{OUT}/nb995_summary.json", "w"), indent=2)
    print(f"saved -> {OUT}/similarity_map_umap.html")


if __name__ == "__main__":
    main()
