"""nb994 — VIZ Phase C: 3D interaction panels for the test compounds that have Boltz-2 predicted poses.
82 activity-test compounds overlap the 184 structure-test -> we have their predicted PXR-ligand complexes
(structure_baseline_v1.zip). For each: compute which polar anchors the PREDICTED pose engages, cross-ref the
activity + nb3200 prediction, and render rotatable py3Dmol panels. Tests: do active compounds' predicted poses
engage the anchors more than inactives? (Does the model's STRUCTURE explain the pEC50?)
Outputs -> C:/pxr_struct/pose3d/.
"""
import os, sys, json, zipfile, io
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pxr.data import load_test
import gemmi
from rdkit import Chem
from rdkit.Chem import inchi
from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")
import py3Dmol

D = "data/processed"; OUT = "C:/pxr_struct/pose3d"; os.makedirs(OUT, exist_ok=True)
ZIP = "submissions/structure_baseline_v1.zip"
# Boltz-2 construct numbering (native - 141); display the native name
ANCHORS = {106: "Ser247", 144: "Gln285", 186: "His327", 266: "His407", 269: "Arg410"}


def ik(s):
    m = Chem.MolFromSmiles(str(s)); return inchi.MolToInchiKey(m)[:14] if m else None


def is_aa(r):
    info = gemmi.find_tabulated_residue(r.name); return info is not None and info.is_amino_acid()


def anchors_engaged(pdb_text):
    st = gemmi.read_pdb_string(pdb_text); st.setup_entities(); model = st[0]
    lig = None; prot = None
    for ch in model:
        aa = [r for r in ch if is_aa(r)]
        if aa: prot = ch
        else:
            for r in ch:
                if r.name == "LIG" or sum(1 for a in r if a.element.name != "H") >= 8:
                    lig = r
    if lig is None or prot is None: return None, None
    lpos = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in lig if a.element.name != "H"])
    hit = []
    for r in prot:
        if int(r.seqid.num) in ANCHORS:
            rp = np.array([[a.pos.x, a.pos.y, a.pos.z] for a in r if a.element.name != "H"])
            if len(rp) and np.linalg.norm(rp[:, None] - lpos[None], axis=2).min() <= 4.5:
                hit.append(int(r.seqid.num))
    return hit, pdb_text


def render_panel(pdb_text, title, sub):
    v = py3Dmol.view(width=560, height=440)
    v.addModel(pdb_text, "pdb")
    v.setStyle({"cartoon": {"color": "lightgrey", "opacity": 0.5}})
    v.setStyle({"resn": "LIG"}, {"stick": {"colorscheme": "cyanCarbon", "radius": 0.22}})
    for num, nm in ANCHORS.items():
        v.addStyle({"resi": str(num)}, {"stick": {"colorscheme": "redCarbon", "radius": 0.15}})
        v.addResLabels({"resi": str(num)}, {"fontSize": 10, "backgroundColor": "white", "fontColor": "red"})
    v.zoomTo({"resn": "LIG"}); v.zoom(0.8)
    return f"<div style='display:inline-block;margin:6px;border:1px solid #ccc;padding:4px'><b>{title}</b><br><span style='font-size:11px'>{sub}</span><br>{v._make_html()}</div>"


def main():
    te = load_test(); unb = np.load(f"{D}/_audit_unblind_idx.npy"); y253 = np.load(f"{D}/_audit_unblind_y.npy")
    nb3200 = np.load(f"{D}/nb3200_pred_oof.npy")
    pos253 = {int(p): k for k, p in enumerate(unb)}
    te_ik = {ik(s): i for i, s in enumerate(te["smiles"])}
    st = pd.read_csv("data/raw/pxr-challenge_structure_TEST_BLINDED.csv")
    z = zipfile.ZipFile(ZIP); names = set(z.namelist())

    recs = []
    for _, row in st.iterrows():
        k = ik(row["smiles"])
        if k not in te_ik: continue
        ai = te_ik[k]                       # activity-test index
        sid = str(row["structure"])
        pf = f"{sid}.pdb"
        if pf not in names: continue
        pdb_text = z.read(pf).decode("utf-8", "replace")
        hit, _ = anchors_engaged(pdb_text)
        if hit is None: continue
        in253 = ai in pos253; kk = pos253.get(ai)
        recs.append({"sid": sid, "act_idx": ai, "in253": in253,
                     "actual": float(y253[kk]) if in253 else None,
                     "pred": float(nb3200[kk]) if in253 else None,
                     "n_anchors": len(hit), "anchors": [ANCHORS[h] for h in hit],
                     "pdb": pdb_text})
    print(f"{len(recs)} test compounds with Boltz-2 poses + anchor analysis")

    # active vs inactive anchor engagement (does the predicted pose explain activity?)
    lab = [r for r in recs if r["in253"]]
    act = [r for r in lab if r["actual"] >= 5.0]; ina = [r for r in lab if r["actual"] < 4.0]
    print(f"labeled {len(lab)}: active(>=5) {len(act)} mean_anchors {np.mean([r['n_anchors'] for r in act]):.2f}; "
          f"inactive(<4) {len(ina)} mean_anchors {np.mean([r['n_anchors'] for r in ina]):.2f}")

    # curated panels: 3 active well-pred, 3 inactive, 3 worst-pred
    lab_sorted = sorted(lab, key=lambda r: r["actual"])
    pick = []
    pick += sorted(act, key=lambda r: abs(r["pred"] - r["actual"]))[:3]
    pick += sorted(ina, key=lambda r: abs(r["pred"] - r["actual"]))[:3]
    pick += sorted(lab, key=lambda r: -abs(r["pred"] - r["actual"]))[:3]
    seen = set(); panels = []
    for r in pick:
        if r["sid"] in seen: continue
        seen.add(r["sid"])
        tag = "ACTIVE" if r["actual"] >= 5 else ("INACTIVE" if r["actual"] < 4 else "mid")
        title = f"{r['sid']} — {tag}"
        sub = (f"actual pEC50 {r['actual']:.2f} / nb3200 pred {r['pred']:.2f} (err {r['pred']-r['actual']:+.2f})<br>"
               f"predicted pose engages {r['n_anchors']}/5 anchors: {', '.join(r['anchors']) or 'none'}")
        panels.append(render_panel(r["pdb"], title, sub))

    html = ("<html><head><meta charset='utf-8'><title>PXR predicted poses</title></head><body style='font-family:sans-serif'>"
            "<h2>PXR test compounds — Boltz-2 PREDICTED poses + anchor engagement</h2>"
            "<p>cyan = ligand (predicted pose) · <span style='color:red'>red = polar anchors Ser247/Gln285/His327/His407/Arg410</span>. "
            "Drag to rotate. Does the predicted pose engage the anchors more for ACTIVE compounds?</p>"
            f"<p><b>Active (pEC50&ge;5): mean {np.mean([r['n_anchors'] for r in act]):.2f} anchors engaged · "
            f"Inactive (&lt;4): mean {np.mean([r['n_anchors'] for r in ina]):.2f}</b></p>"
            + "".join(panels) + "</body></html>")
    open(f"{OUT}/pose_panels.html", "w", encoding="utf-8").write(html)
    # save the full anchor-engagement table
    pd.DataFrame([{k: v for k, v in r.items() if k != "pdb"} for r in recs]).to_csv(f"{OUT}/nb994_anchor_engagement.csv", index=False)
    json.dump({"n_poses": len(recs), "active_mean_anchors": float(np.mean([r['n_anchors'] for r in act])),
               "inactive_mean_anchors": float(np.mean([r['n_anchors'] for r in ina]))},
              open(f"{OUT}/nb994_summary.json", "w"), indent=2)
    print(f"saved -> {OUT}/pose_panels.html ({len(panels)} panels) + nb994_anchor_engagement.csv")


if __name__ == "__main__":
    main()
