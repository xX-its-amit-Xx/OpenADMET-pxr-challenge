"""modal_deeppurpose.py — DeepPurpose Mode B cross-target NR-panel fingerprint (the user's PCM ask).
Score each compound vs each of the 12 NR LBD sequences with a pretrained BindingDB DTI model -> a 12-dim
"NR-superfamily engagement" profile. The target VARIES = orthogonal to our single-target 2D chemprop.
Morgan_CNN (no dgl/MPNN). Run: modal run scripts/modal_deeppurpose.py
"""
import modal

app = modal.App("pxr-deeppurpose")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libxrender1", "libxext6", "libsm6", "libgomp1", "git", "wget")
    .pip_install("numpy==1.23.5", "scipy==1.10.1", "pandas==1.5.3", "scikit-learn==1.1.3", "tqdm", "prettytable",
                 "torch==2.0.1", "rdkit-pypi", "pandas-flavor",
                 "git+https://github.com/bp-kelley/descriptastorus", "DeepPurpose")
    .add_local_file("data/external/nr_panel/nr_panel.json", "/root/nr_panel.json")
    .add_local_file("data/processed/unimol_train.csv", "/root/train.csv")
    .add_local_file("data/processed/unimol_test513.csv", "/root/test.csv")
)
MODEL = "Morgan_CNN_BindingDB"


@app.function(image=image, cpu=8.0, timeout=5400)
def score():
    import json, numpy as np, pandas as pd, warnings
    warnings.filterwarnings("ignore")
    from DeepPurpose import DTI as models
    from DeepPurpose import utils
    panel = json.load(open("/root/nr_panel.json"))
    names = sorted(panel)
    tr = pd.read_csv("/root/train.csv"); te = pd.read_csv("/root/test.csv")
    net = models.model_pretrained(model=MODEL)
    out = {}
    for tag, df in [("train", tr), ("test", te)]:
        smi = df["smiles"].astype(str).tolist()
        cols = []
        for nm in names:
            seq = panel[nm]
            X = utils.data_process(X_drug=smi, X_target=[seq] * len(smi), y=[0.0] * len(smi),
                                   drug_encoding="Morgan", target_encoding="CNN", split_method="no_split")
            p = np.asarray(net.predict(X), dtype=np.float32)
            cols.append(p.reshape(-1, 1))
            print(f"  {tag}/{nm}: mean={p.mean():.3f} std={p.std():.3f}", flush=True)
        out[tag] = np.hstack(cols)
    return {"names": names, "train": out["train"].tolist(), "test": out["test"].tolist(),
            "n_tr": len(tr), "n_te": len(te)}


@app.local_entrypoint()
def main():
    import os, json, numpy as np
    OUT = "C:/pxr_struct/deeppurpose"; os.makedirs(OUT, exist_ok=True)
    print("running DeepPurpose Mode B on Modal (py3.8 + DeepPurpose)...")
    r = score.remote()
    np.save(f"{OUT}/dp_modeB_train.npy", np.array(r["train"], np.float32))
    np.save(f"{OUT}/dp_modeB_test.npy", np.array(r["test"], np.float32))
    json.dump(r["names"], open(f"{OUT}/dp_modeB_cols.json", "w"))
    print(f"saved: train {r['n_tr']}x12, test {r['n_te']}x12 ({r['names']}) -> {OUT}/")
    print("NEXT: nb1009 integration test (12 NR cols, no PCA needed) on combined+chempropembed")
