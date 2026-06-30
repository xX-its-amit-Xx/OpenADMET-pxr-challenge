"""modal_unimol_repr.py — extract Uni-Mol 3D molecular EMBEDDINGS (UniMolRepr) for 4139 train + 513 test,
on Modal GPU. The prediction washed out (nb1004), but the full learned representation may be decorrelated
from ChempropEmbed where the prediction isn't -> test if Uni-Mol EMBEDDINGS add to the chemprop substrate
(the real absorption test for the representation path). Run: modal run scripts/modal_unimol_repr.py
"""
import modal

app = modal.App("pxr-unimol-repr")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libxrender1", "libxext6", "libsm6", "libgomp1")
    .pip_install("numpy==1.26.4", "scipy==1.11.4", "pandas", "scikit-learn", "rdkit",
                 "huggingface_hub", "joblib", "addict", "pyyaml", "tqdm")
    .pip_install("torch==2.2.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("unimol_tools")
    .add_local_file("data/processed/unimol_train.csv", "/root/data/unimol_train.csv")
    .add_local_file("data/processed/unimol_test513.csv", "/root/data/unimol_test513.csv")
)


@app.function(image=image, gpu="A10G", timeout=5400)
def extract_repr():
    import pandas as pd, numpy as np
    from unimol_tools import UniMolRepr
    tr = pd.read_csv("/root/data/unimol_train.csv"); te = pd.read_csv("/root/data/unimol_test513.csv")
    clf = UniMolRepr(data_type="molecule", remove_hs=False, use_gpu=True)
    def emb(smis):
        r = clf.get_repr(list(smis), return_atomic_reprs=False)
        return np.asarray(r["cls_repr"] if isinstance(r, dict) else r)
    e_tr = emb(tr["smiles"].astype(str)); e_te = emb(te["smiles"].astype(str))
    return {"train_repr": e_tr.tolist(), "test_repr": e_te.tolist(),
            "dim": int(e_tr.shape[1]), "n_tr": int(e_tr.shape[0]), "n_te": int(e_te.shape[0])}


@app.local_entrypoint()
def main():
    import json, os, numpy as np
    OUT = "C:/pxr_struct/unimol"; os.makedirs(OUT, exist_ok=True)
    print("extracting Uni-Mol embeddings on Modal A10G ...")
    r = extract_repr.remote()
    np.save(f"{OUT}/unimol_emb_train.npy", np.array(r["train_repr"], dtype=np.float32))
    np.save(f"{OUT}/unimol_emb_test.npy", np.array(r["test_repr"], dtype=np.float32))
    print(f"saved embeddings: train {r['n_tr']}x{r['dim']}, test {r['n_te']}x{r['dim']} -> {OUT}/")
    print("NEXT (local): does unimol_emb ADD to combined+chempropembed on the 253 (nb964 protocol)?")
