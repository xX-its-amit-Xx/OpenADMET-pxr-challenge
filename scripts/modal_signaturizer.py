"""modal_signaturizer.py — Chemical Checker Signaturizer on Modal (pinned py3.10 + TF2.15, classic Keras).
Local py3.12 forces TF>=2.16 (Keras-3, breaks signaturizer's KerasLayer). Featurizes 4139 train + 513 test
across NR/CYP/xenobiotic CC spaces -> 128-d signature + applicability (alpha) per space.
Run: modal run scripts/modal_signaturizer.py
"""
import modal

app = modal.App("pxr-signaturizer")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libxrender1", "libxext6", "libsm6", "libgomp1")
    .pip_install("numpy==1.26.4", "pandas", "rdkit", "protobuf<5",
                 "tensorflow==2.15.0", "tensorflow-hub==0.15.0", "signaturizer")
    .add_local_file("data/processed/unimol_train.csv", "/root/data/train.csv")
    .add_local_file("data/processed/unimol_test513.csv", "/root/data/test.csv")
)
SPACES = ["B1", "B2", "B4", "B5", "C3", "D2", "E3"]


@app.function(image=image, cpu=4.0, timeout=3600)
def featurize():
    import ssl, certifi, os
    os.environ["SSL_CERT_FILE"] = certifi.where()
    ssl._create_default_https_context = ssl._create_unverified_context
    import pandas as pd, numpy as np
    from signaturizer import Signaturizer
    tr = pd.read_csv("/root/data/train.csv"); te = pd.read_csv("/root/data/test.csv")
    out = {}
    for tag, df in [("train", tr), ("test", te)]:
        smi = df["smiles"].astype(str).tolist()
        sigs, alphas = [], []
        for sp in SPACES:
            s = Signaturizer(sp); res = s.predict(smi)
            sigs.append(np.asarray(res.signature, dtype=np.float32))
            a = getattr(res, "applicability", None)
            alphas.append(np.asarray(a, dtype=np.float32).reshape(-1, 1) if a is not None else np.full((len(smi), 1), np.nan, np.float32))
            print(f"  {tag}/{sp}: done", flush=True)
        out[tag] = (np.hstack(sigs), np.hstack(alphas))
    return {"train_sig": out["train"][0].tolist(), "train_alpha": out["train"][1].tolist(),
            "test_sig": out["test"][0].tolist(), "test_alpha": out["test"][1].tolist(),
            "n_tr": len(tr), "n_te": len(te), "dim": out["test"][0].shape[1]}


@app.local_entrypoint()
def main():
    import os, numpy as np
    OUT = "C:/pxr_struct/signaturizer"; os.makedirs(OUT, exist_ok=True)
    print("running CC Signaturizer on Modal (py3.10 + TF2.15)...")
    r = featurize.remote()
    np.save(f"{OUT}/sign_signatures_4139.npy", np.array(r["train_sig"], np.float32))
    np.save(f"{OUT}/sign_applicability_4139.npy", np.array(r["train_alpha"], np.float32))
    np.save(f"{OUT}/sign_signatures_513.npy", np.array(r["test_sig"], np.float32))
    np.save(f"{OUT}/sign_applicability_513.npy", np.array(r["test_alpha"], np.float32))
    print(f"saved: train {r['n_tr']}x{r['dim']}, test {r['n_te']}x{r['dim']} -> {OUT}/")
    print("NEXT: python scripts/nb1007_sign_integration.py")
