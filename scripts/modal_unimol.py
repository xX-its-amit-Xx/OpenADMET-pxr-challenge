"""modal_unimol.py — fine-tune Uni-Mol (3D molecular foundation model) on PXR pEC50, on Modal GPU.
The 'better base representation' lever (would supersede the chempropembed sink). 5 scaffold folds run
in PARALLEL on separate GPUs -> ~25 min wall. Returns scaffold-CV OOF (4139) + 513 deploy preds so we
can test it on the nb952 degradation curve and stack/SE-weight downstream.

Run:  modal run scripts/modal_unimol.py
Outputs land in C:/pxr_struct/unimol/ via the local entrypoint.
"""
import modal

app = modal.App("pxr-unimol")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "libxrender1", "libxext6", "libsm6", "libgomp1")
    .pip_install("numpy==1.26.4", "scipy==1.11.4", "pandas", "scikit-learn",
                 "rdkit", "huggingface_hub", "joblib", "addict", "pyyaml", "tqdm")
    .pip_install("torch==2.2.0", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("unimol_tools")
    .add_local_file("data/processed/unimol_train.csv", "/root/data/unimol_train.csv")
    .add_local_file("data/processed/unimol_eval253.csv", "/root/data/unimol_eval253.csv")
    .add_local_file("data/processed/unimol_test513.csv", "/root/data/unimol_test513.csv")
)

GPU = "A10G"
EPOCHS = 40


@app.function(image=image, gpu=GPU, timeout=5400)
def train_fold(fold_k: int):
    """Fine-tune Uni-Mol on scaffold fold-train (folds != k), predict fold-val + the 513 test."""
    import pandas as pd, numpy as np, os, shutil
    from unimol_tools import MolTrain, MolPredict
    tr = pd.read_csv("/root/data/unimol_train.csv")
    te = pd.read_csv("/root/data/unimol_test513.csv")
    va = tr["scaffold_fold"].to_numpy() == fold_k
    trn = pd.DataFrame({"SMILES": tr.loc[~va, "smiles"].values, "TARGET": tr.loc[~va, "pec50"].values})
    val = pd.DataFrame({"SMILES": tr.loc[va, "smiles"].values})
    tst = pd.DataFrame({"SMILES": te["smiles"].values})
    trn.to_csv("/tmp/trn.csv", index=False); val.to_csv("/tmp/val.csv", index=False); tst.to_csv("/tmp/tst.csv", index=False)
    exp = f"/tmp/exp{fold_k}"
    MolTrain(task="regression", data_type="molecule", epochs=EPOCHS, batch_size=32,
             learning_rate=1e-4, metrics="mae", save_path=exp).fit(data="/tmp/trn.csv")
    mp = MolPredict(load_model=exp)
    val_pred = np.asarray(mp.predict("/tmp/val.csv")).ravel()
    tst_pred = np.asarray(mp.predict("/tmp/tst.csv")).ravel()
    shutil.rmtree(exp, ignore_errors=True)
    return {"fold": fold_k, "val_idx": np.where(va)[0].tolist(),
            "val_pred": val_pred.tolist(), "tst_pred": tst_pred.tolist()}


@app.function(image=image, gpu=GPU, timeout=5400)
def train_deploy():
    """Fine-tune on ALL 4139 -> predict 513 (the deploy model)."""
    import pandas as pd, numpy as np
    from unimol_tools import MolTrain, MolPredict
    tr = pd.read_csv("/root/data/unimol_train.csv"); te = pd.read_csv("/root/data/unimol_test513.csv")
    pd.DataFrame({"SMILES": tr["smiles"], "TARGET": tr["pec50"]}).to_csv("/tmp/all.csv", index=False)
    pd.DataFrame({"SMILES": te["smiles"]}).to_csv("/tmp/tst.csv", index=False)
    MolTrain(task="regression", data_type="molecule", epochs=EPOCHS, batch_size=32,
             learning_rate=1e-4, metrics="mae", save_path="/tmp/dep").fit(data="/tmp/all.csv")
    dep = np.asarray(MolPredict(load_model="/tmp/dep").predict("/tmp/tst.csv")).ravel()
    return {"deploy513": dep.tolist()}


@app.local_entrypoint()
def main():
    import json, os, numpy as np
    OUT = "C:/pxr_struct/unimol"; os.makedirs(OUT, exist_ok=True)
    print("launching 5 scaffold folds + deploy in parallel on Modal", GPU, "...")
    fold_results = list(train_fold.map([0, 1, 2, 3, 4]))
    dep = train_deploy.remote()
    # assemble OOF (4139) + mean test pred across folds
    import pandas as pd
    tr = pd.read_csv("data/processed/unimol_train.csv")
    oof = np.full(len(tr), np.nan); tst_stack = []
    for r in fold_results:
        oof[np.array(r["val_idx"])] = r["val_pred"]; tst_stack.append(r["tst_pred"])
    np.save(f"{OUT}/unimol_oof_4139.npy", oof)
    np.save(f"{OUT}/unimol_test_cvmean.npy", np.nanmean(np.array(tst_stack), axis=0))
    np.save(f"{OUT}/unimol_deploy513.npy", np.array(dep["deploy513"]))
    json.dump({"n_oof_finite": int(np.isfinite(oof).sum()), "gpu": GPU, "epochs": EPOCHS},
              open(f"{OUT}/modal_unimol_summary.json", "w"), indent=2)
    print(f"saved -> {OUT}/ (unimol_oof_4139.npy, unimol_test_cvmean.npy, unimol_deploy513.npy)")
    print("NEXT (local): test unimol_oof on the nb952 degradation curve + does it beat/stack-on chemprop_aux?")
