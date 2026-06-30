"""nb960 — predict 41 TDC ADMET properties (ADMET-AI / Chemprop-RDKit) for train+test.
RUN IN THE ISOLATED admet venv: C:/admet_venv/Scripts/python.exe scripts/nb960_admet_predict.py
Reads the CSV bundle (no pyarrow needed), writes C:/admet_out/admet_{train,test}.csv (D: is full).
PXR regulates CYP3A4 -> the CYP3A4/metabolism predictions are a mechanistically-orthogonal
biological axis to test against the nb952 degradation curve.
"""
import os
os.environ.setdefault("HF_HOME", "C:/hf_cache")  # D: is full
import pandas as pd
from admet_ai import ADMETModel

D = "data/processed"
OUT = "C:/admet_out"
os.makedirs(OUT, exist_ok=True)
model = ADMETModel()
for split, path in [("train", f"{D}/unimol_train.csv"), ("test", f"{D}/unimol_test513.csv")]:
    df = pd.read_csv(path)
    smis = df["smiles"].astype(str).tolist()
    uniq = list(dict.fromkeys(smis))                     # unique, order-preserving
    preds = model.predict(smiles=uniq)                   # DataFrame indexed by SMILES
    out = preds.loc[smis].reset_index(drop=True)         # map back to full input order (dupes ok)
    out.insert(0, "smiles", smis)
    out.to_csv(f"{OUT}/admet_{split}.csv", index=False)
    print(f"{split}: {out.shape}  cols0..5={list(out.columns)[:6]}", flush=True)
print("ADMET prediction done.")
