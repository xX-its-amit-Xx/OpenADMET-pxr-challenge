"""sign_featurize.py — Chemical Checker Signaturizer (measured-bioactivity 'models-as-data' feature).
Runs in an ISOLATED venv (TensorFlow dep). Featurizes the 513 test compounds across the NR/CYP/xenobiotic-
relevant CC spaces -> per-space 128-d signature + applicability (alpha). Saves arrays for the main-env
residual test on combined+chempropembed (cycle-291 protocol). Spaces: B1 MoA, B2 drug-metabolizing enzymes
(CYP!), B4 binding, B5 HTS bioassays, C3 metabolic pathways, D2 cell sensitivity/tox, E3 side effects.
"""
import os, sys, ssl
try:
    import certifi
    os.environ["SSL_CERT_FILE"] = certifi.where(); os.environ["CURL_CA_BUNDLE"] = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
except Exception:
    pass
ssl._create_default_https_context = ssl._create_unverified_context  # belt-and-suspenders for TF-Hub
import numpy as np
import pandas as pd

OUT = "C:/pxr_struct/signaturizer"; os.makedirs(OUT, exist_ok=True)
SPACES = ["B1", "B2", "B4", "B5", "C3", "D2", "E3"]
CSV = sys.argv[1] if len(sys.argv) > 1 else "data/processed/unimol_test513.csv"


def main():
    from signaturizer import Signaturizer
    df = pd.read_csv(CSV)
    smi = df["smiles"].astype(str).tolist()
    print(f"loaded {len(smi)} SMILES from {CSV}")
    sigs, alphas = [], []
    for sp in SPACES:
        print(f"  space {sp} ...", flush=True)
        s = Signaturizer(sp)
        res = s.predict(smi)
        sig = np.asarray(res.signature, dtype=np.float32)          # (N,128)
        alpha = np.asarray(getattr(res, "applicability", np.full(len(smi), np.nan)), dtype=np.float32)
        sigs.append(sig); alphas.append(alpha.reshape(-1, 1))
        print(f"    {sp}: sig{sig.shape} alpha_mean={np.nanmean(alpha):.3f}", flush=True)
    S = np.hstack(sigs)              # (N, 7*128=896)
    A = np.hstack(alphas)           # (N, 7)
    np.save(f"{OUT}/sign_signatures_513.npy", S)
    np.save(f"{OUT}/sign_applicability_513.npy", A)
    print(f"saved signatures {S.shape} + applicability {A.shape} -> {OUT}/")


if __name__ == "__main__":
    main()
