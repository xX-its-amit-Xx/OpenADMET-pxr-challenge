"""Pool Boltz-2 cofold embeddings -> (513, D) activity feature matrix.
Each ligand's embeddings_<idx>.npz has s (1, n_tok, 384) + z (1, n_tok, n_tok, 128); tokens =
N_PROT PXR residues then ligand-atom tokens. The LIGAND-conditioned signal = ligand-token rows of s
(mean+std -> 768) + protein x ligand block of z (mean+std -> 256). Run on the cluster after the array.
"""
import numpy as np, glob, os, sys

N_PROT = 434  # PXR sequence length (tokens 0..433 = protein, 434.. = ligand atoms)
N = 513
emb_files = glob.glob("out/*/boltz_results_*/predictions/*/embeddings_*.npz")
feats = np.full((N, 768 + 256), np.nan, dtype=np.float32)
done = 0
for f in emb_files:
    name = os.path.basename(f).replace("embeddings_", "").replace(".npz", "")
    try:
        idx = int(name)
    except ValueError:
        continue
    d = np.load(f)
    s = d["s"][0]                       # (n_tok, 384)
    z = d["z"][0]                       # (n_tok, n_tok, 128)
    if s.shape[0] <= N_PROT:
        continue
    s_lig = s[N_PROT:]                  # (n_lig, 384)
    z_pl = z[N_PROT:, :N_PROT]          # (n_lig, n_prot, 128)
    s_pool = np.concatenate([s_lig.mean(0), s_lig.std(0)])            # 768
    z_pool = np.concatenate([z_pl.mean((0, 1)), z_pl.std((0, 1))])    # 256
    feats[idx] = np.concatenate([s_pool, z_pool]).astype(np.float32)
    done += 1
np.save("boltz_emb_513.npy", feats)
ok = np.isfinite(feats).all(axis=1).sum()
print(f"saved boltz_emb_513.npy {feats.shape}; pooled {done} npz; finite rows {ok}/{N}")
if ok < N:
    miss = [i for i in range(N) if not np.isfinite(feats[i]).all()]
    print(f"MISSING idx ({len(miss)}): {miss[:20]}{'...' if len(miss) > 20 else ''}")
