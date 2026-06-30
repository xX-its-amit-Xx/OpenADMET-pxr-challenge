"""Pool rich-z (512-d, v2) for the cofolded TRAINING ligands (out_train/). Index = unimol_train.csv row order.
CPU-only (no GPU). For the independent validation: feature was selected on the 253, never on these compounds.
"""
import numpy as np, glob, os
N_PROT = 434
N = 4139
files = glob.glob("out_train/*/boltz_results_*/predictions/*/embeddings_*.npz")
feats = np.full((N, 512), np.nan, dtype=np.float32)
done = 0
for f in files:
    name = os.path.basename(f).replace("embeddings_", "").replace(".npz", "")
    try:
        idx = int(name)
    except ValueError:
        continue
    if idx >= N:
        continue
    try:
        z = np.load(f)["z"][0]
    except Exception:
        continue
    if z.shape[0] <= N_PROT:
        continue
    zpl = z[N_PROT:, :N_PROT, :]
    feats[idx] = np.concatenate([zpl.mean((0, 1)), zpl.std((0, 1)),
                                 zpl.max(1).mean(0), zpl.reshape(-1, 128).max(0)]).astype(np.float32)
    done += 1
np.save("boltz_z_rich_train.npy", feats)
print(f"saved boltz_z_rich_train.npy ({N},512); pooled {done} independent train ligands")
