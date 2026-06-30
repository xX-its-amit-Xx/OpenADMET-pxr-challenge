"""Merge the existing 1905 pooled arrays + the per-task cofold_pool slices -> final train_pool_*.npy (4139)."""
import numpy as np, glob, sys
prefix, N = sys.argv[1], int(sys.argv[2])
rich = np.nan_to_num(np.load("boltz_z_rich_train.npy"))      # 1905 filled
edges = np.load("train_edges.npy"); enorm = np.load("train_enorm.npy"); nedge = np.load("train_nedge.npy").copy()
for f in sorted(glob.glob(f"{prefix}_t*_nedge.npy")):
    base = f[:-len("_nedge.npy")]
    ne = np.load(f); fill = ne > 0
    rich[fill] = np.load(base + "_rich.npy")[fill]
    edges[fill] = np.load(base + "_edges.npy")[fill]
    enorm[fill] = np.load(base + "_enorm.npy")[fill]
    nedge[fill] = ne[fill]
for s, a in [("rich", rich), ("edges", edges), ("enorm", enorm), ("nedge", nedge)]:
    np.save(f"{prefix}_{s}.npy", a)
print(f"merged -> {prefix}_*.npy : {int((nedge>0).sum())}/{N} pooled")
