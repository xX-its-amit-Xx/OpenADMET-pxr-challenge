"""General merge of cofold_pool per-task slices -> {prefix}_*.npy (no existing base). Args: <prefix> <N>."""
import numpy as np, glob, sys
prefix, N = sys.argv[1], int(sys.argv[2])
rich = np.zeros((N, 512), np.float32); edges = np.zeros((N, 128, 128), np.float32)
enorm = np.zeros((N, 128), np.float32); nedge = np.zeros(N, np.int32)
for f in sorted(glob.glob(f"{prefix}_t*_nedge.npy")):
    base = f[:-len("_nedge.npy")]; ne = np.load(f); fill = ne > 0
    nedge[fill] = ne[fill]; rich[fill] = np.load(base + "_rich.npy")[fill]
    edges[fill] = np.load(base + "_edges.npy")[fill]; enorm[fill] = np.load(base + "_enorm.npy")[fill]
for s, a in [("rich", rich), ("edges", edges), ("enorm", enorm), ("nedge", nedge)]:
    np.save(f"{prefix}_{s}.npy", a)
print(f"merged {prefix}: {int((nedge>0).sum())}/{N}")
