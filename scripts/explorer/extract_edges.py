"""Extract the top-M strongest protein-ligand INTERACTION EDGES per ligand from the Boltz z tensor,
for the interaction-graph additive head. Each edge = a (ligand-atom, protein-residue) pair with its 128-d z
vector (the 'micro-interaction'). Data-driven pocket: top-M pairs by |z| -> no residue mapping needed.
Args: <out_glob> <N> <prefix>.  CPU-only.
"""
import numpy as np, glob, os, sys
N_PROT = 434
M = 128
out_glob, N, prefix = sys.argv[1], int(sys.argv[2]), sys.argv[3]
edges = np.zeros((N, M, 128), np.float32)
enorm = np.zeros((N, M), np.float32)
nedge = np.zeros(N, np.int32)
done = 0
for f in glob.glob(f"{out_glob}/*/boltz_results_*/predictions/*/embeddings_*.npz"):
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
    zlp = z[N_PROT:, :N_PROT, :].reshape(-1, 128)
    norm = np.linalg.norm(zlp, axis=1)
    k = min(M, len(norm))
    top = np.argpartition(norm, -k)[-k:]
    top = top[np.argsort(norm[top])[::-1]]
    edges[idx, :k] = zlp[top]
    enorm[idx, :k] = norm[top]
    nedge[idx] = k
    done += 1
np.save(f"{prefix}_edges.npy", edges)
np.save(f"{prefix}_enorm.npy", enorm)
np.save(f"{prefix}_nedge.npy", nedge)
print(f"saved {prefix}_edges.npy ({N},{M},128) + enorm + nedge; pooled {done}")
