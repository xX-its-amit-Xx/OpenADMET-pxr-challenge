"""EFFICIENT cofold: run Boltz in chunks to node-local /tmp, POOL the z in-job (rich-512 + top-128 edges),
then DELETE the 107MB tensors. Keeps the GPU busy (local IO) + writes ~KB/ligand not 440GB. Resumable:
loads existing pooled arrays, only cofolds rows with nedge==0. Reusable for PXR-4139 AND PDBbind pretraining.
Args: <yaml_dir> <pooled_prefix> <N>   (pooled_prefix -> _rich/_edges/_enorm/_nedge .npy)
"""
import subprocess, numpy as np, glob, os, sys, shutil
N_PROT = 434; M = 128
yaml_dir, prefix, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
TASK = int(sys.argv[4]) if len(sys.argv) > 4 else 0
NTASKS = int(sys.argv[5]) if len(sys.argv) > 5 else 1
DONE_NEDGE = sys.argv[6] if len(sys.argv) > 6 else None   # global already-done mask from prior cofold
CHUNK = 40
TMP = f"/tmp/{os.environ.get('SLURM_JOB_ID', 'cofold')}_{TASK}"
prefix = f"{prefix}_t{TASK}" if NTASKS > 1 else prefix     # task-specific output


def load_or_init(suffix, shape):
    p = f"{prefix}_{suffix}.npy"
    if os.path.exists(p):
        a = np.load(p)
        if a.shape == shape:
            return a
    return np.zeros(shape, np.float32) if suffix != "nedge" else np.zeros(shape, np.int32)


rich = load_or_init("rich", (N, 512)); edges = load_or_init("edges", (N, M, 128))
enorm = load_or_init("enorm", (N, M)); nedge = load_or_init("nedge", (N,))
done = np.load(DONE_NEDGE) > 0 if DONE_NEDGE and os.path.exists(DONE_NEDGE) else np.zeros(N, bool)


def pool(z):
    zpl = z[N_PROT:, :N_PROT, :]                      # (n_lig, n_prot, 128)
    r = np.concatenate([zpl.mean((0, 1)), zpl.std((0, 1)), zpl.max(1).mean(0), zpl.reshape(-1, 128).max(0)])
    flat = zpl.reshape(-1, 128); nrm = np.linalg.norm(flat, axis=1)
    k = min(M, len(nrm)); top = np.argpartition(nrm, -k)[-k:]; top = top[np.argsort(nrm[top])[::-1]]
    e = np.zeros((M, 128), np.float32); en = np.zeros(M, np.float32)
    e[:k] = flat[top]; en[:k] = nrm[top]
    return r.astype(np.float32), e, en, k


undone = [i for i in range(N) if not done[i] and nedge[i] == 0 and os.path.exists(f"{yaml_dir}/{i:05d}.yaml")]
todo = undone[TASK::NTASKS]
print(f"task {TASK}/{NTASKS}: {len(todo)} to cofold (of {len(undone)} globally undone); {int((nedge>0).sum())} this-task pooled", flush=True)
for c0 in range(0, len(todo), CHUNK):
    chunk = todo[c0:c0 + CHUNK]
    shutil.rmtree(TMP, ignore_errors=True); os.makedirs(f"{TMP}/in")
    for i in chunk:
        os.symlink(os.path.abspath(f"{yaml_dir}/{i:05d}.yaml"), f"{TMP}/in/{i:05d}.yaml")
    subprocess.run(["./env/bin/boltz", "predict", f"{TMP}/in", "--write_embeddings", "--no_kernels",
                    "--out_dir", f"{TMP}/out", "--cache", "./boltz_cache", "--output_format", "pdb",
                    "--num_workers", "4"], capture_output=True)
    got = 0
    for npz in glob.glob(f"{TMP}/out/**/embeddings_*.npz", recursive=True):
        name = os.path.basename(npz).replace("embeddings_", "").replace(".npz", "")
        try:
            i = int(name)
        except ValueError:
            continue
        try:
            z = np.load(npz)["z"][0]
        except Exception:
            continue
        if z.shape[0] <= N_PROT:
            continue
        rich[i], edges[i], enorm[i], nedge[i] = pool(z); got += 1
    shutil.rmtree(TMP, ignore_errors=True)
    np.save(f"{prefix}_rich.npy", rich); np.save(f"{prefix}_edges.npy", edges)
    np.save(f"{prefix}_enorm.npy", enorm); np.save(f"{prefix}_nedge.npy", nedge)
    print(f"  chunk {c0//CHUNK}: +{got}  total pooled {int((nedge>0).sum())}/{N}", flush=True)
print(f"DONE: pooled {int((nedge>0).sum())}/{N}", flush=True)
