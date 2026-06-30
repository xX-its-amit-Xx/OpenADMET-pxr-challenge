"""TERNARY cofold pooling: PXR + SRC-1 coactivator peptide + ligand. Pools THREE z-interface blocks per ligand:
  lig_pxr  = z[449:, :434]      ligand x PXR ('binding' block, in the coactivator-present context)
  pxr_pep  = z[:434, 434:449]   PXR x SRC-1 peptide (the AF-2 COACTIVATOR interface — activation coupling)
  lig_pep  = z[449:, 434:449]   ligand x peptide (does the ligand reach toward AF-2)
Each pooled to 512 (mean/std/per-row-max-mean/global-max over the block). Activation feature (computed later) =
lig_pxr_coact - lig_pxr_binary (existing test_rich) PLUS the new pxr_pep / lig_pep blocks the binary cofold can't see.

Resumable. Args: <yaml_dir> <prefix> <N> [TASK] [NTASKS]
"""
import subprocess, numpy as np, glob, os, sys, shutil

N_PXR = 434; N_PEP = 15; N_PROT = N_PXR + N_PEP   # 449; ligand tokens start at 449
yaml_dir, prefix, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
TASK = int(sys.argv[4]) if len(sys.argv) > 4 else 0
NTASKS = int(sys.argv[5]) if len(sys.argv) > 5 else 1
CHUNK = 40
TMP = f"/tmp/{os.environ.get('SLURM_JOB_ID', 'coact')}_{TASK}"
prefix = f"{prefix}_t{TASK}" if NTASKS > 1 else prefix


def load_or_init(suffix, shape, dt=np.float32):
    p = f"{prefix}_{suffix}.npy"
    if os.path.exists(p):
        a = np.load(p)
        if a.shape == shape:
            return a
    return np.zeros(shape, dt)


lig_pxr = load_or_init("ligpxr", (N, 512)); pxr_pep = load_or_init("pxrpep", (N, 512))
lig_pep = load_or_init("ligpep", (N, 512)); done = load_or_init("done", (N,), np.int32)


def pool_block(zb):
    """zb: (a, b, 128) -> 512-d (mean, std, row-max-mean, global-max)."""
    flat = zb.reshape(-1, 128)
    return np.concatenate([zb.mean((0, 1)), zb.std((0, 1)), zb.max(1).mean(0), flat.max(0)]).astype(np.float32)


todo = [i for i in range(N) if not done[i] and os.path.exists(f"{yaml_dir}/{i:05d}.yaml")][TASK::NTASKS]
print(f"task {TASK}/{NTASKS}: {len(todo)} to cofold; {int(done.sum())} already done", flush=True)
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
        if z.shape[0] <= N_PROT:           # need ligand tokens beyond 449
            continue
        lig_pxr[i] = pool_block(z[N_PROT:, :N_PXR, :])
        pxr_pep[i] = pool_block(z[:N_PXR, N_PXR:N_PROT, :])
        lig_pep[i] = pool_block(z[N_PROT:, N_PXR:N_PROT, :])
        done[i] = 1; got += 1
    shutil.rmtree(TMP, ignore_errors=True)
    for s, a in [("ligpxr", lig_pxr), ("pxrpep", pxr_pep), ("ligpep", lig_pep), ("done", done)]:
        np.save(f"{prefix}_{s}.npy", a)
    print(f"  chunk {c0//CHUNK}: +{got}  total {int(done.sum())}/{N}", flush=True)
print(f"DONE: {int(done.sum())}/{N}", flush=True)
