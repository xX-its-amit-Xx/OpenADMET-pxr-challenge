"""extract_emb_v2 — RICHER pooling of the Boltz z interaction tensor (the verified signal carrier).
v1 pooled only mean+std of the protein x ligand z block. The strongest contacts (max-pool) often dominate
activity, and mean washes them out. v2 keeps the z-only features but adds max/strongest-contact statistics:
  z_pl = z[ligand_tokens, protein_tokens, :128]  ->
   mean(lig,prot) 128 | std(lig,prot) 128 | max-over-prot then mean-over-lig 128 (best protein contact/atom)
   | global max(lig,prot) 128  =>  512-d  (saved separately so nb can A/B vs v1's 256-d).
"""
import numpy as np, glob, os

N_PROT = 434
N = 513
emb_files = glob.glob("out/*/boltz_results_*/predictions/*/embeddings_*.npz")
feats = np.full((N, 512), np.nan, dtype=np.float32)
done = 0
for f in emb_files:
    name = os.path.basename(f).replace("embeddings_", "").replace(".npz", "")
    try:
        idx = int(name)
    except ValueError:
        continue
    z = np.load(f)["z"][0]                 # (n_tok, n_tok, 128)
    if z.shape[0] <= N_PROT:
        continue
    zpl = z[N_PROT:, :N_PROT, :]           # (n_lig, n_prot, 128)
    pooled = np.concatenate([
        zpl.mean((0, 1)),                  # 128 avg interaction
        zpl.std((0, 1)),                   # 128 spread
        zpl.max(1).mean(0),                # 128 strongest protein contact per ligand atom, averaged
        zpl.reshape(-1, 128).max(0),       # 128 single strongest interaction
    ]).astype(np.float32)
    feats[idx] = pooled
    done += 1
np.save("boltz_z_rich_513.npy", feats)
ok = np.isfinite(feats).all(axis=1).sum()
print(f"saved boltz_z_rich_513.npy {feats.shape}; pooled {done}; finite {ok}/{N}")
