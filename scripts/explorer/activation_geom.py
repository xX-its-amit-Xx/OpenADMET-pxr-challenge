"""Creative Stage-B feature: PXR ACTIVATION geometry from the 513 Boltz poses.
NR agonism = the C-terminal AF-2 helix (helix-12) adopting the active position. Docking ENERGY can't see this;
the cofolded POSE can (if Boltz captures ligand-induced conformational change). This script:
  1. Superposes all 513 protein conformations onto a common frame (stable LBD core).
  2. Computes per-residue Cα RMSF -> WHERE does the protein move with ligand? (diagnostic: is H12 mobile?)
  3. Extracts a per-ligand ACTIVATION COORDINATE = displacement of the most-mobile C-terminal region.
If RMSF is ~flat (esp. C-terminus), Boltz folds PXR rigidly regardless of ligand -> structural-activation axis is
dead (cheap, decisive negative). If H12 moves and tracks pEC50 -> the signal docking missed.
"""
import numpy as np, glob, os, json, warnings
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser

N_PROT = 434
parser = PDBParser(QUIET=True)

# index poses by yaml stem
files = {}
for f in glob.glob("out/*/boltz_results_*/predictions/*/*_model_0.pdb"):
    pid = os.path.basename(os.path.dirname(f))
    try:
        files[int(pid)] = f
    except ValueError:
        pass

ca = {}
for idx, f in files.items():
    try:
        chainA = parser.get_structure("x", f)[0]["A"]
        coords = [res["CA"].coord for res in chainA if "CA" in res]
        ca[idx] = np.asarray(coords, dtype=np.float64)
    except Exception:
        pass
idxs = sorted(ca.keys())
nres = min(len(ca[i]) for i in idxs)
X = np.stack([ca[i][:nres] for i in idxs])    # (N, nres, 3)
print(f"poses {len(idxs)}, protein Cα residues {nres}")


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, P.mean(0), Q.mean(0)


# superpose every pose onto pose0 using a stable core (avoid flexible termini/loops)
core = slice(40, min(380, nres))
ref = X[0]
Xa = np.empty_like(X)
for k in range(len(idxs)):
    R, pc, qc = kabsch(X[k][core], ref[core])
    Xa[k] = (X[k] - pc) @ R.T + qc
mean = Xa.mean(0)
rmsf = np.sqrt(((Xa - mean) ** 2).sum(-1).mean(0))   # per-residue RMSF across ligands

cterm = rmsf[-25:]
print(f"RMSF: median={np.median(rmsf):.3f} max={rmsf.max():.3f}@res{int(rmsf.argmax())} "
      f"core_med={np.median(rmsf[core]):.3f} Cterm25_med={np.median(cterm):.3f} Cterm25_max={cterm.max():.3f}")
print("interpretation:", "C-TERMINUS MOBILE (H12 candidate) -> activation signal possible"
      if cterm.max() > 2 * np.median(rmsf[core]) else "C-terminus ~rigid -> Boltz folds PXR ligand-independently")

# per-ligand activation feature: coords of the top-mobile residues, reduced to a few PCs
mob = rmsf.argsort()[-12:]                     # 12 most ligand-mobile residues
F = (Xa[:, mob, :] - mean[mob]).reshape(len(idxs), -1)   # (N, 36) displacement vectors
# order to full 513 (missing -> nan)
out = np.full((513, F.shape[1]), np.nan, np.float32)
for k, idx in enumerate(idxs):
    out[idx] = F[k]
np.save("activation_geom.npy", out)
json.dump({"n_poses": len(idxs), "nres": int(nres), "rmsf_core_med": float(np.median(rmsf[core])),
           "rmsf_cterm25_max": float(cterm.max()), "rmsf_max": float(rmsf.max()),
           "mobile_residues": [int(x) for x in sorted(mob)],
           "cterm_mobile": bool(cterm.max() > 2 * np.median(rmsf[core]))},
          open("activation_geom.json", "w"), indent=2)
np.save("activation_rmsf.npy", rmsf.astype(np.float32))
print(f"saved activation_geom.npy {out.shape} + activation_rmsf.npy + json")
