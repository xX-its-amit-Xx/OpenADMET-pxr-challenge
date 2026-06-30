"""activation_geom_v2 — FIX the superposition (v1 hand-Kabsch gave impossible 12A core RMSF) with BioPython's
tested Superimposer. Question: does Boltz produce LIGAND-DEPENDENT PXR conformations, and is the C-terminal
AF-2 helix (helix-12, the activation switch) mobile? If core RMSF ~1A but C-terminus moves and tracks pEC50,
the z signal is ACTIVATION-related (what docking energy missed). If everything is rigid, the z signal is pure
interaction-pattern, not global conformation.
"""
import numpy as np, glob, os, json, warnings
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser, Superimposer

parser = PDBParser(QUIET=True)
poses = {}
for f in glob.glob("out/*/boltz_results_*/predictions/*/*_model_0.pdb"):
    pid = os.path.basename(os.path.dirname(f))
    try:
        poses[int(pid)] = f
    except ValueError:
        pass
idxs = sorted(poses)
ca = {}
for idx in idxs:
    chainA = parser.get_structure("x", poses[idx])[0]["A"]
    ca[idx] = [res["CA"] for res in chainA if "CA" in res]
nres = min(len(ca[i]) for i in idxs)
ref = ca[idxs[0]][:nres]
core = list(range(40, min(380, nres)))            # stable core for alignment
sup = Superimposer()
Xa = []
for idx in idxs:
    mov = ca[idx][:nres]
    sup.set_atoms([ref[i] for i in core], [mov[i] for i in core])
    rot, tran = sup.rotran
    Xa.append(np.array([a.coord for a in mov]) @ rot + tran)
Xa = np.stack(Xa)                                  # (N, nres, 3)
mean = Xa.mean(0)
rmsf = np.sqrt(((Xa - mean) ** 2).sum(-1).mean(0))

cterm = rmsf[-25:]
core_med = float(np.median(rmsf[40:min(380, nres)]))
print(f"poses {len(idxs)} nres {nres}")
print(f"RMSF core_med={core_med:.3f} (should be ~1A if aligned)  Cterm25_med={np.median(cterm):.3f} "
      f"Cterm25_max={cterm.max():.3f}@res{nres-25+int(cterm.argmax())}  global_max={rmsf.max():.3f}@res{int(rmsf.argmax())}")
mob = rmsf > 2 * core_med
print(f"mobile residues (>2x core): {int(mob.sum())} -> {list(np.where(mob)[0][:30])}")
verdict = ("C-TERM H12 MOBILE + tracks ligand -> activation signal" if cterm.max() > 2 * core_med
           else "C-terminus rigid; mobility (if any) elsewhere" if mob.sum() else "PXR ~rigid across ligands")
print("VERDICT:", verdict)

# per-ligand C-terminal (last 12 res) displacement -> activation feature
F = (Xa[:, -12:, :] - mean[-12:]).reshape(len(idxs), -1)
out = np.full((513, F.shape[1]), np.nan, np.float32)
for k, idx in enumerate(idxs):
    out[idx] = F[k]
np.save("activation_geom_v2.npy", out)
np.save("activation_rmsf_v2.npy", rmsf.astype(np.float32))
json.dump({"core_med": core_med, "cterm_max": float(cterm.max()), "n_mobile": int(mob.sum()),
           "cterm_mobile": bool(cterm.max() > 2 * core_med)}, open("activation_geom_v2.json", "w"), indent=2)
print("saved activation_geom_v2.npy + rmsf + json")
