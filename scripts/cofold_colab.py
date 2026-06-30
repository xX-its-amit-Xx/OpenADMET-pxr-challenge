"""Boltz-2 multi-sample cofold + in-job feature extraction — Colab A100 runner (driven by the Colab CLI).

Mirrors scripts/modal_boltz_cofold.py but as a standalone script for `colab exec -f`. Reads from the runtime:
  /content/pxr_msa.a3m   (uploaded)            — precomputed PXR MSA (reused for every ligand)
  /content/ligands.csv   (uploaded: idx,smiles) — the ligands to cofold
Writes per-ligand /content/feats/<idx>.npz {richz(512), geom(18), rmsf(293)} and a merged /content/feats_merged.npz.
Resumable: skips any idx whose npz already exists. Run after `colab install boltz gemmi biopython 'numpy<2'`.
"""
import subprocess, os, glob, json, csv, time, traceback
import numpy as np

PXR = ("GLTEEQRMMIRELMDAQMKTFDTTFSHFKNFRLPGVLSSGCELPESLQAPSREEAAKWSQVRKDLCSLKVSLQLRGED"
       "GSVWNYKPPADSGGKEIFSLLPHMADMSTYMFKGIISFAKVISYFRDLPIEDQISLLKGAAFELCQLRFNTVFNAETG"
       "TWECGRLSYCLEDTAGGFQQLLLEPMLKFHYMLKKLQLHEEEYVLMQAISLFSPDRPGVLQHRVVDQLQEQFAITLKS"
       "YIECNRPQPAHRFLFLKIMAMLTELRSINAQHTQRLLRIQDIHPFATPLMQELFGITGS")
MSA = "/content/pxr_msa.a3m"
OUT = "/content/feats"; os.makedirs(OUT, exist_ok=True)


def yaml_for(smi):
    return ("version: 1\nsequences:\n"
            f"  - protein:\n      id: A\n      sequence: {PXR}\n      msa: {MSA}\n"
            f"  - ligand:\n      id: B\n      smiles: '{smi}'\n")


def extract(wd, name, nres_emb):
    pdbs = sorted(glob.glob(f"{wd}/**/{name}_model_*.pdb", recursive=True))
    CA, LIG = [], []
    for p in pdbs:
        ca, lig = {}, []
        for l in open(p):
            if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[21] == "A":
                ca[int(l[22:26])] = [float(l[30:38]), float(l[38:46]), float(l[46:54])]
            elif l.startswith(("ATOM", "HETATM")) and l[21] == "B":
                lig.append([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        res = sorted(ca)
        CA.append(np.array([ca[r] for r in res], np.float64)); LIG.append(np.array(lig, np.float64))
    nres = min(len(c) for c in CA); CA = np.stack([c[:nres] for c in CA]); S = len(CA)
    core = slice(40, min(274, nres)); h12 = slice(max(0, nres - 20), nres)
    ref = CA[0]; aligned = [CA[0]]
    for s in range(1, S):
        P, Q = CA[s][core], ref[core]
        Pc, Qc = P - P.mean(0), Q - Q.mean(0)
        U, _, Vt = np.linalg.svd(Pc.T @ Qc)
        R = (Vt.T) @ np.diag([1., 1., np.sign(np.linalg.det((Vt.T) @ (U.T)))]) @ (U.T)
        aligned.append((CA[s] - P.mean(0)) @ R.T + Q.mean(0))
    A = np.stack(aligned); mean = A.mean(0)
    rmsf = np.sqrt(((A - mean) ** 2).sum(-1).mean(0))
    h12_rmsf = float(rmsf[h12].mean()); core_rmsf = float(rmsf[40:min(274, nres)].mean())
    feats = []
    for s in range(S):
        ca, lg = CA[s], LIG[s]
        h12c, corec = ca[h12].mean(0), ca[core].mean(0)
        ligc = lg.mean(0) if len(lg) else corec
        d_hl = float(np.linalg.norm(h12c - ligc)); d_hc = float(np.linalg.norm(h12c - corec))
        rg = float(np.sqrt(((ca[h12] - h12c) ** 2).sum(-1).mean()))
        nc = float((np.sqrt(((lg[:, None, :] - ca[None, :, :]) ** 2).sum(-1)).min(1) < 4.5).sum()) if len(lg) else 0.0
        feats.append([d_hl, d_hc, rg, nc])
    feats = np.array(feats); fm, fs = feats.mean(0), feats.std(0)
    cs = []
    for cj in sorted(glob.glob(f"{wd}/**/confidence_{name}_model_*.json", recursive=True)):
        j = json.load(open(cj))
        cs.append([j.get("confidence_score", 0), j.get("ligand_iptm", 0), j.get("complex_plddt", 0), j.get("complex_pde", 0)])
    cs = np.array(cs) if cs else np.zeros((1, 4)); cm, csd = cs.mean(0), cs.std(0)
    z = np.load(glob.glob(f"{wd}/**/embeddings_{name}.npz", recursive=True)[0])["z"][0]
    zpl = z[nres_emb:, :nres_emb, :].astype(np.float64); flat = zpl.reshape(-1, 128)
    richz = np.concatenate([zpl.mean((0, 1)), zpl.std((0, 1)), zpl.max(1).mean(0), flat.max(0)]).astype(np.float32)
    geom = np.concatenate([fm, fs, [h12_rmsf, core_rmsf], cm, csd]).astype(np.float32)
    return richz, geom, rmsf.astype(np.float32)


def main():
    rows = list(csv.DictReader(open("/content/ligands.csv")))
    print(f"cofold {len(rows)} ligands", flush=True)
    t0 = time.time(); done = 0
    for r in rows:
        idx, smi = r["idx"], r["smiles"]
        outp = f"{OUT}/{idx}.npz"
        if os.path.exists(outp):
            done += 1; continue
        wd = f"/tmp/lig{idx}"; os.system(f"rm -rf {wd}"); os.makedirs(wd, exist_ok=True)
        open(f"{wd}/in.yaml", "w").write(yaml_for(smi))
        cmd = ["boltz", "predict", f"{wd}/in.yaml", "--out_dir", wd, "--diffusion_samples", "5",
               "--recycling_steps", "3", "--sampling_steps", "100", "--output_format", "pdb",
               "--write_embeddings", "--no_kernels", "--override"]
        try:
            rr = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if rr.returncode != 0:
                print(f"idx{idx} rc{rr.returncode}: {rr.stderr[-300:]}", flush=True); os.system(f"rm -rf {wd}"); continue
            richz, geom, rmsf = extract(wd, "in", len(PXR))
            np.savez(outp, richz=richz, geom=geom, rmsf=rmsf); done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        except Exception:
            print(f"idx{idx} FAIL: {traceback.format_exc()[-300:]}", flush=True)
        finally:
            os.system(f"rm -rf {wd}")
    # merge
    fs = sorted(glob.glob(f"{OUT}/*.npz"), key=lambda p: int(os.path.basename(p)[:-4]))
    idxs = [int(os.path.basename(p)[:-4]) for p in fs]
    rz = np.stack([np.load(p)["richz"] for p in fs]); gm = np.stack([np.load(p)["geom"] for p in fs])
    rm = np.stack([np.load(p)["rmsf"] for p in fs])
    np.savez("/content/feats_merged.npz", idx=np.array(idxs), richz=rz, geom=gm, rmsf=rm)
    print(f"DONE {done}/{len(rows)} -> /content/feats_merged.npz ({len(idxs)} total)", flush=True)


if __name__ == "__main__":
    main()
