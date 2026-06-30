"""Boltz-2.1 hosted-API cofold pipeline for the B2 geometric model (activity track).

Async: submit all jobs (idempotency-keyed) -> manifest; poll retrieve until succeeded; download the 5-sample CIFs;
extract per-complex coords (ligand atoms + pocket CA + helix-12) + confidence -> compact npz per compound.
Sidesteps Colab/Modal entirely (hosted, parallel, no session reclaim).

  python scripts/boltz_api_cofold.py submit <csv> <tag>     # csv: idx,smiles  -> manifest
  python scripts/boltz_api_cofold.py poll   <tag>            # poll + download + extract (resumable)
"""
import os, sys, json, csv, subprocess, time, urllib.request, glob
import numpy as np

BA = r"D:\Users\ashenoy00000\AppData\Local\Programs\Boltz\bin\boltz-api.exe"
ROOT = "C:/pxr_struct/boltz_api"
PXR = ("GLTEEQRMMIRELMDAQMKTFDTTFSHFKNFRLPGVLSSGCELPESLQAPSREEAAKWSQVRKDLCSLKVSLQLRGED"
       "GSVWNYKPPADSGGKEIFSLLPHMADMSTYMFKGIISFAKVISYFRDLPIEDQISLLKGAAFELCQLRFNTVFNAETG"
       "TWECGRLSYCLEDTAGGFQQLLLEPMLKFHYMLKKLQLHEEEYVLMQAISLFSPDRPGVLQHRVVDQLQEQFAITLKS"
       "YIECNRPQPAHRFLFLKIMAMLTELRSINAQHTQRLLRIQDIHPFATPLMQELFGITGS")
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def ba(args, timeout=120):
    return subprocess.run([BA] + args, capture_output=True, text=True, env=ENV, timeout=timeout)


def heavy(smi):
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi); return m.GetNumHeavyAtoms() if m else 99
    except Exception:
        return 99


def payload(smi):
    y = ["entities:", "  - type: protein", "    chain_ids: [A]", f"    value: {PXR}",
         "  - type: ligand_smiles", "    chain_ids: [B]", f"    value: '{smi}'"]
    if heavy(smi) < 48:
        y += ["binding:", "  type: ligand_protein_binding", "  binder_chain_id: B"]
    y += ["num_samples: 5"]
    return "\n".join(y) + "\n"


def submit(csvpath, tag):
    pdir = f"{ROOT}/{tag}/payloads"; os.makedirs(pdir, exist_ok=True)
    rows = list(csv.DictReader(open(csvpath)))
    mpath = f"{ROOT}/{tag}/manifest.json"
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    t0 = time.time()
    for i, r in enumerate(rows):
        idx = str(r["idx"])
        if idx in man and man[idx].get("job"):
            continue
        slug = f"pxract-{tag}-{idx}"
        pf = f"{pdir}/{idx}.yaml"; open(pf, "w").write(payload(r["smiles"]))
        rr = ba(["predictions:structure-and-binding", "start", "--model", "boltz-2.1",
                 "--idempotency-key", slug, "--input", f"@yaml://{pf}", "--raw-output", "--transform", "id"])
        job = (rr.stdout or "").strip().splitlines()[-1].strip() if rr.stdout.strip() else ""
        man[idx] = {"job": job, "smiles": r["smiles"]}
        if (i + 1) % 25 == 0:
            json.dump(man, open(mpath, "w")); print(f"  submitted {i+1}/{len(rows)} ({time.time()-t0:.0f}s)", flush=True)
        time.sleep(0.2)
    json.dump(man, open(mpath, "w"))
    print(f"submit DONE: {sum(1 for v in man.values() if v.get('job'))}/{len(rows)} jobs -> {mpath}")


def parse_cif(text):
    """minimal CIF atom_site parse -> (chain, resseq, atom, x,y,z) for CA (protein) + ligand atoms."""
    lines = text.splitlines()
    cols = []; rows = []; in_loop = False; fields = []
    for l in lines:
        if l.startswith("_atom_site."):
            fields.append(l.strip().split(".")[1]); in_loop = True; continue
        if in_loop:
            if l.startswith("#") or l.startswith("loop_") or l.startswith("_") or not l.strip():
                if rows:
                    break
                continue
            rows.append(l.split())
    if not fields or not rows:
        return None
    fi = {f: k for k, f in enumerate(fields)}
    def g(r, name, d=""):
        return r[fi[name]] if name in fi and fi[name] < len(r) else d
    prot, lig = {}, []
    for r in rows:
        if len(r) < len(fields):
            continue
        ch = g(r, "label_asym_id") or g(r, "auth_asym_id")
        atom = g(r, "label_atom_id"); comp = g(r, "label_comp_id")
        try:
            x, y, z = float(g(r, "Cartn_x")), float(g(r, "Cartn_y")), float(g(r, "Cartn_z"))
        except ValueError:
            continue
        if comp != "LIG" and atom == "CA":
            try:
                rs = int(g(r, "label_seq_id"))
                prot[rs] = [x, y, z]
            except ValueError:
                pass
        elif comp == "LIG" or ch == "B":
            lig.append([x, y, z])
    return prot, lig


def poll(tag):
    mpath = f"{ROOT}/{tag}/manifest.json"; man = json.load(open(mpath))
    fdir = f"{ROOT}/{tag}/feats"; os.makedirs(fdir, exist_ok=True)
    pending = [(idx, v["job"]) for idx, v in man.items() if v.get("job") and not os.path.exists(f"{fdir}/{idx}.npz")]
    print(f"poll {tag}: {len(pending)} pending of {len(man)}", flush=True)
    rounds = 0
    while pending and rounds < 120:
        still = []
        for idx, job in pending:
            rr = ba(["predictions:structure-and-binding", "retrieve", "--id", job, "--raw-output"])
            try:
                d = json.loads(rr.stdout)
            except Exception:
                still.append((idx, job)); continue
            st = d.get("status")
            if st in ("running", "queued", "pending", "starting"):
                still.append((idx, job)); continue
            if st != "succeeded" or not d.get("output"):
                np.savez(f"{fdir}/{idx}.npz", failed=True); continue   # mark done (failed)
            samples = d["output"].get("all_sample_results") or [d["output"]["best_sample"]]
            prots, ligs, confs = [], [], []
            for s in samples:
                url = s.get("structure", {}).get("url")
                if not url:
                    continue
                try:
                    cif = urllib.request.urlopen(url, timeout=60).read().decode()
                except Exception:
                    continue
                pr = parse_cif(cif)
                if not pr:
                    continue
                prot, lig = pr
                if not prot or not lig:
                    continue
                prots.append(prot); ligs.append(np.array(lig, np.float32))
                m = s.get("metrics", {})
                confs.append([m.get("structure_confidence", 0), m.get("ligand_iptm", 0),
                              m.get("complex_plddt", 0), m.get("complex_pde", 0)])
            if prots:
                res = sorted(set.intersection(*[set(p) for p in prots]))
                ca = np.stack([[p[r] for r in res] for p in prots]).astype(np.float32)   # (S, nres, 3)
                np.savez(f"{fdir}/{idx}.npz", ca=ca, resids=np.array(res), conf=np.array(confs, np.float32),
                         lig0=ligs[0], nlig=np.array([len(l) for l in ligs]),
                         ligpad=_pad(ligs))
            else:
                np.savez(f"{fdir}/{idx}.npz", failed=True)
        pending = still
        done = len(glob.glob(f"{fdir}/*.npz"))
        print(f"  round {rounds}: {done}/{len(man)} done, {len(pending)} pending", flush=True)
        if pending:
            time.sleep(45)
        rounds += 1
    print(f"poll DONE: {len(glob.glob(f'{fdir}/*.npz'))}/{len(man)} -> {fdir}")


def _pad(ligs):
    mx = max(len(l) for l in ligs); out = np.zeros((len(ligs), mx, 3), np.float32)
    for i, l in enumerate(ligs):
        out[i, :len(l)] = l
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "submit":
        submit(sys.argv[2], sys.argv[3])
    elif cmd == "poll":
        poll(sys.argv[2])
