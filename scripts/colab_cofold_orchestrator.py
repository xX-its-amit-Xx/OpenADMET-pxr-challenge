"""Robust Colab-A100 cofold orchestrator — survives Colab's ephemeral sessions.

Colab reclaims idle runtimes (a DETACHED cofold doesn't keep the session alive -> lost work). Fix: run in small
FOREGROUND batches (the active `exec` connection keeps the session warm) and DOWNLOAD after every batch, so a
session death costs <= 1 batch. Re-provisions automatically if the session is gone. Source of truth = the LOCAL
feats dir (downloaded per-idx npz), so re-provisioning only re-cofolds genuinely-missing idx.

Usage: python scripts/colab_cofold_orchestrator.py <input_csv> <local_feats_dir> [batch=12]
  input_csv: idx,smiles   (e.g. C:/pxr_struct/boltz/eval_missing.csv)
"""
import sys, os, csv, subprocess, time, glob
import numpy as np

COLAB = r"D:\Users\ashenoy00000\.local\bin\colab.exe"
S = "pxr-activity-cofold"
ENV = {**os.environ, "MSYS_NO_PATHCONV": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
ASSETS = {"C:/pxr_struct/boltz/pxr_msa.a3m": "/content/pxr_msa.a3m",
          "scripts/cofold_colab.py": "/content/cofold_colab.py"}


def run(args, timeout):
    return subprocess.run([COLAB, "--auth", "adc"] + args, capture_output=True, text=True, env=ENV, timeout=timeout)


def ping():
    """A trivial exec — this both health-checks AND makes the CLI clean up a lost (404/401) session,
    so a zombie session (listed but dead) is removed and we re-provision next."""
    try:
        r = run(["exec", "-s", S, "-f", "C:/pxr_struct/boltz/ping.py", "--timeout", "60"], 120)
        return "PONG" in r.stdout
    except Exception:
        return False


def ensure_session():
    if ping():
        return True
    print("[orch] session not responsive; provisioning fresh A100...", flush=True)
    try:
        run(["stop", "-s", S], 60)
    except Exception:
        pass
    r = run(["new", "-s", S, "--gpu", "A100"], 400)
    if "READY" not in (r.stdout + r.stderr):
        print("[orch] provision FAILED:", (r.stdout + r.stderr)[-200:], flush=True); return False
    for L, R in ASSETS.items():
        run(["upload", "-s", S, L, R], 150)
    print("[orch] installing boltz...", flush=True)
    run(["install", "-s", S, "boltz", "gemmi", "biopython"], 900)
    return ping()


def main():
    inp, fdir = sys.argv[1], sys.argv[2]
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    os.makedirs(fdir, exist_ok=True)
    rows = list(csv.DictReader(open(inp)))
    done = set(int(os.path.basename(p)[:-4]) for p in glob.glob(f"{fdir}/*.npz"))
    todo = [r for r in rows if int(r["idx"]) not in done]
    print(f"[orch] {len(rows)} total, {len(done)} done, {len(todo)} todo", flush=True)

    def out(r):
        return (getattr(r, "stdout", "") or "") + (getattr(r, "stderr", "") or "")

    bi = 0
    while todo:
      try:
        if not ensure_session():
            print("[orch] cannot get session; sleeping 120s", flush=True); time.sleep(120); continue
        chunk = todo[:batch]
        # write+upload batch CSV
        bcsv = "C:/pxr_struct/boltz/_batch.csv"
        with open(bcsv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["idx", "smiles"])
            for r in chunk:
                w.writerow([r["idx"], r["smiles"]])
        if "Uploaded" not in out(run(["upload", "-s", S, bcsv, "/content/ligands.csv"], 120)):
            print("[orch] batch upload failed (session likely dead); re-check next loop", flush=True); time.sleep(10); continue
        # reset /content/feats so merged = just this batch, then cofold (foreground keeps session warm)
        t0 = time.time()
        try:
            r = run(["exec", "-s", S, "-f", "scripts/cofold_colab.py", "--timeout", str(60 * len(chunk) + 300)],
                    timeout=60 * len(chunk) + 360)
        except subprocess.TimeoutExpired:
            print("[orch] exec timeout; re-check next loop", flush=True); time.sleep(10); continue
        if "DONE" not in out(r):
            print(f"[orch] batch exec no DONE ({out(r)[-200:]}); re-check next loop", flush=True); time.sleep(10); continue
        # download merged + split to local per-idx npz (unique name per batch -> no file-lock)
        dl = f"C:/pxr_struct/boltz/_merged_{bi}.npz"
        run(["download", "-s", S, "/content/feats_merged.npz", dl], 150)
        if os.path.exists(dl):
            with np.load(dl) as d:
                idxs, rz, gm, rm = d["idx"], d["richz"], d["geom"], d["rmsf"]
                for k, ix in enumerate(idxs):
                    np.savez(f"{fdir}/{int(ix)}.npz", richz=rz[k], geom=gm[k], rmsf=rm[k])
            try:
                os.remove(dl)
            except OSError:
                pass
            done = set(int(os.path.basename(p)[:-4]) for p in glob.glob(f"{fdir}/*.npz"))
            todo = [r for r in rows if int(r["idx"]) not in done]
            bi += 1
            print(f"[orch] batch {bi}: +{len(chunk)} in {time.time()-t0:.0f}s | done {len(done)}/{len(rows)} | left {len(todo)}", flush=True)
        else:
            print("[orch] download failed; retry", flush=True)
      except Exception as e:
        print(f"[orch] batch error ({type(e).__name__}: {e}); retry next loop", flush=True); time.sleep(15)
    print(f"[orch] ALL DONE: {len(done)}/{len(rows)} -> {fdir}", flush=True)


if __name__ == "__main__":
    main()
