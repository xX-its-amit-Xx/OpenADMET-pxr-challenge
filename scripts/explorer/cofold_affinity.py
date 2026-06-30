"""Run Boltz-2 affinity cofold on a yaml dir slice (TASK NTASKS), extract affinity_pred_value + prob."""
import subprocess, glob, json, os, sys, csv, shutil
yaml_dir, out_csv = sys.argv[1], sys.argv[2]
TASK=int(sys.argv[3]) if len(sys.argv)>3 else 0
NT=int(sys.argv[4]) if len(sys.argv)>4 else 1
allf=sorted(glob.glob(f"{yaml_dir}/*.yaml")); mine=allf[TASK::NT]
TMP=f"/tmp/aff_{os.environ.get('SLURM_JOB_ID','x')}_{TASK}"; shutil.rmtree(TMP,ignore_errors=True); os.makedirs(f"{TMP}/in")
for y in mine: os.symlink(os.path.abspath(y), f"{TMP}/in/{os.path.basename(y)}")
subprocess.run(["./env/bin/boltz","predict",f"{TMP}/in","--no_kernels","--out_dir",f"{TMP}/out",
                "--cache","./boltz_cache","--output_format","pdb","--num_workers","4"],check=False)
rows=[]
for jf in glob.glob(f"{TMP}/out/**/affinity_*.json",recursive=True):
    name=os.path.basename(jf).replace("affinity_","").replace(".json","")
    try: d=json.load(open(jf))
    except: continue
    rows.append((name, d.get("affinity_pred_value"), d.get("affinity_probability_binary")))
out=f"{out_csv}.t{TASK}" if NT>1 else out_csv
with open(out,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["idx","affinity_pred_value","affinity_prob_binary"]); w.writerows(rows)
print(f"task {TASK}: extracted {len(rows)} -> {out}")
shutil.rmtree(TMP,ignore_errors=True)
