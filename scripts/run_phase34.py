"""Run nb95 (all-feature fusion) and nb96 (grand ensemble v8) only."""
import os, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

ROOT    = Path(__file__).parent.parent
JUPYTER = ROOT / ".venv" / "Scripts" / "jupyter"
NB_DIR  = ROOT / "notebooks"
LOG_DIR = ROOT / "scripts" / "nb_logs"
LOG_DIR.mkdir(exist_ok=True)
TIMEOUT = 7200

def run_nb(nb_name):
    nb_path = NB_DIR / nb_name
    log_path = LOG_DIR / nb_name.replace(".ipynb", ".log")
    err_path = LOG_DIR / nb_name.replace(".ipynb", ".err")
    print(f"  [START] {nb_name}", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            [str(JUPYTER), "nbconvert", "--to", "notebook", "--execute",
             "--inplace", f"--ExecutePreprocessor.timeout={TIMEOUT}",
             "--ExecutePreprocessor.kernel_name=pxr-challenge", str(nb_path)],
            capture_output=True, text=True, timeout=TIMEOUT + 120, cwd=str(ROOT),
        )
        elapsed = time.time() - t0
        ok = result.returncode == 0
        log_path.write_text(f"=== STDOUT ===\n{result.stdout}\n=== STDERR ===\n{result.stderr}\n")
        err_path.write_text(result.stderr)
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {nb_name}  ({elapsed:.0f}s)", flush=True)
        if not ok:
            for ln in result.stderr.strip().split("\n")[-20:]:
                print(f"    | {ln}", flush=True)
        return nb_name, ok, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] {nb_name}  ({elapsed:.0f}s)", flush=True)
        return nb_name, False, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [ERROR] {nb_name}: {e}", flush=True)
        return nb_name, False, elapsed

if __name__ == "__main__":
    print("Phase 3+4 runner started", flush=True)
    results = {}
    for nb in ["95_all_feature_fusion.ipynb", "96_grand_ensemble_v8.ipynb"]:
        name, ok, t = run_nb(nb)
        results[name] = ok
        if not ok:
            print(f"  STOPPING: {name} failed", flush=True)
            break
    print("\n=== SUMMARY ===", flush=True)
    for nb, ok in results.items():
        print(f"  {'OK' if ok else 'FAIL'} {nb}", flush=True)
    sys.exit(0 if all(results.values()) else 1)
