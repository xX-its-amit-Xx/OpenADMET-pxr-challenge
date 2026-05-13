"""Run nb103-nb106: delta-ML extensions (Chemprop, tiered, uncertainty, reverse).

nb103 — Delta-ML with Chemprop features (CPU, small network)
nb104 — Tiered delta-ML by similarity bucket (HIGH/MED/LOW)
nb105 — Delta-ML with uncertainty-weighted blending
nb106 — Reverse delta-ML (test compounds as templates)
"""
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
TIMEOUT = 7200  # 2 hours per notebook — nb103 (Chemprop) may be slow on CPU

NOTEBOOKS = [
    "103_delta_chemprop_cpu.ipynb",
    "104_delta_similarity_tiers.ipynb",
    "105_delta_with_uncertainty.ipynb",
    "106_reverse_delta_ml.ipynb",
]


def run_nb(nb_name, stop_on_fail=True):
    nb_path  = NB_DIR / nb_name
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
        log_path.write_text(
            f"=== STDOUT ===\n{result.stdout}\n=== STDERR ===\n{result.stderr}\n",
            encoding="utf-8"
        )
        err_path.write_text(result.stderr, encoding="utf-8")
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
    import argparse
    parser = argparse.ArgumentParser(description="Run phase v8 notebooks (nb103-nb106)")
    parser.add_argument("--no-stop-on-fail", action="store_true",
                        help="Continue running even if a notebook fails")
    parser.add_argument("--notebooks", nargs="+", default=None,
                        help="Run specific notebooks only (e.g. 104_delta_similarity_tiers.ipynb)")
    args = parser.parse_args()

    nbs_to_run = args.notebooks if args.notebooks else NOTEBOOKS
    stop_on_fail = not args.no_stop_on_fail

    print("=" * 60)
    print("Phase v8 runner  (nb103–nb106 delta-ML extensions)")
    print(f"Notebooks: {nbs_to_run}")
    print(f"Stop on fail: {stop_on_fail}")
    print("=" * 60, flush=True)

    results = {}
    t_total = time.time()
    for nb in nbs_to_run:
        name, ok, t = run_nb(nb, stop_on_fail=stop_on_fail)
        results[name] = ok
        if not ok and stop_on_fail:
            print(f"\n  STOPPING: {name} failed", flush=True)
            break

    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}", flush=True)
    print(f"=== SUMMARY ({elapsed_total/60:.1f}m total) ===", flush=True)
    for nb, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {nb}", flush=True)
    n_ok = sum(results.values())
    n_total = len(results)
    print(f"\n{n_ok}/{n_total} notebooks succeeded.", flush=True)
    sys.exit(0 if all(results.values()) else 1)
