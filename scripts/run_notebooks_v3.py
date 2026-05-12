"""
Run notebooks nb63–nb75 sequentially after run_notebooks_v2.py completes.

Dependency order:
  Phase 0: nb63 (data fetch — fixes PubChem, BindingDB, Papyrus)
  Phase 1: nb64–nb74 (LGBM/Chemprop combinations, sequential, 1 at a time)
  Phase 2: nb75 (grand metrics comparison, needs all OOFs)

Resource limit: max_workers=1, OMP_NUM_THREADS=4.
"""

import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def run_seq(names, label):
    print(f"\n=== {label} ===", flush=True)
    results = {}
    for name in names:
        nb, ok, elapsed = run_nb(name)
        results[nb] = ok
    return results


if __name__ == "__main__":
    all_results = {}

    # Phase -1: Re-run v2 failures (nb48 TabNet fix, nb51 DART, nb61 cliff analysis)
    r_patch = run_seq(
        ["48_tabnet_pxr.ipynb", "51_xgboost_dart.ipynb",
         "61_cliff_analysis.ipynb", "62_grand_ensemble_v7.ipynb"],
        "Phase -1: Re-run v2 failures (TabNet + DART + cliff analysis + ensemble)"
    )
    all_results.update(r_patch)

    r0 = run_seq(["63_expanded_data_fetch.ipynb"],
                 "Phase 0: Expanded data fetch")
    all_results.update(r0)

    phase1 = [
        "64_lgbm_full_metrics_baseline.ipynb",
        "65_lgbm_crc_singleconc_fdr.ipynb",
        "66_lgbm_chembl_pxr_direct.ipynb",
        "67_lgbm_chembl_all_nr_weighted.ipynb",
        "68_lgbm_pubchem_pxr_fixed.ipynb",
        "69_lgbm_counter_soft_labels.ipynb",
        "70_lgbm_crc_sp_chembl_pxr.ipynb",
        "71_lgbm_all_external_v2.ipynb",
        "72_lgbm_cliff_aware_external.ipynb",
        "73_lgbm_multitask_heads.ipynb",
        "74_chemprop_chembl_nr_multitask.ipynb",
    ]
    r1 = run_seq(phase1, "Phase 1: Combinatorial LGBM/Chemprop (sequential)")
    all_results.update(r1)

    r2 = run_seq(["75_grand_metrics_comparison.ipynb"],
                 "Phase 2: Grand metrics comparison")
    all_results.update(r2)

    print("\n=== FINAL SUMMARY ===", flush=True)
    ok_count = sum(1 for v in all_results.values() if v)
    for nb, ok in all_results.items():
        print(f"  {'OK' if ok else 'FAIL'} {nb}", flush=True)
    print(f"\n{ok_count}/{len(all_results)} notebooks succeeded.", flush=True)
    sys.exit(0 if ok_count == len(all_results) else 1)
