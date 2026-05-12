"""
Run notebooks nb76–nb85 sequentially after run_notebooks_v3.py completes.

Creative approaches:
  nb76: Delta-ML template prediction
  nb77: Sparse Gaussian Process (Tanimoto kernel)
  nb78: 7-fingerprint diversity ensemble
  nb79: 3D conformer shape descriptors
  nb80: SMILES enumeration augmentation
  nb81: Pseudo-label self-training
  nb82: Selectivity-aware prediction
  nb83: Transductive graph label spreading
  nb84: Free-Wilson scaffold decomposition
  nb85: Creative mega-ensemble

Resource limit: max_workers=1, OMP_NUM_THREADS=4.
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

    # Phase -1: remaining v3 failures (nb61 cliff analysis, nb69/73 LGBM, nb75 comparison)
    r_patch = run_seq(
        ["61_cliff_analysis.ipynb", "69_lgbm_counter_soft_labels.ipynb",
         "73_lgbm_multitask_heads.ipynb", "75_grand_metrics_comparison.ipynb"],
        "Phase -1: Re-run remaining v3 failures"
    )
    all_results.update(r_patch)

    creative = [
        "76_delta_ml_template.ipynb",
        "77_gaussian_process_tanimoto.ipynb",
        "78_multi_fingerprint_ensemble.ipynb",
        "79_3d_shape_descriptors.ipynb",
        "80_smiles_enumeration_augmentation.ipynb",
        "81_pseudo_label_self_training.ipynb",
        "82_selectivity_aware_prediction.ipynb",
        "83_graph_label_spreading.ipynb",
        "84_free_wilson_decomposition.ipynb",
    ]
    r_creative = run_seq(creative, "Creative approaches (nb76–nb84)")
    all_results.update(r_creative)

    r_mega = run_seq(["85_creative_mega_ensemble.ipynb"],
                     "Mega-ensemble (nb85)")
    all_results.update(r_mega)

    print("\n=== FINAL SUMMARY ===", flush=True)
    ok_count = sum(1 for v in all_results.values() if v)
    for nb, ok in all_results.items():
        print(f"  {'OK' if ok else 'FAIL'} {nb}", flush=True)
    print(f"\n{ok_count}/{len(all_results)} notebooks succeeded.", flush=True)
    sys.exit(0 if ok_count == len(all_results) else 1)
