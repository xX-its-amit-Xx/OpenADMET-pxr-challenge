"""
Run nb86-nb96 locally (CPU), then push GPU notebooks to Kaggle.

Local sequence (max_workers=1, OMP_NUM_THREADS=4):
  nb86: proper nested-CV ensemble (fast, ~60s)
  nb87: ChEMBL NR biological fingerprint (~200s)
  nb88: 3D conformer shape descriptors (~900s, conformer gen)
  nb89: PXR pharmacophore + SMARTS features (~120s)
  nb90: Tox21 fetch + bio-FP (~300s, includes download)
  nb91: cliff-proximity adaptive ensemble (~60s)
  nb92: multi-NR transfer LGBM (~200s)
  nb95: all-feature fusion (depends on nb87/88/89/92 outputs)
  nb96: grand ensemble v8 (depends on all above OOF files)

GPU notebooks nb93, nb94 are pushed to Kaggle separately.
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

    # Phase 1: independent feature notebooks
    r1 = run_seq(
        ["86_nested_cv_ensemble.ipynb",
         "87_bio_nr_fingerprint.ipynb",
         "89_pxr_pharmacophore.ipynb",
         "90_tox21_bio_fp.ipynb",
         "91_cliff_adaptive_blend.ipynb",
         "92_multi_nr_transfer.ipynb"],
        "Phase 1: nested ensemble + bio-FP + pharmacophore + Tox21 + cliff blend + NR transfer"
    )
    all_results.update(r1)

    # Phase 2: 3D shape (slow — conformer gen)
    r2 = run_seq(
        ["88_3d_shape_conformer.ipynb"],
        "Phase 2: 3D conformer shape descriptors"
    )
    all_results.update(r2)

    # Phase 3: all-feature fusion (depends on nb87/88/89/92 outputs)
    r3 = run_seq(
        ["95_all_feature_fusion.ipynb"],
        "Phase 3: all-feature fusion"
    )
    all_results.update(r3)

    # Phase 4: grand ensemble v8 (depends on all above)
    r4 = run_seq(
        ["96_grand_ensemble_v8.ipynb"],
        "Phase 4: grand ensemble v8"
    )
    all_results.update(r4)

    print("\n=== FINAL SUMMARY ===", flush=True)
    ok_count = sum(1 for v in all_results.values() if v)
    for nb, ok in all_results.items():
        print(f"  {'OK' if ok else 'FAIL'} {nb}", flush=True)
    print(f"\n{ok_count}/{len(all_results)} notebooks succeeded.", flush=True)
    sys.exit(0 if ok_count == len(all_results) else 1)
