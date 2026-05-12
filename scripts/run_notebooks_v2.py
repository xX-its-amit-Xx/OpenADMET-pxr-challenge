"""
Run notebooks nb37–nb62 with dependency management and resource-aware concurrency.

Dependency graph:
  Phase 1 (independent LightGBM/tree): nb43, nb50, nb51, nb55, nb57, nb58
  Phase 2 (independent NNs):           nb39, nb47, nb48, nb49
  Phase 3 (needs cliff_labels):        nb41, nb46, nb54, nb59
  Phase 4 (grand ensemble):            nb61, nb62

Skipped (too long / no GPU): nb52, nb53, nb56 (Chemprop pretrain variants)
nb60 (deep graph cliff) is run in Phase 3 if time allows.

Resource limit: max_workers=1 (sequential) to leave headroom for Windsurf IDE and
other Claude Code sessions. OMP_NUM_THREADS=4 caps LightGBM/XGBoost thread usage.
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT   = Path(__file__).parent.parent
JUPYTER = ROOT / '.venv' / 'Scripts' / 'jupyter'
NB_DIR  = ROOT / 'notebooks'
LOG_DIR = ROOT / 'scripts' / 'nb_logs'
LOG_DIR.mkdir(exist_ok=True)

TIMEOUT = 7200   # 2 h max per notebook
MAX_WORKERS = 1  # sequential: keep headroom for Windsurf IDE and other sessions


def run_nb(nb_name: str) -> tuple[str, bool, float]:
    nb_path = NB_DIR / nb_name
    log_path = LOG_DIR / nb_name.replace('.ipynb', '.log')
    err_path = LOG_DIR / nb_name.replace('.ipynb', '.err')
    print(f'  [START] {nb_name}', flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            [
                str(JUPYTER), 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',
                f'--ExecutePreprocessor.timeout={TIMEOUT}',
                '--ExecutePreprocessor.kernel_name=pxr-challenge',
                str(nb_path),
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT + 120,
            cwd=str(ROOT),
        )
        elapsed = time.time() - t0
        ok = result.returncode == 0
        log_path.write_text(f'=== STDOUT ===\n{result.stdout}\n=== STDERR ===\n{result.stderr}\n')
        err_path.write_text(result.stderr)
        status = 'OK' if ok else 'FAIL'
        print(f'  [{status}] {nb_name}  ({elapsed:.0f}s)', flush=True)
        if not ok:
            # Print last 20 lines of stderr for quick diagnosis
            lines = result.stderr.strip().split('\n')
            for ln in lines[-20:]:
                print(f'    | {ln}', flush=True)
        return nb_name, ok, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f'  [TIMEOUT] {nb_name}  ({elapsed:.0f}s)', flush=True)
        return nb_name, False, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f'  [ERROR] {nb_name}: {e}  ({elapsed:.0f}s)', flush=True)
        return nb_name, False, elapsed


def run_group(names: list[str], label: str, workers: int = MAX_WORKERS) -> dict[str, bool]:
    print(f'\n=== {label} ===', flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_nb, n): n for n in names}
        for fut in as_completed(futures):
            nb, ok, elapsed = fut.result()
            results[nb] = ok
    return results


if __name__ == '__main__':
    all_results: dict[str, bool] = {}

    # Phase 1: LightGBM / tree-based — run sequentially
    # OMP_NUM_THREADS=4 caps OpenMP threads for LightGBM/XGBoost
    import os
    os.environ.setdefault('OMP_NUM_THREADS', '4')

    phase1 = [
        '43_focal_loss_lgbm.ipynb',
        '50_catboost_cliff.ipynb',
        '55_lgbm_all_external.ipynb',
        '57_lgbm_pubchem_pxr.ipynb',
        '58_lgbm_bindingdb.ipynb',
    ]
    r1 = run_group(phase1, 'Phase 1: LightGBM / tree-based', workers=1)
    all_results.update(r1)

    # Phase 1b: XGBoost DART alone (memory-heavy, run solo)
    r1b = run_group(['51_xgboost_dart.ipynb'], 'Phase 1b: XGBoost DART (solo)', workers=1)
    all_results.update(r1b)

    # Phase 2: Neural networks (run solo to avoid memory pressure)
    phase2 = [
        '39_hard_negative_augmentation.ipynb',
        '47_multitask_mlp.ipynb',
        '48_tabnet_pxr.ipynb',
        '49_wide_deep_mlp.ipynb',
    ]
    for nb in phase2:
        r = run_group([nb], f'Phase 2: {nb}', workers=1)
        all_results.update(r)

    # Phase 3: Models needing cliff_labels (already exists from nb38)
    phase3 = [
        '41_chemprop_cliff_heads.ipynb',
        '46_siamese_cliff_net.ipynb',
        '54_deep_ensemble_uncertainty.ipynb',
        '59_lgbm_cliff_oversample.ipynb',
    ]
    for nb in phase3:
        r = run_group([nb], f'Phase 3: {nb}', workers=1)
        all_results.update(r)

    # Phase 4: Cliff analysis (needs OOF files from prior notebooks)
    r4 = run_group(['61_cliff_analysis.ipynb'], 'Phase 4: cliff analysis', workers=1)
    all_results.update(r4)

    # Phase 5: Grand ensemble v7
    r5 = run_group(['62_grand_ensemble_v7.ipynb'], 'Phase 5: grand ensemble v7', workers=1)
    all_results.update(r5)

    # Summary
    print('\n=== FINAL SUMMARY ===', flush=True)
    ok_count = sum(1 for v in all_results.values() if v)
    for nb, ok in all_results.items():
        print(f'  {"OK" if ok else "FAIL"} {nb}', flush=True)
    print(f'\n{ok_count}/{len(all_results)} notebooks succeeded.', flush=True)
    sys.exit(0 if ok_count == len(all_results) else 1)
