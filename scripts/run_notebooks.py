"""
Orchestrate execution of notebooks nb26-nb36 with dependency management.

Dependency graph:
  Independent:  nb26, nb27, nb28, nb33, nb34, nb35
  Requires nb29: nb30, nb31, nb32
  Requires all:  nb36
"""

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
JUPYTER = ROOT / '.venv' / 'Scripts' / 'jupyter'
NB_DIR  = ROOT / 'notebooks'
LOG_DIR = ROOT / 'scripts' / 'nb_logs'
LOG_DIR.mkdir(exist_ok=True)

TIMEOUT = 7200  # 2 hours max per notebook


def run_nb(nb_name: str) -> tuple[str, bool, float]:
    nb_path = NB_DIR / nb_name
    log_path = LOG_DIR / nb_name.replace('.ipynb', '.log')
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
            timeout=TIMEOUT + 60,
            cwd=str(ROOT),
        )
        elapsed = time.time() - t0
        ok = result.returncode == 0
        with open(log_path, 'w') as f:
            f.write(f'=== STDOUT ===\n{result.stdout}\n=== STDERR ===\n{result.stderr}\n')
        status = 'OK' if ok else 'FAIL'
        print(f'  [{status}] {nb_name}  ({elapsed:.0f}s)', flush=True)
        return nb_name, ok, elapsed
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f'  [TIMEOUT] {nb_name}  ({elapsed:.0f}s)', flush=True)
        return nb_name, False, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f'  [ERROR] {nb_name}: {e}  ({elapsed:.0f}s)', flush=True)
        return nb_name, False, elapsed


def run_group(names: list[str], label: str, max_workers: int = 4) -> dict[str, bool]:
    print(f'\n=== {label} ===', flush=True)
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_nb, n): n for n in names}
        for fut in as_completed(futures):
            nb, ok, elapsed = fut.result()
            results[nb] = ok
    return results


if __name__ == '__main__':
    all_results = {}

    # ── Phase 1: independent notebooks ───────────────────────────────────────
    phase1 = [
        '26_singleconc_lgbm.ipynb',
        '27_nr_weighted_lgbm.ipynb',
        '28_auxiliary_features_lgbm.ipynb',
        '33_crossattn_chemberta_esm2.ipynb',
        '34_crossattn_grover_esm2.ipynb',
        '35_chemprop_auxiliary.ipynb',
    ]
    r1 = run_group(phase1, 'Phase 1: independent notebooks', max_workers=3)
    all_results.update(r1)

    # ── Phase 2: protein embeddings (must complete before nb30-32) ───────────
    r2 = run_group(['29_protein_embeddings.ipynb'], 'Phase 2: protein embeddings', max_workers=1)
    all_results.update(r2)

    # ── Phase 3: protein-aware multi-NR notebooks ────────────────────────────
    if r2.get('29_protein_embeddings.ipynb', False):
        phase3 = [
            '30_morgan_esm2_multinr.ipynb',
            '31_chemberta_esm2_multinr.ipynb',
            '32_protbert_multinr.ipynb',
        ]
        r3 = run_group(phase3, 'Phase 3: multi-NR protein-aware notebooks', max_workers=3)
        all_results.update(r3)
    else:
        print('\n[SKIP] Phase 3 — nb29 failed, skipping nb30-32', flush=True)

    # ── Phase 4: grand ensemble v6 ───────────────────────────────────────────
    r4 = run_group(['36_grand_ensemble_v6.ipynb'], 'Phase 4: grand ensemble v6', max_workers=1)
    all_results.update(r4)

    # ── Summary ──────────────────────────────────────────────────────────────
    print('\n=== FINAL SUMMARY ===', flush=True)
    ok_count = sum(1 for v in all_results.values() if v)
    for nb, ok in all_results.items():
        print(f'  {"✓" if ok else "✗"} {nb}', flush=True)
    print(f'\n{ok_count}/{len(all_results)} notebooks succeeded.', flush=True)
    sys.exit(0 if ok_count == len(all_results) else 1)
