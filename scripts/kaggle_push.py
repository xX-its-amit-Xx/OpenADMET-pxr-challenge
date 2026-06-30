"""Push processed data and/or notebooks to Kaggle for GPU execution.

Usage:
    python scripts/kaggle_push.py --data              # create/update dataset
    python scripts/kaggle_push.py --nb 63             # push notebook as GPU kernel
    python scripts/kaggle_push.py --nb 63 --poll      # push and wait for completion
    python scripts/kaggle_push.py --nb 63 --pull      # push, poll, and download outputs
    python scripts/kaggle_push.py --data --nb 63      # sync data then push notebook

Requirements:
    pip install kaggle psutil
    Place ~/.kaggle/kaggle.json (or D:/Users/ashenoy00000/.kaggle/kaggle.json on Windows)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
SUBMISSIONS    = ROOT / "submissions"
NB_DIR         = ROOT / "notebooks"
SRC_DIR        = ROOT / "src"

USERNAME     = "knowledgegraphlover"
DATASET_SLUG = "pxr-challenge-data"
DATASET_REF  = f"{USERNAME}/{DATASET_SLUG}"

CPU_THRESHOLD  = int(os.environ.get("KAGGLE_CPU_THRESHOLD", "70"))   # % — abort if above after grace period
POLL_INTERVAL  = 60   # seconds between kernel status checks


# ---------------------------------------------------------------------------
# CPU guard
# ---------------------------------------------------------------------------

def _check_cpu(label: str = "") -> None:
    load = psutil.cpu_percent(interval=1)
    tag  = f"[{label}] " if label else ""
    if load > CPU_THRESHOLD:
        print(f"{tag}CPU at {load:.0f}% (threshold {CPU_THRESHOLD}%). "
              "Waiting 30 s for headroom…")
        time.sleep(30)
        load = psutil.cpu_percent(interval=1)
        if load > CPU_THRESHOLD:
            print(f"{tag}CPU still at {load:.0f}%. Aborting to protect other sessions.")
            sys.exit(1)
    print(f"{tag}CPU {load:.0f}% — OK")


# ---------------------------------------------------------------------------
# Kaggle CLI wrapper
# ---------------------------------------------------------------------------

def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["kaggle"] + args, check=True, text=True, **kwargs)


def _dataset_exists() -> bool:
    r = subprocess.run(
        ["kaggle", "datasets", "list", "--user", USERNAME, "--search", DATASET_SLUG],
        capture_output=True, text=True,
    )
    return DATASET_SLUG in r.stdout


# ---------------------------------------------------------------------------
# Data sync
# ---------------------------------------------------------------------------

def push_data() -> None:
    _check_cpu("data")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # processed parquets
        n_pq = 0
        for f in DATA_PROCESSED.glob("*.parquet"):
            shutil.copy(f, tmp / f.name)
            n_pq += 1

        # submission CSVs
        subs_tmp = tmp / "submissions"
        subs_tmp.mkdir()
        n_csv = 0
        for f in SUBMISSIONS.glob("*.csv"):
            shutil.copy(f, subs_tmp / f.name)
            n_csv += 1

        # src/pxr library so Kaggle notebooks can import it
        src_tmp = tmp / "src"
        shutil.copytree(SRC_DIR, src_tmp)

        # ChemBERTa model directory — Kaggle auto-extracts dirs zipped by --dir-mode zip
        # Uploads chemberta_mtr/ → accessible at .../pxr-challenge-data/chemberta_mtr/ on Kaggle
        model_dir = ROOT / "data" / "external" / "chemberta_mtr"
        if model_dir.is_dir() and any(model_dir.iterdir()):
            shutil.copytree(model_dir, tmp / "chemberta_mtr")
            model_mb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) // (1024 * 1024)
            print(f"[data] chemberta_mtr/ directory ({model_mb}MB)")

        meta = {
            "title":    "PXR Challenge Data",
            "id":       DATASET_REF,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (tmp / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        print(f"[data] {n_pq} parquets · {n_csv} CSVs · src/pxr -> {DATASET_REF}")

        if _dataset_exists():
            _run(["datasets", "version", "-p", str(tmp), "-m", "auto-sync from kaggle_push.py",
                  "--dir-mode", "zip"])
            print(f"[data] Dataset version updated: {DATASET_REF}")
        else:
            _run(["datasets", "create", "-p", str(tmp), "--dir-mode", "zip"])
            print(f"[data] Dataset created: {DATASET_REF}")


# ---------------------------------------------------------------------------
# Notebook kernel push
# ---------------------------------------------------------------------------

def _find_notebook(nb_num: int) -> Path:
    for pattern in (f"{nb_num:02d}_*.ipynb", f"{nb_num}_*.ipynb"):
        matches = sorted(NB_DIR.glob(pattern))
        if matches:
            return matches[0]
    print(f"[nb] No notebook matching {nb_num:02d}_*.ipynb found in notebooks/")
    sys.exit(1)


def _patch_notebook(nb_path: Path, dst: Path) -> None:
    """Inject a setup cell so Kaggle kernels can find src/pxr, and fix kernelspec."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    # Kaggle only has 'python3', not our local 'pxr-challenge' kernel
    nb.setdefault("metadata", {})["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    ds = DATASET_SLUG
    setup_src = f"""\
# === Kaggle self-contained setup (no external dataset dependency) ===
import sys, os, subprocess, types
import numpy as _np, pandas as _pd
from pathlib import Path as _Path

WORK = _Path('/kaggle/working')
SUBMISSIONS = WORK / 'submissions'
SUBMISSIONS.mkdir(exist_ok=True)
DATA_PROCESSED = WORK / 'processed'
DATA_PROCESSED.mkdir(exist_ok=True)

# 1) Try dataset path if mounted
# Kaggle mounts at /kaggle/input/datasets/USERNAME/SLUG
_ds_base = f'/kaggle/input/datasets/{USERNAME}/{ds}'
_ds_src = _ds_base + '/src'
if _Path(_ds_src + '/pxr').is_dir():
    sys.path.insert(0, _ds_src)
    os.environ['PXR_DATA_ROOT'] = _ds_base
    print(f'pxr loaded from dataset: {{_ds_src}}')

# 2) Ensure rdkit is available
try:
    from rdkit import Chem as _Chem
    from rdkit.Chem import AllChem as _AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold as _MS
    _RDKIT_OK = True
    print('rdkit available')
except ImportError:
    print('rdkit not found — installing...')
    subprocess.run([sys.executable,'-m','pip','install','rdkit','-q'], check=False)
    try:
        from rdkit import Chem as _Chem
        from rdkit.Chem import AllChem as _AllChem
        from rdkit.Chem.Scaffolds import MurckoScaffold as _MS
        _RDKIT_OK = True
        print('rdkit installed OK')
    except ImportError:
        _RDKIT_OK = False
        print('WARNING: rdkit unavailable — scaffold/Morgan shims will be stubs')

# 3) Download raw data from HuggingFace if pxr not available
_HF_BASE = 'https://huggingface.co/datasets/openadmet/pxr-challenge-train-test/resolve/main'
_DATA_DIR = WORK / 'rawdata'
_DATA_DIR.mkdir(exist_ok=True)

def _fetch(fname):
    p = _DATA_DIR / fname
    if not p.exists():
        import urllib.request
        url = f'{{_HF_BASE}}/{{fname}}'
        print(f'Downloading {{fname}}...')
        urllib.request.urlretrieve(url, p)
    return p

# 4) Inject pxr shims (always safe to do; local pxr will override if imported later)
_pxr_data = types.ModuleType('pxr.data')
def load_train():
    p = _fetch('pxr-challenge_TRAIN.csv')
    df = _pd.read_csv(p)
    return df.rename(columns={{'Molecule Name':'name','SMILES':'smiles','pEC50':'pec50',
        'pEC50_std.error (-log10(molarity))':'pec50_se','Emax_estimate (log2FC vs. baseline)':'emax'}})
def load_test():
    p = _fetch('pxr-challenge_TEST_BLINDED.csv')
    df = _pd.read_csv(p)
    return df.rename(columns={{'Molecule Name':'name','SMILES':'smiles'}})
def load_counter():
    p = _fetch('pxr-challenge_counter-assay_TRAIN.csv')
    df = _pd.read_csv(p)
    return df.rename(columns={{'Molecule Name':'name','SMILES':'smiles','pEC50':'pec50'}})
_pxr_data.load_train   = load_train
_pxr_data.load_test    = load_test
_pxr_data.load_counter = load_counter

_pxr_eval = types.ModuleType('pxr.eval')
def rae(yt, yp):
    yt, yp = _np.asarray(yt, float), _np.asarray(yp, float)
    return float(_np.mean(_np.abs(yt-yp)) / _np.mean(_np.abs(yt-yt.mean())))
def scaffold_kfold_indices(scaffolds, n_splits=5, seed=42):
    from collections import defaultdict
    s2idx = defaultdict(list)
    for i, s in enumerate(scaffolds): s2idx[s].append(i)
    buckets = sorted(s2idx.values(), key=len, reverse=True)
    folds = [[] for _ in range(n_splits)]; sizes = [0]*n_splits
    for b in buckets:
        f = min(range(n_splits), key=lambda k: sizes[k])
        folds[f].extend(b); sizes[f] += len(b)
    all_idx = list(range(len(scaffolds)))
    return [(sorted(set(all_idx)-set(folds[k])), folds[k]) for k in range(n_splits)]
_pxr_eval.rae = rae
_pxr_eval.scaffold_kfold_indices = scaffold_kfold_indices

_pxr_chem = types.ModuleType('pxr.chem')
if _RDKIT_OK:
    def bemis_murcko(smi):
        try:
            mol = _Chem.MolFromSmiles(str(smi))
            return _MS.GetScaffoldForMol(mol) if mol else smi
        except: return smi
    def morgan_fp_batch(smiles, radius=2, n_bits=2048):
        fps = []
        for s in smiles:
            mol = _Chem.MolFromSmiles(str(s))
            if mol:
                fp = _AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                fps.append(list(fp))
            else:
                fps.append([0]*n_bits)
        return _np.array(fps, dtype=_np.float32)
else:
    def bemis_murcko(smi): return str(smi)[:20]  # stub
    def morgan_fp_batch(smiles, radius=2, n_bits=2048):
        return _np.zeros((len(smiles), n_bits), dtype=_np.float32)
_pxr_chem.bemis_murcko = bemis_murcko
_pxr_chem.morgan_fp_batch = morgan_fp_batch

_pxr_paths = types.ModuleType('pxr.paths')
_pxr_paths.DATA_PROCESSED = DATA_PROCESSED
_pxr_paths.SUBMISSIONS    = SUBMISSIONS

_pxr = types.ModuleType('pxr')
sys.modules.update({{'pxr':_pxr,'pxr.data':_pxr_data,'pxr.eval':_pxr_eval,
                    'pxr.chem':_pxr_chem,'pxr.paths':_pxr_paths}})
print('Shims ready. Data will download from HuggingFace on first load_train()/load_test() call.')
"""
    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": setup_src,
    }
    nb["cells"] = [setup_cell] + nb["cells"]
    dst.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def push_notebook(nb_num: int, poll: bool, pull: bool) -> None:
    _check_cpu("nb")

    nb_path     = _find_notebook(nb_num)
    kernel_slug = f"pxr-challenge-nb{nb_num:02d}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        patched = tmp / nb_path.name
        _patch_notebook(nb_path, patched)

        accel = os.environ.get("KAGGLE_ACCEL", "gpu").lower()
        meta = {
            "id":               f"{USERNAME}/{kernel_slug}",
            "title":            f"PXR Challenge nb{nb_num:02d}",
            "code_file":        nb_path.name,
            "language":         "python",
            "kernel_type":      "notebook",
            "is_private":       True,
            "enable_gpu":       (accel == "gpu"),
            "enable_tpu":       (accel == "tpu"),
            "enable_internet":  True,
            "dataset_sources":  [DATASET_REF],
            "competition_sources": [],
            "kernel_sources":   [],
        }
        # Kaggle's default "GPU" is a P100 (cc6.0) — modern torch has no sm_60 kernels and crashes on the
        # first GPU op. Request a T4 (cc7.5, torch-compatible, 2x16GB) explicitly. Override via KAGGLE_GPU_TYPE.
        gpu_type = os.environ.get("KAGGLE_GPU_TYPE", "NvidiaTeslaT4")
        push_cmd = ["kernels", "push", "-p", str(tmp)]
        if accel == "gpu" and gpu_type:
            push_cmd += ["--accelerator", gpu_type]
        (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

        tag = f"{accel.upper()}{':' + gpu_type if accel == 'gpu' and gpu_type else ''}"
        print(f"[nb] Pushing {nb_path.name} -> {USERNAME}/{kernel_slug} ({tag})")
        _run(push_cmd)
        print(f"[nb] Kernel live: https://www.kaggle.com/code/{USERNAME}/{kernel_slug}")

    if poll or pull:
        _poll_kernel(kernel_slug, pull, nb_num)


# ---------------------------------------------------------------------------
# Polling + output download
# ---------------------------------------------------------------------------

def _poll_kernel(kernel_slug: str, pull: bool, nb_num: int) -> None:
    ref = f"{USERNAME}/{kernel_slug}"
    print(f"[poll] Checking every {POLL_INTERVAL}s — Ctrl+C to stop watching")
    while True:
        r = subprocess.run(
            ["kaggle", "kernels", "status", ref],
            capture_output=True, text=True,
        )
        line = r.stdout.strip()
        ts   = time.strftime("%H:%M:%S")
        print(f"  [{ts}] {line}")
        if "complete" in line.lower():
            print("[poll] Done.")
            if pull:
                _pull_outputs(kernel_slug, nb_num)
            break
        elif "error" in line.lower() or "cancel" in line.lower():
            print("[poll] Kernel did not complete successfully.")
            break
        time.sleep(POLL_INTERVAL)


def _pull_outputs(kernel_slug: str, nb_num: int) -> None:
    out_dir = SUBMISSIONS / f"kaggle_nb{nb_num:02d}"
    out_dir.mkdir(exist_ok=True)
    _run(["kernels", "output", f"{USERNAME}/{kernel_slug}", "-p", str(out_dir)])
    print(f"[pull] Outputs -> {out_dir.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Push data / notebooks to Kaggle for GPU execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", action="store_true",
                        help="Sync data/processed + submissions + src/pxr to Kaggle dataset")
    parser.add_argument("--nb",   type=int, metavar="N",
                        help="Push notebook N as a private GPU kernel")
    parser.add_argument("--poll", action="store_true",
                        help="Wait for kernel to finish (implies watching)")
    parser.add_argument("--pull", action="store_true",
                        help="Download outputs when done (implies --poll)")
    args = parser.parse_args()

    if not args.data and args.nb is None:
        parser.print_help()
        sys.exit(0)

    if args.pull:
        args.poll = True

    if args.data:
        push_data()

    if args.nb is not None:
        push_notebook(args.nb, poll=args.poll, pull=args.pull)
