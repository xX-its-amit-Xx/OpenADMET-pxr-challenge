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

CPU_THRESHOLD  = 70   # % — abort if above after grace period
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

        meta = {
            "title":    "PXR Challenge Data",
            "id":       DATASET_REF,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (tmp / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

        print(f"[data] {n_pq} parquets · {n_csv} CSVs · src/pxr -> {DATASET_REF}")

        if _dataset_exists():
            _run(["datasets", "version", "-p", str(tmp), "-m", "auto-sync from kaggle_push.py"])
            print(f"[data] Dataset version updated: {DATASET_REF}")
        else:
            _run(["datasets", "create", "-p", str(tmp)])
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
    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Kaggle: wire up src/pxr from dataset\n",
            "import sys, os\n",
            f"sys.path.insert(0, f'/kaggle/input/{DATASET_SLUG}/src')\n",
            "os.environ.setdefault('PXR_DATA_ROOT', f'/kaggle/input/{DATASET_SLUG}')\n",
        ],
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

        meta = {
            "id":               f"{USERNAME}/{kernel_slug}",
            "title":            f"PXR Challenge nb{nb_num:02d}",
            "code_file":        nb_path.name,
            "language":         "python",
            "kernel_type":      "notebook",
            "is_private":       True,
            "enable_gpu":       True,
            "enable_internet":  True,
            "dataset_sources":  [DATASET_REF],
            "competition_sources": [],
            "kernel_sources":   [],
        }
        (tmp / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

        print(f"[nb] Pushing {nb_path.name} -> {USERNAME}/{kernel_slug} (GPU T4)")
        _run(["kernels", "push", "-p", str(tmp)])
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
