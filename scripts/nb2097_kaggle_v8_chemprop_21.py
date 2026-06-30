"""
nb2097 — Kaggle v8 chemprop 2.1.x dispatcher (Option C fallback if v7 fails)
=============================================================================
Generates Kaggle push payload for chemprop 2.1.x retrain in case v7 (2.0.x)
fails on the Kaggle kernel runtime. Targets:
  - chemprop==2.1.0
  - lightning>=2.2
  - torch CPU (Kaggle CPU runtime fallback) OR cu124 (P100 GPU)
  - PXR pEC50 + counter-assay multitask head

This script ONLY assembles the payload. Kaggle execution is gated by a check
of v7 status; if v7 is RUNNING or SUCCEEDED, this script writes the dispatch
file but does NOT push.

Output: data/processed/nb2097_kaggle_v8_payload.json
        scripts/kaggle/nb2097_v8_runner.py  (Kaggle kernel script)
"""
import json
from pathlib import Path

ROOT = Path("d:/Users/ashenoy00000/.windsurf/OpenADMET-pxr-challenge")
PROC = ROOT / "data" / "processed"
KAG = ROOT / "scripts" / "kaggle"
KAG.mkdir(exist_ok=True)

# Check v7 status
v7_status_path = PROC / "kaggle_v7_status.json"
v7_status = "UNKNOWN"
if v7_status_path.exists():
    v7_status = json.loads(v7_status_path.read_text()).get("status", "UNKNOWN")

payload = {
    "name": "pxr-challenge-chemprop-v8",
    "chemprop_version": "2.1.0",
    "lightning_version": ">=2.2",
    "torch_index": "https://download.pytorch.org/whl/cpu",
    "gpu_fallback_index": "https://download.pytorch.org/whl/cu124",
    "kernel_runtime": "cpu",  # safer than P100 for chemprop 2.1.x
    "tasks": ["pEC50", "pEC50_null"],
    "n_folds": 5,
    "scaffold_seed": 42,
    "hyperparams": {
        "depth": 3,
        "d_h": 300,
        "dropout": 0.1,
        "epochs": 50,
        "batch_size": 64,
        "lr": 1e-3,
    },
    "dispatch_gated_by_v7": True,
    "v7_status": v7_status,
    "should_dispatch": v7_status in ("FAILED", "UNKNOWN", "NOT_FOUND"),
}

(PROC / "nb2097_kaggle_v8_payload.json").write_text(json.dumps(payload, indent=2))

# Kaggle kernel runner stub (not pushed unless v7 fails)
runner = '''"""
nb2097 Kaggle v8 runner — chemprop 2.1.x retrain
Run via Kaggle dataset `knowledgegraphlover/pxr-challenge-data`
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "chemprop==2.1.0", "lightning>=2.2", "rdkit"])
# ... full training loop omitted for stub; will be filled when v7 fails
print("nb2097 v8 dispatcher stub — fill in when v7 verdict known")
'''
(KAG / "nb2097_v8_runner.py").write_text(runner)

print(json.dumps(payload, indent=2))
print(f"\nDISPATCH GATE: should_dispatch={payload['should_dispatch']}")
print(f"  v7 status read from {v7_status_path}: {v7_status}")
print(f"  Runner stub written: {KAG / 'nb2097_v8_runner.py'}")
