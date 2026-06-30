#!/usr/bin/env bash
# PXR Challenge — Codespace bootstrap (the D:-full escape hatch).
# Re-assembles the gitignored data + deps so the repo is runnable in a fresh Codespace.
# Codespaces gives 32GB+ disk, so there is no D:-full problem here.
set -uo pipefail
echo "==================== PXR Codespace bootstrap ===================="

# ---- 0. caches into the workspace volume (ample disk) ----
export HF_HOME="$PWD/.cache/hf"; export PIP_CACHE_DIR="$PWD/.cache/pip"
mkdir -p .cache/hf .cache/pip data/raw data/external/pdb64_structures

# ---- 1. python deps ----
echo "--- installing deps (uv) ---"
pip install -q uv 2>/dev/null || true
if [ -f pyproject.toml ]; then uv sync 2>/dev/null || pip install -q -e . 2>/dev/null || true; fi
pip install -q numpy pandas scipy scikit-learn lightgbm rdkit gemmi py3Dmol matplotlib requests 2>/dev/null || true

# ---- 2. re-clone the gitignored challenge data (HF) ----
if [ ! -f data/raw/pxr-challenge_TRAIN.csv ]; then
  echo "--- cloning HF train/test dataset ---"
  git clone https://huggingface.co/datasets/openadmet/pxr-challenge-train-test data/raw \
    || echo "!! HF clone failed — clone data/raw manually (needs HF access)"
fi
[ -d tutorial ] || git clone https://github.com/OpenADMET/PXR-Challenge-Tutorial.git tutorial 2>/dev/null || true

# ---- 3. re-download the 64 PXR holo structures from RCSB (public) ----
PDBS="1m13 1nrl 1skx 2o9i 2qnv 3hvl 3r8d 4ny9 4s0t 4x1f 4x1g 4xhd 5a86 5x0r 6bns 6dup 6hj2 6hty 6nx1 6p2b 6s41 6tfi 6xp9 7ax9 7axa 7axb 7axc 7axd 7axe 7axf 7axg 7axh 7axi 7axj 7axk 7axl 7n2a 7rio 7riu 7riv 7yfk 8cct 8cf9 8ch8 8e3n 8eqz 8f5y 8fpe 8r00 8r81 8r82 8svo 8svp 8svq 8svr 8svs 8svt 8svx 8szv 9beq 9fzg 9fzh 9fzi 9fzj"
echo "--- fetching 64 PXR CIF structures from RCSB ---"
n=0
for id in $PDBS; do
  f="data/external/pdb64_structures/${id}.cif"
  [ -f "$f" ] || curl -sf "https://files.rcsb.org/download/${id}.cif" -o "$f" && n=$((n+1))
done
echo "    fetched/present: $(ls data/external/pdb64_structures/*.cif 2>/dev/null | wc -l)/64"

# ---- 4. kernel ----
python -m ipykernel install --user --name pxr-challenge --display-name "pxr-challenge" 2>/dev/null || true

echo "==================== bootstrap done ===================="
echo "STILL NEEDED (secrets, not in git):"
echo "  • Kaggle creds   -> ~/.kaggle/kaggle.json   (Codespaces > Settings > Secrets: KAGGLE_USERNAME/KAGGLE_KEY)"
echo "  • Gradio submit token for the activity/structure auto-submit scripts"
echo "  • data/processed/ derived artifacts (te_*.npy, oof, _audit_*) are NOT in git — re-run the notebooks"
echo "    or upload the needed .npy. The 64 structures + raw data above are sufficient for the MedChem report."
echo "TO RE-HOME CRONS here: run the 3 scripts under a scheduler (cron/systemd-timer); they are rate-limit-safe."
