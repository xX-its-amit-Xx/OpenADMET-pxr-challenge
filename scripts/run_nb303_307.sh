#!/bin/bash
LOG=data/processed/nb303_307_run.log
echo "Starting at $(date -u)" > $LOG

# Light/medium first (A, C, E)
ORDER=(
  nb305_mope       # CPU LGBM, ~5 min
  nb307_soci       # CPU pairwise scan, ~30 min for 4139 x 4139
  nb303_ts_ada     # PyTorch DANN 5-fold, ~30 min CPU
  nb304_cel        # network lookup, depends on cache, ~5 min
  nb306_ce_psmim   # PyTorch MIL + 8 conformers x compound, ~45 min CPU
)

for s in "${ORDER[@]}"; do
  echo "----------" >> $LOG
  echo "[$(date -u)] starting $s" >> $LOG
  timeout 3600 python scripts/${s}.py >> $LOG 2>&1
  rc=$?
  echo "[$(date -u)] finished $s (rc=$rc)" >> $LOG
done
echo "All done at $(date -u)" >> $LOG
