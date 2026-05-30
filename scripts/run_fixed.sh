#!/bin/bash
LOG=data/processed/nb_fixed_run.log
echo "Starting at $(date -u)" > $LOG
for s in nb293_conformal_stacking nb290_mmp_transform_model nb297_pysr_symbolic_residual nb292_molrule_loss; do
  echo "----------" >> $LOG
  echo "[$(date -u)] starting $s" >> $LOG
  timeout 1200 python scripts/${s}.py >> $LOG 2>&1
  rc=$?
  echo "[$(date -u)] finished $s (rc=$rc)" >> $LOG
done
echo "All done at $(date -u)" >> $LOG
