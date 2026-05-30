#!/bin/bash
# Queued runner: nb285-nb301 sequential with logging.
# Heavy scripts go later; lighter ones first.
LOG=data/processed/nb285_to_301_run.log
echo "Starting at $(date -u)" > $LOG

ORDER=(
  nb292_molrule_loss
  nb293_conformal_stacking
  nb294_heteroscedastic_nll
  nb295_rag_qsar
  nb297_pysr_symbolic_residual
  nb298_pose_ifp_gbsa
  nb289_test_time_finetune
  nb290_mmp_transform_model
  nb288_gp_uncertainty
  nb286_smiles_quality_prep
  nb291_biotype_3d_atom_features
  nb296_tda_persistent_homology
  nb300_diffusion_counterfactual
  nb299_nr_clip
  nb285_se3_egnn
  nb287_evoformer_pxr
  nb301_denoising_3d_backbone
)

for s in "${ORDER[@]}"; do
  echo "----------" >> $LOG
  echo "[$(date -u)] starting $s" >> $LOG
  timeout 1800 python scripts/${s}.py >> $LOG 2>&1
  rc=$?
  echo "[$(date -u)] finished $s (rc=$rc)" >> $LOG
done
echo "All done at $(date -u)" >> $LOG
