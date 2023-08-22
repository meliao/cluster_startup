#!/bin/bash

#SBATCH --job-name=2023-08-24_long_model_train
#SBATCH --partition=general
#SBATCH --gres=gpu:1
# TODO: Delete this line and choose files for the output stdout and stderr logs
#SBATCH --output=
#SBATCH --error=


# This command logs some information which is helpful for debugging
echo "`date` Starting Job"
echo "SLURM Info: Job name:${SLURM_JOB_NAME}"
echo "    JOB ID: ${SLURM_JOB_ID}"
echo "    Host list: ${SLURM_JOB_NODELIST}"
echo "    CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

# This command tells which installation of python the script is calling. Also helpful for debugging.
which python

python train_network.py \
# TODO: delete this line and fill out the missing parameters
-n_epochs XXX \
-n_epochs_per_log XXX \ 
-L_vals XXX \
-train_results_dir results_long \
-models_dir models_long