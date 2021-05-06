#!/bin/sh
#SBATCH --partition=gpu # Name of cluster partition; default: big-cpu
#SBATCH --gres=gpu:2 # Number of GPUs to allocate
#SBATCH --job-name SST # Job Name
#SBATCH --cpus-per-task 20
#SBATCH --ntasks 1
#SBATCH --mem 120000
#SBATCH --time=1000:00:00 # Time after which the job will be aborted
#
#
# Actual singularity call with nvidia capabilities, mounted folder and call to script
singularity exec \
  --nv \
  --bind ~/checkpoints/:/checkpoints/ \
  --bind /cfs/share/cache/db_xds/data_original/:/data_original/ \
  --bind /cfs/share/cache/db_xds/data_prepared/:/data_prepared/ \
  --bind ~/Scribosermo/:/Scribosermo/ \
  --bind ~/corcua/:/corcua/ \
  ~/images/scribosermo.sif \
  /bin/bash -c 'export TF_GPU_THREAD_MODE=gpu_private && python3 /Scribosermo/training/run_train.py'
