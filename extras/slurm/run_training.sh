#!/bin/sh
#SBATCH --partition=gpu # Name of cluster partition; default: big-cpu
#SBATCH --gres=gpu:2 # Number of GPUs to allocate
#SBATCH --job-name DSGT # Job Name
#SBATCH --cpus-per-task 18
#SBATCH --ntasks 1
#SBATCH --mem 128000
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
  /cfs/share/cache/db_xds/images/scribosermo.sif \
  /bin/bash -c 'python3 /Scribosermo/training/run_train.py'
