#!/bin/sh
#SBATCH --partition=gpu # Name of cluster partition; default: big-cpu
#SBATCH --gres=gpu:4 # Number of GPUs to allocate
#SBATCH --job-name DSGT # Job Name
#SBATCH --cpus-per-task 36
#SBATCH --ntasks 1
#SBATCH --mem 128000
#SBATCH --time=1000:00:00 # Time after which the job will be aborted
#
#
# Actual singularity call with nvidia capabilities, mounted folder and call to script
singularity exec \
  --nv \
  --bind ~/checkpoints/:/DeepSpeech/checkpoints/ \
  --bind /cfs/share/cache/db_xds/data_original/:/DeepSpeech/data_original/ \
  --bind /cfs/share/cache/db_xds/data_prepared/:/DeepSpeech/data_prepared/ \
  --bind ~/deepspeech-polyglot/:/DeepSpeech/deepspeech-polyglot/ \
  /cfs/share/cache/db_xds/images/deepspeech_polyglot.sif \
  /bin/bash -c 'chmod +x /DeepSpeech/deepspeech-polyglot/training/train.sh && /DeepSpeech/deepspeech-polyglot/training/train.sh'
