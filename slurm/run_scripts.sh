#!/bin/sh
#SBATCH --partition=small-cpu # Name of cluster partition; default: big-cpu
#SBATCH --job-name DSGS # Job Name
#SBATCH --ntasks 1
#SBATCH --time=1000:00:00 # Time after which the job will be aborted
#
#
# Actual singularity call, mounted folder and call to script
singularity exec \
  --bind ~/checkpoints/:/DeepSpeech/checkpoints/ \
  --bind /cfs/share/cache/db_xds/data_original/:/DeepSpeech/data_original/ \
  --bind /cfs/share/cache/db_xds/data_prepared/:/DeepSpeech/data_prepared/ \
  --bind ~/deepspeech-polyglot/:/DeepSpeech/deepspeech-polyglot/ \
  /cfs/share/cache/db_xds/images/deepspeech_polyglot.sif \
  /bin/bash -c 'python3 /DeepSpeech/deepspeech-polyglot/preprocessing/noise_to_csv.py'
