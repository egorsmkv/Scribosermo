#!/bin/sh
#SBATCH --partition=small-cpu # Name of cluster partition; default: big-cpu
#SBATCH --job-name DSGS # Job Name
#SBATCH --cpus-per-task 8
#SBATCH --ntasks 1
#SBATCH --mem 16000
#SBATCH --time=1000:00:00 # Time after which the job will be aborted
#
#
# Actual singularity call, mounted folder and call to script
singularity exec \
  --bind ~/checkpoints/:/checkpoints/ \
  --bind /cfs/share/cache/db_xds/data_original/:/data_original/ \
  --bind /cfs/share/cache/db_xds/data_prepared/:/data_prepared/ \
  --bind ~/Scribosermo/:/Scribosermo/ \
  --bind ~/corcua/:/corcua/ \
  ~/images/scribosermo.sif \
  /bin/bash -c 'python3 /Scribosermo/preprocessing/noise_to_csv.py'
