#!/bin/sh
#SBATCH --partition=gpu # Name of cluster partition; default: big-cpu
#SBATCH --gres=gpu:8 # Number of GPUs to allocate
#SBATCH --job-name DSG # Job Name
#SBATCH --cpus-per-task 64
#SBATCH --ntasks 1
#SBATCH --mem 262144
#SBATCH --time=1000:00:00 # Time after which the job will be aborted
#
#
# Actual singularity call with nvidia capabilities, amounted folder and call to script
singularity exec \
  --nv \
  --bind ~/checkpoints/:/DeepSpeech/checkpoints/ \
  --bind /cfs/share/cache/db_xds/data_original/:/DeepSpeech/data_original/ \
  --bind /cfs/share/cache/db_xds/data_prepared/:/DeepSpeech/data_prepared/ \
  --bind ~/deepspeech-german/:/DeepSpeech/deepspeech-german/ \
  /cfs/share/cache/db_xds/images/mds_slurm.sif \
  /bin/bash /DeepSpeech/deepspeech-german/training/train.sh /DeepSpeech/checkpoints/voxforge/ /DeepSpeech/data_prepared/voxforge/train_azce.csv /DeepSpeech/data_prepared/voxforge/dev_azce.csv /DeepSpeech/data_prepared/voxforge/test_azce.csv 1 --
