#!/bin/sh
#SBATCH --partition=gpu # Name of cluster partition; default: big-cpu
#SBATCH --gres=gpu:8 # Number of GPUs to allocate
#SBATCH --job-name DSG # Job Name
#SBATCH --cpus-per-task 64
#SBATCH --ntasks 1
#SBATCH --mem 262144
#SBATCH --time=1000:00:00 # Time after which the job will be aborted
#
# Actual singularity call with nvidia capabilities, amounted folder and call to script
singularity exec --nv \
  -B /cfs/share/cache/db_xds/checkpoints/:/DeepSpeech/checkpoints/ \
  -B /cfs/share/cache/db_xds/data_original/:/DeepSpeech/data_original/ \
  -B /cfs/share/cache/db_xds/data_prepared/:/DeepSpeech/data_prepared/ \
  -B /cfs/share/cache/db_xds/deepspeech-german/:/DeepSpeech/deepspeech-german/ \
  -B /cfs/share/cache/db_xds/DeepSpeech/evaluate.py:/DeepSpeech/evaluate.py \
  -B /cfs/share/cache/db_xds/DeepSpeech/DeepSpeech.py:/DeepSpeech/DeepSpeech.py \
  /cfs/share/cache/db_xds/images/deep_speech_german.sif \
  /bin/bash deepspeech-german/training/train.sh checkpoints/voxforge/ data_prepared/voxforge/train_azce.csv data_prepared/voxforge/dev_azce.csv data_prepared/voxforge/test_azce.csv 1 --
