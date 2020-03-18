#!/bin/sh
#SBATCH --partition=gpu # Name of cluster partition; default: big-cpu
#SBATCH --gres=gpu:8 # Number of GPUs to allocate
#SBATCH --job-name DSG # Job Name
#SBATCH --cpus-per-task 64
#SBATCH --ntasks 1
#SBATCH --mem 262144
#SBATCH --time=07:00:00 # Time after which the job will be aborted
#
# Actual singularity call with nvidia capabilities, amounted folder and the url to the docker container + python call and script
singularity exec --nv \
-B /cfs/share/cache/db_xds/checkpoints/:/DeepSpeech/checkpoints/ \
-B /cfs/share/cache/db_xds/data_original/:/DeepSpeech/data_original/ \
-B /cfs/share/cache/db_xds/data_prepared/:/DeepSpeech/data_prepared/ \
-B /cfs/share/cache/db_xds/tests/evaluate.py:/DeepSpeech/evaluate.py \
/cfs/share/cache/db_xds/images/mds05.sif \
python3 /DeepSpeech/DeepSpeech.py --test_files /DeepSpeech/data_prepared/tuda-voxforge-swc-mailabs-common_voice/test_mix.csv --checkpoint_dir /DeepSpeech/checkpoints/dsg05_models/checkpoints/ --alphabet_config_path /DeepSpeech/checkpoints/dsg05_models/alphabet.txt --lm_trie_path /DeepSpeech/checkpoints/dsg05_models/trie --lm_binary_path /DeepSpeech/checkpoints/dsg05_models/lm.binary --test_batch_size 36

