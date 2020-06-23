#! /bin/bash

CHECKPOINT_DIR=${1:-"/DeepSpeech/checkpoints/voxforge/"}
TEST_FILE=${2:-"/DeepSpeech/data_prepared/voxforge/test_azce.csv"}

BATCH_SIZE=24
NOISE_FILE="/DeepSpeech/data_prepared/noise/test.csv"

# Using speech or noise augmentation for testing requires the 'noiseaugmaster' branch container
AUG_SPEECH="--test_augmentation_speech_files ${TEST_FILE} \
            --audio_aug_min_speech_snr_db 9 \
            --audio_aug_max_speech_snr_db 30 \
            --audio_aug_limit_speech_peak_dbfs 1.0 \
            --audio_aug_min_n_speakers 0 \
            --audio_aug_max_n_speakers 3"
AUG_NOISE="--test_augmentation_noise_files ${NOISE_FILE} \
           --audio_aug_min_noise_snr_db 6 \
           --audio_aug_max_noise_snr_db 20 \
           --audio_aug_limit_noise_peak_dbfs 1.0 \
           --audio_aug_min_n_noises 1 \
           --audio_aug_max_n_noises 2"
#AUG_SPEECH=""
#AUG_NOISE=""

DSARGS="--test_files ${TEST_FILE} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --scorer /DeepSpeech/data_prepared/lm/kenlm_az.scorer \
        --alphabet_config_path /DeepSpeech/deepspeech-german/data/alphabet_az.txt \
        --test_batch_size ${BATCH_SIZE} \
        --use_allow_growth  \
        ${AUG_SPEECH} \
        ${AUG_NOISE}"

echo ""
echo ""
echo "Running test with arguments:" ${DSARGS}
echo ""
echo ""
python3 -u /DeepSpeech/DeepSpeech.py ${DSARGS}
