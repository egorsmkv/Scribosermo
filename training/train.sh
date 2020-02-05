#! /bin/bash

CHECKPOINT_DIR=${1:-"checkpoints/voxforge/"}
TRAIN_FILE=${2:-"data_prepared/voxforge/train_az.csv"}
DEV_FILE=${3:-"data_prepared/voxforge/dev_az.csv"}
TEST_FILE=${4:-"data_prepared/voxforge/test_az.csv"}

DELETE_OLD_CHECKPOINTS=${5:-0}
START_FROM_ENGLISH_CHECKPOINT=${6:-0}

BATCH_SIZE=12
USE_AUGMENTATION=1

if [ "${DELETE_OLD_CHECKPOINTS}" = "1" ] || [ "${START_FROM_ENGLISH_CHECKPOINT}" = "1" ]; then
    rm -rf ${CHECKPOINT_DIR}
fi;

if [ "${START_FROM_ENGLISH_CHECKPOINT}" = "1" ]; then
    cp -r "checkpoints/deepspeech-0.6.0-checkpoint/" ${CHECKPOINT_DIR}
fi;

if [ "${USE_AUGMENTATION}" = "1" ]; then
    AUG_AUDIO="--data_aug_features_additive 0.2 \
                 --data_aug_features_multiplicative 0.2 \
                 --augmentation_speed_up_std 0.2"
    AUG_FREQ_TIME="--augmentation_freq_and_time_masking \
                     --augmentation_freq_and_time_masking_freq_mask_range 5 \
                     --augmentation_freq_and_time_masking_number_freq_masks 3 \
                     --augmentation_freq_and_time_masking_time_mask_range 2 \
                     --augmentation_freq_and_time_masking_number_time_masks 3"
    AUG_PITCH_TEMPO="--augmentation_pitch_and_tempo_scaling \
                       --augmentation_pitch_and_tempo_scaling_min_pitch 0.95 \
                       --augmentation_pitch_and_tempo_scaling_max_pitch 1.2 \
                       --augmentation_pitch_and_tempo_scaling_max_tempo 1.2"
    AUG_SPEC_DROP="--augmentation_spec_dropout_keeprate 0.9"
    AUG_NOISE="--audio_aug_mix_noise_walk_dirs data_original/noise/"
else
    AUG_AUDIO=""
    AUG_FREQ_TIME=""
    AUG_PITCH_TEMPO=""
    AUG_SPEC_DROP=""
    AUG_NOISE=""
fi;

DSARGS="--train_files ${TRAIN_FILE} \
        --dev_files ${DEV_FILE} \
        --test_files ${TEST_FILE} \
        --alphabet_config_path deepspeech-german/data/alphabet_az.txt \
        --lm_trie_path data_prepared/trie_az \
        --lm_binary_path data_prepared/lm_az.binary \
        --test_batch_size ${BATCH_SIZE} \
        --train_batch_size ${BATCH_SIZE} \
        --dev_batch_size ${BATCH_SIZE} \
        --epochs 100 \
        --es_steps 4 \
        --learning_rate 0.0001 \
        --dropout_rate 0.25 \
        --use_allow_growth  \
        --use_cudnn_rnn \
        --export_dir ${CHECKPOINT_DIR} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        ${AUG_AUDIO} \
        ${AUG_FREQ_TIME} \
        ${AUG_PITCH_TEMPO} \
        ${AUG_SPEC_DROP} \
        --show_progressbar True \
        ${AUG_NOISE}"

echo ""
echo ""
echo "Running training with arguments:" ${DSARGS}
echo ""
echo ""
python3 -u DeepSpeech.py ${DSARGS}
