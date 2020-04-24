#! /bin/bash

CHECKPOINT_DIR=${1:-"/DeepSpeech/checkpoints/voxforge/"}
TRAIN_FILE=${2:-"/DeepSpeech/data_prepared/voxforge/train_azce.csv"}
DEV_FILE=${3:-"/DeepSpeech/data_prepared/voxforge/dev_azce.csv"}
TEST_FILE=${4:-"/DeepSpeech/data_prepared/voxforge/test_azce.csv"}

DELETE_OLD_CHECKPOINTS=${5:-0}
START_FROM_CHECKPOINT=${6:-"/DeepSpeech/checkpoints/deepspeech-0.6.0-checkpoint/"}

BATCH_SIZE=60
USE_AUGMENTATION=1

if [[ "${DELETE_OLD_CHECKPOINTS}" == "1" ]] || [[ "${START_FROM_CHECKPOINT}" != "--" ]]; then
  rm -rf ${CHECKPOINT_DIR}
  mkdir -p ${CHECKPOINT_DIR}
fi

if [[ "${START_FROM_CHECKPOINT}" != "--" ]]; then
  cp -a ${START_FROM_CHECKPOINT}"." ${CHECKPOINT_DIR}
fi

if [[ "${USE_AUGMENTATION}" == "1" ]]; then
  AUG_PITCH_TEMPO="--augmentation_pitch_and_tempo_scaling \
                   --augmentation_pitch_and_tempo_scaling_min_pitch 0.98 \
                   --augmentation_pitch_and_tempo_scaling_max_pitch 1.1 \
                   --augmentation_pitch_and_tempo_scaling_max_tempo 1.2"
  AUG_ADD_DROP="--data_aug_features_additive 0.2 \
                --augmentation_spec_dropout_keeprate 0.95"
  AUG_NOISE="--train_augmentation_noise_files /DeepSpeech/data_prepared/voxforge/train_azce.csv \
             --dev_augmentation_noise_files /DeepSpeech/data_prepared/voxforge/dev_azce.csv \
             --test_augmentation_noise_files /DeepSpeech/data_prepared/voxforge/test_azce.csv \
             --train_augmentation_speech_files /DeepSpeech/data_prepared/voxforge/train_azce.csv \
             --dev_augmentation_speech_files /DeepSpeech/data_prepared/voxforge/dev_azce.csv \
             --test_augmentation_speech_files /DeepSpeech/data_prepared/voxforge/test_azce.csv \
             --audio_aug_max_audio_dbfs -5 \
             --audio_aug_min_audio_dbfs -40 \
             --audio_aug_min_noise_snr_db 3 \
             --audio_aug_max_noise_snr_db 30 \
             --audio_aug_min_speech_snr_db 10 \
             --audio_aug_max_speech_snr_db 30 \
             --audio_aug_limit_audio_peak_dbfs 3.0 \
             --audio_aug_limit_noise_peak_dbfs 1.0 \
             --audio_aug_limit_speech_peak_dbfs 1.0 \
             --audio_aug_min_n_noises 0 \
             --audio_aug_max_n_noises 2 \
             --audio_aug_min_n_speakers 0 \
             --audio_aug_max_n_speakers 2"
  AUG_FREQ_TIME="--augmentation_freq_and_time_masking True"

  #  Easy disabling of single flags only
  #  AUG_ADD_DROP=""
  #  AUG_NOISE=""
  #  AUG_FREQ_TIME=""
  #  AUG_PITCH_TEMPO=""
else
  AUG_PITCH_TEMPO=""
  AUG_ADD_DROP=""
  AUG_NOISE=""
  AUG_FREQ_TIME=""
fi

DSARGS="--train_files ${TRAIN_FILE} \
        --dev_files ${DEV_FILE} \
        --test_files ${TEST_FILE} \
        --scorer /DeepSpeech/data_prepared/lm/kenlm_az.scorer \
        --alphabet_config_path /DeepSpeech/deepspeech-german/data/alphabet_az.txt \
        --test_batch_size ${BATCH_SIZE} \
        --train_batch_size ${BATCH_SIZE} \
        --dev_batch_size ${BATCH_SIZE} \
        --epochs 100 \
        --early_stop True \
        --es_epochs 7 \
        --reduce_lr_on_plateau True \
        --plateau_epochs 3 \
        --force_initialize_learning_rate True \
        --learning_rate 0.0001 \
        --dropout_rate 0.25 \
        --use_allow_growth  \
        --drop_source_layers 0 \
        --train_cudnn \
        --export_dir ${CHECKPOINT_DIR} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --summary_dir ${CHECKPOINT_DIR} \
        --max_to_keep 3 \
        --review_audio_steps 7 \
        ${AUG_FREQ_TIME} \
        ${AUG_PITCH_TEMPO} \
        ${AUG_ADD_DROP} \
        ${AUG_NOISE}"

echo ""
echo ""
echo "Running training with arguments:" ${DSARGS}
echo ""
echo ""
python3 -u /DeepSpeech/DeepSpeech.py ${DSARGS}

# Convert output graph for inference
if [[ -f ${CHECKPOINT_DIR}"output_graph.pb" ]]; then
  echo ""
  echo "Converting output graph for inference:"
  echo ""
  /DeepSpeech/convert_graphdef_memmapped_format --in_graph=${CHECKPOINT_DIR}"output_graph.pb" --out_graph=${CHECKPOINT_DIR}"output_graph.pbmm"
fi
