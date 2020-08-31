#! /bin/bash

LANGUAGE="de"

CHECKPOINT_DIR=${1:-"/DeepSpeech/checkpoints/${LANGUAGE}/voxforge/"}
TRAIN_FILE=${2:-"/DeepSpeech/data_prepared/${LANGUAGE}/voxforge/train_azce.csv"}
DEV_FILE=${3:-"/DeepSpeech/data_prepared/${LANGUAGE}/voxforge/dev_azce.csv"}
TEST_FILE=${4:-"/DeepSpeech/data_prepared/${LANGUAGE}/voxforge/test_azce.csv"}

DELETE_OLD_CHECKPOINTS=${5:-0}
START_FROM_CHECKPOINT=${6:-"/DeepSpeech/checkpoints/deepspeech-0.8.1-checkpoint/"}

BATCH_SIZE=36
USE_AUGMENTATION=1
FREEZE_SOURCE_LAYERS=0
#LOAD_FROZEN_GRAPH="--load_frozen_graph True"
LOAD_FROZEN_GRAPH=""
NOISE_FILE="/DeepSpeech/data_prepared/noise/train.csv"

if [[ "${DELETE_OLD_CHECKPOINTS}" == "1" ]] || [[ "${START_FROM_CHECKPOINT}" != "--" ]]; then
  rm -rf "${CHECKPOINT_DIR}"
  mkdir -p "${CHECKPOINT_DIR}"
fi

if [[ "${START_FROM_CHECKPOINT}" != "--" ]]; then
  cp -a "${START_FROM_CHECKPOINT}." "${CHECKPOINT_DIR}"
fi

if [[ "${LANGUAGE}" == "de" ]] || [[ "${LANGUAGE}" == "it" ]]; then
  DROP_SOURCE_LAYERS=0
else
  DROP_SOURCE_LAYERS=1
fi

if [[ ${DROP_SOURCE_LAYERS} != 0 ]]; then
  FREEZE_SOURCE_LAYERS=${DROP_SOURCE_LAYERS}
fi

if [[ "${USE_AUGMENTATION}" == "1" ]]; then
    AUG_AUDIO="--augmentation_pitch_and_tempo_scaling \
      --augmentation_pitch_and_tempo_scaling_min_pitch 0.95 \
     --augmentation_pitch_and_tempo_scaling_max_pitch 1.1 \
     --augmentation_pitch_and_tempo_scaling_max_tempo 1.25"
    AUG_ADD_DROP="--data_aug_features_additive 0.25 \
      --augmentation_spec_dropout_keeprate 0.95"
    AUG_FREQ_TIME="--augmentation_freq_and_time_masking True"
    AUG_EXTRA="--augment reverb[p=0.1,delay=50.0~30.0,decay=10.0:2.0~1.0] \
      --augment gaps[p=0.05,n=1:3~2,size=10:100] \
      --augment resample[p=0.1,rate=12000:8000~4000] \
      --augment codec[p=0.1,bitrate=48000:16000] \
      --augment volume[p=0.1,dbfs=-10:-40]"
    AUG_SPEECH="--augment overlay[p=0.3,source=$TRAIN_FILE,layers=7:1,snr=30:15~9]"
    AUG_NOISE="--augment overlay[p=0.5,source=$NOISE_FILE,layers=2:1,snr=18:9~5]"
    CACHING="--feature_cache /tmp/ \
      --augmentations_per_epoch 10"


  #  Easy disabling of single flags only
  #  AUG_AUDIO=""
  #  AUG_ADD_DROP=""
  #  AUG_FREQ_TIME=""
  #  AUG_EXTRA=""
    AUG_SPEECH=""
  #  AUG_NOISE=""
else
  AUG_AUDIO=""
  AUG_ADD_DROP=""
  AUG_FREQ_TIME=""
  AUG_EXTRA=""
  AUG_SPEECH=""
  AUG_NOISE=""
  CACHING=""
fi

DSARGS="--train_files ${TRAIN_FILE} \
        --dev_files ${DEV_FILE} \
        --test_files ${TEST_FILE} \
        --scorer /DeepSpeech/data_prepared/texts/${LANGUAGE}/kenlm_${LANGUAGE}.scorer \
        --alphabet_config_path /DeepSpeech/deepspeech-polyglot/data/alphabet_${LANGUAGE}.txt \
        --test_batch_size ${BATCH_SIZE} \
        --train_batch_size ${BATCH_SIZE} \
        --dev_batch_size ${BATCH_SIZE} \
        --epochs 1000 \
        --early_stop True \
        --es_epochs 7 \
        --reduce_lr_on_plateau True \
        --plateau_epochs 3 \
        --force_initialize_learning_rate True \
        --learning_rate 0.0001 \
        --dropout_rate 0.25 \
        --use_allow_growth  \
        --drop_source_layers ${DROP_SOURCE_LAYERS} \
        --freeze_source_layers ${FREEZE_SOURCE_LAYERS} \
        ${LOAD_FROZEN_GRAPH} \
        --train_cudnn \
        --export_dir ${CHECKPOINT_DIR} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --summary_dir ${CHECKPOINT_DIR} \
        --max_to_keep 3 \
        ${AUG_AUDIO} \
        ${AUG_ADD_DROP} \
        ${AUG_FREQ_TIME} \
        ${AUG_EXTRA} \
        ${AUG_SPEECH} \
        ${AUG_NOISE} \
        ${CACHING}"

echo ""
echo ""
echo "Running training with arguments: ${DSARGS}"
echo ""
echo ""
/bin/bash -c "python3 -u /DeepSpeech/DeepSpeech.py ${DSARGS}"

# Convert output graph for inference
echo ""
echo "Converting output graph for inference:"
echo ""
if [[ -f ${CHECKPOINT_DIR}"best_dev_checkpoint" ]]; then
  python3 -u /DeepSpeech/DeepSpeech.py --checkpoint_dir "${CHECKPOINT_DIR}" \
    --scorer /DeepSpeech/data_prepared/texts/${LANGUAGE}/kenlm_${LANGUAGE}.scorer \
    --alphabet_config_path /DeepSpeech/deepspeech-polyglot/data/alphabet_${LANGUAGE}.txt \
    --export_tflite --export_dir "${CHECKPOINT_DIR}" \
    && mv "${CHECKPOINT_DIR}output_graph.tflite" "${CHECKPOINT_DIR}output_graph_${LANGUAGE}.tflite"
fi
echo ""
if [[ -f ${CHECKPOINT_DIR}"output_graph.pb" ]]; then
  /DeepSpeech/convert_graphdef_memmapped_format --in_graph="${CHECKPOINT_DIR}output_graph.pb" \
    --out_graph="${CHECKPOINT_DIR}output_graph_${LANGUAGE}.pbmm"
fi

echo ""
echo "FINISHED TRAINING"
