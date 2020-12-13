#! /bin/bash

# Fixing issue #3088 after few training steps
export TF_CUDNN_RESET_RND_GEN_STATE=1

CHECKPOINT_DIR=${1:-"/DeepSpeech/checkpoints/${LANGUAGE}/voxforge/"}
TRAIN_FILE=${2:-"/DeepSpeech/data_prepared/${LANGUAGE}/voxforge/train_azce.csv"}
DEV_FILE=${3:-"/DeepSpeech/data_prepared/${LANGUAGE}/voxforge/dev_azce.csv"}
TEST_FILE=${4:-"/DeepSpeech/data_prepared/${LANGUAGE}/voxforge/test_azce.csv"}

DELETE_OLD_CHECKPOINTS=${5:-0}
START_FROM_CHECKPOINT=${6:-"/DeepSpeech/checkpoints/deepspeech-0.8.1-checkpoint/"}

BATCH_SIZE=24
# Training will normally be stopped by early-stopping but augmentations use this too,
# so try to keep this close the the estimated stopping epoch
MAX_EPOCHS=35

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
    AUG_AUDIO="--augment volume[p=0.1,dbfs=-10:-40] \
      --augment pitch[p=0.1,pitch=1.05~0.1] \
      --augment tempo[p=0.1,factor=1.1~0.25]"
    AUG_ADD_DROP="--augment dropout[p=0.1,rate=0.05] \
      --augment add[p=0.1,domain=signal,stddev=0~0.5]"
    AUG_FREQ_TIME="--augment frequency_mask[p=0.1,n=1:3,size=1:5] \
      --augment time_mask[p=0.1,domain=signal,n=3:10~2,size=50:100~40]"
    AUG_EXTRA="--augment reverb[p=0.1,delay=50.0~30.0,decay=10.0:2.0~1.0] \
      --augment resample[p=0.1,rate=12000:8000~4000] \
      --augment codec[p=0.1,bitrate=48000:16000]"
    AUG_SPEECH="--augment overlay[p=0.3,source=$TRAIN_FILE,layers=7:1,snr=30:15~9]"
    AUG_NOISE="--augment overlay[p=0.5,source=$NOISE_FILE,layers=2:1,snr=18:9~5]"

# Got error: AttributeError: module 'tensorflow._api.v1.io.gfile' has no attribute 'remove_remote'
#     CACHING="--feature_cache /tmp/ \
#       --cache_for_epochs 2"
   CACHING=""

  #  Easy disabling of single flags only
  #  AUG_AUDIO=""
  #  AUG_ADD_DROP=""
  #  AUG_FREQ_TIME=""
  #  AUG_EXTRA=""
   AUG_SPEECH=""
   AUG_NOISE=""
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
        --epochs ${MAX_EPOCHS} \
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
    --scorer "/DeepSpeech/data_prepared/texts/${LANGUAGE}/kenlm_${LANGUAGE}.scorer" \
    --alphabet_config_path "/DeepSpeech/deepspeech-polyglot/data/alphabet_${LANGUAGE}.txt" \
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
