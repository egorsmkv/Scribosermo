#! /bin/bash

LANGUAGE="de"

CHECKPOINT_DIR=${1:-"/DeepSpeech/checkpoints/${LANGUAGE}/voxforge/"}
TRAIN_FILE=${2:-"/DeepSpeech/data_prepared/${LANGUAGE}/voxforge/train_azce.csv"}
DEV_FILE=${3:-"/DeepSpeech/data_prepared/${LANGUAGE}/voxforge/dev_azce.csv"}
TEST_FILE=${4:-"/DeepSpeech/data_prepared/${LANGUAGE}/voxforge/test_azce.csv"}

DELETE_OLD_CHECKPOINTS=${5:-0}
START_FROM_CHECKPOINT=${6:-"/DeepSpeech/checkpoints/deepspeech-0.7.3-checkpoint/"}

BATCH_SIZE=24
USE_AUGMENTATION=1
NOISE_FILE="/DeepSpeech/data_prepared/noise/train.csv"

if [[ "${DELETE_OLD_CHECKPOINTS}" == "1" ]] || [[ "${START_FROM_CHECKPOINT}" != "--" ]]; then
  rm -rf ${CHECKPOINT_DIR}
  mkdir -p ${CHECKPOINT_DIR}
fi

if [[ "${START_FROM_CHECKPOINT}" != "--" ]]; then
  cp -a ${START_FROM_CHECKPOINT}"." ${CHECKPOINT_DIR}
fi

if [[ "${LANGUAGE}" == "de" ]]; then
  DROP_SOURCE_LAYERS=0
else
  DROP_SOURCE_LAYERS=1
fi

#if [[ "${USE_AUGMENTATION}" == "1" ]]; then
#    AUG_AUDIO="--augment volume[p=0.1,dbfs=-10:-40] \
#      --augment pitch[p=0.1,pitch=1.1~0.95] \
#      --augment tempo[p=0.1,factor=1.25~0.75]"
#    AUG_ADD_DROP="--augment dropout[p=0.1,rate=0.05] \
#      --augment add[p=0.1,domain=signal,stddev=0~0.5] \
#      --augment multiply[p=0.1,domain=features,stddev=0~0.5]"
#    AUG_FREQ_TIME="--augment frequency_mask[p=0.1,n=1:3,size=1:5] \
#      --augment time_mask[p=0.1,domain=signal,n=3:10~2,size=50:100~40]"
#    AUG_EXTRA="--augment reverb[p=0.1,delay=50.0~30.0,decay=10.0:2.0~1.0] \
#      --augment resample[p=0.1,rate=12000:8000~4000] \
#      --augment codec[p=0.1,bitrate=48000:16000]"
#    AUG_SPEECH="--augment overlay[p=0.3,source=$TRAIN_FILE,layers=10:1,snr=50:20~9]"
#    AUG_NOISE="--augment overlay[p=0.5,source=$NOISE_FILE,layers=2:1,snr=50:20~6]"

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
  AUG_NOISE="--augment overlay[p=0.5,source=$NOISE_FILE,layers=2:1,snr=18:9~6]"

  #  Easy disabling of single flags only
  #  AUG_AUDIO=""
  #  AUG_ADD_DROP=""
  #  AUG_FREQ_TIME=""
  #  AUG_EXTRA=""
  #  AUG_SPEECH=""
  #  AUG_NOISE=""
else
  AUG_AUDIO=""
  AUG_ADD_DROP=""
  AUG_FREQ_TIME=""
  AUG_EXTRA=""
  AUG_SPEECH=""
  AUG_NOISE=""
fi

DSARGS="--train_files ${TRAIN_FILE} \
        --dev_files ${DEV_FILE} \
        --test_files ${TEST_FILE} \
        --scorer /DeepSpeech/data_prepared/texts/${LANGUAGE}/kenlm_az.scorer \
        --alphabet_config_path /DeepSpeech/deepspeech-german/data/alphabet_${LANGUAGE}.txt \
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
        --drop_source_layers ${DROP_SOURCE_LAYERS} \
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
