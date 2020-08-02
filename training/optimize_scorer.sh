#! /bin/bash

LANGUAGE="de"
BATCH_SIZE=24
TRIALS=800

CHECKPOINT_DIR=${1:-"/DeepSpeech/checkpoints/${LANGUAGE}/voxforge/"}
DEV_FILE=${2:-"/DeepSpeech/data_prepared/${LANGUAGE}/voxforge/dev_azce.csv"}

DSARGS="--test_files ${DEV_FILE} \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --scorer /DeepSpeech/data_prepared/texts/${LANGUAGE}/kenlm_${LANGUAGE}.scorer \
        --alphabet_config_path /DeepSpeech/deepspeech-polyglot/data/alphabet_${LANGUAGE}.txt \
        --test_batch_size ${BATCH_SIZE} \
        --n_trials ${TRIALS} \
        --use_allow_growth"

echo ""
echo ""
echo "Running optimization with arguments:" ${DSARGS}
echo ""
echo ""
python3 -u /DeepSpeech/lm_optimizer.py ${DSARGS}

echo ""
echo "FINISHED OPTIMIZATION"
