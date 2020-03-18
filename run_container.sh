#! /bin/bash

docker run \
  --gpus all \
  --network host \
  --name dsg_container \
  --rm \
  --mount type=bind,src="$(pwd)"/deepspeech-german/,dst=/DeepSpeech/deepspeech-german/ \
  --mount type=bind,src="$(pwd)"/checkpoints/,dst=/DeepSpeech/checkpoints/ \
  --mount type=bind,src="$(pwd)"/data_original/,dst=/DeepSpeech/data_original/ \
  --mount type=bind,src="$(pwd)"/data_prepared/,dst=/DeepSpeech/data_prepared/ \
  --mount type=bind,src="$(pwd)"/DeepSpeech/DeepSpeech.py,dst=/DeepSpeech/DeepSpeech.py \
  --mount type=bind,src="$(pwd)"/DeepSpeech/evaluate.py,dst=/DeepSpeech/evaluate.py \
  --mount type=bind,src="$(pwd)"/DeepSpeech/util/feeding.py,dst=/DeepSpeech/util/feeding.py \
  --mount type=bind,src="$(pwd)"/DeepSpeech/util/evaluate_tools.py,dst=/DeepSpeech/util/evaluate_tools.py \
  --mount type=bind,src="$(pwd)"/DeepSpeech/util/flags.py,dst=/DeepSpeech/util/flags.py \
  --mount type=bind,src="$(pwd)"/DeepSpeech/data/lm/generate_lm.py,dst=/DeepSpeech/data/lm/generate_lm.py \
  --mount type=bind,src="$(pwd)"/DeepSpeech/data/lm/generate_package.py,dst=/DeepSpeech/data/lm/generate_package.py \
  -it deep_speech_german \
  bash
