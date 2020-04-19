#! /bin/bash

# Add "--gpus all" to use with docker + gpu (its not working with podman)

podman run \
  --network host \
  --name dsg_container \
  --rm \
  --mount type=bind,src="$(pwd)"/deepspeech-german/,dst=/DeepSpeech/deepspeech-german/ \
  --mount type=bind,src="$(pwd)"/checkpoints/,dst=/DeepSpeech/checkpoints/ \
  --mount type=bind,src="$(pwd)"/data_original/,dst=/DeepSpeech/data_original/ \
  --mount type=bind,src="$(pwd)"/data_prepared/,dst=/DeepSpeech/data_prepared/ \
  -it deep_speech_german_slurm \
  bash
