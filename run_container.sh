#! /bin/bash

docker run \
  --network host --rm -it \
  --gpus all --privileged --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  --volume `pwd`/DeepSpeech/training/:/DeepSpeech/training/ \
  --volume `pwd`/deepspeech-polyglot/:/DeepSpeech/deepspeech-polyglot/ \
  --volume `pwd`/checkpoints/:/DeepSpeech/checkpoints/ \
  --volume `pwd`/data_original/:/DeepSpeech/data_original/ \
  --volume `pwd`/data_prepared/:/DeepSpeech/data_prepared/ \
  deepspeech_polyglot \
  bash
