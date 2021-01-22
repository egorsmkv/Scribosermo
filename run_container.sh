#! /bin/bash

docker run --privileged --rm --network host -it \
  --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  --volume "$(pwd)"/deepspeech-polyglot/:/deepspeech-polyglot/ \
  --volume "$(pwd)"/corcua/:/corcua/ \
  --volume "$(pwd)"/checkpoints/:/checkpoints/ \
  --volume "$(pwd)"/data_original/:/data_original/ \
  --volume "$(pwd)"/data_prepared/:/data_prepared/ \
  dspol
