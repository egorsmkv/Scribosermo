Build and use our docker container:

```bash
docker build -f deepspeech-polyglot/Containerfile -t dspol ./deepspeech-polyglot/

docker run --privileged --rm --network host -it \
  --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  --volume `pwd`/deepspeech-polyglot/:/deepspeech-polyglot/ \
  --volume `pwd`/checkpoints/:/checkpoints/ \
  --volume `pwd`/data_original/:/data_original/ \
  --volume `pwd`/data_prepared/:/data_prepared/ \
  dspol
```

Edit `dspol/config/train_config.template.yaml` and save as `train_config.yaml`.

Run training or tests:

```bash
python3 /deepspeech-polyglot/dspol/run_train.py

# Besides the normal network test, there are also some debugging tests you can uncomment
python3 /deepspeech-polyglot/dspol/run_tests.py
```
