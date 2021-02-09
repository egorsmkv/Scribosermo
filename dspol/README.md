# Training

Instructions for training networks. The training datasets have to be in `tab` separated `.csv` format,
containing at least the columns `filepath` (absolute), `duration` (seconds), `text`.
All other columns will be ignored automatically.
Audio files have to be in `.wav` format, with 16kHz recording rate and a single channel only.

Go to `config/`, edit the `train_config.template.yaml` file and save as `train_config.yaml`.

Run training or tests:

```bash
python3 /deepspeech-polyglot/dspol/run_train.py

# Besides the normal network test, there are also some debugging tests you can uncomment
python3 /deepspeech-polyglot/dspol/run_tests.py
```

Some helpful commands:

```bash
# Watch gpu utilization (run in another shell tab)
watch -n 1 nvidia-smi

# Restrict training/test to selected gpus
export CUDA_VISIBLE_DEVICES=1

# Start Tensorboard
tensorboard --logdir /checkpoints/en/tmp/
```
