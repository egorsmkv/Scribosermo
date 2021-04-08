# Training

Instructions for training networks. The training datasets have to be in `tab` separated `.csv` format,
containing at least the columns `filepath` (absolute), `duration` (seconds), `text`.
All other columns will be ignored automatically.
Audio files have to be in `.wav` format, with 16kHz recording rate and a single channel only.

Go to `config/`, edit the `train_config.template.yaml` file and save as `train_config.yaml`.

Run training or tests:

```bash
# By default all gpus are used automatically
python3 /Scribosermo/training/run_train.py

# Besides the normal network test, there are also some debugging tests you can uncomment
# Testing always uses a sinlge gpu only
python3 /Scribosermo/training/run_tests.py
```

Some helpful commands:

```bash
# Watch gpu utilization (run in another shell tab)
watch -n 1 nvidia-smi

# Restrict training/test to selected gpus
export CUDA_VISIBLE_DEVICES=1

# Start Tensorboard
tensorboard --logdir /checkpoints/en/tmp/

# Print important log infos (with additional lines before and after)
cat log.txt | grep -B 7 -A 1 "Saved"

# Run as detached process
# (edit the run_container.sh file before: remove the "-it" flag and append training command, that training is directly started)
nohup ./Scribosermo/run_container.sh > nohup.out 2>&1 &
```
