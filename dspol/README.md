# Training

Edit `dspol/config/train_config.template.yaml` and save as `train_config.yaml`.

Run training or tests:

```bash
python3 /deepspeech-polyglot/dspol/run_train.py

# Besides the normal network test, there are also some debugging tests you can uncomment
python3 /deepspeech-polyglot/dspol/run_tests.py

# Watch gpu utilization (run in another shell tab)
watch -n 1 nvidia-smi

# Restrict training/test to selected gpus
export CUDA_VISIBLE_DEVICES=1
```
