Build and use our docker container:

```bash
docker build -f deepspeech-polyglot/Containerfile -t dspol ./deepspeech-polyglot/

docker run --privileged --rm --network host -it \
  --gpus all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  --volume `pwd`/deepspeech-polyglot/:/deepspeech-polyglot/ \
  --volume `pwd`/corcua/:/corcua/ \
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

# Watch gpu utilization (run in another shell tab)
watch -n 1 nvidia-smi
```

Other commands:

```bash
export LANGUAGE="de"

# To shuffle and replace non alphabet characters and clean the files run (repeat for all 3 csv files):
python3 /deepspeech-polyglot/preprocessing/dataset_operations.py "/data_prepared/en/librispeech/train-all.csv" \
  "/data_prepared/en/librispeech/train-all_azce.csv" --replace --clean --exclude

# Combine specific csv files:
python3 /deepspeech-polyglot/preprocessing/combine_datasets.py --file_output "/data_prepared/en/librispeech/train-all.csv" \
  --files "/data_prepared/en/librispeech/train-clean-100.csv /data_prepared/en/librispeech/train-clean-360.csv /data_prepared/en/librispeech/train-other-500.csv"

# Run unit tests
cd /deepspeech-polyglot/ && pytest --cov=preprocessing
```
