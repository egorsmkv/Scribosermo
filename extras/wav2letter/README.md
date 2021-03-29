# Some experiments

I did run some experiments using [Wav2Letter](https://github.com/facebookresearch/wav2letter/),
but results weren't promising enough to continue further.
The code is not maintained anymore, so use it at your own risk.

<br/>

- Build the container:

  ```bash
  git clone https://github.com/facebookresearch/wav2letter.git
  cd wav2letter && git checkout v0.2
  docker build --no-cache -f ./Dockerfile-CUDA -t wav2letter .
  cd ..
  ```

- Create training resources (Run in polyglot's container):

```bash
export LANGUAGE="de"

# Convert csv files to lst files:
python3 Scribosermo/extras/wav2letter/ds_to_w2l.py /data_prepared/${LANGUAGE}/voxforge/train_azce.csv /data_prepared/${LANGUAGE}/w2l_voxforge/train_azce.lst

# Generate lexicon file:
python3 Scribosermo/extras/wav2letter/create_lexicon.py /data_prepared/${LANGUAGE}/voxforge/train_azce.csv /data_prepared/${LANGUAGE}/voxforge/dev_azce.csv /data_prepared/texts/${LANGUAGE}/lexicon.txt
```

- Run training:

```bash
docker run \
  --network host --rm -it \
  --gpus all --privileged --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  --volume `pwd`/checkpoints/:/root/checkpoints/ \
  --volume `pwd`/Scribosermo/:/root/Scribosermo/ \
  --volume `pwd`/data_original/:/data_original/ \
  --volume `pwd`/data_prepared/:/data_prepared/ \
  wav2letter /bin/bash -c "cd /root/; bash"

Run training with single gpu:
mpirun -n 1 --allow-run-as-root /root/wav2letter/build/Train continue --flagsfile /root/Scribosermo/extras/wav2letter/training/train.cfg
mpirun -n 1 --allow-run-as-root /root/wav2letter/build/Decoder --flagsfile /root/Scribosermo/extras/wav2letter/training/decode.cfg

Continue training:
mpirun -n 1 --allow-run-as-root /root/wav2letter/build/Train continue checkpoints/w2l/voxforge_conv_glu/
```
