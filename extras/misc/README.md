# Various scripts

Collection of various scripts that have no special topics.

### Crossbuild CTC-Decoder for RasPi

Build DeepSpeech's building container:

```bash
cd DeepSpeech/
make Dockerfile.build
cd ..

# Now edit the Dockerfile and remove the KenLM building part at the bottom
# With the KenLM installation I had some problems building the native-client again

docker build -t dsbuild - < DeepSpeech/Dockerfile.build
```

Enable container cross-building:

```bash
sudo podman run --security-opt label=disable --rm --privileged multiarch/qemu-user-static --reset -p yes
```

Build `swig` for raspbian:

```bash
podman build --cgroup-manager=cgroupfs -f deepspeech-polyglot/extras/misc/Containerfile_DecoderCrossbuild1 -t xbuildctc1

podman run --rm -it \
  --volume "$(pwd)"/deepspeech-polyglot/extras/misc/:/dspol/misc/ \
  xbuildctc1

cp /ds-swig.tar.gz /dspol/misc/
```

Now build the `ds_ctcdecoder.whl` package and extract it from the container:

```bash
docker build -f deepspeech-polyglot/extras/misc/Containerfile_DecoderCrossbuild2 -t xbuildctc2 deepspeech-polyglot/extras/misc/

docker run --rm -it \
  --volume "$(pwd)"/deepspeech-polyglot/extras/misc/:/dspol/misc/ \
  xbuildctc2

cp /DeepSpeech/native_client/ctcdecode/dist/*.whl /dspol/misc/
```

### Testing transcription on raspberry pi

Enable cross-building if you don't want to build the Container on the RasPi itself:

```bash
sudo podman run --security-opt label=disable --rm --privileged multiarch/qemu-user-static --reset -p yes
```

Build container:

```bash
podman build --cgroup-manager=cgroupfs -f deepspeech-polyglot/extras/misc/Containerfile_Raspbian -t dspol_raspbian
```

Run script:

```bash
sudo podman run --privileged --rm --network host -it \
  --volume "$(pwd)"/deepspeech-polyglot/extras/exporting/:/deepspeech-polyglot/extras/exporting/:ro \
  --volume "$(pwd)"/deepspeech-polyglot/data/:/deepspeech-polyglot/data/:ro \
  --volume "$(pwd)"/checkpoints/:/checkpoints/:ro \
  --volume "$(pwd)"/data_prepared/texts/:/data_prepared/texts/:ro \
  dspol_raspbian

python3 /deepspeech-polyglot/extras/exporting/testing_tflite.py
```
