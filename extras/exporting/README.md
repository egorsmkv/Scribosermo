# Exporting

Tools to export the model for easier inference.

Notes:

- Enabling tflite optimization slightly reduces performance
- Models containing a LSTM layer don't work in quantized tflite format (might be fixed after a tensorflow update)
- For production use, the tflite runtime is recommended for single-board computers, as well as for desktop computers.
  Installation and initialization is much faster, while the speed difference on desktop computers is only minimal.

Edit the files to your needs:

```bash
# Important: Since the tf=2.4 update, the models have to be loaded from the '.index' and '.data-xxx' files instead of '.pb' and 'variables/'

# For some reason exporting to tflite doesn't work with tf=2.4, use version 2.3 instead and also downgrade the tf-io package
pip3 install "tensorflow<2.4" "tensorflow-io<0.17"

# Fix some tensorflow problem
sed -i '/with layer._metrics_lock/i\      if not hasattr(layer, "_metrics_lock"): continue' \
  /usr/local/lib/python3.8/dist-packages/tensorflow/python/keras/engine/base_layer.py

# Speed up exporting by disabling the gpu
export CUDA_VISIBLE_DEVICES=""

# Export with either mode=pb or mode=tflite
python3 /Scribosermo/extras/exporting/export.py \
  --checkpoint_dir "/checkpoints/en/qnetp5/" \
  --export_dir "checkpoints/en/qnetp5/exported/" \
  --mode "pb"

# Test exported models
python3 /Scribosermo/extras/exporting/testing_pb.py
python3 /Scribosermo/extras/exporting/testing_tflite.py
```

<br>

Performance test options: \
(For performance tests, restart the computer every time, to prevent caching speedups)

```bash
# Disable gpus inside the container
export CUDA_VISIBLE_DEVICES=""

# Options to start container using only the first cpu core
--cpus=1.0 --cpuset-cpus=0

# Option to start container using all cpus with only an equivalent quota of one cpu core
--cpus=$(LC_NUMERIC=C awk "BEGIN {print 1/`nproc`}")
```

<br>

Run streaming inference: \
(If an audio file is already existing, or for very short audio streams, the file-based transcription approach is recommended)

```bash
# Test streaming
python3 /Scribosermo/extras/exporting/testing_stream.py
```

<br>

### Collecting test audios

- Go to: https://commonvoice.mozilla.org/de/listen and find a nice audio file. \
  (Languages can be switched directly in the link)

- Use the browsers _inspect the website_ tool to get the link of the audio file and download it from there

- Convert from `.mp3` to `.wav` format:
  ```bash
  ffmpeg -i test_de.mp3 -acodec pcm_s16le -ac 1 -ar 16000 test_de.wav
  ```
