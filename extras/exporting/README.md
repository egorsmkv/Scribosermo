# Exporting

Tools to export the model for easier inference.

Notes:

- Enabling tflite optimization slightly reduces performance
- Models containing a LSTM layer don't work in quantized tflite format (might be fixed after a tensorflow update)
- For production use, the tflite runtime is recommended for single-board computers, as well as for desktop computers.
  Installation and initialization is much faster, while the speed difference on desktop computers is only minimal.

Edit the files to your needs:

```bash
# Export
python3 /Scribosermo/extras/exporting/export.py

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
