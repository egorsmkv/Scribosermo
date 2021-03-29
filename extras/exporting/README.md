# Exporting

Tools to export the model for easier inference.

Notes:

- Using fixed normalization (required for streaming) slightly reduces WER from 3.7% to 4.4%
- Enabling tflite optimization reduces performance too
- Models containing a LSTM layer don't work in quantized tflite format (might be fixed after a tensorflow update)

Edit the files to your needs:

```bash
# Export
python3 /Scribosermo/extras/exporting/export.py

# Test exported models
python3 /Scribosermo/extras/exporting/testing_pb.py
python3 /Scribosermo/extras/exporting/testing_tflite.py

# Disable gpus for performance tests
export CUDA_VISIBLE_DEVICES=""
```

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
