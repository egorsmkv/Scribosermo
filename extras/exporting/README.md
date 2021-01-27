# Exporting

Tools to export the model for easier inference.

Notes:

- Using fixed normalization (required for streaming) slightly reduces WER from 3.7% to 4.4%
- Exporting to tflite currently only works with a fixed sized input signal length

Edit the files to your needs:

```bash
# Export
python3 /deepspeech-polyglot/extras/exporting/export.py

# Test exported models
python3 /deepspeech-polyglot/extras/exporting/testing_pb.py
python3 /deepspeech-polyglot/extras/exporting/testing_tflite.py
```
