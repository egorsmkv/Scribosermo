# NeMo Model Conversion

The official ASR tutorial (containing also instructions for transfer-learning) can be found under this
[link](https://colab.research.google.com/github/NVIDIA/NeMo/blob/master/tutorials/asr/01_ASR_with_NeMo.ipynb). \
The goal here is to use the pretrained NeMo models from Nvidia with our tensorflow implementation.

- Get model [here](https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels/files) and save it in `models` folder.

- Convert `.nemo` to `.onnx`:

  ```bash
  docker build -f ./Scribosermo/extras/nemo/Containerfile_Nemo -t dsp_nemo ./Scribosermo/

  docker run --gpus all -it --rm -p 8888:8888 -p 6006:6006 \
    --ulimit memlock=-1 --ulimit stack=67108864  --shm-size=8g \
    --volume `pwd`/Scribosermo/extras/nemo/:/dsp_nemo/ \
    --volume `pwd`/data_prepared/:/data_prepared/ \
    --device=/dev/snd dsp_nemo

  # Convert pretrained model
  python3 /NeMo/scripts/convasr_to_single_onnx.py \
    --nemo_file /dsp_nemo/models/QuartzNet15x5Base-En.nemo --onnx_file /dsp_nemo/models/QuartzNet15x5Base-En.onnx

  # Test model and some debugging for our pipeline
  python3 /dsp_nemo/testing_nemo.py
  ```

- Go to https://netron.app/ and look at the graph structure.
  This is also a nice way to compare two graph implementations with each other.

- Build and start conversion container:

  ```bash
  docker build -f ./Scribosermo/extras/nemo/Containerfile_Onnx -t onnx-tf ./Scribosermo/

  docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    --volume `pwd`/Scribosermo/:/Scribosermo/ \
    --volume `pwd`/Scribosermo/extras/nemo/:/nemo/ \
    --volume `pwd`/checkpoints/:/checkpoints/ -it onnx-tf
  ```

- Transfer the pretrained weights, or run some tests to check the .onnx model and data pipeline: \
  (Note that currently the calculated spectrogram differs between the pipeline here and the one from nemo)

  ```bash
  # Uncomment the required calls at the bottom
  python3 /nemo/testing_models.py
  ```

- Go to the exported checkpoint, edit the `config_export.json` and then do a full test run.

- Convert between `.pb` and `.onnx`:

  ```bash
  # From .pb to .onnx (can be used for better visualisation in above web-tool)
  python3 -m tf2onnx.convert --opset 12 --saved-model /checkpoints/tmp/ --output /nemo/models/tfmodel.onnx

  # From .onnx to .pb (loading it into our tensorflow model didn't work)
  onnx-tf convert -i /nemo/models/QuartzNet5x5LS-En.onnx -o /nemo/models/tfpb/
  ```
