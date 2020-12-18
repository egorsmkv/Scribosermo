# NeMo Model Conversion

* Get nemo docker container from [here](https://ngc.nvidia.com/catalog/containers/nvidia:nemo).
* Get model [here](https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels/files) and save it in `models` folder.

* Convert `.nemo` to `.onnx`:
  ```bash
  docker run --gpus all -it --rm -p 8888:8888 -p 6006:6006 \
    --ulimit memlock=-1 --ulimit stack=67108864  --shm-size=8g \
    --volume `pwd`/deepspeech-polyglot/extras/nemo/:/dsp_nemo/ \
    --device=/dev/snd nvcr.io/nvidia/nemo:v1.0.0b2
  
  # We need to clone the repo because it's not included in the container
  cd / && git clone --depth 1 https://github.com/NVIDIA/NeMo.git
    
  python3 /NeMo/scripts/convasr_to_single_onnx.py \
    --nemo_file /dsp_nemo/models/QuartzNet15x5Base-En.nemo --onnx_file /dsp_nemo/models/QuartzNet15x5Base-En.onnx
  ```
  
* Go to https://netron.app/ and look at the graph structure.
  This is also a nice way to compare two graph implementations with each other.

* Build conversion container:
  ```bash
  docker build -f ./deepspeech-polyglot/extras/nemo/Containerfile -t onnx-tf ./deepspeech-polyglot/
  ```
  
* Convert between `.pb` and `.onnx`:
  ```bash
  docker run --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all \
    --volume `pwd`/deepspeech-polyglot/:/deepspeech-polyglot/ \
    --volume `pwd`/deepspeech-polyglot/extras/nemo/:/nemo/ \
    --volume `pwd`/checkpoints/:/checkpoints/ -it onnx-tf
  
  # From .pb to .onnx
  python3 -m tf2onnx.convert --opset 12 --saved-model /checkpoints/tmp/ --output /checkpoints/model.onnx

  # From .onnx to .pb
  onnx-tf convert -i /nemo/models/QuartzNet5x5LS-En.onnx -o /nemo/models/tfpb/
  onnx-tf convert -i /nemo/models/QuartzNet15x5Base-En.onnx -o /nemo/models/tfpb/
  onnx-tf convert -i /checkpoints/model.onnx -o /checkpoints/tfpb/
  ```
  
The goal was to use the pretrained NeMo models from Nvidia with the tensorflow implementation here.
By running `python3 /nemo/testing_onnx.py` in the container, you can do some inference tests, 
but I wasn't able to replicate the input pipeline, so the outputs won't make sense.

The official ASR tutorial (containing also instructions for transfer-leraning) can be found under this
[link](https://colab.research.google.com/github/NVIDIA/NeMo/blob/master/tutorials/asr/01_ASR_with_NeMo.ipynb).
