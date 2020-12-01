# NeMo Model Conversion

* Get nemo docker container from [here](https://ngc.nvidia.com/catalog/containers/nvidia:nemo).
* Get model from [here](https://ngc.nvidia.com/catalog/models/nvidia:nemospeechmodels/files) and save it in `models` folder.

* Convert `.nemo` to `.onnx`:
  ```bash
  python3 ./NeMo/scripts/convasr_to_single_onnx.py \
    --nemo_file ./models/QuartzNet15x5Base-En.nemo --onnx_file ./models/QuartzNet15x5Base-En.onnx  
  ```

* Build conversion container:
  ```bash
  docker build -f ./Dockerfile -t onnx-tf ./
  ```

* Convert `.onnx` to `.pb`:
  ```bash
  docker run --rm -v `pwd`/:/nemo/ -it onnx-tf /bin/bash -c \
    "onnx-tf convert -i /nemo/models/QuartzNet15x5Base-En.onnx -o /nemo/models/tf.pb"
  ```

* Convert `.pb` to `.pbmm`:
  ```bash
  # Can't parse /models/tf/saved_model.pb as binary proto
  docker run --rm -v `pwd`/:/nemo/ -it deepspeech_polyglot \
    /bin/bash -c '/DeepSpeech/convert_graphdef_memmapped_format \
      --in_graph="/nemo/models/tf/saved_model.pb" --out_graph="/nemo/models/tf/saved_model.pbmm"'
  ```
