FROM nvcr.io/nvidia/tensorflow:20.11-tf2-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
WORKDIR /

RUN apt-get update && apt-get install -y libsndfile1

# Use some tools from DeepSpeech project
RUN git clone --depth 1 https://github.com/mozilla/DeepSpeech.git
# CTC decoder (the next line is required for building with shallow git clone)
RUN sed -i 's/git describe --long --tags/git describe --long --tags --always/g' /DeepSpeech/native_client/bazel_workspace_status_cmd.sh
RUN apt-get update && apt-get install -y libmagic-dev
RUN cd /DeepSpeech/native_client/ctcdecode && make NUM_PROCESSES=$(nproc) bindings
RUN pip3 install --upgrade /DeepSpeech/native_client/ctcdecode/dist/*.whl
# KenLM
RUN apt-get update && apt-get install -y libboost-all-dev
RUN cd /DeepSpeech/native_client/ && \
  rm -rf kenlm && \
  git clone --depth 1  https://github.com/kpu/kenlm && \
  mkdir -p kenlm/build && \
  cd kenlm/build && \
  cmake .. && \
  make -j $(nproc)
## Graph converter
#RUN python3 /DeepSpeech/util/taskcluster.py --source tensorflow --branch r1.15 \
#  --artifact convert_graphdef_memmapped_format  --target /DeepSpeech/

# Solve broken pip "ImportError: No module named pip._internal.cli.main"
RUN python3 -m pip install --upgrade pip

# Pre-install some libraries for faster installation time of dspol package
RUN pip3 install --no-cache-dir pandas
RUN pip3 install --no-cache-dir librosa
RUN pip3 install --no-cache-dir tensorflow
RUN pip3 install --no-cache-dir tensorflow-io

COPY dspol/ /deepspeech-polyglot/dspol/
RUN pip3 install -e /deepspeech-polyglot/dspol/

CMD ["/bin/bash"]
