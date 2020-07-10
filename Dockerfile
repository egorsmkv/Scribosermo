FROM mozilla_deepspeech:latest
#FROM mds_slurm:latest
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y --no-install-recommends nano file zip
RUN apt-get update && apt-get install -y --no-install-recommends sox libsox-dev

# Dependencies for noise normalization
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg
RUN pip install --no-cache-dir --upgrade pydub

# Tool to convert output graph for inference
RUN python3 util/taskcluster.py --source tensorflow --artifact convert_graphdef_memmapped_format --branch r1.15 --target .

# Delete apt cache to save space
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Update pip
RUN python3 -m pip install --upgrade pip

# Install python packages
RUN pip3 install --no-cache-dir --upgrade \
    num2words \
    google-cloud-texttospeech

# Fix error: AttributeError: module 'gast' has no attribute 'Num'
RUN pip3 install --no-cache-dir gast==0.2.2

# Parallel pandas functions
RUN pip3 install --no-cache-dir pandarallel

# Upgrade setuptools for tensorboard
RUN pip3 install --upgrade --no-cache-dir setuptools

# Update pandas version to fix an error
RUN pip3 install --upgrade --no-cache-dir pandas

# Build kenlm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake libboost-all-dev
RUN cd /DeepSpeech/native_client/ && rm -r kenlm/ \
    && git clone --depth 1 https://github.com/kpu/kenlm \
    && cd kenlm \
    && mkdir -p build \
    && cd build \
    && cmake .. \
    && make -j 4

RUN pip3 install --upgrade --no-cache-dir pytest pytest-cov

# Install audiomate
RUN pip3 install --upgrade git+https://github.com/danbmh/audiomate.git@new_features
#RUN pip3 install --no-cache-dir audiomate

COPY . /DeepSpeech/deepspeech-polyglot/

CMD ["/bin/bash"]
