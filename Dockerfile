FROM mozilla_deepspeech:latest
#FROM mds_slurm:latest
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y --no-install-recommends nano file zip
RUN apt-get update && apt-get install -y --no-install-recommends sox libsox-dev

# Dependencies for noise normalization
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg
RUN pip install --no-cache-dir --upgrade pydub

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

RUN pip3 install --upgrade --no-cache-dir pytest pytest-cov
RUN pip3 install --upgrade --no-cache-dir progressist

# Install audiomate
RUN pip3 install --upgrade git+https://github.com/danbmh/audiomate.git@new_features
#RUN pip3 install --no-cache-dir audiomate

# Download scorer generator script
RUN cd /DeepSpeech/data/lm/ \
    && curl -LO https://github.com/mozilla/DeepSpeech/releases/latest/download/native_client.amd64.cpu.linux.tar.xz \
    && tar xvf native_client.*.tar.xz

COPY . /DeepSpeech/deepspeech-polyglot/

CMD ["/bin/bash"]
