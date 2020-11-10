FROM mozilla_deepspeech:latest
#FROM mds_slurm:latest
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y
RUN apt-get update && apt-get install -y --no-install-recommends nano file zip
RUN apt-get update && apt-get install -y --no-install-recommends sox libsox-dev

# Dependencies for noise normalization and some dataset preparations
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg
RUN pip install --no-cache-dir --upgrade pydub

# Update pip
RUN python3 -m pip install --upgrade pip

# Install python packages
RUN pip3 install --no-cache-dir --upgrade \
    num2words \
    google-cloud-texttospeech \
    pytest pytest-cov \
    progressist

# Fix error: AttributeError: module 'gast' has no attribute 'Num'
RUN pip3 install --no-cache-dir gast==0.2.2

# Parallel pandas functions
RUN pip3 install --no-cache-dir pandarallel

# Upgrade setuptools for tensorboard
RUN pip3 install --upgrade --no-cache-dir setuptools

# Update pandas version to fix an error
RUN pip3 install --upgrade --no-cache-dir pandas

# Install audiomate
RUN pip3 install --upgrade git+https://github.com/danbmh/audiomate.git@new_features
#RUN pip3 install --no-cache-dir audiomate

# Download scorer generator script
RUN cd /DeepSpeech/data/lm/ \
    && curl -LO https://github.com/mozilla/DeepSpeech/releases/latest/download/native_client.amd64.cpu.linux.tar.xz \
    && tar xvf native_client.*.tar.xz

# Youtube downloading requirements
RUN pip3 install --upgrade --no-cache-dir youtube_dl
RUN pip3 install --upgrade --no-cache-dir youtube_transcript_api
RUN pip3 install --upgrade git+https://github.com/DanBmh/aud-crawler@some_improvements

CMD ["/bin/bash"]
