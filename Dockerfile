FROM mozilla_deep_speech:latest
ARG DEBIAN_FRONTEND=noninteractive

# Update pip
RUN pip3 install --upgrade --no-cache-dir pip

# Install python packages
RUN pip3 install --no-cache-dir --upgrade \
    num2words \
    google-cloud-texttospeech

# Fix error: AttributeError: module 'gast' has no attribute 'Num'
RUN pip3 install --no-cache-dir gast==0.2.2

RUN apt-get update && apt-get install -y --no-install-recommends file
RUN apt-get update && apt-get install -y --no-install-recommends zip

# Dependencies for noise normalization
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg
RUN pip install --no-cache-dir --upgrade pydub

# Tool to convert output graph for inference
RUN python3 util/taskcluster.py --source tensorflow --artifact convert_graphdef_memmapped_format --branch r1.15 --target .

# Delete apt cache to save space
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

# Parallel pandas functions
RUN pip3 install --no-cache-dir pandarallel

# Upgrade setuptools for tensorboard
RUN pip3 install --upgrade --no-cache-dir setuptools

# Install audiomate
#RUN pip3 install git+https://github.com/danbmh/audiomate
RUN pip3 install --no-cache-dir audiomate

CMD ["/bin/bash"]
