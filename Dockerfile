FROM mozilla_deep_speech:latest
ARG DEBIAN_FRONTEND=noninteractive

# Update pip
RUN pip3 install --upgrade pip

# Install python packages
RUN pip3 install --no-cache-dir --upgrade \
    num2words \
    google-cloud-texttospeech

# Fix error: AttributeError: module 'gast' has no attribute 'Num'
RUN pip3 install gast==0.2.2

RUN apt-get update && apt-get install -y file
RUN apt-get update && apt-get install -y zip

# Dependencies for noise normalization
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --no-cache-dir --upgrade pydub

# Tool to convert output graph for inference
RUN python3 util/taskcluster.py --source tensorflow --artifact convert_graphdef_memmapped_format --branch r1.15 --target .

# Install audiomate
RUN pip3 install git+https://github.com/danbmh/audiomate

# Using mfcc to find invalid files
RUN pip3 install --no-cache-dir --upgrade python_speech_features

CMD ["/bin/bash"]
