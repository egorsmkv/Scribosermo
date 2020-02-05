FROM mozilla_deep_speech:latest
ARG DEBIAN_FRONTEND=noninteractive

# Update pip
RUN pip install --upgrade pip

# Install python packages
RUN pip install --no-cache-dir --upgrade \
    num2words

# Fix error: AttributeError: module 'gast' has no attribute 'Num'
RUN pip3 install gast==0.2.2

RUN apt-get update && apt-get install -y file

# Install audiomate
RUN pip3 install git+https://github.com/danbmh/audiomate

CMD ["/bin/bash"]
