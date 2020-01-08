FROM mozilla_deep_speech:latest
ARG DEBIAN_FRONTEND=noninteractive

## Install kenlm
#RUN git clone https://github.com/kpu/kenlm.git
#RUN cd kenlm; mkdir -p build
#RUN cd kenlm/build; cmake ..; make -j `nproc`

# Install python packages
RUN pip install --no-cache-dir --upgrade \
    num2words

# Fix error: AttributeError: module 'gast' has no attribute 'Num'
RUN pip3 install gast==0.2.2

RUN apt-get update && apt-get install -y file

# Install audiomate
RUN pip3 install git+https://github.com/danbmh/audiomate

CMD ["/bin/bash"]
