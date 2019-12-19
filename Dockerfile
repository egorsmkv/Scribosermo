FROM mozilla_deep_speech:latest
ARG DEBIAN_FRONTEND=noninteractive

# Install kenlm
RUN git clone https://github.com/kpu/kenlm.git
RUN cd kenlm; mkdir -p build
RUN cd kenlm/build; cmake ..; make -j `nproc`

# Install python packages
RUN pip3 install git+https://github.com/ynop/audiomate.git
RUN pip install --no-cache-dir --upgrade \
    num2words

RUN apt-get update && apt-get install -y file

CMD ["/bin/bash"]
