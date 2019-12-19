FROM mozilla_deep_speech:latest
ARG DEBIAN_FRONTEND=noninteractive

# Install kenlm
RUN git clone https://github.com/kpu/kenlm.git
RUN cd kenlm; mkdir -p build
RUN cd kenlm/build; cmake ..; make -j `nproc`

# Install python packages
# Use my own repo for audiomate until my pull request to fix encoding error is accepted
RUN pip3 install git+https://github.com/DanBmh/audiomate
RUN pip install --no-cache-dir --upgrade \
    num2words

RUN apt-get update && apt-get install -y file

# Fix error: AttributeError: module 'gast' has no attribute 'Num'
RUN pip3 install gast==0.2.2

CMD ["/bin/bash"]
