FROM nvcr.io/nvidia/tensorflow:20.11-tf2-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
WORKDIR /

COPY dspol/ /deepspeech-polyglot/dspol/
RUN pip3 install -e /deepspeech-polyglot/dspol/

CMD ["/bin/bash"]
