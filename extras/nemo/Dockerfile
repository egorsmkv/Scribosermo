FROM nvcr.io/nvidia/tensorflow:20.11-tf2-py3

WORKDIR /
RUN git clone https://github.com/onnx/onnx-tensorflow.git
RUN cd onnx-tensorflow && pip3 install -e .

CMD ["/bin/bash"]
