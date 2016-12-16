FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04

RUN which wget || (apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
    && rm -rf /var/lib/apt/lists/*)

RUN wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl1_1.2.3-1.cuda8.0_amd64.deb -O libnccl.deb \
    && dpkg -i libnccl.deb \
    && rm libnccl.deb \
    && ldconfig

RUN wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl-dev_1.2.3-1.cuda8.0_amd64.deb -O libnccl-dev.deb \
    && dpkg -i libnccl-dev.deb \
    && rm libnccl-dev.deb \
    && ldconfig
