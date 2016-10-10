FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        libnccl-dev \
    && rm -rf /var/lib/apt/lists/*
