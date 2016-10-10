#!/bin/bash
set -e

export DOCKER_BASE=caffe-nv-debuild-xenial-cuda80
docker pull nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
docker build -t $DOCKER_BASE -f Dockerfile.xenial-cuda80 .
./build.sh
