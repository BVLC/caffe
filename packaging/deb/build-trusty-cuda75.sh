#!/bin/bash
set -e

export DOCKER_BASE=caffe-nv-debuild-trusty-cuda75
docker pull nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
docker build -t $DOCKER_BASE -f Dockerfile.trusty-cuda75 .
./build.sh
