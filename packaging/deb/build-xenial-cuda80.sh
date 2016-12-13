#!/bin/bash
set -e

export DOCKER_BASE=caffe-nv-debuild-xenial-cuda80
docker build --pull -t $DOCKER_BASE -f Dockerfile.xenial-cuda80 .
./build.sh
