#!/bin/bash
set -e

export DOCKER_BASE=caffe-nv-debuild-trusty-cuda75
docker build --pull -t $DOCKER_BASE -f Dockerfile.trusty-cuda75 .
./build.sh
