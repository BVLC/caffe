#!/usr/bin/env sh
# 
# All modification made by Intel Corporation: © 2016 Intel Corporation
# 
# All contributions by the University of California:
# Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
# All rights reserved.
# 
# All other contributions:
# Copyright (c) 2014, 2015, the respective contributors
# All rights reserved.
# For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
# 
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
set -e
# The following example allows for the MNIST example (using LeNet) to be
# trained using the caffe docker image instead of building from source.
#
# The GPU-enabled version of Caffe can be used, assuming that nvidia-docker
# is installed, and the GPU-enabled Caffe image has been built.
# Setting the GPU environment variable to 1 will enable the use of nvidia-docker.
# e.g.
#   GPU=1 ./examples/mnist/train_lenet_docker.sh [ADDITIONAL_CAFFE_ARGS]
#
# With any arguments following the script being passed directly to caffe
# when training the network.
#
# The steps that are performed by the script are as follows:
# 1. The MNIST data set is downloaded
#    (see data/mnist/get_mnist.sh)
# 2. An LMDB database is created from the downloaded data
#    (see examples/mnist/create_mnist.sh.
# 3. A caffe network based on the LeNet solver is trained.
#    (see examples/mnist/lenet_solver.prototxt)
#
# For each of these, a step is executed to ensure that certain prerequisites
# are available, after which a command that actually performs the work is
# executed.
#
# In order to provide additional flexibility, the following shell (environment)
# variables can be used to controll the execution of each of the phases:
#
# DOWNLOAD_DATA: Enable (1) or disable (0) the downloading of the MNIST dataset
# CREATE_LMDB: Enable (1) or disable (0) the creation of the LMDB database
# TRAIN: Enable (1) or disable (0) the training of the LeNet networkd.
#
# As an example, assuming that the data set has been downloaded, and an LMDB
# database created, the following command can be used to train the LeNet
# network with GPU computing enabled.
#
# DOWNLOAD_DATA=0 CREATE_LMDB=0 GPU=1 ./examples/mnist/train_lenet_docker.sh
#


if [ x"$(uname -s)" != x"Linux" ]
then
echo ""
echo "This script is designed to run on Linux."
echo "There may be problems with the way Docker mounts host volumes on other"
echo "systems which will cause the docker commands to fail."
echo ""
read -p "Press [ENTER] to continue..." key
echo ""
fi


# Check if GPU mode has been enabled and set the docker executable accordingly
if [ ${GPU:-0} -eq 1 ]
then
DOCKER_CMD=nvidia-docker
IMAGE=caffe:gpu
else
DOCKER_CMD=docker
IMAGE=caffe:cpu
fi
echo "Using $DOCKER_CMD to launch $IMAGE"

# On non-Linux systems, the Docker host is typically a virtual machine.
# This means that the user and group id's may be different.
# On OS X, for example, the user and group are 1000 and 50, respectively.
if [ x"$(uname -s)" != x"Linux" ]
then
CUID=1000
CGID=50
else
CUID=$(id -u)
CGID=$(id -g)
fi

# Define some helper variables to make the running of the actual docker
# commands less verbose.
# Note:
#   -u $CUID:$CGID             runs the docker image as the current user to ensure
#                              that the file permissions are compatible with the
#                              host system. The variables CUID and CGID have been
#                              set above depending on the host operating system.
#   --volume $(pwd):/workspace mounts the current directory as the docker volume
#                              /workspace
#   --workdir /workspace       Ensures that the docker container starts in the right
#                              working directory
DOCKER_OPTIONS="--rm -ti -u $CUID:$CGID --volume=$(pwd):/workspace --workdir=/workspace"
DOCKER_RUN="$DOCKER_CMD run $DOCKER_OPTIONS $IMAGE"

# Download the data
if [ ${DOWNLOAD_DATA:-1} -eq 1 ]
then
$DOCKER_RUN bash -c "mkdir -p ./data/mnist;
                     cp -ru \$CAFFE_ROOT/data/mnist/get_mnist.sh ./data/mnist/"
$DOCKER_RUN ./data/mnist/get_mnist.sh
fi

# Create the LMDB database
if [ ${CREATE_LMDB:-1} -eq 1 ]
then
$DOCKER_RUN bash -c "mkdir -p ./examples/mnist;
                     cp -ru \$CAFFE_ROOT/examples/mnist/create_mnist.sh ./examples/mnist/;
                     sed -i s#BUILD=build#BUILD=\$CAFFE_ROOT/build## ./examples/mnist/create_mnist.sh"
$DOCKER_RUN ./examples/mnist/create_mnist.sh
fi

# Train the network
if [ ${TRAIN:-1} -eq 1 ]
then
$DOCKER_RUN bash -c "cp \$CAFFE_ROOT/examples/mnist/lenet_solver.prototxt ./examples/mnist/;
                     cp \$CAFFE_ROOT/examples/mnist/lenet_train_test.prototxt ./examples/mnist/"
    # Ensure that the solver_mode is compatible with the desired GPU mode.
    if [ ${GPU:-0} -eq 0 ]
    then
    $DOCKER_RUN sed -i 's#solver_mode: GPU#solver_mode: CPU##' ./examples/mnist/lenet_solver.prototxt
    fi
$DOCKER_RUN caffe train --solver=examples/mnist/lenet_solver.prototxt $*
fi
