#!/bin/bash
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
#### https://github.com/Yangqing/mincepie/wiki/Launch-Your-Mapreducer

# If you encounter error that the address already in use, kill the process.
# 11235 is the port of server process
# https://github.com/Yangqing/mincepie/blob/master/mincepie/mince.py
#     sudo netstat -ap | grep 11235
# The last column of the output is  PID/Program name
#     kill -9 PID
# Second solution: 
#     nmap localhost
#     fuser -k 11235/tcp
# Or just wait a few seconds.

## Launch your Mapreduce locally
# num_clients: number of processes
# image_lib: OpenCV or PIL, case insensitive. The default value is the faster OpenCV.
# input: the file containing one image path relative to input_folder each line
# input_folder: where are the original images
# output_folder: where to save the resized and cropped images
./resize_and_crop_images.py --num_clients=8 --image_lib=opencv --input=/home/user/Datasets/ImageNet/ILSVRC2010/ILSVRC2010_images.txt --input_folder=/home/user/Datasets/ImageNet/ILSVRC2010/ILSVRC2010_images_train/ --output_folder=/home/user/Datasets/ImageNet/ILSVRC2010/ILSVRC2010_images_train_resized/

## Launch your Mapreduce with MPI
# mpirun -n 8 --launch=mpi resize_and_crop_images.py --image_lib=opencv --input=/home/user/Datasets/ImageNet/ILSVRC2010/ILSVRC2010_images.txt --input_folder=/home/user/Datasets/ImageNet/ILSVRC2010/ILSVRC2010_images_train/ --output_folder=/home/user/Datasets/ImageNet/ILSVRC2010/ILSVRC2010_images_train_resized/
