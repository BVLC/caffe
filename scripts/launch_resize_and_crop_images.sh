#!/bin/bash
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
