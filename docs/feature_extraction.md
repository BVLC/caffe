---
layout: default
title: Caffe
---

Extracting Features Using Pre-trained Model
===========================================

CAFFE represents Convolution Architecture For Feature Extraction. Extracting features using pre-trained model is one of the strongest requirements users ask for.

Because of the record-breaking image classification accuracy and the flexible domain adaptability of [the network architecture proposed by Krizhevsky, Sutskever, and Hinton](http://books.nips.cc/papers/files/nips25/NIPS2012_0534.pdf), Caffe provides a pre-trained reference image model to save you from days of training. 

If you need detailed usage help information of the involved tools, please read the source code of them which provide everything you need to know about.

Get the Reference Model
-----------------------

Assume you are in the root directory of Caffe.

    cd models
    ./get_caffe_reference_imagenet_model.sh

After the downloading is finished, you will have models/caffe_reference_imagenet_model.

Preprocess the Data
-------------------

Generate a list of the files to process. 

    build/tools/generate_file_list.py /your/images/dir /your/images.txt

The network definition of the reference model only accepts 256*256 pixel images stored in the leveldb format. First, resize your images if they do not match the required size.

    build/tools/resize_and_crop_images.py --num_clients=8 --image_lib=opencv --output_side_length=256 --input=/your/images.txt --input_folder=/your/images/dir --output_folder=/your/resized/images/dir_256_256

Set the num_clients to be the number of CPU cores on your machine. Run "nproc" or "cat /proc/cpuinfo | grep processor | wc -l" to get the number on Linux.

    build/tools/generate_file_list.py /your/resized/images/dir_256_256 /your/resized/images_256_256.txt
    build/tools/convert_imageset /your/resized/images/dir_256_256 /your/resized/images_256_256.txt /your/resized/images_256_256_leveldb 1

In practice, subtracting the mean image from a dataset significantly improves classification accuracies.

    build/tools/compute_image_mean.bin /your/resized/images_256_256_leveldb /your/resized/images_256_256_mean.binaryproto

Define the Feature Extraction Network Architecture
--------------------------------------------------

If you do not want to change the reference model network architecture , simply copy examples/imagenet into examples/your_own_dir. Then point the source and meanfile field of the data layer in imagenet_val.prototxt to /your/resized/images_256_256_leveldb and /your/resized/images_256_256_mean.binaryproto respectively. 

Extract Features
----------------

Now everything necessary is in place.

    build/tools/extract_features.bin models/caffe_reference_imagenet_model examples/feature_extraction/imagenet_val.prototxt fc7 examples/feature_extraction/features 10

The name of feature blob that you extract is fc7 which represents the highest level feature of the reference model. Any other blob is also applicable. The last parameter above is the number of data mini-batches.
