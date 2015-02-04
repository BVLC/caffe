#!/bin/bash

# I like to work in the example directory, not in root
[ -d examples/xor_hello_world ] && cd examples/xor_hello_world 

[ -f xor_data.hdf5 ] && echo "Found xor_data.hdf5 file" || ./create_hdf5_data.py
h5dump xor_data.hdf5

# The only data file we use for training here:
echo 'xor_data.hdf5' > xor_data.filelist.txt

# solver.prototxt links to network xor_net.prototxt
# xor_net.prototxt specifies training filelist xor_data.filelist.txt
# TODO training files should NOT be part of the NET definition!!

# Make sure caffe is found
type caffe || PATH=$PATH:../../build/tools/

caffe train -solver=solver.prototxt

# For a discussion of the xor net see here:
# See https://groups.google.com/forum/?utm_medium=email&utm_source=footer#!msg/caffe-users/4KnV4BOhI60/7C5d7FqNIioJ