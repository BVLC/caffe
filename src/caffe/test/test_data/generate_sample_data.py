# 
# All modification made by Intel Corporation: Copyright (c) 2016 Intel Corporation
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
"""
Generate data used in the HDF5DataLayer and GradientBasedSolver tests.
"""
import os
import numpy as np
import h5py

script_dir = os.path.dirname(os.path.abspath(__file__))

# Generate HDF5DataLayer sample_data.h5

num_cols = 8
num_rows = 10
height = 6
width = 5
total_size = num_cols * num_rows * height * width

data = np.arange(total_size)
data = data.reshape(num_rows, num_cols, height, width)
data = data.astype('float32')

# We had a bug where data was copied into label, but the tests weren't
# catching it, so let's make label 1-indexed.
label = 1 + np.arange(num_rows)[:, np.newaxis]
label = label.astype('float32')

# We add an extra label2 dataset to test HDF5 layer's ability
# to handle arbitrary number of output ("top") Blobs.
label2 = label + 1

print data
print label

with h5py.File(script_dir + '/sample_data.h5', 'w') as f:
    f['data'] = data
    f['label'] = label
    f['label2'] = label2

with h5py.File(script_dir + '/sample_data_2_gzip.h5', 'w') as f:
    f.create_dataset(
        'data', data=data + total_size,
        compression='gzip', compression_opts=1
    )
    f.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1,
        dtype='uint8',
    )
    f.create_dataset(
        'label2', data=label2,
        compression='gzip', compression_opts=1,
        dtype='uint8',
    )

with open(script_dir + '/sample_data_list.txt', 'w') as f:
    f.write('src/caffe/test/test_data/sample_data.h5\n')
    f.write('src/caffe/test/test_data/sample_data_2_gzip.h5\n')

# Generate GradientBasedSolver solver_data.h5

num_cols = 3
num_rows = 8
height = 10
width = 10

data = np.random.randn(num_rows, num_cols, height, width)
data = data.reshape(num_rows, num_cols, height, width)
data = data.astype('float32')

targets = np.random.randn(num_rows, 1)
targets = targets.astype('float32')

print data
print targets

with h5py.File(script_dir + '/solver_data.h5', 'w') as f:
    f['data'] = data
    f['targets'] = targets

with open(script_dir + '/solver_data_list.txt', 'w') as f:
    f.write('src/caffe/test/test_data/solver_data.h5\n')
