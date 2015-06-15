"""
Generate data used in the HDF5DataLayer test.
"""
import os
import numpy as np
import h5py

script_dir = os.path.dirname(os.path.abspath(__file__))

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
        compression='gzip', compression_opts=1
    )
    f.create_dataset(
        'label2', data=label2,
        compression='gzip', compression_opts=1
    )

with open(script_dir + '/sample_data_list.txt', 'w') as f:
    f.write(script_dir + '/sample_data.h5\n')
    f.write(script_dir + '/sample_data_2_gzip.h5\n')
