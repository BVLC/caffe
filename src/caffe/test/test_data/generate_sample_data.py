"""
Generate data used in the HDF5DataLayer test.
"""
import os
import numpy as np
import h5py

num_cols = 8
num_rows = 10
height = 5
width = 5
total_size = num_cols * num_rows * height * width

data = np.arange(total_size)
data = data.reshape(num_rows, num_cols, height, width)
data = data.astype('float32')
label = np.arange(num_rows)[:, np.newaxis]
label = label.astype('float32')

print data
print label

with h5py.File(os.path.dirname(__file__) + '/sample_data.h5', 'w') as f:
    f['data'] = data
    f['label'] = label

with h5py.File(os.path.dirname(__file__) + '/sample_data_2_gzip.h5', 'w') as f:
    f.create_dataset(
        'data', data=data + total_size,
        compression='gzip', compression_opts=1
    )
    f.create_dataset(
        'label', data=label,
        compression='gzip', compression_opts=1
    )

with open(os.path.dirname(__file__) + '/sample_data_list.txt', 'w') as f:
    f.write(os.path.dirname(__file__) + '/sample_data.h5\n')
    f.write(os.path.dirname(__file__) + '/sample_data_2_gzip.h5\n')
