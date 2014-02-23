"""
Generate data used in the HDF5DataLayer test.
"""

import numpy as np
import h5py

num_cols = 8
num_rows = 10
data = np.arange(num_cols * num_rows).reshape(num_rows, num_cols)
label = np.arange(num_rows)[:, np.newaxis]
print data
print label

with h5py.File('./sample_data.h5', 'w') as f:
    f['data'] = data.astype('float32')
    f['label'] = label.astype('float32')
