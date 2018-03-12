"""
Generate distance matrix in hdf5 format.
"""
import os
import numpy as np
import h5py

script_dir = os.path.dirname(os.path.abspath(__file__))

# Generate HDF5DataLayer distance_matrix.h5

d = 4
data = np.arange(d * d)
data = data.reshape(d, d)
data = data.astype('float32')


for i in range(d):
  for j in range(d):
    data[i][j] = abs(i-j)

with h5py.File(script_dir + '/wasserstein_ground_metric.h5', 'w') as f:
    f['data'] = data
