#!/usr/bin/env python
import h5py
import numpy as np

train_filename = 'xor_data.hdf5'
data = np.array(((0, 0), (0, 1), (1, 0), (1, 1)))
labels = np.array((0, 1, 1, 0))
# data = f.create_dataset("data", (4,2), dtype='f') # needs float later!

with h5py.File(train_filename, 'w') as f:
  f['data'] = data.astype(np.float32)
  f['label'] = labels.astype(np.float32)