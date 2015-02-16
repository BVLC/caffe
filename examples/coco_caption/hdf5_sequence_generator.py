#!/usr/bin/env python

import h5py
import numpy as np
import os
import random
import sys

class SequenceGenerator():
  def __init__(self):
    self.dimension = 10
    self.batch_stream_length = 2000
    self.batch_num_streams = 8
    self.min_stream_length = 13
    self.max_stream_length = 17
    self.substream_names = None
    self.streams_initialized = False

  def streams_exhausted(self):
    return False

  def init_streams(self):
    self.streams = [None] * self.batch_num_streams
    self.stream_indices = [0] * self.batch_num_streams
    self.reset_stream(0)
    self.streams_initialized = True

  def reset_stream(self, stream_index):
    streams = self.get_streams()
    stream_names = sorted(streams.keys())
    if self.substream_names is None:
      assert len(stream_names) > 0
      self.substream_names = stream_names
    assert self.substream_names == stream_names
    if self.streams[stream_index] is None:
      self.streams[stream_index] = {}
    stream_length = len(streams[stream_names[0]])
    for k, v in streams.iteritems():
      assert stream_length == len(v)
      self.streams[stream_index][k] = v
    self.stream_indices[stream_index] = 0

  # Pad with zeroes by default -- override this to pad with soemthing else
  # for a particular stream
  def get_pad_value(self, stream_name):
    return 0

  def get_next_batch(self, truncate_at_exhaustion=True):
    if not self.streams_initialized:
      self.init_streams()
    batch_size = self.batch_num_streams * self.batch_stream_length
    batch = {}
    batch_indicators = np.zeros((self.batch_stream_length, self.batch_num_streams))
    for name in self.substream_names:
      batch[name] = self.get_pad_value(name) * np.ones_like(batch_indicators)
    exhausted = [False] * self.batch_num_streams
    all_exhausted = False
    reached_exhaustion = False
    num_completed_streams = 0
    for t in range(self.batch_stream_length):
      all_exhausted = True
      for i in range(self.batch_num_streams):
        if not exhausted[i]:
          if self.streams[i] is None or \
              self.stream_indices[i] == len(self.streams[i][self.substream_names[0]]):
            self.stream_indices[i] = 0
            reached_exhaustion = reached_exhaustion or self.streams_exhausted()
            if reached_exhaustion: exhausted[i] = True
            if not reached_exhaustion or not truncate_at_exhaustion:
              self.reset_stream(i)
            else:
              continue
          for name in self.substream_names:
            batch[name][t, i] = self.streams[i][name][self.stream_indices[i]]
          batch_indicators[t, i] = 0 if self.stream_indices[i] == 0 else 1
          self.stream_indices[i] += 1
          if self.stream_indices[i] == len(self.streams[i][self.substream_names[0]]):
            num_completed_streams += 1
        if not exhausted[i]: all_exhausted = False
      if all_exhausted and truncate_at_exhaustion:
        print ('Exhausted all data; cutting off batch at timestep %d ' +
               'with %d streams completed') % (t, num_completed_streams)
        for name in self.substream_names:
          batch[name] = batch[name][:t, :]
        batch_indicators = batch_indicators[:t, :]
        break
    return batch, batch_indicators

  def get_streams(self):
    raise Exception('get_streams should be overridden to return a dict ' +
                    'of equal-length iterables.')

class HDF5SequenceWriter():
  def __init__(self, sequence_generator, output_dir=None, verbose=False):
    self.generator = sequence_generator
    assert output_dir is not None  # required
    self.output_dir = output_dir
    if os.path.exists(output_dir):
      raise Exception('Output directory already exists: ' + output_dir)
    os.makedirs(output_dir)
    self.verbose = verbose
    self.filenames = []

  def write_batch(self, stop_at_exhaustion=False):
    batch_comps, cont_indicators = self.generator.get_next_batch()
    batch_index = len(self.filenames)
    filename = '%s/batch_%d.h5' % (self.output_dir, batch_index)
    self.filenames.append(filename)
    h5file = h5py.File(filename, 'w')
    dataset = h5file.create_dataset('cont', shape=cont_indicators.shape, dtype=cont_indicators.dtype)
    dataset[:] = cont_indicators
    dataset = h5file.create_dataset('buffer_size', shape=(1,), dtype=np.int)
    dataset[:] = self.generator.batch_num_streams
    for key, batch in batch_comps.iteritems():
      if self.verbose:
        for s in range(self.generator.batch_num_streams):
          stream = np.array(self.generator.streams[s][key])
          print 'batch %d, stream %s, index %d: ' % (batch_index, key, s), stream
      h5dataset = h5file.create_dataset(key, shape=batch.shape, dtype=batch.dtype)
      h5dataset[:] = batch
    h5file.close()

  def write_to_exhaustion(self):
    while not self.generator.streams_exhausted():
      self.write_batch(stop_at_exhaustion=True)

  def write_filelists(self):
    assert self.filenames is not None
    filelist_filename = '%s/hdf5_chunk_list.txt' % self.output_dir
    with open(filelist_filename, 'w') as listfile:
      for filename in self.filenames:
        listfile.write('%s\n' % filename)
