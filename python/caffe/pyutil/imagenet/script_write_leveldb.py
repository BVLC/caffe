"""Writes to a leveldb from a bunch of files.

This program converts a set of images to a leveldb by storing them as Datum
proto buffers. The input file should be a list of files as well as their labels,
in the format of
    file1.JPEG 0
    file2.JPEG 1
    ....
For the leveldb, the keys will be a monotonically increasing id followed by the
filename. If --shuffle, we will shuffle the lines before writing to leveldb,
which will make a random order easier for training.

To make the output consistent with the C++ code, we will store the images in
BGR format.

Copyright 2013 Yangqing Jia
"""

import gflags
import leveldb
import numpy as np
import os
import random
from skimage import io
import sys

from caffe.pyutil import convert

BATCH_SIZE=256

gflags.DEFINE_string("filename", "", "The input file name.")
gflags.DEFINE_string("input_folder", "", "The input folder that stores images.")
gflags.DEFINE_string("db_name", "", "The output leveldb name.")
gflags.DEFINE_bool("shuffle", False,
    "If True, shuffle the lines before writing.")
FLAGS = gflags.FLAGS

def write_db():
  """The main script to write the leveldb database."""
  db = leveldb.LevelDB(FLAGS.db_name, write_buffer_size=268435456,
      create_if_missing=True, error_if_exists=True)
  lines = [line.strip() for line in open(FLAGS.filename)]
  if FLAGS.shuffle:
    random.shuffle(lines)
  total = len(lines)
  key_format = '%%0%dd_%%s' % len(str(total))
  batch = leveldb.WriteBatch()
  for line_id, line in enumerate(lines):
    imagename, label = line.split(' ')
    label = int(label)
    img = io.imread(os.path.join(FLAGS.input_folder, imagename))
    if img.ndim == 2:
      img = np.tile(img, (1,1,3))
    # convert to BGR, and then swap the axes.
    img = img[::-1].swapaxes(1,2).swapaxes(0,1)
    datum = convert.array_to_datum(img, label=label)
    batch.Put(key_format % (line_id, imagename), datum.SerializeToString())
    if line_id > 0 and line_id % 1000 == 0:
      print '%d of %d done.' % (line_id, total)
    if line_id > 0 and line_id % BATCH_SIZE == 0:
      # Write the current batch and start a new batch.
      db.Write(batch)
      batch = leveldb.WriteBatch()
  # finishing the job.
  del db
  return

if __name__ == '__main__':
  FLAGS(sys.argv)
  write_db()
