"""This script converts images stored in the ILSVRC format to a leveldb,
converting every image to a 256*256 image as well as converting them to channel
first storage. The output will be shuffled - so that a sequential read will
result in pseudo-random minibatches.
"""

from decaf import util
from decaf.util import transform
import glob
import leveldb
import numpy as np
import random
import os
from skimage import io
import sys

from caffe.proto import caffe_pb2

def main(argv):
  root = argv[0]
  db_name = argv[1]
  db = leveldb.LevelDB(db_name)
  synsets = glob.glob(os.path.join(root, "n????????"))
  synsets.sort()
  print 'A total of %d synsets' % len(synsets)
  all_files = [glob.glob(os.path.join(root, synset, "*.JPEG"))
      for synset in synsets]
  all_labels = [[i] * len(files) for i, files in enumerate(all_files)]
  all_files = sum(all_files, [])
  all_labels = sum(all_labels, [])
  print 'A total of %d files' % len(all_files)
  random_indices = list(range(len(all_files)))
  random.shuffle(random_indices)
  datum = caffe_pb2.Datum()
  datum.blob.num = 1
  datum.blob.channels = 3
  datum.blob.height = 256
  datum.blob.width = 256
  my_timer = util.Timer()
  batch = leveldb.WriteBatch()
  for i in range(len(all_files))[:1281]:
    filename = all_files[random_indices[i]]
    basename = os.path.basename(filename)
    label = all_labels[random_indices[i]]
    image = io.imread(filename)
    image = transform.scale_and_extract(transform.as_rgb(image), 256)
    image = np.ascontiguousarray(image.swapaxes(1,2).swapaxes(0,1))
    del datum.blob.data[:]
    datum.blob.data.extend(list(image.flatten()))
    datum.label = label
    batch.Put('%d_%d_%s' % (i, label, basename),
        datum.SerializeToString())
    print '(%d %s) Wrote file %s' % (i, my_timer.total(), basename)
    if (i % 256 and i > 0):
      # write and start a new batch
      db.Write(batch)
      batch = leveldb.WriteBatch()

if __name__ == '__main__':
  if len(sys.argv) != 3:
    print 'Usage: convert_ilsvrc.py DATA_ROOT OUTPUT_DB'
  else:
    main(sys.argv[1:])