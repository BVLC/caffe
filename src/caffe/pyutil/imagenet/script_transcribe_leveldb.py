"""This script is an ad-hoc solution to translate an old leveldb database to a
new one. The use case is that py-leveldb is compiled against leveldb 1.7 while
newer versions changed the storage format. As a result I compiled py-leveldb
with leveldb 1.14 and had to transcribe the database.

To use this, put the old python leveldb library as old/leveldb.so and the new one
as new/leveldb.so, and run this script.

Copyright 2013 Yangqing Jia
"""

import gflags
import sys

import old.leveldb
import new.leveldb

BATCH_SIZE=256

gflags.DEFINE_string("in_db_name", "", "The output leveldb name.")
gflags.DEFINE_string("out_db_name", "", "The output leveldb name.")
FLAGS = gflags.FLAGS

def transcribe_db():
  """The main script to write the leveldb database."""
  in_db = old.leveldb.LevelDB(FLAGS.in_db_name,
      create_if_missing=False, error_if_exists=Fallse)
  out_db = new.leveldb.LevelDB(FLAGS.out_db_name, write_buffer_size=268435456,
      create_if_missing=False, error_if_exists=Fallse)
  batch = new.leveldb.WriteBatch()
  count = 0
  for key, value in in_db.RangeIter():
    batch.Put(key, value)
    if count % BATCH_SIZE == 0 and count > 0:
      # Write the current batch and start a new batch.
      out_db.Write(batch)
      batch = new.leveldb.WriteBatch()
    count += 1
  return

if __name__ == '__main__':
  FLAGS(sys.argv)
  transcribe_db()