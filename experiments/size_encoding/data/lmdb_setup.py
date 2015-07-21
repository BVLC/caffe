
import os
import sys
import lmdb
import itertools
import caffe.proto.caffe_pb2
from PIL import Image

MAP_SIZE = int(2 ** 32)

im_dir = sys.argv[1]
encoded_lmdb_file = sys.argv[2]
unencoded_lmdb_file = sys.argv[3]

env_encoded = lmdb.open(encoded_lmdb_file, map_size=MAP_SIZE)
env_unencoded = lmdb.open(unencoded_lmdb_file, map_size=MAP_SIZE)

txn_encoded = env_encoded.begin(write=True)
txn_unencoded = env_unencoded.begin(write=True)

for x, f in enumerate(os.listdir(im_dir)):
	r = os.path.join(im_dir, f)
	im_encoded = open(r, 'rb').read()
	im = Image.open(r)
	im.load()

	key = str(x)
	d1 = caffe.proto.caffe_pb2.Datum()
	d2 = caffe.proto.caffe_pb2.Datum()

	# encoded image
	d1.channels = 3
	d1.height = im.size[1]
	d1.width = im.size[0]
	d1.data = im_encoded
	d1.encoded = True

	d2.channels = 3
	d2.height = im.size[1]
	d2.width = im.size[0]
	pixel_vals = list(itertools.chain(*(list(im.getdata()))))
	pixel_vals = "".join(map(chr, pixel_vals))
	d2.data = pixel_vals

	txn_encoded.put(key, d1.SerializeToString())
	txn_unencoded.put(key, d2.SerializeToString())

txn_encoded.commit()
env_encoded.close()
txn_unencoded.commit()
env_unencoded.close()

