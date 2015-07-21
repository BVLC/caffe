
import lmdb
import caffe.proto.caffe_pb2


SIZE=127
DB1="data/parallel_test_data_lmdb"
DB2="data/parallel_test_labels_lmdb"

env1 = lmdb.open(DB1, readonly=False)
env2 = lmdb.open(DB2, readonly=False)

txn1 = env1.begin(write=True)
txn2 = env2.begin(write=True)

for x in xrange(SIZE):
	key = str(x)
	d1 = caffe.proto.caffe_pb2.Datum()
	d2 = caffe.proto.caffe_pb2.Datum()

	d1.channels = 1
	d1.height = 1
	d1.width = 1
	d1.data = chr(x)

	d2.channels = 1
	d2.height = 1
	d2.width = 1
	d2.data = chr(SIZE - x)

	txn1.put(key, d1.SerializeToString())
	txn2.put(key, d2.SerializeToString())

txn1.commit()
txn2.commit()
env1.close()
env2.close()

