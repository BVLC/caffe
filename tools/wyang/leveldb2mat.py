import leveldb
import feat_helper_pb2
import numpy as np
import scipy.io as sio
import time

def main(argv):
	leveldb_name = sys.argv[1]
	print "%s" % sys.argv[1]
	batch_num = int(sys.argv[2]);
	batch_size = int(sys.argv[3]);
	window_num = batch_num*batch_size;

	start = time.time()
	if 'db' not in locals().keys():
		db = leveldb.LevelDB(leveldb_name)
		datum = feat_helper_pb2.Datum()

	ft = np.zeros((window_num, int(sys.argv[4])))
	for im_idx in range(window_num):
		datum.ParseFromString(db.Get('%d' %(im_idx)))
		ft[im_idx, :] = datum.float_data

	print 'time 1: %f' %(time.time() - start)
	sio.savemat(sys.argv[5], {'feats':ft})
	print 'time 2: %f' %(time.time() - start)
	print 'done!'

	#leveldb.DestroyDB(leveldb_name)

if __name__ == '__main__':
	import sys
	main(sys.argv)
