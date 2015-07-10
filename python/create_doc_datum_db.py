import os
import sys
import argparse
import caffe
import lmdb
import StringIO
import caffe.proto.caffe_pb2
import random
from PIL import Image

def get_size(size, args):
	longer = max(size)
	shorter = min(size)
	if 'x' in args.size_str:
		out_size = tuple(map(int, args.size_str.split('x')))
	elif args.size_str.endswith('l'):
		s = int(args.size_str[:-1])
		scale = float(s) / longer
		new_shorter = int(shorter * scale)
		mod = (new_shorter - s) % args.aspect_ratio_bin
		if mod >= args.aspect_ratio_bin / 2:
			new_shorter += (args.aspect_ratio_bin - mod)
		else:
			new_shorter -= mod
		if longer == size[0]:
			out_size = (s, new_shorter)
		else:
			out_size = (new_shorter, s)
	elif args.size_str.endswith('s'):
		s = int(args.size_str[:-1])
		scale = float(s) / shorter
		new_longer = int(longer * scale)
		mode = (new_longer - s) % args.aspect_ratio_bin
		if mod >= args.aspect_ratio_bin / 2:
			new_longer += (args.aspect_ratio_bin - mod)
		else:
			new_longer -= mod
		if shorter == size[0]:
			out_size = (s, new_longer)
		else:
			out_size = (new_longer, s)
	else:
		out_size = (int(args.size_str), int(args.size_str))
	return out_size
	

def process_im(im_file, args):
	im = Image.open(im_file)
	size = get_size(im.size, args)
	im = im.resize(size)
	if args.gray:
		im = im.convert('L')
	else:
		im = im.convert('RGB')
	return im


def process_labels(label_file, args):
	return {}


def open_db(db_file):
	env = lmdb.open(db_file, readonly=False, map_size=int(2 ** 38))
	txn = env.begin(write=True)
	return env, txn
	
def package(im, label_info, args):
	doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
	datum_im = doc_datum.image

	datum_im.channels = 3 if im.mode == 'RGB' else 1
	datum_im.width = im.size[0]
	datum_im.height = im.size[1]
	datum_im.encoding = args.encoding

	if args.encoding != 'none':
		buf = StringIO.StringIO()
		im.save(buf, args.encoding)
		datum_im.data = buf.getvalue()
	else:
		pix = np.array(im).transpose(2, 0, 1)
		datum_im.data = pix.tostring()

	return doc_datum
	

def main(args):
	dbs = {}
	if args.multiple_db:
		try:
			os.makedirs(args.outdb)
		except:
			pass

	print "Reading Manifest..."
	lines = open(args.manifest, 'r').readlines()
	if args.shuffle:
		print "Shuffling Data..."
		random.shuffle(lines)

	for x,line in enumerate(lines):
		if x and x % 1000:
			print "Processed %d images" % x
		line = line.rstrip()
		tokens = line.split()
		im_file = os.path.join(args.imroot, tokens[0])
		im = process_im(im_file, args)
		if len(tokens) > 1:
			label_file = tokens[1]
			label_info = process_labels(label_file, args)
		else:
			label_info = {}
		doc_datum = package(im, label_info, args)
		if args.multiple_db:
			db_file = os.path.join(args.outdb, "%dx%d_lmdb" % im.size)
		else:
			db_file = args.outdb
		if db_file not in dbs:
			dbs[db_file] = open_db(db_file)
		env, txn = dbs[db_file]
		key = "%d:%s" % (x, os.path.splitext(os.path.basename(im_file))[0])
		txn.put(key, doc_datum.SerializeToString())

	print "Done Processing Images"

	for key, val in dbs.items():
		print "Closing DB: ", key
		env, txn = val
		txn.commit()
		env.close()


def get_args():
	parser = argparse.ArgumentParser(description="Creates an LMDB of DocumentDatums")
	parser.add_argument('manifest', type=str,
						help='file listing image-paths and metadata-paths, one per line')
	parser.add_argument('imroot', type=str,
						help='path to the root of the image dir')
	parser.add_argument('outdb', type=str,
						help='where to put the db')
	parser.add_argument('-g', '--gray', default=False, action="store_true",
						help='Force images to be grayscale.  Force color if ommited')
	parser.add_argument('-s', '--size-str', type=str, default="",
						help='The size string: e.g. 256, 256x384, 256l, 384s')
	parser.add_argument('-b', '--aspect-ratio-bin', type=int, default=32,
						help='When sizing image, round it to the nearest aspect ratio')
	parser.add_argument('-m', '--multiple-db', default=False, action="store_true",
						help='Output a db for each image size in the $outdb directory')
	parser.add_argument('-e', '--encoding', type=str, default='none',
						help='How to store the image in the DocumentDatum')
	parser.add_argument('--no-shuffle', dest="shuffle", default=True, action="store_false",
						help='How to store the image in the DocumentDatum')
	args = parser.parse_args()
	return args



if __name__ == "__main__":
	args = get_args();
	main(args)

