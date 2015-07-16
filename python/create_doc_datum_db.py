import os
import sys
import argparse
import caffe
import lmdb
import StringIO
import caffe.proto.caffe_pb2
import random
import json
import traceback
import collections
import pickle
from PIL import Image

LABEL_DELIM = '|'

GL_TYPE_MAP = {
	"1940USFedCen" : {}
}

_countries = list()
_languages = list()
_decade_scale = 1. / 500
_decade_shift = -1500 * _decade_scale
_collections = list()
_col_count_scale = 1. / 44
_col_count_shift = 0
_record_scale = 1. / 40
_record_shift = 0
_per_image_scale = 1. / 2
_per_image_shift = 0
_machine_text = {'None': 0, 'Some': 0.5, 'Most': 1.0}
_hand_text = {'None': 0, 'Some': 0.5, 'Most': 1.0}
_layout_categories = list()
_layout_types = list()
_record_type_broads = list()
_record_type_fines = list()
_media_types = list()

def choose_layout_type(d):
	training_name = d['TrainingSetName']
	if d.get('LayoutTypeDerivedFromManualLabeling'):
		return d['LayoutTypeDerivedFromManualLabeling']
	if d.get('GLLayoutType'):
		gl_type = d['GLLayoutType']
		if training_name in GL_TYPE_MAP:
			mapping = GL_TYPE_MAP[training_name]
			if gl_type in mapping:
				return mapping[gl_type]
	if d.get('LayoutTypeDerivedFromMetadata'):
		return "%s_%s" % (training_name, d.get('LayoutTypeDerivedFromMetadata'))
	return ""
			

def get_size(size, args):
	longer = max(size)
	shorter = min(size)
	if 'x' in args.size_str:
		out_size = tuple(map(int, args.size_str.split('x')))
	elif args.size_str.endswith('l'):
		s = int(args.size_str[:-1])
		scale = float(s) / longer
		new_shorter = int(max(shorter * scale, s / args.truncate))
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
		new_longer = int(min(longer * scale, s * args.truncate))
		mod = (new_longer - s) % args.aspect_ratio_bin
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
	try:
		lines = open(label_file, 'r').readlines()
		l1 = lines[0].rstrip()
		l2 = lines[1].rstrip()
		field_names = l1.split(LABEL_DELIM)
		field_values = l2.split(LABEL_DELIM)
		d = {field_names[i] : field_values[i] for i in xrange(len(field_names))}
		d['layout_type'] = choose_layout_type(d)
		return d
	except:
		return {}


def open_db(db_file):
	env = lmdb.open(db_file, readonly=False, map_size=int(2 ** 38), writemap=True)
	txn = env.begin(write=True)
	return env, txn
	
def package(im, label_info, args):
	doc_datum = caffe.proto.caffe_pb2.DocumentDatum()
	datum_im = doc_datum.image

	datum_im.channels = 3 if im.mode == 'RGB' else 1
	datum_im.width = im.size[0]
	datum_im.height = im.size[1]
	datum_im.encoding = args.encoding

	# image data
	if args.encoding != 'none':
		buf = StringIO.StringIO()
		im.save(buf, args.encoding)
		datum_im.data = buf.getvalue()
	else:
		pix = np.array(im).transpose(2, 0, 1)
		datum_im.data = pix.tostring()

	# meta information
	if label_info.get('id'):
		doc_datum.id = label_info.get('id')
	if label_info.get('DBID'):
		doc_datum.dbid = int(label_info.get('DBID'))
	if label_info.get('ImageName'):
		doc_datum.image_name = label_info.get('ImageName')

	# label fields
	if label_info.get('Country'):
		country = label_info.get('Country')
		doc_datum.country_str = country
		doc_datum.country = one_of_k(country, _countries)
	if label_info.get('Language'):
		language = label_info.get('Language')
		doc_datum.language_str = language
		doc_datum.language = one_of_k(language, _languages)
	if label_info.get('Decade'):
		decade = label_info.get('Decade')
		doc_datum.decade_str = decade
		doc_datum.decade = regression(float(decade), _decade_scale, _decade_shift)
	if label_info.get('TrainingSetName'):
		name = label_info.get('TrainingSetName')
		doc_datum.collection_str = name
		doc_datum.collection = one_of_k(name, _collections)
	if label_info.get('ColumnCount'):
		col_count = label_info.get('ColumnCount')
		doc_datum.column_count_str = col_count
		doc_datum.column_count = regression(float(col_count), _col_count_scale, _col_count_shift)
	if label_info.get('PossibleRecords'):
		poss_records = label_info.get('PossibleRecords')
		doc_datum.possible_records_str = poss_records
		doc_datum.possible_records = regression(float(poss_records), _record_scale, _record_shift)
	if label_info.get('ActualRecords'):
		actual_records = label_info.get('ActualRecords')
		doc_datum.actual_records_str = actual_records
		doc_datum.actual_records = regression(float(actual_records), _record_scale, _record_shift)
	if label_info.get('PagesPerImage'):
		pages = label_info.get('PagesPerImage')
		doc_datum.pages_per_image_str = pages
		doc_datum.pages_per_image = regression(float(pages), _per_image_scale, _per_image_shift)
	if label_info.get('DocsPerImage'):
		docs = label_info.get('DocsPerImage')
		doc_datum.docs_per_image_str = docs
		doc_datum.docs_per_image = regression(float(docs), _per_image_scale, _per_image_shift)
	if label_info.get('MachineText'):
		mt = label_info.get('MachineText')
		doc_datum.machine_text_str = mt
		doc_datum.machine_text = substitute(mt, _machine_text)
	if label_info.get('HWText'):
		hw = label_info.get('HWText')
		doc_datum.hand_text_str = hw
		doc_datum.hand_text = substitute(hw, _hand_text)
	if label_info.get('LayoutCategory'):
		cat = label_info.get('LayoutCategory')
		doc_datum.layout_category_str = cat
		doc_datum.layout_category = one_of_k(cat, _layout_categories)
	if label_info.get('layout_type'):
		ty = label_info.get('layout_type')
		doc_datum.layout_type_str = ty
		doc_datum.layout_type = one_of_k(ty, _layout_types)
	if label_info.get('RecordTypeBroad'):
		rtype = label_info.get('RecordTypeBroad')
		doc_datum.record_type_broad_str = rtype
		doc_datum.record_type_broad = one_of_k(rtype, _record_type_broads)
	if label_info.get('RecordTypeFine'):
		rtype = label_info.get('RecordTypeFine')
		doc_datum.record_type_fine_str = rtype
		doc_datum.record_type_fine = one_of_k(rtype, _record_type_fines)
	if label_info.get('MediaType'):
		mtype = label_info.get('MediaType')
		doc_datum.media_type_str = mtype
		doc_datum.media_type = one_of_k(mtype, _media_types)
	if label_info.get('IsDocument'):
		pred = label_info.get('IsDocument')
		doc_datum.is_document_str = pred
		doc_datum.is_document = binary(pred)
	if label_info.get('IsGraphical'):
		pred = label_info.get('IsGraphical')
		doc_datum.is_graphical_document_str = pred
		doc_datum.is_graphical_document = binary(pred)
	if label_info.get('IsHistorical'):
		pred = label_info.get('IsHistorical')
		doc_datum.is_historical_document_str = pred
		doc_datum.is_historical_document = binary(pred)
	if label_info.get('IsTextual'):
		pred = label_info.get('IsTextual')
		doc_datum.is_textual_document_str = pred
		doc_datum.is_textual_document = binary(pred)

	return doc_datum

def substitute(val, lookup):
	return lookup[val]
	
def one_of_k(val, l):
	if val not in l:
		l.append(val)
	return l.index(val)

def binary(val):
	return 1 if val == 'Y' else 0

def regression(val, scale, shift):
	return val * scale + shift

def list_to_str(l):
	return "\t" + "\n\t".join(map(lambda tup: "%d:\t%s" % tup, enumerate(l)))

def serialize_encoding(args):

	d = {
			"Country": _countries,
			"Language": _languages,
			"Decade": {"shift": _decade_shift, "scale": _decade_scale},
			"TrainingSetName": _collections,
			"ColumnCount": {"shift": _col_count_shift, "scale": _col_count_scale},
			"Records": {"shift": _record_shift, "scale": _record_scale},
			"PerImage": {"shift": _per_image_shift, "scale": _per_image_scale},
			"MachineText": _machine_text,
			"HWText": _hand_text,
			"LayoutCategory": _layout_categories,
			"layout_type": _layout_types,
			"RecordTypeBroad": _record_type_broads,
			"RecordTypeFine": _record_type_fines,
			"MediaType": _media_types
		}
	out = open(args.out_encoding, 'w')
	pickle.dump(d, out)
	out.close()

def load_encoding(args):
	global _countries
	global _languages
	global _decade_scale
	global _decade_shift
	global _collections
	global _col_count_scale
	global _col_count_shift
	global _record_scale
	global _record_shift
	global _per_image_scale
	global _per_image_shift
	global _machine_text
	global _hand_text
	global _layout_categories
	global _layout_types
	global _record_type_broads
	global _record_type_fines
	global _media_types
	if args.initial_encoding:
		_in = open(args.initial_encoding, 'rb')
		d = pickle.load(_in)
		_countries = d["Country"]
		_languages = d["Language"]
		_decade_shift = d["Decade"]["shift"]
		_decade_scale = d["Decade"]["scale"]
		_collections = d["TrainingSetName"]
		_col_count_shift = d["ColumnCount"]["shift"]
		_col_count_scale = d["ColumnCount"]["scale"]
		_record_shift = d["Records"]["shift"]
		_record_scale = d["Records"]["scale"]
		_per_image_shift = d["PerImage"]["shift"]
		_per_image_scale = d["PerImage"]["scale"]
		_machine_text = d["MachineText"]
		_hand_text = d["HWText"]
		_layout_categories = d["LayoutCategory"]
		_layout_types = d["layout_type"]
		_record_type_broad = d["RecordTypeBroad"]
		_record_type_fine = d["RecordTypeFine"]
		_media_types = d["MediaType"]



def print_encoding(args):

	# layout types
	print "Grid Line Layout Type Mappings\n%s\n\n" % json.dumps(GL_TYPE_MAP, sort_keys=True,
																indent=4)
	print "layout_type:\n%s\n" % list_to_str(_layout_types)
	print "Country:\n%s\n" % list_to_str(_countries)
	print "Languag:\n%s\n" % list_to_str(_languages)
	print "TrainingSetName:\n%s\n" % list_to_str(_collections)
	print "LayoutCategory:\n%s\n" % list_to_str(_layout_categories)
	print "RecordTypeBroad:\n%s\n" % list_to_str(_record_type_broads)
	print "RecordTypeFine:\n%s\n" % list_to_str(_record_type_fines)
	print "MediaTypes:\n%s\n\n" % list_to_str(_media_types)

	print "Decade Scale: %f\tShift: %f\n" % (_decade_scale, _decade_shift)
	print "Col Count Scale: %f\tShift: %f\n" % (_col_count_scale, _col_count_shift)
	print "Record Scale: %f\tShift: %f\n" % (_record_scale, _record_shift)
	print "Per Image Scale: %f\tShift: %f\n\n" % (_per_image_scale, _per_image_shift)

	print "Machine Text:\n%s\n" % json.dumps(_machine_text, sort_keys=True, indent=4)
	print "Handwritten Text:\n%s\n\n" % json.dumps(_hand_text, sort_keys=True, indent=4)

	print "Is* : Y=1, N=0\n"

def main(args):
	dbs = {}
	if args.multiple_db:
		try:
			os.makedirs(args.outdb)
		except:
			pass

	load_encoding(args)

	print "Reading Manifest..."
	lines = open(args.manifest, 'r').readlines()
	if args.shuffle:
		print "Shuffling Data..."
		random.shuffle(lines)

	c = collections.Counter()
	for x,line in enumerate(lines):
		if x and x % 1000 == 0:
			print "Processed %d images" % x
		try:
			line = line.rstrip()
			tokens = line.split()
			im_file = os.path.join(args.imroot, tokens[0])
			im = process_im(im_file, args)
			if len(tokens) > 1:
				label_file = tokens[1]
				label_info = process_labels(label_file, args)
			else:
				label_info = {}
			if args.collection_folder and "TrainingSetName" not in label_info:
				label_info["TrainingSetName"] = os.path.basename(os.path.dirname(im_file))
			doc_datum = package(im, label_info, args)
			if args.multiple_db:
				db_file = os.path.join(args.outdb, "%dx%d_lmdb" % im.size)
			else:
				db_file = args.outdb
			if db_file not in dbs:
				dbs[db_file] = open_db(db_file)
				print "\tOpening %s" % db_file
			env, txn = dbs[db_file]
			key = "%d:%s" % (x, os.path.splitext(os.path.basename(im_file))[0])
			txn.put(key, doc_datum.SerializeToString())
			c[db_file] += 1
			if c[db_file] % 1000 == 0:
				txn.commit()
				env.sync()
				print db_file
				print env.stat()
				print env.info()
				txn = env.begin(write=True)
				dbs[db_file] = (env, txn)
		except Exception as e:
			print e
			print traceback.print_exc(file=sys.stdout)


	print "Done Processing Images"

	print_encoding(args)
	serialize_encoding(args)

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
	parser.add_argument('out_encoding', type=str,
						help='where to put the encoding file (pickle file)')
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
	parser.add_argument('-t', '--truncate', type=float, default='2',
						help='Upper/Lower bound on the image AR')
	parser.add_argument('-c', '--collection-folder', default=False, action="store_true",
						help='If the collection or metadata file is missing, use the directory containing the image as the collection')
	parser.add_argument('-i', '--initial-encoding', default="", type=str,
						help='Use the specified encoding of metadata to labels (pickle)')
	parser.add_argument('--no-shuffle', dest="shuffle", default=True, action="store_false",
						help='How to store the image in the DocumentDatum')
	args = parser.parse_args()
	return args



if __name__ == "__main__":
	args = get_args();
	main(args)

