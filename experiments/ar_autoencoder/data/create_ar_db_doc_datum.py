
import sys
import lmdb
import caffe.proto.caffe_pb2
import Image
import StringIO

#sizes=[128,192,256,320,384,448,512]
sizes=[int(sys.argv[3])]
resolution=1./32
start = 0.5
start = 1
end = 1

dims = set()
for d1 in sizes:
	ar = start
	while ar <= end:
		d2 = int(ar * d1)
		dims.add( (d1, d2) )
		dims.add( (d2, d1) )
		ar += resolution

in_image = sys.argv[1]
db_file = sys.argv[2]

im = Image.open(in_image)
im = im.convert("RGB")

env = lmdb.open(db_file, readonly=False, map_size=int(2 ** 32))
txn = env.begin(write=True)

for dim in dims:
	print dim
	key = str(dim)
	d = caffe.proto.caffe_pb2.DocumentDatum()
	d.id = dim[0]
	i = d.image

	d.image_name = in_image

	d.country_str = "USA"
	d.country = 0

	d.language_str = "english"
	d.language = 1

	d.decade_str = "1940"
	d.decade = 0.75

	d.column_count_str = "35"
	d.column_count = 35

	d.possible_records_str = "5"
	d.possible_records = 5

	d.actual_records_str = "10"
	d.actual_records = 10

	d.pages_per_image_str = "2"
	d.pages_per_image = 2

	d.docs_per_image_str = "2"
	d.docs_per_image = 2

	d.machine_text_str = "some"
	d.machine_text = 0.5

	d.hand_text_str = "most"
	d.hand_text = 1

	d.layout_category_str = "table"
	d.layout_category = 3

	d.layout_type_str = "1940_general"
	d.layout_type = 5

	d.record_type_broad_str = "census"
	d.record_type_broad = 7

	d.record_type_fine_str = "census_population"
	d.record_type_fine = 14

	d.media_type_str = "microfilm"
	d.media_type = 3

	d.is_document_str = "y"
	d.is_document = 1

	d.is_graphical_document_str = "n"
	d.is_graphical_document = 0

	d.is_historical_document_str = "y"
	d.is_historical_document = 1

	d.is_textual_document_str = "y"
	d.is_textual_document = 1

	d.collection_str = "1234"
	d.collection = 1

	buf = StringIO.StringIO()
	r = im.resize(dim)
	r.save(buf, "JPEG")
	i.data = buf.getvalue()
	i.encoding = "JPEG"

	i.channels = 3
	i.width = r.size[0]
	i.height = r.size[1]

	txn.put(key, d.SerializeToString())
	buf.close()

txn.commit()
env.close()

