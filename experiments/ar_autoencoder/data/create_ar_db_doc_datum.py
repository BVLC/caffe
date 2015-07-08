
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

