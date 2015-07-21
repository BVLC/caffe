from PIL import Image
import sys

for infile in sys.argv[1:]:
	im = Image.open(infile)
	print infile, im.format, "%dx%d" % im.size, im.mode
