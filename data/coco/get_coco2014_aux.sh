#!/usr/bin/env sh
#
# Downloads Andrej Karpathy's train/val/test splits of COCO2014 as text files.

echo "Downloading..."

wget http://dl.caffe.berkeleyvision.org/coco2014_aux.tar.gz

echo "Unzipping..."

tar -xf coco2014_aux.tar.gz && rm -f coco2014_aux.tar.gz

echo "Done."
