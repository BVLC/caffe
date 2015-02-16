#!/usr/bin/env bash
#
# Downloads Andrej Karpathy's train/val/test splits of COCO2014 as text files.

# change to directory $DIR where this script is stored
pushd .
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR

FILENAME=coco2014_aux.tar.gz

echo "Downloading..."

wget http://dl.caffe.berkeleyvision.org/$FILENAME

echo "Unzipping to $DIR"

tar -xf $FILENAME && rm -f $FILENAME

echo "Done."

# change back to original working directory
popd
