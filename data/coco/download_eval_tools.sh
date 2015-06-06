#!/usr/bin/env bash

# change to directory $DIR where this script is stored
pushd .
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR

OUTFILE=coco_caption_eval.zip
wget --no-check-certificate https://github.com/jeffdonahue/coco-caption/archive/master.zip -O $OUTFILE
unzip $OUTFILE
mv coco-caption-master coco-caption-eval

# change back to original working directory
popd

echo "Downloaded COCO evaluation tools to: $DIR/coco-caption-eval"
