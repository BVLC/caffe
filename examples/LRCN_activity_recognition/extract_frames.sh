#!/bin/bash

EXPECTED_ARGS=2
E_BADARGS=65

if [ $# -lt $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` video frames/sec [size=256]"
  exit $E_BADARGS
fi

NAME=${1%.*}
FRAMES=$2
BNAME=`basename $NAME`
echo $BNAME
mkdir -m 755 $BNAME

ffmpeg -i $1 -r $FRAMES $BNAME/$BNAME.%4d.jpg
