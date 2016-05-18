#!/usr/bin/env sh
# This scripts downloads the mnist data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

url=http://host.robots.ox.ac.uk/pascal/VOC/voc2012

target=VOC2012
for fname in VOCtrainval_11-May-2012
do
    if [ ! -e ${target} ]; then
        wget -c --no-check-certificate ${url}/${fname}.tar
        tar --strip-components=1 -xf ${fname}.tar
    fi
done
