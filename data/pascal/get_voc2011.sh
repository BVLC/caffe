#!/usr/bin/env sh
# This scripts downloads the pascal voc data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

# Filenames to download
FN_VOC=VOCtrainval_25-May-2011.tar

# Name of extracted file
EX_VOC=TrainVal

# Name of the datafile
DF_VOC=VOC2011

if [ ! -e $EX_VOC ]; then
  wget --no-check-certificate http://host.robots.ox.ac.uk/pascal/VOC/voc2011/${FN_VOC}
  tar -xf ${FN_VOC}
fi

if [ ! -e $DF_VOC ]; then
  ln -s ${EX_VOC}/VOCdevkit/${DF_VOC} ${DF_VOC}
  cp seg11valid.txt ${DF_VOC}/ImageSets/Segmentation/
fi

