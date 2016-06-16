#!/usr/bin/env sh
# This scripts downloads the semantic countours dataset and extracts it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading..."

# Filenames to download
FN_SBD=benchmark.tgz

# Name of extracted file
EX_SDB=benchmark_RELEASE

# The final datafile
DF_SDB=dataset

#
if [ ! -e $EX_SDB ]; then
  wget --no-check-certificate http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/${FN_SBD}

  tar -xf ${FN_SBD}
fi

if [ ! -e $DF_SDB ]; then
    ln -s ${EX_SDB}/${DF_SDB} ${DF_SDB}
fi
