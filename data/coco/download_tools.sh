#!/usr/bin/env bash

# change to directory $DIR where this script is stored
pushd .
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
cd $DIR

git clone https://github.com/pdollar/coco.git

# change back to original working directory
popd

echo "Cloned COCO tools to: $DIR/coco"
echo "To setup COCO tools (and optionally download data), run:"
echo "    cd $DIR/coco"
echo "    python setup.py install"
echo "and follow the prompts."
