#!/usr/bin/env sh
# Train the pascal multi-label classification task.
# This script is run from the caffe root as:
#   ./examples/pascal/train_pascal_image_layer.sh
#
# This requires that the pascal data has been created:
#   ./data/pascal/get_pascal.sh
# And the input image file lists have been created:
#   python ./examples/pascal/create_file_list.py
#

./build/tools/caffe train --solver=examples/pascal/solver_image_layer.prototxt --weights=models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel
