build/tools/caffe train \
    -solver examples/kitti/detectnet_solver.prototxt \
    -weights models/bvlc_googlenet/bvlc_googlenet.caffemodel \
    $@
