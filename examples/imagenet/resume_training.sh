#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=models/bvlc_reference_caffenet/solver.prototxt \
<<<<<<< HEAD
    --snapshot=models/bvlc_reference_caffenet/caffenet_train_10000.solverstate.h5
=======
<<<<<<< HEAD
<<<<<<< HEAD
    --snapshot=models/bvlc_reference_caffenet/caffenet_train_10000.solverstate.h5
=======
    --snapshot=models/bvlc_reference_caffenet/caffenet_train_10000.solverstate
>>>>>>> origin/BVLC/parallel
=======
    --snapshot=models/bvlc_reference_caffenet/caffenet_train_10000.solverstate.h5
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
