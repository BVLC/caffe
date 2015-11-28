#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver.prototxt

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr1.prototxt \
<<<<<<< HEAD
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
=======
<<<<<<< HEAD
<<<<<<< HEAD
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
=======
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate
>>>>>>> origin/BVLC/parallel
=======
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr2.prototxt \
<<<<<<< HEAD
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
=======
<<<<<<< HEAD
<<<<<<< HEAD
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
=======
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate
>>>>>>> origin/BVLC/parallel
=======
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
>>>>>>> caffe
>>>>>>> pod-caffe-pod.hpp-merge
