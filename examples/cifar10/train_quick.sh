#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt

# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
<<<<<<< HEAD
<<<<<<< HEAD
  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate.h5
=======
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate.h5
=======
=======
>>>>>>> pod/caffe-merge
  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate
>>>>>>> origin/BVLC/parallel
=======
  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate.h5
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
