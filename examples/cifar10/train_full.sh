#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver.prototxt

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr1.prototxt \
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
=======
>>>>>>> pod/device/blob.hpp
=======
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
=======
>>>>>>> pod/caffe-merge
=======
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
=======
>>>>>>> pod/device/blob.hpp
=======
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
=======
>>>>>>> pod/device/blob.hpp
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate
>>>>>>> origin/BVLC/parallel
=======
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate.h5
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr2.prototxt \
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
=======
>>>>>>> pod/device/blob.hpp
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
=======
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> pod-caffe-pod.hpp-merge
<<<<<<< HEAD
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
=======
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
=======
>>>>>>> pod/device/blob.hpp
=======
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
=======
>>>>>>> pod/caffe-merge
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate
>>>>>>> origin/BVLC/parallel
=======
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod/caffe-merge
=======
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
>>>>>>> device-abstraction
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/caffe-merge
=======
=======
>>>>>>> pod-caffe-pod.hpp-merge
=======
>>>>>>> pod/device/blob.hpp
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
=======
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate
>>>>>>> origin/BVLC/parallel
=======
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate.h5
>>>>>>> caffe
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> pod-caffe-pod.hpp-merge
>>>>>>> pod/device/blob.hpp
=======
=======
>>>>>>> pod/device/blob.hpp
>>>>>>> pod-caffe-pod.hpp-merge
