#!/usr/bin/env sh

TOOLS=./build/tools

echo "executing 4 nodes with mpirun"

OMP_NUM_THREADS=1 \
mpirun -l -host 127.0.0.1 -n 4 \
$TOOLS/caffe train --solver=examples/cifar10/cifar10_full_solver.prototxt --param_server=mpi
