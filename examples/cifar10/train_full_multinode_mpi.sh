#!/usr/bin/env sh

TOOLS=./build/tools

echo "starting data server"
OMP_NUM_THREADS=1 $TOOLS/caffe data_server \
    --solver=examples/cifar10/cifar10_full_solver_data_server.prototxt  \
    --listen_address=tcp://*:8888 2>/dev/null &

sleep 1
echo "executing 4 nodes with mpirun"

OMP_NUM_THREADS=1 \
mpirun -v \
-host 127.0.0.1 -n 4 \
$TOOLS/caffe train -v 0 --solver=examples/cifar10/cifar10_full_solver_sync_param_server.prototxt --param_server=mpi

