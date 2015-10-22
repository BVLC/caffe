#!/usr/bin/env sh

TOOLS=./build/tools

echo "starting data server"
OMP_NUM_THREADS=1 $TOOLS/caffe data_server \
    --solver=examples/cifar10/cifar10_full_solver_data_server.prototxt  \
    --listen_address=tcp://*:8888 2>/dev/null &

sleep 1

echo "starting param server"
OMP_NUM_THREADS=1 $TOOLS/caffe param_server \
    --solver=examples/cifar10/cifar10_full_solver_sync_param_server.prototxt  \
    --listen_address=tcp://*:7777 2>/dev/null &

sleep 1
echo "starting train node"
OMP_NUM_THREADS=1 $TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_sync_client.prototxt \
    --param_server=tcp://127.0.0.1:7777 2>/dev/null &

echo "starting train node"
OMP_NUM_THREADS=1 $TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_sync_client.prototxt \
    --param_server=tcp://127.0.0.1:7777 2>/dev/null &

echo "starting train node"
OMP_NUM_THREADS=1 $TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_sync_client.prototxt \
    --param_server=tcp://127.0.0.1:7777 2>/dev/null &

echo "starting train node"
OMP_NUM_THREADS=1 $TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_sync_client.prototxt \
    --param_server=tcp://127.0.0.1:7777

