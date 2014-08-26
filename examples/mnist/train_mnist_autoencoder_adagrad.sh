#!/bin/bash
TOOLS=../../build/tools

$TOOLS/caffe.bin train --solver=mnist_autoencoder_solver.prototxt
