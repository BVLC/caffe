#!/usr/bin/env sh

./build/tools/caffe train \
	--solver=models/SqueezeNet/RistrettoDemo/solver_finetune.prototxt \
	--weights=models/SqueezeNet/squeezenet_v1.0.caffemodel
