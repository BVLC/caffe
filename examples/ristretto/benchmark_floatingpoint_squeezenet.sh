#!/usr/bin/env sh

./build/tools/caffe test \
	--model=models/SqueezeNet/train_val.prototxt \
	--weights=models/SqueezeNet/squeezenet_v1.0.caffemodel \
	--gpu=0 --iterations=2000
