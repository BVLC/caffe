#!/usr/bin/env sh

./build/tools/caffe test --model=models/SqueezeNet/demo/quantized.prototxt \
	--weights=models/SqueezeNet/demo/squeezenet_finetuned.caffemodel \
	--gpu=0 --iterations=2000
