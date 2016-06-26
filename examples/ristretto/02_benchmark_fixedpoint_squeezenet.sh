#!/usr/bin/env sh

./build/tools/caffe test \
	--model=models/SqueezeNet/RistrettoDemo/quantized.prototxt \
	--weights=models/SqueezeNet/RistrettoDemo/squeezenet_finetuned.caffemodel \
	--gpu=0 --iterations=2000
