cd ~/workspace/caffe-rpi;
nice -20 \
./build/examples/cpp_classification/standalone_classify.bin \
 SqueezeNet/SqueezeNet_v1.0/deploy.prototxt \
 SqueezeNet/SqueezeNet_v1.0/squeezenet_v1.0.caffemodel \
 data/ilsvrc12/imagenet_mean.binaryproto \
 data/ilsvrc12/synset_words_short.txt \
 examples/images/cat.jpg
