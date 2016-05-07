nice -20 \
./build/examples/cpp_classification/standalone_classify.bin \
 models/bvlc_googlenet/deploy.prototxt \
 models/bvlc_googlenet/bvlc_googlenet.caffemodel \
 data/ilsvrc12/imagenet_mean.binaryproto \
 data/ilsvrc12/synset_words.txt \
 examples/images/cat.jpg
