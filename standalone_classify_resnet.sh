nice -20 \
./build/examples/cpp_classification/standalone_classify.bin \
 models/ResNet/ResNet-50-deploy.prototxt \
 models/ResNet/ResNet-50-model.caffemodel \
 models/ResNet/ResNet_mean.binaryproto \
 data/ilsvrc12/synset_words.txt \
 examples/images/cat.jpg
