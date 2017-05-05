# Using model fuse to run inference-ontpimzed caffe

The example use fused-model prototxt and weightfile to using layer-fused classification.

Take googlenet as an example:

1. Download GoogleNet model form "Model Zoo" using following script:
```
 $CAFFE_ROOT/scripts/download_model_binary.py models/bvlc_googlenet
```
2. ImageNet label file required by:
```
 $CAFFE_ROOT/data/ilsvrc12/get_ilsvrc_aux.sh
```
3. Use model_fuse.py to generate fused model and cpp_classifcation to test the clasify funtionality with script:
```
 ./googlenet_inference_test.sh
``` 
