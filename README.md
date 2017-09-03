# Caffe for DeepDetect

This is a slightly modified version of Caffe as used by the [Deep Learning API & server Deepdetect](https://github.com/beniz/deepdetect). The repository is kept up to date with the original Caffe master branch.

Improvements and new features include:

- Switch from `LOG(FATAL)` error to `CaffeErrorException` thrown on every recoverable errors. This allows the safe use of Caffe as a C++ library from external applications, and in production
- Various fixes, including ability to run the exact same job in parallel
- Makefile fixes with default build supporting all NVIDIA architectures
- Sparse inputs and CPU/GPU computations
- Support for class weights applied to Softmax loss, useful for training over imbalanced datasets
- SSD: Single Shot MultiBox Detector for object detection in images
- Support for lightweight nets via accelerated depthwise convolutions (https://github.com/BVLC/caffe/pull/5665) and shufflenet layer (https://github.com/farmingyard/ShuffleNet).
- Support for image segmentation, via PSPNet, U-Net, SegNet, etc...
- Support for Squeeze & Excitation Nets (https://github.com/hujie-frank/SENet).

While this is intended to be used with DeepDetect, this is a great alternative to the original Caffe if you'd like to avoid uncaptured errors, train from text or sparse data, need built-in image detection.
