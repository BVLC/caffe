# Confusion Matrix Layer
Written by *Boris Ginsburg*

Loss layer, which  produces [Confusion Matrix](https://en.wikipedia.org/wiki/Confusion_matrix).
 - input : the same as all loss layers
 - output: 2D Matrix (*Blob*)

Sources:
 - include/caffe/loss_layers.hpp: added ConfusionMatrix Layer
 - src/caffe/layers/confusion_matrix_layer.cpp
 - src/caffe/test/test_confusion_matrix_layer.cpp: tests
 - src/caffe/proto/caffe.proto:  added proto parameters
 - src/caffe/solver.cpp: modifed *Solver::Test(.)* to support layers with 2D output
 - plot_conf_matrix.py: visualization

Example:
 - examples/cifar10/cifar10_cm_solver.prototxt
 - examples/cifar10/cifar10_cm_train_test.prototxt
 - examples/cifar10/train_cm.sh
