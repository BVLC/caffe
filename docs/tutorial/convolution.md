---
title: Convolution
---
# Caffeinated Convolution

The Caffe strategy for convolution is to reduce the problem to matrix-matrix multiplication.
This linear algebra computation is highly-tuned in BLAS libraries and efficiently computed on GPU devices.

For more details read Yangqing's [Convolution in Caffe: a memo](https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo).

As it turns out, this same reduction was independently explored in the context of conv. nets by

> K. Chellapilla, S. Puri, P. Simard, et al. High performance convolutional neural networks for document processing. In Tenth International Workshop on Frontiers in Handwriting Recognition, 2006.
