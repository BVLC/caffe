This branch contains code of fast forward and backward propagation for 
pixelwise classification of the following paper:

H. Li, R. Zhao, X. Wang, "Highly Efficient Forward and Backward 
Propagation of Convolutional Neural Networks for Pixelwise 
Classification", arXiv:1412.4526

The implementation is based on Caffe's conv_layer and pooling_layer



Two new layer types "CONVOLUTION_SK" and "POOLING_SK" are added.

	1. A "kstride" variable is added to ConvolutionParameters
	   and PoolingParameters. It corresponds to the "d" value
	   in the paper.

	2. Only max-pooling was implemented by now.

	3. Only GPU function of the two layers were implemented by now.

	3. "pad" for the two new layer types must be set to 0.



For questions, please contact lihongsheng AT gmail DOT com.
