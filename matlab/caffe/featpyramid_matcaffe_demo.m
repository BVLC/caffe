function pyra = featpyramid_matcaffe_demo(imfn, use_gpu)
% scores = matcaffe_demo(im, use_gpu)
% 
% Demo of the matlab wrapper using the ILSVRC network to produce a feature pyramid.
%
% input
%   imfn     image filename
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   N row structure array with two fields: scale, feats
%	 scale: scalar
%        feats: 2D numeric array
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  imfn = '../../examples/cat.jpg';
%  pyra = featpyramid_matcaffe_demo(imfn, 1);

if ~exist('imfn', 'var') 
   imfn = '/home/moskewcz/svn_work/boda/test/pascal/000001.jpg'
end

%model_def_file = '../../examples/imagenet_deploy.prototxt';
model_def_file = '../../python/caffe/imagenet/imagenet_rcnn_batch_1_input_2000x2000_output_conv5.prototxt' 
% NOTE: you'll have to get the pre-trained ILSVRC network
model_file = '../../examples/alexnet_train_iter_470000';

% init caffe network (spews logging info)
caffe('init', model_def_file, model_file);

% set to use GPU or CPU
if exist('use_gpu', 'var') && use_gpu
  caffe('set_mode_gpu');
else
  caffe('set_mode_cpu');
end

% put into test mode
caffe('set_phase_test');

pyra = caffe('convnet_featpyramid', imfn );

