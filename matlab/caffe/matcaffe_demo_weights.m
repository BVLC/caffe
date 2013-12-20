function layers = matcaffe_demo_weights(use_gpu)
% layers = matcaffe_demo_weights(use_gpu)
% 
% Demo of how to extract network parameters ("weights") using the matlab
% wrapper.
%
% input
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   layers   struct array of layers and their weights
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system

% init caffe network (spews logging info)
if caffe('is_initialized') == 0
  model_def_file = '../../examples/imagenet_deploy.prototxt';
  model_file = '../../examples/alexnet_train_iter_470000';
  caffe('init', model_def_file, model_file);
end

% set to use GPU or CPU
if exist('use_gpu', 'var') && use_gpu
  caffe('set_mode_gpu');
else
  caffe('set_mode_cpu');
end

% put into test mode
caffe('set_phase_test');

layers = caffe('get_weights');
