function scores = matcaffe_demo(im, use_gpu)
% scores = matcaffe_demo(im, use_gpu)
% 
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   im       color image as uint8 HxWx3
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000-dimensional ILSVRC score vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);

model_def_file = '../../examples/imagenet_deploy.prototxt';
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

% prepare oversampled input
tic;
input_data = {prepare_image(im)};
toc;

% do forward pass to get scores
tic;
scores = caffe('forward', input_data);
toc;

% average output scores
scores = reshape(scores{1}, [1000 10]);
scores = mean(scores, 2);


% ------------------------------------------------------------------------
function images = prepare_image(im)
% ------------------------------------------------------------------------
d = load('ilsvrc_2012_mean');
IMAGE_MEAN = d.image_mean;
IMAGE_DIM = 256;
CROPPED_DIM = 227;

% resize to fixed input size
im = single(im);
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
im = im(:,:,[3 2 1]) - IMAGE_MEAN;

% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
curr = 1;
for i = indices
  for j = indices
    images(:, :, :, curr) = ...
        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
    curr = curr + 1;
  end
end
center = floor(indices(2) / 2)+1;
images(:,:,:,5) = ...
    permute(im(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), ...
        [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);
