function [scores, weights] = matcaffe_demo(im, use_gpu)
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
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda-5.5/lib64
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  im = imread('../../examples/cat.jpg');
%  scores = matcaffe_demo(im, 1);
%  [score, class] = max(scores);
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order 
%   [width, height, channels, images]
% where width is the fastest dimension.
% Here is the rough matlab for putting image data into the correct
% format:
%   % convert from uint8 to single
%   im = single(im);
%   % reshape to a fixed size (e.g., 227x227)
%   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
%   % permute from RGB to BGR and subtract the data mean (already in BGR)
%   im = im(:,:,[3 2 1]) - data_mean;
%   % flip width and height to make width the fastest dimension
%   im = permute(im, [2 1 3]);

% If you have multiple images, cat them with cat(4, ...)

% The actual forward function. It takes in a cell array of 4-D arrays as
% input and outputs a cell array. 


% init caffe network (spews logging info)
if nargin < 1
  % For demo purposes we will use the peppers image
  im = imread('peppers.png');
end

if caffe('is_initialized') == 0
  model_def_file = '../../models/imagenet.prototxt';
  model_file = '../../models/caffe_reference_imagenet_model';
  if exist(model_file, 'file') == 0
    % NOTE: you'll have to get the pre-trained ILSVRC network
    error('You need a network model file');
  end
  if ~exist(model_def_file,'file')
    % NOTE: you'll have to get network definition
    error('You need the network prototxt definition');
  end
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

% prepare oversampled input
% input_data is Height x Width x Channel x Num
tic;
input_data = {prepare_image(im)};
toc;

% do forward pass to get scores
% scores are now Width x Height x Channels x Num
tic;
scores = caffe('forward', input_data);
toc;

% One can permute back Width and Height.
scores = scores{1};
size(scores)
scores = squeeze(scores);
scores = mean(scores,2);

% you can also get network weights by calling
if nargout == 2
  weights = caffe('get_weights');
end

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
