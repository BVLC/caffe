function scores = matcaffe_demo_vgg(im, use_gpu, model_def_file, model_file, mean_file)
% scores = matcaffe_demo_vgg(im, use_gpu, model_def_file, model_file, mean_file)
%
% Demo of the matlab wrapper using the networks described in the BMVC-2014 paper "Return of the Devil in the Details: Delving Deep into Convolutional Nets"
%
% INPUT
%   im - color image as uint8 HxWx3
%   use_gpu - 1 to use the GPU, 0 to use the CPU
%   model_def_file - network configuration (.prototxt file)
%   model_file - network weights (.caffemodel file)
%   mean_file - mean BGR image as uint8 HxWx3 (.mat file)
%
% OUTPUT
%   scores   1000-dimensional ILSVRC score vector
%
% EXAMPLE USAGE
%  model_def_file = 'zoo/VGG_CNN_F_deploy.prototxt';
%  model_file = 'zoo/VGG_CNN_F.caffemodel';
%  mean_file = 'zoo/VGG_mean.mat';
%  use_gpu = true;
%  im = imread('../../examples/images/cat.jpg');
%  scores = matcaffe_demo_vgg(im, use_gpu, model_def_file, model_file, mean_file);
% 
% NOTES
%  the image crops are prepared as described in the paper (the aspect ratio is preserved)
%
% PREREQUISITES
%  You may need to do the following before you start matlab:
%   $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%   $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
%  Or the equivalent based on where things are installed on your system

% init caffe network (spews logging info)
matcaffe_init(use_gpu, model_def_file, model_file);

% prepare oversampled input
% input_data is Height x Width x Channel x Num
tic;
input_data = {prepare_image(im, mean_file)};
toc;

% do forward pass to get scores
% scores are now Width x Height x Channels x Num
tic;
scores = caffe('forward', input_data);
toc;

scores = scores{1};
% size(scores)
scores = squeeze(scores);
% scores = mean(scores,2);

% [~,maxlabel] = max(scores);

% ------------------------------------------------------------------------
function images = prepare_image(im, mean_file)
% ------------------------------------------------------------------------
IMAGE_DIM = 256;
CROPPED_DIM = 224;

d = load(mean_file);
IMAGE_MEAN = d.image_mean;

% resize to fixed input size
im = single(im);

if size(im, 1) < size(im, 2)
    im = imresize(im, [IMAGE_DIM NaN]);
else
    im = imresize(im, [NaN IMAGE_DIM]);
end

% RGB -> BGR
im = im(:, :, [3 2 1]);

% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');

indices_y = [0 size(im,1)-CROPPED_DIM] + 1;
indices_x = [0 size(im,2)-CROPPED_DIM] + 1;
center_y = floor(indices_y(2) / 2)+1;
center_x = floor(indices_x(2) / 2)+1;

curr = 1;
for i = indices_y
  for j = indices_x
    images(:, :, :, curr) = ...
        permute(im(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :)-IMAGE_MEAN, [2 1 3]);
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
    curr = curr + 1;
  end
end
images(:,:,:,5) = ...
    permute(im(center_y:center_y+CROPPED_DIM-1,center_x:center_x+CROPPED_DIM-1,:)-IMAGE_MEAN, ...
        [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);
