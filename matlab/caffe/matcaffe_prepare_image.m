function images = matcaffe_prepare_image(img, img_mean)
% Prepare an image for input to caffe
% images = matcaffe_prepare_image(img, img_mean)
% 
% img_mean should be a 256x256x3 floating point matrix with values ranging from 0 to 255
% 
% Five things to be aware of:
%   caffe uses row-major order
%   matlab uses column-major order
%   caffe uses BGR color channel order
%   matlab uses RGB color channel order
%   images need to have the data mean subtracted

% Data coming in from matlab needs to be in the order 
%   [width, height, channels, images]
% where width is the fastest dimension.

% By Daniel Golden (dan at cellscope dot com) August 2014

%% Error checking
assert(isfloat(img_mean) && ...
       any(img_mean(:) > 1) && ...
       all(img_mean(:) >= 0 & img_mean(:) <= 255) && ...
       isequal(size(img_mean), [256 256 3]), ...
       'img_mean should be a 256x256x3 floating point matrix with values ranging from 0 to 255');


%% Go

IMAGE_MEAN = img_mean;
IMAGE_DIM = 256;
CROPPED_DIM = 227;

% resize to fixed input size
img = single(img);
img = imresize(img, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR (IMAGE_MEAN is already BGR)
img = img(:,:,[3 2 1]) - IMAGE_MEAN;

% oversample (4 corners, center, and their x-axis flips)
images = zeros(CROPPED_DIM, CROPPED_DIM, 3, 10, 'single');
indices = [0 IMAGE_DIM-CROPPED_DIM] + 1;
curr = 1;
for i = indices
  for j = indices
    images(:, :, :, curr) = ...
        permute(img(i:i+CROPPED_DIM-1, j:j+CROPPED_DIM-1, :), [2 1 3]);
    images(:, :, :, curr+5) = images(end:-1:1, :, :, curr);
    curr = curr + 1;
  end
end
center = floor(indices(2) / 2)+1;
images(:,:,:,5) = ...
    permute(img(center:center+CROPPED_DIM-1,center:center+CROPPED_DIM-1,:), ...
        [2 1 3]);
images(:,:,:,10) = images(end:-1:1, :, :, curr);
