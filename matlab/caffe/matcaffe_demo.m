function res = matcaffe_demo(im, gpu)

% load image net mean
%  // In matlab, reading an image gives [height, width, channels] where height is the fastest dimension
%  //  - want to have the order as [width, height, channels, images]
%  //    (channels in BGR order)
%  //  - 

% 1: swap channel order to BGR
% 2: extract 5 crops and their flips
% 3: swap rows and columns and concat along 4th dim
% 4: wrap in cell aray

caffe('init');
if gpu
  caffe('set_mode_gpu');
else
  caffe('set_mode_cpu');
end
caffe('set_phase_test');
tic;
blob = {prepare_image(im)};
toc;
tic;
res = caffe('forward', blob);
toc;
res = reshape(res{1}, [1000 10]);
res = mean(res, 2);


function images = prepare_image(im)
d = load('ilsvrc_2012_mean');
image_mean = d.image_mean;
IMAGE_DIM = 256;
CROPPED_DIM = 227;

% resize to fixed input size
im = single(im);
im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
% permute from RGB to BGR
im = im(:,:,[3 2 1]) - image_mean;

% oversample
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
