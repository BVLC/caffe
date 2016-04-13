function images = tvg_prepare_image_fixed(im)

INPUT_DIM = 500;

% mean BGR pixel
mean_pix = [103.939, 116.779, 123.68];

im = single(im);
% RGB -> BGR
im = im(:, :, [3 2 1]);

% mean BGR pixel subtraction
for c = 1:3
    im(:, :, c) = im(:, :, c) - mean_pix(c);
end

images = zeros(INPUT_DIM, INPUT_DIM, 3, 1, 'single');
[h, w, ~] = size(im);

images(1:w, 1:h, :, 1) = permute(im, [2 1 3]);
