function [scores,list_im] = matcaffe_batch(list_im, use_gpu)
% scores = matcaffe_batch(list_im, use_gpu)
%
% Demo of the matlab wrapper using the ILSVRC network.
%
% input
%   list_im  list of images files
%   use_gpu  1 to use the GPU, 0 to use the CPU
%
% output
%   scores   1000 x num_images ILSVRC output vector
%
% You may need to do the following before you start matlab:
%  $ export LD_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/usr/local/cuda/lib64
%  $ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
% Or the equivalent based on where things are installed on your system
%
% Usage:
%  scores = matcaffe_batch({'peppers.png','onion.png'});
%  scores = matcaffe_batch('list_images.txt', 1);
if nargin < 1
  % For test purposes
  list_im = {'peppers.png','onions.png'};
end
if ischar(list_im)
    %Assume it is a file contaning the list of images
    filename = list_im;
    list_im = read_cell(filename);
end
% Adjust the batch size to match with imagenet_deploy.prototxt
batch_size = 10;
% Adjust dim to the output size of imagenet_deploy.prototxt
dim = 1000;
disp(list_im)
if mod(length(list_im),batch_size)
    warning(['Assuming batches of ' num2str(batch_size) ' images rest will be filled with zeros'])
end

% init caffe network (spews logging info)
if exist('use_gpu', 'var')
  matcaffe_init(use_gpu);
else
  matcaffe_init();
end

d = load('ilsvrc_2012_mean');
IMAGE_MEAN = d.image_mean;

% prepare input

num_images = length(list_im);
scores = zeros(dim,num_images,'single');
num_batches = ceil(length(list_im)/batch_size)
initic=tic;
for bb = 1 : num_batches
    batchtic = tic;
    range = 1+batch_size*(bb-1):min(num_images,batch_size * bb);
    tic
    input_data = prepare_batch(list_im(range),IMAGE_MEAN,batch_size);
    toc, tic
    fprintf('Batch %d out of %d %.2f%% Complete ETA %.2f seconds\n',...
        bb,num_batches,bb/num_batches*100,toc(initic)/bb*(num_batches-bb));
    output_data = caffe('forward', {input_data});
    toc
    output_data = squeeze(output_data{1});
    scores(:,range) = output_data(:,mod(range-1,batch_size)+1);
    toc(batchtic)
end
toc(initic);

if exist('filename', 'var')
    save([filename '.probs.mat'],'list_im','scores','-v7.3');
end



