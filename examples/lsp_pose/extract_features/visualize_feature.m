close all; clear all;
addpath(genpath('../../../tools/wyang'));

num_output  = 96; %  num_output: 96 for conv 1 (refer to the proto)
layer = 'features-fc7';
load([layer '/features.mat']);
load('/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test/pos.mat');
width = size(feats, 2);
nmap  = width / num_output;

imdir = '/home/wyang/Code/PE1.41DBN_human_detector/LSP/';
imlist = dir([imdir '/*.jpg']);

for i = 1:length(pos)
    [p, name, ext] = fileparts(pos(i).im);
    feat = feats(i, :);
    feat = reshape(feat, [nmap num_output]);
    figure('name', name);
	im = imread([imdir pos(i).im]); % original image
    subplot(1, 2, 1);
    imshow(im);
    subplot(1, 2, 2)
    h = display_network(feat);
    saveas(gcf, [layer '/' name '.png']);
    pause;
    close;
end
