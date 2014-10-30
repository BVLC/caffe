close all; clear all;
addpath(genpath('../../../tools/wyang'));

num_output  = 4096; %  num_output: 96 for conv 1 (refer to the proto)
layer = 'features-fc7';
load([layer '/features.mat']);
load('/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test/pos.mat');
width = size(feats, 2);
sz = sqrt(num_output);


D = pdist2(feats, feats);
[v I] = sort(D, 2);

imdir = '/home/wyang/Code/PE1.41DBN_human_detector/LSP/';
imlist = dir([imdir '/*.jpg']);


imdir = '/home/wyang/Code/PE1.41DBN_human_detector/LSP/';
imlist = dir([imdir '/*.jpg']);

for i = 1:length(pos)
    [p, name, ext] = fileparts(pos(i).im);
    feat = feats(i, :);
    feat = reshape(feat, [sz sz]);
    figure('name', name);
	im = imread([imdir pos(i).im]); % original image
    subplot(1, 2, 1);
    imshow(im);
    subplot(1, 2, 2)    
    imagesc(feat)
    saveas(gcf, [layer '/' name '.png']);
    pause;
    close;
end

% 
% for i = 1:length(pos)
%     [p, name, ext] = fileparts(pos(i).im);
%     figure('name', name);
%     
%     feat1 = feats(i, :);
%     feat1 = reshape(feat1, [sz sz]);
% 	im = imread([imdir pos(i).im]); % original image
%     subplot(2, 2, 1);
%     imshow(im);
%     subplot(2, 2, 2)    
%     imagesc(feat1)
%     
%     idx = I(i, 2);
%     feat2 = feats(idx, :);
%     feat2 = reshape(feat2, [sz sz]);
% 	im = imread([imdir pos(idx).im]); % original image
%     subplot(2, 2, 3);
%     imshow(im);
%     subplot(2, 2, 4)    
%     imagesc(feat2)
%     saveas(gcf, [layer '/' name '.png']);
%     
%     pause;
%     close;
% end
% 

