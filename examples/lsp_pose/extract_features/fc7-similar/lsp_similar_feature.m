close all; clear all;
addpath(genpath('../../../tools/wyang'));

mkdir('cache');

num_output  = 4096; %  num_output: 96 for conv 1 (refer to the proto)
layer = 'features-fc7';
load(['lsp_test_features.mat']);
lsp_test_feats = feats;

load(['lsp_ext_features.mat']);
lsp_ext_feats = feats;

imdir = '/home/wyang/Code/PE1.41DBN_human_detector/LSP/';
load('/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test/pos.mat');
lsp_pos = pos;

load('/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_train_no_fr/pos.mat');
lsp_ext_pos = pos;

sz = sqrt(num_output);

if ~exist('D.mat', 'file')
    D = pdist2(lsp_test_feats, lsp_ext_feats);
    save('D.mat', 'D');
else
    load('D.mat');
end

[v I] = sort(D, 2);

sim_idx = unique( I(:, 2) );

save('sim_idx.mat', 'sim_idx');

for i = 1:length(lsp_pos)
    [p, name, ext] = fileparts(lsp_pos(i).im);
    figure('name', name);
    
    feat1 = lsp_test_feats(i, :);
    feat1 = reshape(feat1, [sz sz]);
	im = imread([imdir lsp_pos(i).im]); % original image
    subplot(2, 2, 1);
    imshow(im);
    subplot(2, 2, 2)    
    imagesc(feat1)
    
    idx = I(i, 2);
    feat2 = lsp_ext_feats(idx, :);
    feat2 = reshape(feat2, [sz sz]);
	im = imread([imdir lsp_ext_pos(idx).im]); % original image
    subplot(2, 2, 3);
    imshow(im);
    subplot(2, 2, 4)    
    imagesc(feat2)
    saveas(gcf, [ 'cache/' name '.png']);
    
    pause;
    close;
end


