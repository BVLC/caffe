function visualize_groundtruth
cropsize = 227;
addpath(genpath('../../../build/wyang'));

nsample     = 128;
layer = 'groundtruth_0928';
load([layer '/features.mat']);
labelsize = size(feats, 2);

imdir = '/home/wyang/Code/PE1.41DBN_human_detector/LSP/';
load('/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test/pos.mat');

for i = 1:length(pos)
    [p, name, ext] = fileparts(pos(i).im);
    tmp = feats(i, :);
    predpose = reshape(tmp, [14 2])';
	im = imread([imdir pos(i).im]);
    f = visualize_pose(imresize(im, [cropsize, cropsize]), predpose(1:2, :)*cropsize, ones(1, 14));
    pause; close(f);
%     saveas(gcf, [layer '/' name '.png']);
end

end
