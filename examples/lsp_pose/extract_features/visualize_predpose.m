function visualize_predpose
cropsize = 227;
addpath(genpath('../../../build/wyang'));

nsample = 128;
layer = 'features_fc8_0927';
load([layer '/features.mat']);
labelsize = size(feats, 2);


imdir = '/home/wyang/Code/PE1.41DBN_human_detector/LSP/';
load('/home/wyang/Code/PE1.41DBN_human_detector/LSP/cache_test/pos.mat');

    
for i = 1:length(pos)
    i
    [p, name, ext] = fileparts(pos(i).im);
    tmp = feats(i, :);
    predpose = reshape(tmp, [14 2])';
	im = imread([imdir pos(i).im]);
    f = visualize_pose(imresize(im, [cropsize, cropsize]), predpose(1:2, :)*cropsize, ones(1, 14));
    saveas(gcf, [layer '/cache/' name '.png']);
    pause; close(f);
    close(gcf)
end

end
