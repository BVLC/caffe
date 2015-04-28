addpath('/rmt/work/deeplabel/code/matlab/my_script');
SetupEnv;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if is_server
  %map_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', feature_type, sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std)); 
  map_folder = fullfile('/rmt/work/deeplabel/exper', dataset, 'res', feature_name, feature_type, sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std)); 
else 
  map_folder = '../result';
end

map_dir = dir(fullfile(map_folder, '*.bin'));

save_root_folder = map_folder;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to change values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(dataset, 'voc12')
  seg_res_dir = [save_root_folder '/results/VOC2012/'];
elseif strcmp(dataset, 'coco')
  seg_res_dir = [save_root_folder '/results/COCO2014/'];
else
  error('Wrong dataset!')
end

save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

if ~exist(save_result_folder, 'dir')
  mkdir(save_result_folder);
end

for i = 1 : numel(map_dir)
  fprintf(1, 'processing %d (%d)...\n', i, numel(map_dir));
  map = LoadBinFile(fullfile(map_folder, map_dir(i).name), 'int16');

  img_fn = map_dir(i).name(1:end-4);
  imwrite(uint8(map), colormap, fullfile(save_result_folder, [img_fn, '.png']));
end
