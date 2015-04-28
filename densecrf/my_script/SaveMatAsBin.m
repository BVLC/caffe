% save mat score maps as bin file for cpp
%
addpath('/rmt/work/deeplabel/code/matlab/my_script');
SetupEnv;
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if is_server
  mat_folder  = fullfile('/rmt/work/deeplabel/exper', dataset, feature_name, model_name, testset, feature_type);
  if strcmp(dataset, 'voc12')
    img_folder  = '/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages';
  elseif strcmp(dataset, 'coco')
    img_folder  = '/rmt/data/coco/JPEGImages';
  else
    error('Wrong dataset!');
  end
  save_folder = fullfile(mat_folder, 'bin');
else
  mat_folder  = fullfile('~/workspace/deeplabeling/exper', dataset, feature_name, model_name, testset, feature_type);
  if strcmp(dataset, 'voc12')
    img_folder  = '~/dataset/PASCAL/VOCdevkit/VOC2012/JPEGImages';
  elseif strcmp(dataset, 'coco')
    img_folder  = '~/dataset/coco/JPEGImages';
  else
    error('Wrong dataset!');
  end
  save_folder = fullfile(mat_folder, 'bin');
end

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

fprintf(1, 'Saving to %s\n', save_folder);

mat_dir = dir(fullfile(mat_folder, '*.mat'));

for i = 1 : numel(mat_dir)
    fprintf(1, 'processing %d (%d)...\n', i, numel(mat_dir));
    data = load(fullfile(mat_folder, mat_dir(i).name));
    data = data.data;
    data = permute(data, [2 1 3]);    
    %Transform data to probability
    data = exp(data);
    data = bsxfun(@rdivide, data, sum(data, 3));
    
    img_fn = mat_dir(i).name(1:end-4);
    img_fn = strrep(img_fn, '_blob_0', '');
    img = imread(fullfile(img_folder, [img_fn, '.jpg']));
    img_row = size(img, 1);
    img_col = size(img, 2);
    
    data = data(1:img_row, 1:img_col, :);
    
    save_fn = fullfile(save_folder, [img_fn, '.bin']);
    
    SaveBinFile(data, save_fn, 'float');
end

