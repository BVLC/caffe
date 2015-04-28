% downsample the bin files for faster cross-validation and not overfit val set
% 
addpath('/rmt/work/deeplabel/code/matlab/my_script');
SetupEnv;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if is_server
  mat_folder  = fullfile('/rmt/work/deeplabel/exper', dataset, feature_name, model_name, testset, feature_type);
  if crf_load_mat
    save_folder = mat_folder;
    file_name   = 'mat';    
  else
    save_folder = fullfile(mat_folder, 'bin');
    file_name   = 'bin';
  end
  folder_name = fullfile(mat_folder, file_name);
else
  mat_folder  = '../feature';
  save_folder = '../feature_bin';
end

if down_sample_method == 1
  dest_folder = [folder_name, sprintf('_downsampleBy%d', down_sample_rate)];
elseif down_sample_method == 2
  dest_folder = [folder_name, sprintf('_numSample%d', num_sample)];
else
   error('Wrong down_sample_method\n');
end

if ~exist(dest_folder, 'dir')
  mkdir(dest_folder)
end

save_dir = dir(fullfile(save_folder, ['*.' file_name]));

if down_sample_method == 1
  save_dir = save_dir(1:down_sample_rate:end);
elseif down_sample_method == 2
  ind = randperm(length(save_dir));
  ind = ind(1:num_sample);
  save_dir = save_dir(ind);
end

for i = 1 : numel(save_dir)
  fprintf(1, 'processing %d (%d)...\n', i, numel(save_dir));
  copyfile(fullfile(save_folder, save_dir(i).name), fullfile(dest_folder, save_dir(i).name));
end
