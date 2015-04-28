% save jpg images as bin file for cpp
%

img_folder = '../img';
save_folder = '../img_bin';

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

img_dir = dir(fullfile(img_folder, '*.jpg'));

for i = 1 : numel(img_dir)
    img = imread(fullfile(img_folder, img_dir(i).name));
    
    img_fn = img_dir(i).name(1:end-4);
    save_fn = fullfile(save_folder, [img_fn, '.bin']);
    
    SaveBinFile(img, save_fn, 'uint8');
end
    