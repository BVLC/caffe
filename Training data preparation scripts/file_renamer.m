% Author: Adnan Chaudhry
% Date: November 25, 2016
%% Rename image files
%%
clc;
clear;

% numeric offset to add to file name
offset = 808;

% Prompt user to naviagte to the images' directory
imagesDir = uigetdir('.', 'Select images'' directory');
imagesDir = strcat(imagesDir, '/');

% Output renamed files to this directory
outputDir = strcat(imagesDir, 'out/');
% if output directory does not exist then create it
if(exist(outputDir, 'dir') == 0)
    mkdir(outputDir);
end

% Get all .bmp files in the selected directory
files = dir(strcat(imagesDir, '*.bmp'));

% Loop through all images
for id = 1:length(files)
    % Get the image name (minus the extension)
    [~, imgName, ext] = fileparts(files(id).name);

      % Get image number i.e. skip the 'img' part in the name
      num = str2double(imgName(4:end));
      if ~isnan(num)
          % If numeric, rename with the offset added
          movefile(strcat(imagesDir, imgName, ext), strcat(outputDir, 'img', num2str(num + offset), ext));
      end

end
