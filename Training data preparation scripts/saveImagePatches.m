% Author: Adnan Chaudhry
% Date: September 22, 2016
%% Save image patches to disk in bmp format
% imagePatches -- input image patches's cell array
% outputDir -- output directory where patches are to be saved
% imageName -- name of the image file to which the patches belong
function saveImagePatches(imagePatches, outputDir, imageName)
[rowDim, colDim] = size(imagePatches);

% Create a sub-directory with the image name which will hold the image
% patches
patchesOutputDir = strcat(outputDir, imageName(1 : end - 4), '/');
% if the directory does not exist then create it
if(exist(patchesOutputDir, 'dir') == 0)
    mkdir(patchesOutputDir);
end

% Run through all patches 
for r = 1 : rowDim
    for c = 1 : colDim
        % form name for the patch image
        patchName = strcat(imageName(1 : end - 4), '_patch_', num2str(((r - 1) * colDim) + c));
        outFile = strcat(patchesOutputDir, patchName, '.bmp');
        % Save to disk
        imwrite(imagePatches{r,c}, outFile);
    end
end

end