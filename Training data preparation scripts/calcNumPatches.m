% Author: Adnan Chaudhry
% Date: September 27, 2016
%% Calculate the number of patches in an image
% image -- is the input RGB image
% patchSize -- is the size of the square patches in which the image is to be
% divided
% nPatches -- number of patches of size (patchSize x patchSize) which could be 
% formed in that image
function nPatches = calcNumPatches(image, patchSize)
[rows, cols, ~] = size(image);

nRowPatchDim = double(int32(rows / patchSize));
nColPatchDim = double(int32(cols / patchSize));

nPatches = nRowPatchDim * nColPatchDim;
end