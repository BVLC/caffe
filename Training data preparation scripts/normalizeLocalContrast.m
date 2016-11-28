% Author: Adnan Chaudhry
% Date: September 22, 2016
%% Apply local contrast normalization
% imagePatches -- patches of input image
% windowSize -- local window size used in the contrast normalization
% patchSize -- size of each patch along one dimension (each patch is
% assumed to be square)
% normalizedPatches -- output normalized patches
function normalizedPatches = normalizeLocalContrast(imagePatches, windowSize, patchSize)
% pre-allocate output memory
[nRowPatches, nColPatches] = size(imagePatches);
normalizedPatches = zeros(nRowPatches * patchSize, nColPatches * patchSize);
rowPatchSizeVector = patchSize * ones(1, nRowPatches);
colPatchSizeVector = patchSize * ones(1, nColPatches);
normalizedPatches = mat2cell(normalizedPatches, rowPatchSizeVector, colPatchSizeVector);

% Define local neighbourhood
nHood = ones(windowSize, windowSize);
% Define local mean filter
meanFilter = ones(windowSize, windowSize) / (windowSize * windowSize);

% Run through all patches
for r = 1 : nRowPatches
    for c = 1 : nColPatches
        patch = imagePatches{r, c};
        % Compute local mean
        meanPatch = conv2(patch, meanFilter, 'same');
        % Compute local standard deviation
        stdDevPatch = stdfilt(patch, nHood);
        % Subtract local mean and divide by local standard deviation and add a
        % small constant to denominator in order to avoid division by zero
        normalizedPatches{r,c} = (patch - meanPatch) ./ (stdDevPatch + 1e-8);
    end
end

end