% Author: Adnan Chaudhry
% Date: September 27, 2016
%% Create training and validation data files
% Write image and score data to a file
% scoresFile -- output file to which data would be written
% imageName -- name of the image whose patches are formed
% nPatches -- number of patches in the image
% scoreValue -- score associated with the image
function writeScoreData(scoresFile, imageName, nPatches, scoreValue)

% for all patches
for i = 1 : nPatches
    % Form the image patch path string with name
    imName = imageName(1 : end - 4); % strip extension
    imPathWithName = strcat(imName, '/', imName, '_patch_', num2str(i), '.bmp');
    % Write image path containing image name along with corresponding score
    % value to the output file
    fprintf(scoresFile, '%s %.4f\n', imPathWithName, scoreValue);        
end

end