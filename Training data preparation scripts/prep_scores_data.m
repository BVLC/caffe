% Author: Adnan Chaudhry
% Date: September 27, 2016
%% Prepare training and validation scores' data files for NR IQA CNN
% The training images should already be divided into patches with local
% contrast normalization applied to each patch before running this script
% dmos_local.mat containing dmos scores for the all the images in the distortion
% directory selected, should be present in every distortion directory and
% there should also be a file named "info.txt" in the distortion images's
% directory which has the names of the reference images and their
% corresponding distorted levels

%%
clc;
clear;

% Every training image is patchSize x patchSize patch of the original
% images in the dataset
patchSize = 32;

% It is recommended to seed the random number generator using rng()
% function if you want the same sequence of files to be allocated to the
% training and validation sets every time you run this script
randomSeed = 7;
rng(randomSeed);

% The number of training and validation set allocation files you want to
% generate (the number if train-validate iterations in k fold cross
% validation
trainValIters = 100;

% Prompt user to naviagte to the refernce image data directory
refDataDir = uigetdir('.', 'Select reference image data directory');
refDataDir = strcat(refDataDir, '/');

% Prompt user to naviagte to the distortion image data directory
% For example 'fastfading' directory
distDataDir = uigetdir('.', 'Select distortion image data directory');
distDataDir = strcat(distDataDir, '/');
% output directory 
outputDir = strcat(distDataDir, 'mappings/');
% if output directory does not exist then create it
if(exist(outputDir, 'dir') == 0)
    mkdir(outputDir);
end

% load dmos scores
dmos = load(strcat(distDataDir, 'dmos_local.mat'));
fNames = fieldnames(dmos);
dmosArray = dmos.(fNames{1});

% Get all image files in the reference image data directory
refImageFiles = dir(strcat(refDataDir, '*.bmp'));

display('Creating training and validation sets'' files ...');

%%
nRefImages = length(refImageFiles);
% Get list of reference images' names
refImNames = {refImageFiles.name};
% Load mappings of the reference images to their corresponding distorted
% versions
mappingFile = fopen(strcat(distDataDir, 'info.txt'), 'r');
if(mappingFile == -1)
    error('Could not open info.txt mapping file');
end
ref2DistMapping = textscan(mappingFile, '%s %s %s');
fclose(mappingFile);

% Some error checking
nDistImages = length(ref2DistMapping{2});
if(nDistImages ~= length(dmosArray))
    error('Mismatch in the number of images and the number of corresponding DMOS scores');
end

% For all train-validate iterations
for i = 1 : trainValIters
    % scores' File for training
    scoresFileTrain = fopen(strcat(outputDir, 'scores_train_', num2str(i), '.txt'), 'wt');
    if(scoresFileTrain == -1)
        error('Could not create/open training file for writing');
    end
    % scores' File for validation
    scoresFileVal = fopen(strcat(outputDir, 'scores_val_', num2str(i), '.txt'), 'wt');
    if(scoresFileVal == -1)
        error('Could not create/open validation file for writing');
    end
    % Run thorugh all reference images and assign 80% of the reference images
    % and their distorted versions to the training set and the rest 20% to the
    % validation set
    for j = 1 : nRefImages
        isTrain = 0;
        if(rand() < 0.8)
            % assign to training set
            isTrain = 1;
        end
        refName = refImNames{j};
        % Find distorted images corresponding to the current reference
        % image
        cellIdx = strfind(ref2DistMapping{1}, refName);
        distIdx = find(~cellfun('isempty', cellIdx));
        nDistIdx = length(distIdx);
        for k = 1 : nDistIdx
            % Index in the mapping table
            distImgIdx = distIdx(k);
            distImgName = ref2DistMapping{2}{distImgIdx};
            % distorted image ID i.e. the number xx in imgxx.bmp
            distImgId = str2double(distImgName(4 : end - 4));
            distImg = imread(strcat(distDataDir, distImgName));
            nPatches = calcNumPatches(distImg, patchSize);
            % Write image patch names alongwith their dmos score to the output
            % training or validation files
            if(isTrain)
                writeScoreData(scoresFileTrain, distImgName, nPatches, dmosArray(distImgId));
            else
                writeScoreData(scoresFileVal, distImgName, nPatches, dmosArray(distImgId));
            end
        end
    end
    fclose(scoresFileTrain);
    fclose(scoresFileVal);
    percentageDone = (i / trainValIters) * 100;
    disp(['Progress = ', num2str(percentageDone), '%']);
end

disp('Done.');
