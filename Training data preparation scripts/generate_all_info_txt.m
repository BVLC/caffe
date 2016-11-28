% Author: Adnan Chaudhry
% Date: November 25, 2016
%% Combine info.txt files for all distortion types for live dataset

clc;
clear;

% All info.txt files to be combined should be in the current directory and
% labeled correctly according to the order in which they are to be combined
% i.e. info1.txt , info2.txt, info3.txt ,... 
% Output file name shall be info_all.txt and the files shall be combined in
% the order in which they are named as described above

infoFiles = dir('info*.txt');
numInfoFiles = length(infoFiles);

% offset to be added to distorted image file name
offset = 0;

% output file
outFile = fopen('info_all.txt', 'wt');

% Loop through all files
for fileID = 1 : numInfoFiles
    % Read the mappings in the file
    infoFile = fopen(infoFiles(fileID).name, 'r');
    ref2DistMapping = textscan(infoFile, '%s %s %s');
    fclose(infoFile);
    % Output the mappings to the output file with appropriate offset added
    % to distorted image names
    nMappings = length(ref2DistMapping{1});
    for mappingNumber = 1 : nMappings
        imgName = ref2DistMapping{2}{mappingNumber};
        imgNumber = str2double(imgName(4 : end - 4));
        imgName = strcat('img', num2str(imgNumber + offset), '.bmp');
        fprintf(outFile, '%s %s %s\n', ref2DistMapping{1}{mappingNumber}, imgName, ref2DistMapping{3}{mappingNumber});
    end
    offset = offset + nMappings;
end
fclose(outFile);