% Author: Adnan Chaudhry
% Date: October 13, 2016
%% Compute LCC and SROCC for the outputs of IQA CNN

clc;
clear;

% number of trained CNN's being tested
nNets = 10;
% Create an output file
out = fopen('outputs/out.txt', 'wt');
if(out == -1)
    error('unable to open output file for writing');
end

% Compute LCC and SROCC for the outputs of all nets
for i = 1 : nNets

    % Read network's output file into an array
    outFile = fopen(strcat('outputs/output_', num2str(i), '.txt'),'r');
    output = fscanf(outFile, '%f');
    fclose(outFile);

    % Read score file into an array (ground truth)
    gTruthFile = fopen(strcat('outputs/scores_', num2str(i), '.txt'),'r');
    gTruth = fscanf(gTruthFile, '%f');
    fclose(gTruthFile);

    % Compute LCC and SROCC for the output
    fprintf(out, '%d LCC = %f\n', i, corr(output, gTruth, 'type', 'Pearson'));
    fprintf(out, '%d SROCC = %f\n', i, corr(output, gTruth, 'type', 'Spearman'));

end

fclose(out);