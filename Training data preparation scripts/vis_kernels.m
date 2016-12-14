% Author: Adnan Chaudhry
% Date: December 14, 2016
%% Visualize learned filter kernels for NR IQA CNN
clc;
clear;

kernelSize = 7;
numKernels = 50;
% Number of kernels displayed along row dimension
nRowKernels = 10;
% Number of kernels displayed along column dimension
nColKernels = 5;

% Get kernel dump file
[dumpFileName, path, ~] = uigetfile('*.*', 'Select kernel dump file');

% Read in weights
weightsFile = fopen(strcat(path, dumpFileName), 'r');
weights = fscanf(weightsFile, '%f');
fclose(weightsFile);

% Organize data into an intuitive layout
weights = weights(1 : kernelSize * kernelSize * numKernels);
weightsMat = reshapeKernelArray(weights, nRowKernels, nColKernels, kernelSize);
rowKernelSizeVector = kernelSize * ones(1, nRowKernels);
colKernelSizeVector = kernelSize * ones(1, nColKernels);
filterKernels = mat2cell(weightsMat, rowKernelSizeVector, colKernelSizeVector);
visFilterKernels(filterKernels);
