% Author: Adnan Chaudhry
% Date: October 13, 2016
%% Compute LCC and SROCC for the outputs of IQA CNN
function compute_LCC_SROCC()

clc;
clear;

% number of trained CNN's being tested
nNets = 100;

% Create an output file
out = fopen('outputs/out.txt', 'wt');
if(out == -1)
    error('unable to open output file for writing');
end

pCorrs = zeros(1, nNets);
sCorrs = zeros(1, nNets);

% Compute LCC and SROCC for the outputs of all nets
for i = 1 : nNets

    % Read network's output for image patches file into an array
    outFile = fopen(strcat('outputs/output_', num2str(i), '.txt'),'r');
    outputPatchScores = fscanf(outFile, '%f');
    fclose(outFile);

    % Read score file for image patches into an array (ground truth)
    gTruthFile = fopen(strcat('outputs/scores_', num2str(i), '.txt'),'r');
    gTruthPatchScores = fscanf(gTruthFile, '%f');
    fclose(gTruthFile);

    % Combine patch scores to form image scores
    [outputImageScores, gTruthImageScores] = convPatchToImageScores(outputPatchScores, gTruthPatchScores);

    % Compute LCC and SROCC for the output image scores
    pCorrs(i) = corr(outputImageScores, gTruthImageScores, 'type', 'Pearson');
    sCorrs(i) = corr(outputImageScores, gTruthImageScores, 'type', 'Spearman');

    % Write results to file
    fprintf(out, '%d LCC = %f\n', i, pCorrs(i));
    fprintf(out, '%d SROCC = %f\n', i, sCorrs(i));

end

%% 
% Output maximum, minimum and average LCC and SROCC

[max_pCorr, max_pCorrNetId] = max(pCorrs);
[max_sCorr, max_sCorrNetId] = max(sCorrs);
avg_pCorr = mean(pCorrs);
avg_sCorr = mean(sCorrs);
med_pCorr = median(pCorrs);
med_sCorr = median(sCorrs);

fprintf(out, 'Network with highest LCC is No. %d, corr value = %f\n', max_pCorrNetId, max_pCorr);
fprintf(out, 'Network with highest SROCC is No. %d, corr value = %f\n', max_sCorrNetId, max_sCorr);
fprintf(out, 'Mean LCC for all networks = %f\n', avg_pCorr);
fprintf(out, 'Mean SROCC for all networks = %f\n', avg_sCorr);
fprintf(out, 'Median LCC for all networks = %f\n', med_pCorr);
fprintf(out, 'Median SROCC for all networks = %f\n', med_sCorr);

fclose(out);

disp(['Network with highest LCC is No. ' num2str(max_pCorrNetId) ', corr value = ' num2str(max_pCorr)]);
disp(['Network with highest SROCC is No. ' num2str(max_sCorrNetId) ', corr value = ' num2str(max_sCorr)]);
disp(['Mean LCC for all networks = ' num2str(avg_pCorr)]);
disp(['Mean SROCC for all networks = ' num2str(avg_sCorr)]);
disp(['Median LCC for all networks = ' num2str(med_pCorr)]);
disp(['Median SROCC for all networks = ' num2str(med_sCorr)]);

end

%% Convert patch-wise scores to image-wise scores (mean of all image patch scores)
% Inputs
% oPatchScores -- patch scores output by the network
% tPatchScores -- ground truth patch scores
% Outputs
% oImageScores -- image-wise scores output by the network
% tImageScores -- image-wise scores ground truth
function [oImageScores, tImageScores] = convPatchToImageScores(oPatchScores, tPatchScores)

prev_tPatchScore = tPatchScores(1);
nPatches = length(tPatchScores);
outIndex = 1;
oImageScores(outIndex) = 0;
tImageScores(outIndex) = tPatchScores(outIndex);

% All patches of a single image are consecutive in the output so when the
% ground truth score changes, it means we have moved to the next image
i = 1;
num = 0;
while(i <= nPatches)
    if(prev_tPatchScore ~= tPatchScores(i))
        oImageScores(outIndex) = oImageScores(outIndex) / num;
        outIndex = outIndex + 1;
        num = 1;
        oImageScores(outIndex) = oPatchScores(i);
        tImageScores(outIndex) = tPatchScores(i);
        prev_tPatchScore = tPatchScores(i);
        i = i + 1;
    else
        oImageScores(outIndex) = oImageScores(outIndex) + oPatchScores(i);
        num = num + 1;
        i = i + 1;
    end
end
oImageScores(outIndex) = oImageScores(outIndex) / num;

oImageScores = oImageScores';
tImageScores = tImageScores';

end
