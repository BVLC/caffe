function [scores index] = BoxBestOverlap(gtBoxes, testBoxes)
% [scores index] = BoxBestOverlap(gtBox, testBoxes)
% 
% Get overlap scores (Pascal-wise) for testBoxes bounding boxes
%
% gtBoxes:                 Ground truth bounding boxes
% testBoxes:               Test bounding boxes
%
% scores:                  Highest overlap scores for each testBoxes bbox.
% index:                   Index for each testBoxes box which ground truth box is best
%
%     Jasper Uijlings - 2013

numGT = size(gtBoxes,1);
numTest = size(testBoxes,1);

scoreM = zeros(numGT, numTest);


for j=1:numGT
    scoreM(j,:) = PascalOverlap(gtBoxes(j,:), testBoxes);
end


[scores index] = max(scoreM, [], 2);


