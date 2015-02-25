function scores = PascalOverlap(targetBox, testBoxes)
% scores = PascalOverlap(targetBox, testBoxes)
%
% Function obtains the pascal overlap scores between the targetBox and
% all testBoxes
%
% targetBox:            1 x 4 array containing target box
% testBoxes:            N x 4 array containing test boxes
%
% scores:               N x 1 array containing for each testBox the pascal
%                       overlap score.
%
%     Jasper Uijlings - 2013

intersectBoxes = BoxIntersection(targetBox, testBoxes);
overlapI = intersectBoxes(:,1) ~= -1; % Get which boxes overlap

% Intersection size
[nr nc intersectionSize] = BoxSize(intersectBoxes(overlapI,:));

% Union size
[nr nc testBoxSize] = BoxSize(testBoxes(overlapI,:));
[nr nc targetBoxSize] = BoxSize(targetBox);
unionSize = testBoxSize + targetBoxSize - intersectionSize;

scores = zeros(size(testBoxes,1),1);
scores(overlapI) = intersectionSize ./ unionSize;
