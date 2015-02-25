function [boxesOut uniqueIdx] = BoxRemoveDuplicates(boxesIn)
% function boxOut = BoxRemoveDuplicates(boxIn)
%
% Removes duplicate boxes. Leaves the boxes in the same order
% Keeps the first box of each kind.
%
% boxesIn:          N x 4 array containing boxes
% 
% boxexOut:         M x 4 array of boxes witout duplicates. M <= N
% uniqueIdx:        Indices of retained boxes from boxesIn
%
%     Jasper Uijlings - 2013

[dummy uniqueIdx] = unique(boxesIn, 'rows', 'first');
uniqueIdx = sort(uniqueIdx);
boxesOut = boxesIn(uniqueIdx,:);
