function [numRows numColumns area] = BoxSize(bbox)
% [numRows numColumns Surface] = BoxSize(bbox)
%
% Retrieves number of rows, columns, and surface area from bounding box
%
% bbox:         4 x N Bounding box as [rowBegin colBegin rowEnd colEnd]
%
% numRows:      Number of rows of boxes
% numColumns:   Number of columns of boxes
% area:         Area of boxes
%
%     Jasper Uijlings - 2013

% Box is empty
if isempty(bbox)
    numRows = 0;
    numColumns = 0;
    area = 0;
    return
end

numRows = bbox(:,3) - bbox(:,1) + 1;
numColumns = bbox(:,4) - bbox(:,2) + 1;
area = numRows .* numColumns;

