function b = NormalizeArray(a)
% Normalizes array a. This means that the minimum value will become 0 and
% the maximum value 1.
%
% a:            Input array.
%
% b:            Normalized output array
%
%     Jasper Uijlings - 2013

minVal = min(a(:));
maxVal = max(a(:));

diffVal = maxVal - minVal;

b = a - minVal;
if diffVal ~= 0
    b = b ./ diffVal;
end
