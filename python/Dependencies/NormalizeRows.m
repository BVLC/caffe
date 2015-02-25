function b = NormalizeRows(a, n)
% Normalizes the rows of a. Makes sure there is no division by zero: b will
% not contain any NaN entries.
%
% a:            data with row vectors
% n:            The rows will sum to n. By default n = 1
% 
% b:            normalized data with row vecors. All rows sum to one except
%               the ones that are zero in the first place: these remain
%               zero.
%
%     Jasper Uijlings - 2013

% Get sums
sumA = sum(a,2);

% Make sure there is no division by zero
sumA(sumA == 0) = 1;

% Do the normalization
if nargin == 1
    b = bsxfun(@rdivide, a, sumA);
else
    b = bsxfun(@rdivide, a, sumA / n);
end

% Do the normalization
% if nargin == 1
%     b = a ./ repmat(sumA, 1, size(a,2));
% else
%     b = a .* n ./ repmat(sumA, 1, size(a,2));
% end