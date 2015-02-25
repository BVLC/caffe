function [cb counts] = CountVisualWordsIndex(indexIm, wordIm, numIndex, numWords)
% cb = CountVisualWordsIndex(indexIm, wordIm, numIndex, numWords)
%
% Counts the number of visual words for the visual words in wordIm.
% wordIm is an array with visual word identities. Zeros will be ignored.
% indexIm is an array with regions to which visual words belong.
%
% WARNING: VERY FEW CHECKS FOR INTEGRETY. WRONG INPUT WILL CRASH THE SYSTEM
%
% indexIm:          Array with indices. Range: [1,numIndex]
% wordIm:           Array with visual word identities. Range: [0,numWords]
% numIndex:         Number of indices in indexIm.
% numWords:         Number of visual words.
%
% cb:               numIndex x numWords array with histogram counts
% counts:           numIndex x 1 array with counts per row of cb.
%
%     Jasper Uijlings - 2013

if size(indexIm,1) ~= size(wordIm,1) | size(indexIm,2) ~= size(wordIm,2)
    error('First two input arguments should have the same 2D dimension');
end

wordIm = double(wordIm);

[cb counts] = mexCountWordsIndex(indexIm, wordIm, numIndex, numWords);
