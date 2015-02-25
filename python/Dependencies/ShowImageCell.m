function ShowImageCell(imageCell, n, m, figurename, imageNames)
% ShowImageCell(imageCell, n, m, figurename, imageNames)
%
% Generate a figure with thumbnails of the images in the imageCell.
%
% imageCell:            Cell array with images which can be displayed
%                       with imshow.
% n:                    number of thumbnail rows per figure.
% m:                    number of thumbnail columns per figure.
% figurename:           Name of the figures (optional).
%
%     Jasper Uijlings - 2013

totImages = length(imageCell);
numFigures = ceil(totImages / (n * m));

if nargin < 4
    figurename = 'untitled';
end

if nargin < 5
    imageNames = cell(length(imageCell));
end

if ~iscell(imageNames)
    imageNamesC = cell(length(imageNames));
    for i=1:length(imageNames)
        imageNamesC{i} = sprintf('%g', imageNames(i));
    end
    imageNames = imageNamesC;
end

idx = 1;
screenSize = get(0, 'ScreenSize');

for i=1:numFigures
    if ispc
        figure('Position', [1, 1, screenSize(3), screenSize(4)], 'Name', figurename);
%         figure('Position', [1 49 1920 946] , 'Name', figurename);
    else
        figure('Position', [1, 1, screenSize(3)/2, screenSize(4)], 'Name', figurename);
    end
    clf;
    for j = 1:n * m
        if(idx <= totImages)
            subplot(n, m, j);
            imshow(imageCell{idx});
            xlabel(imageNames{idx});
            idx = idx + 1;
        end
    end
end
    