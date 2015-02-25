function ShowBlobs(blobs, numRow, numCol, image, minSize, imNames)
% PlotBlobs(blobs, numRow, numCol, image, minSize) plots all blobs in numCol columns
%
%     Jasper Uijlings - 2013

if nargin == 4
    minSize = 0;
end

doNames = exist('imNames', 'var');

if doNames
    if ~iscell(imNames)
        imageNamesC = cell(size(imNames));
        for i=1:length(imNames)
            imageNamesC{i} = sprintf('%g', imNames(i));
        end
        imNames = imageNamesC;
    end    
end
    
% Convert to images
idx = 1;
for i=1:length(blobs)
    if not(isfield(blobs{i}, 'size'))
        blobs{i}.size = sum(sum(blobs{i}.mask));
    end
    if blobs{i}.size > minSize
        images{idx} = Blob2Image(blobs{i}, image);
        if doNames
            iiNames{idx} = imNames{i};
        end
        idx = idx + 1;
    end
end


if doNames;
    ShowImageCell(images, numRow, numCol, '', iiNames);
else
    ShowImageCell(images, numRow, numCol);
end

% totImages = idx - 1;
% 
% numFigures = ceil(totImages / (numCol * numRow))
% 
% n = 1;
% screenSize = get(0, 'ScreenSize');
% 
% for i=1:numFigures
%     figure('Position', [1, 1, screenSize(3)/2, screenSize(4)]);
%     clf;
%     for j = 1:numCol * numRow
%         if(n <= totImages)
%             subplot(numRow, numCol, j);
%             imshow(images{n});
%             n = n + 1;
%         end
%     end
% end
