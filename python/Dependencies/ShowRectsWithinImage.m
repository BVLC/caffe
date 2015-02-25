function ShowRectsWithinImage(rects, numRow, numCol, image, imageNames)
% ShowRects(Rects, numRow, numCol, image)
%
% Shows only the rectangles of the image
%
%     Jasper Uijlings - 2013

if ~exist('imageNames', 'var')
    imageNames = cell(size(rects,1), 1);
    for i=1:size(rects,1)
        imageNames{i} = sprintf('%d', i);
    end
end

% Convert to images
idx = 1;
images = cell(size(rects,1),1);
for i=1:size(rects,1)
    bbox = rects(i,:);
    images{idx} = image(bbox(1):bbox(3),bbox(2):bbox(4),:);
    idx = idx + 1;
end

ShowImageCell(images, numRow, numCol, 'rects', imageNames);