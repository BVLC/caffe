% Author: Adnan Chaudhry
% Date: September 21, 2016
%% Visualize image patches
% imagePatches -- cell array containing image patches
function visImagePatches(imagePatches)
% get row and column dimensions of the input cell array
[rowDim, colDim] = size(imagePatches);
% Show all the patches as subfigures in one figure
plotNumber = 1;
for r = 1 : rowDim
    for c = 1 : colDim        
		% Specify the location for display of the image.
		subplot(rowDim, colDim, plotNumber);
		% Extract the numerical array out of the cell		
		imagePatch = imagePatches{r,c};
		% Display patch
        imshow(imagePatch);
		drawnow;
		% Increment the subplot to the next location.
		plotNumber = plotNumber + 1;
    end
end

end