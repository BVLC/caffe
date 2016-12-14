% Author: Adnan Chaudhry
% Date: December 14, 2016
%% Visualize image patches
% filterKernels -- cell array containing filter kernels
function visFilterKernels(filterKernels)
% get row and column dimensions of the input cell array
[rowDim, colDim] = size(filterKernels);
% Show all the kernels as subfigures in one figure
plotNumber = 1;
for r = 1 : rowDim
    for c = 1 : colDim        
		% Specify the location for display of the kernel image.
		subplot(rowDim, colDim, plotNumber);        
		% Extract the numerical array out of the cell		
		filter = filterKernels{r,c};
		% Display kernel
        imshow(filter, []);
		drawnow;
		% Increment the subplot to the next location.
		plotNumber = plotNumber + 1;
    end
end

end