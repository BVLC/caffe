function [h, display_array] = displayRGBFilters(X, display_cols, contrast)
%displayRGBFilters Display 2D RGB data in a nice grid
%   [h, display_array] = displayRGBFilters(X, filter_width, display_cols,contrast)
%   displays RGB 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

[filter_height filter_width num_channels num_filters] = size(X);

assert(num_channels==3);

% Compute number of items to display
if ~exist('display_cols', 'var')
    display_cols = floor(sqrt(num_filters));
end
display_rows = ceil(num_filters / display_cols);

if ~exist('contrast', 'var')
    contrast = 'local';
end
    
% Between images padding
pad = 1;

% Setup blank display
display_array = ones(pad + display_rows * (filter_height + pad), ...
                       pad + display_cols * (filter_width + pad),3);

% Copy each example into a patch on the display array
curr_ex = 1;
if strcmp(contrast,'global')
    max_val = max(X(:));
    min_val = min(X(:));
end
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > num_filters, 
			break; 
		end
		% Copy the patch
		
        % Get the max value of the patch
        if strcmp(contrast,'local')
            max_val = max(X(:,:,:,curr_ex)(:));
            min_val = min(X(:,:,:,curr_ex)(:));
        end
		display_array(pad + (j - 1) * (filter_height + pad) + (1:filter_height), ...
		              pad + (i - 1) * (filter_width + pad) + (1:filter_width),:) = ...
						(X(:,:,:,curr_ex) - min_val) / (max_val-min_val);
		curr_ex = curr_ex + 1;
	end
	if curr_ex > num_filters, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;

end
