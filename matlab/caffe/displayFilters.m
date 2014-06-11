function displayFilters(X, display_cols)
%displayFilters Display [height * width * num_channels * num_filters] filters 
%  in a nice grid using display_cols, at most it displays 32 filters
%   [h, display_array] = displayFilters(X, display_cols)

% Gray Image
colormap(gray);

% Compute rows, cols
[example_height example_width num_channels num_filters] = size(X);


% Compute number of items to display
if ~exist('display_cols', 'var')
    display_cols = floor(sqrt(num_filters));
end
display_rows = ceil(num_filters / display_cols);
colimage = [];
for n = 1:min(32,num_filters)
    if mod(n,8)== 1
        figure(ceil(n/8))
        colimage = [];
    end
    filter = reshape(X(:,:,:,n),[],num_channels)';
    [~,dsp] = displayData(filter,example_width,num_channels/4);
    colimage = cat(1,colimage,dsp,ones(1,size(dsp,2)));
    if mod(n,8)== 0
        imagesc(colimage, [-1 1]), axis off; drawnow;
    end
end

