
% wrapper for DenseNet convnet_pyramid.
% provides some extra output params that you wouldn't get with caffe('convnet_featpyramid', ...)
% this 'cache' verison enables precomputing features once (not once per pascal category... REALLY ONCE.)

% to enable caching, you simply define 'pyra_params.feature_dir'

% YOU MUST CALL caffe('init', ...) BEFORE RUNNING THIS.
function pyra = convnet_featpyramid_cache(imgFname, pyra_params, feature_dir)
    %set defaults for params not passed by user
    if( ~exist('pyra_params') || ~isfield(pyra_params, 'interval') )
        pyra_params.interval = 5;
    end
    if( ~exist('pyra_params') || ~isfield(pyra_params, 'img_padding') )
        pyra_params.img_padding = 16;
    end

    %save the user's minWidth and minHeight (if it exists) 
    if( exist('pyra_params') && isfield(pyra_params, 'feat_minWidth') )
        local_pyra_params.feat_minWidth = pyra_params.feat_minWidth;
    else
        local_pyra_params.feat_minWidth = 1;
    end
    if( exist('pyra_params') && isfield(pyra_params, 'feat_minHeight') )
        local_pyra_params.feat_minHeight = pyra_params.feat_minHeight;
    else
        local_pyra_params.feat_minHeight = 1;
    end 

    % only use caching if user provided pyra_params.feature_dir.
    if( exist('feature_dir') == 1 ) % ==1 means 'exists in workspace'
        useCache = true;
        [path basename ext] = fileparts(imgFname);
        cache_location = [feature_dir '/' basename '.mat'];
        pyra = try_cache(imgFname, pyra_params, cache_location); %returns [] if cached object does not exist
    else
        useCache = false;
        cache_location = '';
        pyra = [];
    end

    if isempty(pyra) 
        %for 'cache' version, we compute all scales, then prune the too-small ones in Matlab.
        pyra_params.feat_minWidth = 1;
        pyra_params.feat_minHeight = 1;

        % compute the pyramid: 
        pyra = caffe('convnet_featpyramid', imgFname, pyra_params);

        % add DPM-style fields:
        pyra.sbin = 16;
        pyra.padx = pyra.feat_padx; % for DPM conventions
        pyra.pady = pyra.feat_pady;
        pyra.num_levels = length(pyra.scales);
        pyra.valid_levels = true(pyra.num_levels, 1);

        pyra.imsize = [pyra.imheight pyra.imwidth];
        pyra.feat = permute_feat(pyra.feat); % [d h w] -> [h w d]     
        pyra.scales = double(pyra.scales); %get_detection_trees prefers double
        pyra.im = imgFname;

        if useCache == true
            save(cache_location, 'pyra');
        end
    end

    pyra = prune_small_scales(pyra, local_pyra_params);
end

function pyra = try_cache(imgFname, pyra_params, cache_location)
    try
        load(cache_location); %contains pyra
        assert( exist('pyra') == 1 ); %make sure we actually loaded the pyramid.

        [path basename_gold ext] = fileparts(imgFname);
        [path basename_input ext] = fileparts(pyra.im);
        assert( strcmp(basename_input, basename_gold) == 1 ) %make sure we got the right pyramid
        %assert( strcmp(pyra.im, imgFname) == 1 ) %make sure we got the right pyramid
        display('      found cached features');
    catch
        pyra = [];   
    end
end

% input:  pyra.feat{:}, with dims [d h w]
% output: pyra.feat{:}, with dims [h w d]
function feat = permute_feat(feat)
    for featIdx = 1:length(feat)
        feat{featIdx} = permute( feat{featIdx}, [2 3 1] );
    end
end

% needs feat_minHeight and feat_minWidth to be defined in pyra_params.
% prune too-small scales
function pyra = prune_small_scales(pyra, pyra_params)
    
    if( ~exist('pyra_params') || ~isfield(pyra_params, 'feat_minWidth') )
        return;
    end
    if( ~exist('pyra_params') || ~isfield(pyra_params, 'feat_minHeight') )
        return;
    end

    num_scales = length(pyra.scales);
    num_pruned = 0;
 
    for scaleIdx = 1:num_scales
        [h w d] = size(pyra.feat{scaleIdx});
        if (h < pyra_params.feat_minHeight) || (w < pyra_params.feat_minWidth)
            %found a scale that's too small. prune it.
            pyra.valid_levels(scaleIdx) = false;
            pyra.feat{scaleIdx} = []; %clear the too-small scale, just to be safe.
            pyra.scales(scaleIdx) = NaN;
            num_pruned = num_pruned + 1;
        end 
    end
    display(['    had to remove last ' int2str(num_pruned) ' scales, because they were smaller than the minimum that you specified in (feat_minWidth, feat_minHeight)'])
end


