
% wrapper for DenseNet convnet_pyramid.
% provides some extra output params that you wouldn't get with caffe('convnet_featpyramid', ...)

% YOU MUST CALL caffe('init', ...) BEFORE RUNNING THIS.
function pyra = convnet_featpyramid(imgFname, pyra_params)

    %set defaults for params not passed by user
    if( ~exist('pyra_params') || ~isfield(pyra_params, 'interval') )
        pyra_params.interval = 5;
    end
    if( ~exist('pyra_params') || ~isfield(pyra_params, 'img_padding') )
        pyra_params.img_padding = 16;
    end
    if( ~exist('pyra_params') || ~isfield(pyra_params, 'feat_minWidth') )
        pyra_params.feat_minWidth = 1;
    end
    if( ~exist('pyra_params') || ~isfield(pyra_params, 'feat_minHeight') )
        pyra_params.feat_minHeight = 1;
    end

    % compute the pyramid: 
    pyra = caffe('convnet_featpyramid', imgFname, pyra_params);

    % add DPM-style fields:
    pyra.padx = pyra.feat_padx; % for DPM conventions
    pyra.pady = pyra.feat_pady;
    pyra.num_levels = length(pyra.scales);
    pyra.valid_levels = true(pyra.num_levels, 1);

    pyra.imsize = [pyra.imheight pyra.imwidth];
    pyra.feat = permute_feat(pyra.feat); % [d h w] -> [h w d]     
    pyra.scales = double(pyra.scales); %get_detection_trees prefers double
end

% input:  pyra.feat{:}, with dims [d h w]
% output: pyra.feat{:}, with dims [h w d]
function feat = permute_feat(feat)
    for featIdx = 1:length(feat)
        feat{featIdx} = permute( feat{featIdx}, [2 3 1] );
    end
end

