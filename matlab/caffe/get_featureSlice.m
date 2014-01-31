

% @param pyra... should contain 'sbin' and 'scales'
% @param bbox.{x1 x2 y1 y2} -- in image coordinates. -- must be round numbers.
% @param templateSize = desired shape of output in feature descriptor units (e.g. [6 10] root filter)
% (could pass in image_size ... though we currently don't use it, and imsize is typically in pyra)
function [featureSlice, scaleIdx] = get_featureSlice(pyra, bbox, templateSize)

%experiment:
%pyra.sbin = 8;

  %1. select a scale in the pyramid

    bbox_desc = bbox_mult(bbox, 1.0/single(pyra.sbin)); % unscaled image -> unscaled descriptor coords.
    bbox_desc_dim = [bbox_desc.y2 - bbox_desc.y1, bbox_desc.x2 - bbox_desc.x1]; %bbox dim in the space of scale=1 descriptors. 
    bbox_desc_scale = templateSize / bbox_desc_dim; %scale factor from (scale=1) to a scale where bbox fits in templateSize

    scale_to_use = mean(bbox_desc_scale); %avg of scale factors for x and y dims.
    [scale_to_use, scaleIdx] = findNearestScale(scale_to_use, pyra.scales); %best precomputed approx of scale_to_use

    %padx_desc_scaled = (pyra.padx / pyra.sbin) * scale_to_use; %in descriptor coords, at scale_to_use.
    %pady_desc_scaled = (pyra.pady / pyra.sbin) * scale_to_use;
    padx_desc_scaled = pyra.padx; %AHA, this is really in HOG cells and not image pixels.
    pady_desc_scaled = pyra.pady;


  %2. rip out a slice from the appropriate scale
   
    bbox_to_use.x1 = round(bbox_desc.x1 * scale_to_use + padx_desc_scaled); 
    bbox_to_use.x2 = bbox_to_use.x1 + templateSize(2) - 1;
    bbox_to_use.y1 = round(bbox_desc.y1 * scale_to_use + pady_desc_scaled); 
    bbox_to_use.y2 = bbox_to_use.y1 + templateSize(1) - 1;
    bbox_to_use
    scale_to_use
    scaleIdx

    featureSlice = pyra.feat{scaleIdx}(bbox_to_use.y1 : bbox_to_use.y2, bbox_to_use.x1 : bbox_to_use.x2, :); %stub
end

% divide a bbox.{x1 x2 y1 y2} by some value.
function bbox_out = bbox_mult(bbox, multiplier)
    
    bbox_out.x1 = bbox.x1 * multiplier;
    bbox_out.x2 = bbox.x2 * multiplier;
    bbox_out.y1 = bbox.y1 * multiplier;
    bbox_out.y2 = bbox.y2 * multiplier;
end

% @param scale_to_use = a scale that maps bbox to templateSize
% @param scales = list of scales that we computed in our pyramid
function [scale_to_use, scaleIdx] = findNearestScale(scale_to_use, scales)

    %scaleIdx = index of nearestScale in the list of scales
    [nearestScale_distance, scaleIdx] = min(abs(scale_to_use - scales));
    scale_to_use = scales(scaleIdx);
end

