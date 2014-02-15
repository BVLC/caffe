

% @param pyra... should contain 'sbin' and 'scales'
% @param bbox.{x1 x2 y1 y2} -- in image coordinates. -- must be round numbers.
% @param templateSize = desired shape of output in feature descriptor units (e.g. [6 10] root filter)
% (could pass in image_size ... though we currently don't use it, and imsize is typically in pyra)
function [featureSlice, scaleIdx, roundedBox_in_px] = get_featureSlice(pyra, bbox, templateSize)

%experiment:
%pyra.sbin = 8;

  %1. tweak bbox to match the aspect ratio of templateSize
    %bbox = match_aspect_ratio(bbox, templateSize);

  %2. select a scale in the pyramid

    bbox_desc = bbox_mult(bbox, 1.0/single(pyra.sbin)); % unscaled image -> unscaled descriptor coords.
    bbox_desc_dim = [bbox_desc.y2 - bbox_desc.y1, bbox_desc.x2 - bbox_desc.x1]; %bbox dim in the space of scale=1 descriptors. 
    %bbox_desc_scale = templateSize / bbox_desc_dim; %scale factor from (scale=1) to a scale where bbox fits in templateSize
    bbox_desc_scale = (templateSize-1) / bbox_desc_dim; %pretend that template size is slightly smaller... so featureSlice will be slightly outside orig bbox

    scale_to_use = mean(bbox_desc_scale); %avg of scale factors for x and y dims.
    [scale_to_use, scaleIdx] = findNearestScale(scale_to_use, pyra.scales); %best precomputed approx of scale_to_use

    padx_desc_scaled = pyra.padx; %this is in descriptor cells and not image pixels.
    pady_desc_scaled = pyra.pady;

    scale_width_desc = size(pyra.feat{scaleIdx},2);
    scale_height_desc = size(pyra.feat{scaleIdx},1);

  %3. rip out a slice from the appropriate scale
  
    % center bbox_to_use on (bbox_desc.x2 - bbox_desc.x1), (bbox_desc.y2 - bbox_desc.y1)

    bbox_desc_x_center = bbox_desc.x1 + (bbox_desc.x2 - bbox_desc.x1)/2.0;
    bbox_desc_y_center = bbox_desc.y1 + (bbox_desc.y2 - bbox_desc.y1)/2.0;
    bbox_to_use.x1 = ceil(bbox_desc_x_center*scale_to_use - templateSize(2)/2.0 + padx_desc_scaled); %(center - templateWidth/2)
      bbox_to_use.x1 = max(1, bbox_to_use.x1); % make sure we didn't fall off the edge
    bbox_to_use.x2 = bbox_to_use.x1 + templateSize(2) - 1;
    bbox_to_use.y1 = ceil(bbox_desc_y_center*scale_to_use - templateSize(1)/2.0 + pady_desc_scaled); %(center - templateHeight/2)
      bbox_to_use.y1 = max(1, bbox_to_use.y1); % make sure we didn't fall off the edge
    bbox_to_use.y2 = bbox_to_use.y1 + templateSize(1) - 1;

  %4. make sure the slice fits into the overall space.

    if(bbox_to_use.x2 > scale_width_desc) %if we fell off the edge...
        x_offset = bbox_to_use.x2 - scale_width_desc; %amount by which we fell off the edge
        bbox_to_use.x1 = bbox_to_use.x1 - x_offset;
        bbox_to_use.x2 = bbox_to_use.x2 - x_offset;
    end
    if(bbox_to_use.y2 > scale_height_desc) %if we fell off the edge...
        y_offset = bbox_to_use.y2 - scale_height_desc; %amount by which we fell off the edge
        bbox_to_use.y1 = bbox_to_use.y1 - y_offset;
        bbox_to_use.y2 = bbox_to_use.y2 - y_offset;
    end

    %bbox_to_use
    %scale_to_use
    %scaleIdx
    try
        featureSlice = pyra.feat{scaleIdx}(bbox_to_use.y1 : bbox_to_use.y2, bbox_to_use.x1 : bbox_to_use.x2, :); 
    catch
        display('problem with get_featureSlice... you can debug it') 
        keyboard
    end

  %4. project rounded box back to img space (approx) -- for debugging
    roundedBox_in_px = bbox_mult(bbox_to_use, (pyra.sbin / scale_to_use)); 
    %roundedBox_in_px = bbox_add(roundedBox_in_px, -pyra.padx*pyra.sbin/scale_to_use); %assume padx==pady
    roundedBox_in_px.x1 = roundedBox_in_px.x1 - pyra.padx*pyra.sbin/scale_to_use;
    roundedBox_in_px.x2 = roundedBox_in_px.x2 - pyra.padx*pyra.sbin/scale_to_use;
    roundedBox_in_px.y1 = roundedBox_in_px.y1 - pyra.pady*pyra.sbin/scale_to_use;
    roundedBox_in_px.y2 = roundedBox_in_px.y2 - pyra.pady*pyra.sbin/scale_to_use;
end

% multiply a bbox.{x1 x2 y1 y2} by some value.
function bbox_out = bbox_mult(bbox, multiplier)
    bbox_out.x1 = bbox.x1 * multiplier;
    bbox_out.x2 = bbox.x2 * multiplier;
    bbox_out.y1 = bbox.y1 * multiplier;
    bbox_out.y2 = bbox.y2 * multiplier;
end

% add some value to a bbox.{x1 x2 y1 y2}.
function bbox_out = bbox_add(bbox, num_to_add)
    bbox_out.x1 = bbox.x1 + num_to_add;
    bbox_out.x2 = bbox.x2 + num_to_add;
    bbox_out.y1 = bbox.y1 + num_to_add;
    bbox_out.y2 = bbox.y2 + num_to_add;
end

% @param scale_to_use = a scale that maps bbox to templateSize
% @param scales = list of scales that we computed in our pyramid
function [scale_to_use, scaleIdx] = findNearestScale(scale_to_use, scales)

    %scaleIdx = index of nearestScale in the list of scales
    [nearestScale_distance, scaleIdx] = min(abs(scale_to_use - scales));
    scale_to_use = scales(scaleIdx);
end

