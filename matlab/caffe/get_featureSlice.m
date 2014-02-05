

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
    bbox_desc_scale = templateSize / bbox_desc_dim; %scale factor from (scale=1) to a scale where bbox fits in templateSize

    scale_to_use = mean(bbox_desc_scale); %avg of scale factors for x and y dims.
    [scale_to_use, scaleIdx] = findNearestScale(scale_to_use, pyra.scales); %best precomputed approx of scale_to_use

    %padx_desc_scaled = (pyra.padx / pyra.sbin) * scale_to_use; %in descriptor coords, at scale_to_use.
    %pady_desc_scaled = (pyra.pady / pyra.sbin) * scale_to_use;
    padx_desc_scaled = pyra.padx; %AHA, this is really in HOG cells and not image pixels.
    pady_desc_scaled = pyra.pady;


  %3. rip out a slice from the appropriate scale
  
    %TODO: center it on (bbox_desc.x2 - bbox_desc.x1), (bbox_desc.y2 - bbox_desc.y1)

    bbox_desc_x_center = bbox_desc.x1 + (bbox_desc.x2 - bbox_desc.x1)/2.0;
    bbox_desc_y_center = bbox_desc.y1 + (bbox_desc.y2 - bbox_desc.y1)/2.0;
    bbox_to_use.x1 = round(bbox_desc_x_center*scale_to_use - templateSize(2)/2.0 + padx_desc_scaled); %(center - templateWidth/2)
    bbox_to_use.x2 = bbox_to_use.x1 + templateSize(2) - 1;
    bbox_to_use.y1 = round(bbox_desc_y_center*scale_to_use - templateSize(1)/2.0 + pady_desc_scaled); %(center - templateHeight/2)
    bbox_to_use.y2 = bbox_to_use.y1 + templateSize(1) - 1;
    %display('new...') 
    %bbox_to_use

    %bbox_to_use.x1 = round(bbox_desc.x1 * scale_to_use + padx_desc_scaled); 
    %bbox_to_use.x2 = bbox_to_use.x1 + templateSize(2) - 1;
    %bbox_to_use.y1 = round(bbox_desc.y1 * scale_to_use + pady_desc_scaled); 
    %bbox_to_use.y2 = bbox_to_use.y1 + templateSize(1) - 1;
    %display('old...')
    %bbox_to_use
    %scale_to_use
    %scaleIdx

    featureSlice = pyra.feat{scaleIdx}(bbox_to_use.y1 : bbox_to_use.y2, bbox_to_use.x1 : bbox_to_use.x2, :); 

  %4. project rounded box back to img space (approx) -- for debugging
    roundedBox_in_px = bbox_mult(bbox_to_use, (pyra.sbin / scale_to_use)); 
    %roundedBox_in_px = bbox_add(roundedBox_in_px, -pyra.padx*pyra.sbin/scale_to_use); %assume padx==pady
    roundedBox_in_px.x1 = roundedBox_in_px.x1 - pyra.padx*pyra.sbin/scale_to_use;
    roundedBox_in_px.x2 = roundedBox_in_px.x2 - pyra.padx*pyra.sbin/scale_to_use;
    roundedBox_in_px.y1 = roundedBox_in_px.y1 - pyra.pady*pyra.sbin/scale_to_use;
    roundedBox_in_px.y2 = roundedBox_in_px.y2 - pyra.pady*pyra.sbin/scale_to_use;
end

% @param gt_bbox = example bbox to slice out of pyramid (in pixel coords)
% @param templateSize = shape desired in terms of HOG cells
function gt_bbox_new = match_aspect_ratio(gt_bbox, templateSize)

    aspect_ratio_template = templateSize(1)/templateSize(2) % height/width
    aspect_ratio_gt = (gt_bbox.y2 - gt_bbox.y1) / (gt_bbox.x2 - gt_bbox.x1)

    gt_bbox_new = gt_bbox; 
    if(aspect_ratio_template < aspect_ratio_gt) %gt_bbox is too tall
        %make gt_bbox wider (height stays unchanged)
        gt_width = gt_bbox.x2 - gt_bbox.x1;
        gt_new_width = gt_width * (aspect_ratio_gt / aspect_ratio_template); %wider
        gt_bbox_new.x2 = gt_bbox.x2 - gt_width + gt_new_width;
    
    elseif(aspect_ratio_template > aspect_ratio_gt) %gt_bbox is too wide
        %make gt_bbox taller (width stays unchanged)
        gt_height = gt_bbox.y2 - gt_bbox.y1;
        gt_new_height = gt_height * (aspect_ratio_template / aspect_ratio_gt); %taller
        gt_bbox_new.y2 = gt_bbox.y2 - gt_height + gt_new_height;
    end
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

