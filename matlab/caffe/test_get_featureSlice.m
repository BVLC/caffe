
function test_getFeatureSlice()

    %TODO: add voc-release5/features to path.
    addpath('../../voc-release5/bin')
    addpath('../../voc-release5/features')
    addpath('../../voc-release5/vis')

    im = imread('./pascal_000001.jpg');

    %bare bones model to run DPM's HOG code
    model.sbin = 4;
    %model.features.sbin = model.sbin;
    model.padx = 8;
    model.pady = 8;
    model.interval = 10;    
    model.features.extra_octave = 0;
    model.features.dim = 32;
    model.features.truncation_dim = 32;

    pyra = featpyramid(im, model, model.padx, model.pady);

    templateSize = [6 10]; 
    imageSize = size(im);
    bbox.x1 = 110;
    bbox.x2 = 190;
    bbox.y1 = 100;
    bbox.y2 = 148;

    pyra.sbin = model.sbin; 
    [featureSlice, scaleIdx] = get_featureSlice(pyra, bbox, templateSize, imageSize);

    % if using convnets...
    %figure(1)
    %colormap(gray);    
    %imagesc(sum(featureSlice,3))

    % if using HOG...
    figure(2)
    w = foldHOG(featureSlice);
    visualizeHOG(w);

    figure(3);
    w = foldHOG(pyra.feat{scaleIdx});
    visualizeHOG(w);

end 



