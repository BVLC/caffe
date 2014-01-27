


This is Forrest's hack to do the following...

1. Input: image filename from PASCAL VOC detection challenge
2. sample multiscale
3. Output: stich multiscale onto same-sized planes.
    (TODO: add a text file that explains where the images were placed in planes!)
    

    saves something like this:
    inputFilename_plane0.jpg
    inputFilename_plane1.jpg
    ...

DONE:
 find/replace NbFeatures for NbChannels.
 set NbChannels=3

 in Patchwork.{cpp, h}, change the definition of 'Plane' to JPEGImage

 in stitch_pyramid.cpp, call Patchwork(), with MaxRows_ and MaxCols_ as 'biggest pyra scale, rounded up to a factor of 16'

 add a JPEGImage::pad() function that creates a padded copy.

 in stitch_pyramid.cpp, replace {hog, HOG, Hog} to JPEG


