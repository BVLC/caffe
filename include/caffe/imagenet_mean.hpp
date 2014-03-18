#ifndef IMAGENET_MEAN_H
#define IMAGENET_MEAN_H

//mean of all imagenet classification images
// calculation:
// 1. mean of all imagenet images, per pixel location
// 2. take the mean image, and get the mean pixel for R,G,B
// (did it this way because we already had the 'mean of all images, per pixel location')


#define IMAGENET_MEAN_R 122.67f
#define IMAGENET_MEAN_G 116.66f
#define IMAGENET_MEAN_B 104.00f

#endif
