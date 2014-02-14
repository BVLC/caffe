#ifndef PYRAMID_STITCHER_H
#define PYRAMID_STITCHER_H

#include "SimpleOpt.h"
#include "JPEGPyramid.h"
#include "JPEGImage.h"
#include "Patchwork.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace FFLD;
using namespace std;

// location of a scale within a stitched feature pyramid
// can be used for data in image space or feature decscriptor space
class ScaleLocation{
    public:
        int xMin;
        int xMax;
        int yMin;
        int yMax;
        int width;
        int height;

        int planeID;
        //int scaleIdx; //TODO?
};


//image -> multiscale pyramid -> stitch to same-sized planes for Caffe convnet
Patchwork stitch_pyramid(string file, int img_minWidth=1, int img_minHeight=1, 
                         int padding=8, int interval=10, int planeDim=-1);

// coordinates for unstitching the feature descriptors from planes.
//      sorted in descending order of size. 
//        (well, Patchwork sorts in descending order of size, and that survives here.)
vector<ScaleLocation> unstitch_pyramid_locations(Patchwork &patchwork,
                                                 int sbin);


#endif

