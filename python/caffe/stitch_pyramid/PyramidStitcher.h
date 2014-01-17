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

//image -> multiscale pyramid -> stitch to same-sized planes for Caffe convnet
Patchwork stitch_pyramid(string file, int padding=8, int interval=10, int planeDim=-1);

#endif

