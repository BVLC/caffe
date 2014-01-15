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
//void stitch_pyramid(string file, Patchwork out_patchwork, int padding=8, int interval=10);
Patchwork stitch_pyramid(string file, int padding=8, int interval=10);

#endif

