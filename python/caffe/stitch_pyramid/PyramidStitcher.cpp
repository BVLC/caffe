
#include "SimpleOpt.h"
#include "JPEGPyramid.h"
#include "JPEGImage.h" 
#include "Patchwork.h"
#include "PyramidStitcher.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace FFLD;
using namespace std;

//TODO: have a pyramid stitch class?

//  @param planeDim == width == height of planes to cover with images (optional)
//            if planeDim <= 0, then ignore planeDim and compute plane size based on input image dims
Patchwork stitch_pyramid(string file, int padding, int interval, int planeDim)
{
	JPEGImage image(file);
    if (image.empty()) {
        cerr << "\nInvalid image " << file << endl;
    }

    //image = image.resize(image.width()*4, image.height()*4); //UPSAMPLE so that Caffe's 16x downsampling looks like 4x downsampling
    image = image.resize(image.width()*2, image.height()*2); //UPSAMPLE so that Caffe's 16x downsampling looks like 8x downsampling

  // Compute the downsample+stitch
    JPEGPyramid pyramid(image, padding, padding, interval); //multiscale DOWNSAMPLE with (padx == pady == padding)
    if (pyramid.empty()) {
        cerr << "\nInvalid image " << file << endl;
    }

    int planeWidth; 
    int planeHeight;
   
    if(planeDim > 0){
        planeWidth = planeDim;
        planeHeight = planeDim;
    } 
    else{
        planeWidth = (pyramid.levels()[0].width() + 15) & ~15; //TODO: don't subtract padx, pady? 
        planeHeight = (pyramid.levels()[0].height() + 15) & ~15; 
        planeWidth = max(planeWidth, planeHeight);  //SQUARE planes for Caffe convnet
        planeHeight = max(planeWidth, planeHeight);
    }

    Patchwork::Init(planeHeight, planeWidth); 
    const Patchwork patchwork(pyramid); //STITCH

    return patchwork;
}

//@param convnet_subsampling_ratio = difference between input image dim and convnet feature dim
//         e.g. if input img is 200x200 and conv5 is 25x25 ... 200/25=8 -> 8x downsampling in convnet
//JPEGPyramid unstitch_pyramid(Patchwork image_patchwork, float* convnet_planes, int convnet_subsampling_ratio){
vector<ScaleLocation> unstitch_pyramid_locations(Patchwork &patchwork, 
                                                 int convnet_subsampling_ratio)  
{
    int nbScales = patchwork.nbScales;
    vector<ScaleLocation> scaleLocations(nbScales);

    for(int i=0; i<nbScales; i++)
    {
        scaleLocations[i].xMin  = patchwork.rectangles_[i].first.x() / convnet_subsampling_ratio;
        scaleLocations[i].width = patchwork.rectangles_[i].first.width() / convnet_subsampling_ratio;
        scaleLocations[i].xMax = scaleLocations[i].width + scaleLocations[i].xMin; //already accounts for subsampling ratio

        scaleLocations[i].yMin = patchwork.rectangles_[i].first.y() / convnet_subsampling_ratio; 
        scaleLocations[i].height = patchwork.rectangles_[i].first.height() / convnet_subsampling_ratio;
        scaleLocations[i].yMax = scaleLocations[i].height + scaleLocations[i].yMin; //already accounts for subsampling ratio

        scaleLocations[i].planeID = patchwork.rectangles_[i].second;

    }

    return scaleLocations;
}

