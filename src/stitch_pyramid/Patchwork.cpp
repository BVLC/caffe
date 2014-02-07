//--------------------------------------------------------------------------------------------------
// Implementation of the paper "Exact Acceleration of Linear Object Detectors", 12th European
// Conference on Computer Vision, 2012.
//
// Copyright (c) 2012 Idiap Research Institute, <http://www.idiap.ch/>
// Written by Charles Dubout <charles.dubout@idiap.ch>
//
// This file is part of FFLD (the Fast Fourier Linear Detector)
//
// FFLD is free software: you can redistribute it and/or modify it under the terms of the GNU
// General Public License version 3 as published by the Free Software Foundation.
//
// FFLD is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU General Public License along with FFLD. If not, see
// <http://www.gnu.org/licenses/>.
//--------------------------------------------------------------------------------------------------

#include "Patchwork.h"
#include "JPEGPyramid.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <set>
#include <iostream>
#include <assert.h>

using namespace FFLD;
using namespace std;

int Patchwork::MaxRows_(0);
int Patchwork::MaxCols_(0);
int Patchwork::HalfCols_(0);

Patchwork::Patchwork() : padx_(0), pady_(0), interval_(0)
{
}

Patchwork::Patchwork(JPEGPyramid & pyramid) : padx_(pyramid.padx()), pady_(pyramid.pady()),
interval_(pyramid.interval())
{
    int sbin = 16;  //convnet_downsampling_factor -- TODO: take as user input
    int templateWidth = 5*sbin; //TODO: take as user input
    int templateHeight = 15*sbin; 

    imwidth_ = pyramid.imwidth_;
    imheight_ = pyramid.imheight_;
    scales_ = pyramid.scales_; //keep track of pyra scales
    nbScales = pyramid.levels().size(); //Patchwork class variable

    //printf("        before prune_big_scales(). scales_.size()=%ld, pyramid.levels_.size()=%ld \n", scales_.size(), pyramid.levels_.size());
    prune_big_scales(pyramid); //remove scales that won't fit in (MaxRows, MaxCols)
    prune_small_scales(pyramid, templateWidth, templateHeight);
    cout << "    nbScales = " << nbScales << endl;
    //printf("        after prune_big_scales(). scales_.size()=%ld, pyramid.levels_.size()=%ld \n", scales_.size(), pyramid.levels_.size());

    rectangles_.resize(nbScales);
	
	for (int i = 0; i < nbScales; ++i) {
        rectangles_[i].first.setWidth(pyramid.levels()[i].width()); //stitching includes padding in the img size.
        rectangles_[i].first.setHeight(pyramid.levels()[i].height());
	}
	
	// Build the patchwork planes
	const int nbPlanes = BLF(rectangles_);
	cout << "    nbPlanes = " << nbPlanes << endl;
	
    assert(nbPlanes >= 0);
	
	planes_.resize(nbPlanes);
    #pragma omp parallel for
	for (int i = 0; i < nbPlanes; ++i) {
        planes_[i] = JPEGImage(MaxCols_, MaxRows_, JPEGPyramid::NbChannels);     //JPEGImage(width, height, depth)
        //planes_[i].fill_with_rand(); //random noise that will go between images on plane. (TODO: allow user to enable/disable)
        planes_[i].fill_with_imagenet_mean(); //fill with imagenet avg pixel value
	}

    int depth = JPEGPyramid::NbChannels;

    // [Forrest implemented... ]
    //COPY scaled images -> fixed-size planes
    #pragma omp parallel for
    for (int i = 0; i < nbScales; ++i) {

        //currPlane is destination
        JPEGImage* currPlane = &planes_[rectangles_[i].second]; //TODO: make sure my dereferencing makes sense. (trying to avoid deepcopy)

        //currLevel is source
        const JPEGImage* currLevel = &pyramid.levels()[i]; 

        int srcWidth = currLevel->width();
        int srcHeight = currLevel->height();

        int dstWidth = currPlane->width();
        int dstHeight = currPlane->height();
      
            //rectangle packing offsets: 
        int x_off = rectangles_[i].first.x(); 
        int y_off = rectangles_[i].first.y();
    
        for (int y = 0; y < srcHeight; y++){
            for (int x = 0; x < srcWidth; x++){
                for (int ch = 0; ch < depth; ch++){

                    //currPlane.bits[...] = pyramid.levels()[i].data[...];
                    currPlane->bits()[((y + y_off)*dstWidth*depth) + ((x + x_off)*depth) + ch] = currLevel->bits()[y*srcWidth*depth + x*depth + ch];
                }
            }
        }
    }
}

//remove scales that are bigger than one Patchwork plane
void Patchwork::prune_big_scales(JPEGPyramid & pyramid){

    int first_valid_scale_idx = pyramid.levels_.size();

    for(int i=0; i < pyramid.levels_.size(); i++){
        if( !((pyramid.levels_[i].width() > MaxCols_) || (pyramid.levels_[i].height() > MaxRows_)) )
        {
            first_valid_scale_idx = i;
            break;
        }
    }

    if(first_valid_scale_idx > 0){
        scales_.erase( scales_.begin(), scales_.begin() + first_valid_scale_idx);
        pyramid.levels_.erase( pyramid.levels_.begin(), pyramid.levels_.begin() + first_valid_scale_idx);
        nbScales = nbScales - first_valid_scale_idx;
        printf("had to remove first %d scales, because they didn't fit in Patchwork plane \n", first_valid_scale_idx);
    }
}

//remove scales that are smaller than the template
void Patchwork::prune_small_scales(JPEGPyramid & pyramid, int templateWidth, int templateHeight){

    int last_valid_scale_idx = 0;

    //for(int i=0; i < pyramid.levels_.size(); i++){
    for(int i = pyramid.levels_.size()-1; i >= 0; i--){
//printf("pyramid.levels_[%d].width=%d, templateWidth=%d \n", i, pyramid.levels_[i].width(), templateWidth);
        if( ((pyramid.levels_[i].width() >= templateWidth) && (pyramid.levels_[i].height() >= templateHeight)) )
        {
            last_valid_scale_idx = i;
            break;
        }
    }
printf(" last_valid_scale_idx = %d \n", last_valid_scale_idx);

    if(last_valid_scale_idx < pyramid.levels_.size()){
        scales_.erase( scales_.begin() + last_valid_scale_idx, scales_.end() );
        pyramid.levels_.erase( pyramid.levels_.begin() + last_valid_scale_idx, pyramid.levels_.end());
        int num_scales_to_remove = nbScales - last_valid_scale_idx;
        nbScales = nbScales - num_scales_to_remove;
        printf("had to remove last %d scales, because they were smaller than the template \n", num_scales_to_remove);
    }
}

int Patchwork::padx() const
{
	return padx_;
}

int Patchwork::pady() const
{
	return pady_;
}

int Patchwork::interval() const
{
	return interval_;
}

bool Patchwork::empty() const
{
	return planes_.empty();
}

bool Patchwork::Init(int maxRows, int maxCols)
{
	// It is an error if maxRows or maxCols are too small
	if ((maxRows < 2) || (maxCols < 2))
		return false;
	
	// Temporary matrices
	//JPEGPyramid::Matrix tmp(maxRows * JPEGPyramid::NbChannels, maxCols + 2);
	
	int dims[2] = {maxRows, maxCols};
    MaxRows_ = maxRows;
    MaxCols_ = maxCols;
    HalfCols_ = maxCols / 2 + 1;
}

int Patchwork::MaxRows()
{
	return MaxRows_;
}

int Patchwork::MaxCols()
{
	return MaxCols_;
}

namespace FFLD
{
namespace detail
{
// Order rectangles by decreasing area.
class AreaComparator
{
public:
	AreaComparator(const vector<pair<Rectangle, int> > & rectangles) :
	rectangles_(rectangles)
	{
	}
	
	/// Returns whether rectangle @p a comes before @p b.
	bool operator()(int a, int b) const
	{
		const int areaA = rectangles_[a].first.area();
		const int areaB = rectangles_[b].first.area();
		
		return (areaA > areaB) || ((areaA == areaB) && (rectangles_[a].first.height() >
														rectangles_[b].first.height()));
	}
	
private:
	const vector<pair<Rectangle, int> > & rectangles_;
};

// Order free gaps (rectangles) by position and then by size
struct PositionComparator
{
	// Returns whether rectangle @p a comes before @p b
	bool operator()(const Rectangle & a, const Rectangle & b) const
	{
		return (a.y() < b.y()) ||
			   ((a.y() == b.y()) &&
				((a.x() < b.x()) ||
				 ((a.x() == b.x()) &&
				  ((a.height() > b.height()) ||
				   ((a.height() == b.height()) && (a.width() > b.width()))))));
	}
};
}
}

int Patchwork::BLF(vector<pair<Rectangle, int> > & rectangles)
{
	// Order the rectangles by decreasing area. If a rectangle is bigger than MaxRows x MaxCols
	// return -1
	vector<int> ordering(rectangles.size());
	
	for (int i = 0; i < rectangles.size(); ++i) {
		if ((rectangles[i].first.width() > MaxCols_) || (rectangles[i].first.height() > MaxRows_))
			return -1;
		
		ordering[i] = i;
	}
	
	sort(ordering.begin(), ordering.end(), detail::AreaComparator(rectangles));
	
	// Index of the plane containing each rectangle
	for (int i = 0; i < rectangles.size(); ++i)
		rectangles[i].second = -1;
	
	vector<set<Rectangle, detail::PositionComparator> > gaps;
	
	// Insert each rectangle in the first gap big enough
	for (int i = 0; i < rectangles.size(); ++i) {
		pair<Rectangle, int> & rect = rectangles[ordering[i]];
		
		// Find the first gap big enough
		set<Rectangle, detail::PositionComparator>::iterator g;
		
		for (int i = 0; (rect.second == -1) && (i < gaps.size()); ++i) {
			for (g = gaps[i].begin(); g != gaps[i].end(); ++g) {
				if ((g->width() >= rect.first.width()) && (g->height() >= rect.first.height())) {
					rect.second = i;
					break;
				}
			}
		}
		
		// If no gap big enough was found, add a new plane
		if (rect.second == -1) {
			set<Rectangle, detail::PositionComparator> plane;
			plane.insert(Rectangle(MaxCols_, MaxRows_)); // The whole plane is free
			gaps.push_back(plane);
			g = gaps.back().begin();
			rect.second = gaps.size() - 1;
		}
		
		// Insert the rectangle in the gap
		rect.first.setX(g->x());
		rect.first.setY(g->y());
		
		// Remove all the intersecting gaps, and add newly created gaps
		for (g = gaps[rect.second].begin(); g != gaps[rect.second].end();) {
			if (!((rect.first.right() < g->left()) || (rect.first.bottom() < g->top()) ||
				  (rect.first.left() > g->right()) || (rect.first.top() > g->bottom()))) {
				// Add a gap to the left of the new rectangle if possible
				if (g->x() < rect.first.x())
					gaps[rect.second].insert(Rectangle(g->x(), g->y(), rect.first.x() - g->x(),
													   g->height()));
				
				// Add a gap on top of the new rectangle if possible
				if (g->y() < rect.first.y())
					gaps[rect.second].insert(Rectangle(g->x(), g->y(), g->width(),
													   rect.first.y() - g->y()));
				
				// Add a gap to the right of the new rectangle if possible
				if (g->right() > rect.first.right())
					gaps[rect.second].insert(Rectangle(rect.first.right() + 1, g->y(),
													   g->right() - rect.first.right(),
													   g->height()));
				
				// Add a gap below the new rectangle if possible
				if (g->bottom() > rect.first.bottom())
					gaps[rect.second].insert(Rectangle(g->x(), rect.first.bottom() + 1, g->width(),
													   g->bottom() - rect.first.bottom()));
				
				// Remove the intersecting gap
				gaps[rect.second].erase(g++);
			}
			else {
				++g;
			}
		}
	}
	
	return gaps.size();
}
