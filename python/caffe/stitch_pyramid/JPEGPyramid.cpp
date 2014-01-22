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

#include "JPEGPyramid.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace FFLD;
using namespace std;

JPEGPyramid::JPEGPyramid() : padx_(0), pady_(0), interval_(0)
{
}

JPEGPyramid::JPEGPyramid(int padx, int pady, int interval, const vector<Level> & levels) : padx_(0),
pady_(0), interval_(0)
{
	if ((padx < 1) || (pady < 1) || (interval < 1))
		return;
	
	padx_ = padx;
	pady_ = pady;
	interval_ = interval;
	levels_ = levels;
}

JPEGPyramid::JPEGPyramid(const JPEGImage & image, int padx, int pady, int interval, int upsampleFactor) : padx_(0),
pady_(0), interval_(0)
{
	if (image.empty() || (padx < 1) || (pady < 1) || (interval < 1))
		return;
	
	// Copmute the number of scales such that the smallest size of the last level is 5
	const int numScales = ceil(log(min(image.width(), image.height()) / 40.0) / log(2.0)) * interval; //'max_scale' in voc5 featpyramid.m
	
	// Cannot compute the pyramid on images too small
	if (numScales < interval)
		return;

	padx_ = padx;
	pady_ = pady;
	interval_ = interval;
	levels_.resize(numScales+1);
    scales_.resize(numScales+1);

#pragma omp parallel for 
    for (int i = 0; i <= numScales; ++i){
        //generic pyramid... not stitched.

		double scale = pow(2.0, static_cast<double>(-i) / interval) * upsampleFactor;
		JPEGImage scaled = image.resize(image.width() * scale + 0.5, image.height() * scale + 0.5);
        bool use_randPad = false;
        scaled = scaled.pad(padx, pady, use_randPad); //an additional deepcopy. (for efficiency, could have 'resize()' accept padding too

        scales_[i] = scale;
        levels_[i] = scaled;
    }
}

int JPEGPyramid::padx() const
{
	return padx_;
}

int JPEGPyramid::pady() const
{
	return pady_;
}

int JPEGPyramid::interval() const
{
	return interval_;
}

const vector<JPEGPyramid::Level> & JPEGPyramid::levels() const
{
	return levels_;
}

bool JPEGPyramid::empty() const
{
	return levels().empty();
}

