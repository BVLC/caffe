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
#include "imagenet_mean.hpp" //contains hard-coded imagenet RGB mean. from Caffe.

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

#include<sstream>
template< typename T > inline std::string str(T const & i) { std::stringstream s; s << i; return s.str(); } // convert T i to string

//TODO: make this uchar?
//linear interpolation, "lerp"
void JPEGPyramid::linear_interp(float val0, float val1, int n_elements, float* inout_lerp){

    float n_elements_inv = 1 / (float)n_elements;
    inout_lerp[0] = val0;
    inout_lerp[n_elements-1] = val1;

    for(int i=1; i < (n_elements-1); i++){
        float frac_offset = (n_elements-i) * n_elements_inv;

        inout_lerp[i] = val0 * frac_offset + 
                        val1 * (1 - frac_offset);
    }
}

// call this on each scaled image
// fills the image's padding with linear interpolated values from 'edge of img pixel' to 'imagenet mean'
void JPEGPyramid::AvgLerpPad(JPEGImage & image){

    int width = image.width(); //including padding
    int height = image.height();
    int depth = 3;

    float top_lerp[pady_+1]; //interpolated data to fill in above the current image column
    float bottom_lerp[pady_+1];
    float left_lerp[padx_+1];
    float right_lerp[padx_+1];
    float currPx = 0;

    assert( padx_ == pady_ ); // corner padding assumes this (for now)
    float corner_lerp[padx_+1];

    //top, bottom, left, right
    for(int ch=0; ch<3; ch++){
        float avgPx = IMAGENET_MEAN_RGB[ch];
        for(int x=padx_; x < width-padx_; x++){
  
            //top
            currPx = image.bits()[pady_*width*depth + x*depth + ch];
            linear_interp(currPx, avgPx, pady_+1, top_lerp); //populate top_lerp
            for(int y=0; y<pady_; y++){
                image.bits()[y*width*depth + x*depth + ch] = top_lerp[pady_ - y];
            }

            //bottom
            currPx = image.bits()[(height-pady_-1)*width*depth + x*depth + ch]; //TODO: or height-pady-1?
            linear_interp(currPx, avgPx, pady_+1, bottom_lerp); //populate bottom_lerp
            for(int y=0; y<pady_; y++){
                int imgY = y + height - pady_; //TODO: or height-pady-1?
                image.bits()[imgY*width*depth + x*depth + ch] = bottom_lerp[y];
            }
        }

        for(int y=pady_; y < height-pady_; y++){
            
            //left
            currPx = image.bits()[y*width*depth + padx_*depth + ch];
            linear_interp(currPx, avgPx, padx_+1, left_lerp); //populate top_lerp
            for(int x=0; x<padx_; x++){
                image.bits()[y*width*depth + x*depth + ch] = left_lerp[padx_ - x];
            }

            //right
            currPx = image.bits()[y*width*depth + (width-padx_-1)*depth + ch];
            linear_interp(currPx, avgPx, padx_+1, right_lerp); //populate top_lerp
            for(int x=0; x<padx_; x++){
                int imgX = x + width - padx_;
                image.bits()[y*width*depth + imgX*depth + ch] = right_lerp[x];
            }
        }

        //corner padding...
        // dim 0 --> - side of image
        assert( padx_ == pady_ ); // corner padding assumes this (for now)
        for( uint32_t dx = 0; dx != 2; ++ dx ) { 
          for( uint32_t dy = 0; dy != 2; ++ dy ) {
            // x,y is coord of dx,dy valid corner pixel of un-padded image
            uint32_t const x = dx ? (width-padx_-1) : padx_;
            uint32_t const y = dy ? (height-pady_-1) : pady_;
            for( uint32_t dd = 2; dd <= padx_; ++dd ) {
              uint32_t const cx = x + ( dx ? dd : -dd ); // cx,y is point on existing dx padding, dd outside image
              uint32_t const cy = y + ( dy ? dd : -dd ); // x,cy is point on existing dy padding, dd outside image
              assert( x < width ); assert( y < height );
              assert( cx < width ); assert( cy < height );
              float const cx_y_v = image.bits()[y*width*depth + cx*depth + ch];
              float const x_cy_v = image.bits()[cy*width*depth + x*depth + ch];
              linear_interp(cx_y_v, x_cy_v, dd+1, corner_lerp); // populate corner_lerp
              for( uint32_t ci = 1; ci < dd; ++ci ) { // fill in diagonal corner pixels
            uint32_t const cix = x  + ( dx ?  ci : -ci );
            uint32_t const ciy = cy + ( dy ? -ci :  ci );
            //printf( "dx=%s x=%s cx=%s ci=%s cix=%s dd=%s\n", 
            //	str(dx).c_str(), str(x).c_str(), str(cx).c_str(), str(ci).c_str(), str(cix).c_str(), str(dd).c_str() );
            assert( cix < width ); assert( ciy < height );
                    image.bits()[ciy*width*depth + cix*depth + ch] = corner_lerp[ci];
              }
            }
            // fill in all-mean outer half of corner (a triangle)
            uint32_t const cor_x = dx ? (width-1) : 0;
            uint32_t const cor_y = dy ? (height-1) : 0;
            // cor_x,cor_y is coord of dx,dy corner pixel of image (including padding)
            for( uint32_t dd = 0; dd < padx_; ++dd ) {
              for( uint32_t ddi = 0; ddi < (padx_-dd); ++ddi ) {
            uint32_t const tri_x = cor_x + ( dx ? -dd : dd );
            uint32_t const tri_y = cor_y + ( dy ? -ddi : ddi );
            assert( tri_x < width ); assert( tri_y < height );
                    image.bits()[tri_y*width*depth + tri_x*depth + ch] = avgPx;
              }
            }
          }
        }
    }
}

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
        AvgLerpPad(scaled); //linear interpolate edge pixels to imagenet mean

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

