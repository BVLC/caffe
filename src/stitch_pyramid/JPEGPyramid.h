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

#ifndef FFLD_JPEGPYRAMID_H
#define FFLD_JPEGPYRAMID_H

#include "JPEGImage.h"

using namespace std;

namespace FFLD
{
/// The JPEGPyramid class computes and stores the HOG features extracted from a jpeg image at
/// multiple scales. The scale of the pyramid level of index @c i is given by the following formula:
/// 2^(1 - @c i / @c interval), so that the first scale is at double the resolution of the original
/// image). Each level is padded with zeros horizontally and vertically by a fixed amount. The last
/// feature is special: it takes the value one in the padding and zero otherwise.
/// @note Define the PASCAL_JPEGPYRAMID_FELZENSZWALB_FEATURES flag during compilation to use
/// Felzenszwalb's original features (slower and not as accurate as they do no angular
/// interpolation, provided for compatibility only).
/// @note Define the PASCAL_JPEGPYRAMID_DOUBLE to use double scalar values instead of float (slower,
/// uses twice the amount of memory, and the increase in precision is not necessarily useful).
class JPEGPyramid
{
public:
	static const int NbChannels = 3; //RGB
	
	/// Type of a scalar value.
	typedef float Scalar;

	/// Type of a pyramid level (matrix of cells).
    typedef JPEGImage Level;	

	/// Constructs an empty pyramid. An empty pyramid has no level.
	JPEGPyramid();
	
	/// Constructs a pyramid from parameters and a list of levels.
	/// @param[in] padx Amount of horizontal zero padding (in cells).
	/// @param[in] pady Amount of vertical zero padding (in cells).
	/// @param[in] interval Number of levels per octave in the pyramid.
	/// @param[in] levels List of pyramid levels.
	/// @note The amount of padding and the interval should be at least 1.
	JPEGPyramid(int padx, int pady, int interval, const std::vector<Level> & levels);
	
	/// Constructs a pyramid from the JPEGImage of a Scene.
	/// @param[in] image The JPEGImage of the Scene.
	/// @param[in] padx Amount of horizontal zero padding (in cells).
	/// @param[in] pady Amount of vertical zero padding (in cells).
	/// @param[in] interval Number of levels per octave in the pyramid.
    /// @param[in] upsampleFactor = how big should top scale be? (compared to input img)
	/// @note The amount of padding and the interval should be at least 1.
	JPEGPyramid(const JPEGImage & image, int padx, int pady, int interval = 10, int upsampleFactor = 2);
	
	/// Returns whether the pyramid is empty. An empty pyramid has no level.
	bool empty() const;
	
	/// Returns the amount of horizontal zero padding (in cells).
	int padx() const;
	
	/// Returns the amount of vertical zero padding (in cells).
	int pady() const;
	
	/// Returns the number of levels per octave in the pyramid.
	int interval() const;
	
	/// Returns the pyramid levels.
	/// @note Scales are given by the following formula: 2^(1 - @c index / @c interval).
	const std::vector<Level> & levels() const;

    //upsampling/downsampling multipliers. (scales_[i] corresponds to levels_[i])
    vector<float> scales_;

    //populate inout_lerp with a sliding range from val0 to val1 (for padding images)
    void linear_interp(float val0, float val1, int n_elements, float* inout_lerp);

    //fill an image's padding with linear interpolated data: [from edge of img to imagenet mean]
    void AvgLerpPad(JPEGImage & image);

private:
	int padx_;
	int pady_;
	int interval_;
	std::vector<Level> levels_;
};
}

#endif
