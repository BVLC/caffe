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

#include "JPEGImage.h"

#include <algorithm>
#include <utility>

#include <jpeglib.h>
#include <stdio.h>

using namespace FFLD;
using namespace std;

JPEGImage::JPEGImage() : width_(0), height_(0), depth_(0)
{
}

JPEGImage::JPEGImage(int width, int height, int depth, const uint8_t * bits) : width_(0),
height_(0), depth_(0)
{
	if ((width <= 0) || (height <= 0) || (depth <= 0))
		return;
	
	width_ = width;
	height_ = height;
	depth_ = depth;
	bits_.resize(width * height * depth);
	
	if (bits)
		copy(bits, bits + bits_.size(), bits_.begin());
}

JPEGImage::JPEGImage(const string & filename) : width_(0), height_(0), depth_(0)
{
	// Load the image
	FILE * file = fopen(filename.c_str(), "rb");
	
	if (!file)
		return;
	
	jpeg_decompress_struct cinfo;
	jpeg_error_mgr jerr;
	
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, file);
	
	if ((jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) || (cinfo.data_precision != 8) ||
		!jpeg_start_decompress(&cinfo)) {
		fclose(file);
		return;
	}
	
	vector<uint8_t> bits(cinfo.image_width * cinfo.image_height * cinfo.num_components);
	
	for (int y = 0; y < cinfo.image_height; ++y) {
		JSAMPLE * row = static_cast<JSAMPLE *>(&bits[y * cinfo.image_width * cinfo.num_components]);
		
		if (jpeg_read_scanlines(&cinfo, &row, 1) != 1) {
			fclose(file);
			return;
		}
	}
	
	jpeg_finish_decompress(&cinfo);
	
	fclose(file);
	
	// Recopy everyting if the loading was successful
	width_ = cinfo.image_width;
	height_ = cinfo.image_height;
	depth_ = cinfo.num_components;
	bits_.swap(bits);
}

int JPEGImage::width() const
{
	return width_;
}

int JPEGImage::height() const
{
	return height_;
}

int JPEGImage::depth() const
{
	return depth_;
}

const uint8_t * JPEGImage::bits() const
{
	return empty() ? 0 : &bits_[0];
}

uint8_t * JPEGImage::bits()
{
	return empty() ? 0 : &bits_[0];
}

const uint8_t * JPEGImage::scanLine(int y) const
{
	return (empty() || (y >= height_)) ? 0 : &bits_[y * width_ * depth_];
}

uint8_t * JPEGImage::scanLine(int y)
{
	return (empty() || (y >= height_)) ? 0 : &bits_[y * width_ * depth_];
}

bool JPEGImage::empty() const
{
	return (width() <= 0) || (height() <= 0) || (depth() <= 0);
}

void JPEGImage::save(const string & filename, int quality) const
{
	if (empty())
		return;
	
	FILE * file = fopen(filename.c_str(), "wb");
	
	if (!file)
		return;
	
	jpeg_compress_struct cinfo;
	jpeg_error_mgr jerr;
	
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, file);
	
	cinfo.image_width = width_;
	cinfo.image_height = height_;
	cinfo.input_components = depth_;
	cinfo.in_color_space = (depth_ == 1) ? JCS_GRAYSCALE : JCS_RGB;
	
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, TRUE);
	jpeg_start_compress(&cinfo, TRUE);
	
	for (int y = 0; y < height_; ++y) {
		const JSAMPLE * row = static_cast<const JSAMPLE *>(&bits_[y * width_ * depth_]);
		jpeg_write_scanlines(&cinfo, const_cast<JSAMPARRAY>(&row), 1);
	}
	
	jpeg_finish_compress(&cinfo);
	
	fclose(file);
}

JPEGImage JPEGImage::resize(int width, int height) const
{
	// Empty image
	if ((width <= 0) || (height <= 0))
		return JPEGImage();
	
	// Same dimensions
	if ((width == width_) && (height == height_))
		return *this;
	
	JPEGImage result;
    
	result.width_ = width;
	result.height_ = height;
	result.depth_ = depth_;
	result.bits_.resize(width * height * depth_);
	
	// Resize the image at each octave
	int srcWidth = width_;
	int srcHeight = height_;
	
	vector<uint8_t> tmpSrc;
	vector<uint8_t> tmpDst;
	
	float scale = 0.5f;
	int halfWidth = width_ * scale + 0.5f;
	int halfHeight = height_ * scale + 0.5f;
	
	while ((width <= halfWidth) && (height <= halfHeight)) {
		if (tmpDst.empty())
			tmpDst.resize(halfWidth * halfHeight * depth_);
		
		Resize(tmpSrc.empty() ? &bits_[0] : &tmpSrc[0], srcWidth, srcHeight, &tmpDst[0], halfWidth,
			   halfHeight, depth_);
		
		// Dst becomes src
		tmpSrc.swap(tmpDst);
		srcWidth = halfWidth;
		srcHeight = halfHeight;
		
		// Next octave
		scale *= 0.5f;
		halfWidth = width_ * scale + 0.5f;
		halfHeight = height_ * scale + 0.5f;
	}
	
	Resize(tmpSrc.empty() ? &bits_[0] : &tmpSrc[0], srcWidth, srcHeight, &result.bits_[0], width,
		   height, depth_);
	
	return result;
}

JPEGImage JPEGImage::crop(int x, int y, int width, int height) const
{
	// Empty image
	if ((width <= 0) || (height <= 0) || (x + width <= 0) || (y + height <= 0) || (x >= width_) ||
		(y >= height_))
		return JPEGImage();
	
	// Crop the coordinates to the image
	width = min(x + width - 1, width_ - 1) - max(x, 0) + 1;
	height = min(y + height - 1, height_ - 1) - max(y, 0) + 1;
	x = max(x, 0);
	y = max(y, 0);
	
	JPEGImage result;
    
	result.width_ = width;
	result.height_ = height;
	result.depth_ = depth_;
	result.bits_.resize(width * height * depth_);
	
	for (int y2 = 0; y2 < height; ++y2)
		for (int x2 = 0; x2 < width; ++x2)
			for (int i = 0; i < depth_; ++i)
				result.bits_[(y2 * width + x2) * depth_ + i] =
					bits_[((y + y2) * width_ + x + x2) * depth_ + i];
	
	return result;
}

//TODO: remove const?
JPEGImage JPEGImage::pad(int padx, int pady) const
{
    // empty image
    if( (padx < 0) || (pady < 0) )
        return JPEGImage();

    int srcWidth = this->width();
    int srcHeight = this->height();
	
	// new size with padding
    int dstWidth = srcWidth + 2*padx;
    int dstHeight = srcHeight + 2*pady;

	JPEGImage result;
	result.width_ = dstWidth;
	result.height_ = dstHeight;
	result.depth_ = depth_;
	result.bits_.resize(dstWidth * dstHeight * depth_);

    //rectangle packing offsets: 
    int x_off = padx; 
    int y_off = pady;

    //copy to padded image
    for (int y = 0; y < srcHeight; y++){
        for (int x = 0; x < srcWidth; x++){
            for (int ch = 0; ch < depth_; ch++){

                //result.bits_[...] = this->bits_[...];  
                result.bits_[(y+y_off)*dstWidth*depth_ + (x+x_off)*depth_ + ch] = this->bits_[y*srcWidth*depth_ + x*depth_ + ch];
            }
        }
    }
	return result;
}

// Bilinear interpolation coefficient
namespace FFLD
{
namespace detail
{
struct Bilinear
{
	int x0;
	int x1;
	float a;
	float b;
};
}
}

void JPEGImage::Resize(const uint8_t * src, int srcWidth, int srcHeight, uint8_t * dst,
					   int dstWidth, int dstHeight, int depth)
{
	if ((srcWidth == dstWidth) && (srcHeight == dstHeight)) {
		copy(src, src + srcWidth * srcHeight * depth, dst);
		return;
	}
	
	const float xScale = static_cast<float>(srcWidth) / dstWidth;
	const float yScale = static_cast<float>(srcHeight) / dstHeight;
	
	// Bilinear interpolation coefficients
	vector<detail::Bilinear> cols(dstWidth);
	
	for (int j = 0; j < dstWidth; ++j) {
		const float x = min(max((j + 0.5f) * xScale - 0.5f, 0.0f), srcWidth - 1.0f);
		cols[j].x0 = x;
		cols[j].x1 = min(cols[j].x0 + 1, srcWidth - 1);
		cols[j].a = x - cols[j].x0;
		cols[j].b = 1.0f - cols[j].a;
	}
	
	for (int i = 0; i < dstHeight; ++i) {
		const float y = min(max((i + 0.5f) * yScale - 0.5f, 0.0f), srcHeight - 1.0f);
		const int y0 = y;
		const int y1 = min(y0 + 1, srcHeight - 1);
		const float c = y - y0;
		const float d = 1.0f - c;
		
		for (int j = 0; j < dstWidth; ++j)
			for (int k = 0; k < depth; ++k)
				dst[(i * dstWidth + j) * depth + k] =
					(src[(y0 * srcWidth + cols[j].x0) * depth + k] * cols[j].b +
					 src[(y0 * srcWidth + cols[j].x1) * depth + k] * cols[j].a) * d +
					(src[(y1 * srcWidth + cols[j].x0) * depth + k] * cols[j].b +
					 src[(y1 * srcWidth + cols[j].x1) * depth + k] * cols[j].a) * c + 0.5f;
	}
}
