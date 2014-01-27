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

#include "Rectangle.h"

#include <algorithm>
#include <iostream>

using namespace FFLD;
using namespace std;

Rectangle::Rectangle() : x_(0), y_(0), width_(0), height_(0)
{
}

Rectangle::Rectangle(int width, int height) : x_(0), y_(0), width_(width), height_(height)
{
}

Rectangle::Rectangle(int x, int y, int width, int height) : x_(x), y_(y), width_(width),
height_(height)
{
}

int Rectangle::x() const
{
	return x_;
}

void Rectangle::setX(int x)
{
	x_ = x;
}

int Rectangle::y() const
{
	return y_;
}

void Rectangle::setY(int y)
{
	y_ = y;
}

int Rectangle::width() const
{
	return width_;
}

void Rectangle::setWidth(int width)
{
	width_ = width;
}

int Rectangle::height() const
{
	return height_;
}

void Rectangle::setHeight(int height)
{
	height_ = height;
}

int Rectangle::left() const
{
	return x();
}

void Rectangle::setLeft(int left)
{
	setWidth(right() - left + 1);
	setX(left);
}

int Rectangle::top() const
{
	return y();
}

void Rectangle::setTop(int top)
{
	setHeight(bottom() - top + 1);
	setY(top);
}

int Rectangle::right() const
{
	return x() + width() - 1;
}

void Rectangle::setRight(int right)
{
	setWidth(right - left() + 1);
}

int Rectangle::bottom() const
{
	return y() + height() - 1;
}

void Rectangle::setBottom(int bottom)
{
	setHeight(bottom - top() + 1);
}

bool Rectangle::empty() const
{
	return (width() <= 0) || (height() <= 0);
}

int Rectangle::area() const
{
	return max(width(), 0) * max(height(), 0);
}

ostream & FFLD::operator<<(ostream & os, const Rectangle & rect)
{
	return os << rect.x() << ' ' << rect.y() << ' ' << rect.width() << ' ' << rect.height();
}

istream & FFLD::operator>>(istream & is, Rectangle & rect)
{
	int x, y, width, height;
	
    is >> x >> y >> width >> height;
	
    rect.setX(x);
	rect.setY(y);
	rect.setWidth(width);
	rect.setHeight(height);
	
	return is;
}
