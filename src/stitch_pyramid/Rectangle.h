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

#ifndef FFLD_RECTANGLE_H
#define FFLD_RECTANGLE_H

#include <iosfwd>

namespace FFLD
{
/// The Rectangle class defines a rectangle in the plane using integer precision. If the coordinates
/// of the top left corner of the rectangle are (x, y), the coordinates of the bottom right corner
/// are (x + width - 1, y + height - 1), where width and height are the dimensions of the rectangle.
/// The corners are thus understood as the extremal points still inside the rectangle.
class Rectangle
{
public:
	/// Constructs an empty rectangle. An empty rectangle has no area.
	Rectangle();
	
	/// Constructs a rectangle with the given @p width and @p height.
	Rectangle(int width, int height);
	
	/// Constructs a rectangle with coordinates (@p x, @p y) and the given @p width and @p height.
	Rectangle(int x, int y, int width, int height);
	
	/// Returns the x-coordinate of the rectangle.
	int x() const;
	
	/// Sets the x coordinate of the rectangle to @p x.
	void setX(int x);
	
	/// Returns the y-coordinate of the rectangle.
	int y() const;
	
	/// Sets the y coordinate of the rectangle to @p y.
	void setY(int y);
	
	/// Returns the width of the rectangle.
	int width() const;
	
	/// Sets the height of the rectangle to the given @p width.
	void setWidth(int width);
	
	/// Returns the height of the rectangle.
	int height() const;
	
	/// Sets the height of the rectangle to the given @p height.
	void setHeight(int height);
	
	/// Returns the left side of the rectangle.
	/// @note Equivalent to x().
	int left() const;
	
	/// Sets the left side of the rectangle to @p left.
	/// @note The right side of the rectangle is not modified.
	void setLeft(int left);
	
	/// Returns the top side of the rectangle.
	/// @note Equivalent to y().
	int top() const;
	
	/// Sets the top side of the rectangle to @p top.
	/// @note The bottom side of the rectangle is not modified.
	void setTop(int top);
	
	/// Returns the right side of the rectangle.
	/// @note Equivalent to x() + width() - 1.
	int right() const;
	
	/// Sets the right side of the rectangle to @p right.
	/// @note The left side of the rectangle is not modified.
	void setRight(int right);
	
	/// Returns the bottom side of the rectangle.
	/// @note Equivalent to y() + height() - 1.
	int bottom() const;
	
	/// Sets the bottom side of the rectangle to @p bottom.
	/// @note The top side of the rectangle is not modified.
	void setBottom(int bottom);
	
	/// Returns whether the rectangle is empty. An empty rectangle has no area.
	bool empty() const;
	
	/// Returns the area of the rectangle.
	/// @note Equivalent to max(width(), 0) * max(height(), 0).
	int area() const;
	
private:
	int x_;
	int y_;
	int width_;
	int height_;
};

/// Serializes a rectangle to a stream.
std::ostream & operator<<(std::ostream & os, const Rectangle & rect);

/// Unserializes a rectangle from a stream.
std::istream & operator>>(std::istream & is, Rectangle & rect);
}

#endif
