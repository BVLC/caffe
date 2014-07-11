// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_RECT_HPP_
#define CAFFE_RECT_HPP_

#include <algorithm>

namespace caffe {

class Rect {
 public:
  Rect(const int x1 = 0, const int y1 = 0, const int x2 = 0, const int y2 = 0)
     : x1_(x1), y1_(y1), x2_(x2), y2_(y2) {
  }

  Rect(const Rect& that) : x1_(that.x1()),
    y1_(that.y1()), x2_(that.x2()), y2_(that.y2()) {
  }

  Rect& operator = (const Rect& that) {
    x1_ = that.x1();
    y1_ = that.y1();
    x2_ = that.x2();
    y2_ = that.y2();
    return *this;
  }

  bool operator == (const Rect& that) const {
    return x1_ == that.x1() && y1_ == that.y1() && x2_ == that.x2() &&
        y2_ == that.y2();
  }

  inline int height() const {
    return y2_ - y1_;
  }

  inline int width() const {
    return x2_ - x1_;
  }

  inline float area() const {
    if (empty()) {
      return 0;
    }
    return static_cast<float>(width() * height());
  }

  Rect intersect(const Rect& that) const {
    int rx1 = std::max(x1_, that.x1());
    int ry1 = std::max(y1_, that.y1());
    int rx2 = std::min(x2_, that.x2());
    int ry2 = std::min(y2_, that.y2());
    Rect rect(rx1, ry1, rx2, ry2);
    if (rect.empty()) {
      Rect zero_rect(0, 0, 0, 0);
      return zero_rect;
    }
    return rect;
  }

  bool empty() const {
    if (x1_ >= x2_ || y1_ >= y2_) {
      return true;
    }
    return false;
  }

  inline int x1() const { return x1_; }
  inline int y1() const { return y1_; }
  inline int x2() const { return x2_; }
  inline int y2() const { return y2_; }

 protected:
  // x1_ and y1_ are inclusive
  int x1_;  // left upper corner horizontal
  int y1_;  // left upper corner vertical
  // x2_ and y2_ are exclusive
  int x2_;  // right bottom corner horizontal
  int y2_;  // right bottom corner vertical
};

}  // namespace caffe

#endif  // CAFFE_RECT_HPP_
