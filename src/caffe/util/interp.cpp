// Copyright 2014 George Papandreou

#include "caffe/common.hpp"
#include "caffe/util/interp.hpp"
#include <algorithm>
#include <cmath>

namespace caffe {

// Bi-linear interpolation
// IN : [channels height1 width1] cropped from a bigger [Height1 Width1] image
// OUT: [channels height2 width2] cropped from a bigger [Height2 Width2] image
template <typename Dtype, bool packed>
void caffe_cpu_interp2(const int channels,
    const Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  CHECK(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
  CHECK(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
  // special case: just copy
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
	const int w1 = w2;
	if (packed) {
	  const Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
	  Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
	  for (int c = 0; c < channels; ++c) {
	    pos2[0] = pos1[0];
	    pos1++;
	    pos2++;
	  }
	}
	else {
	  const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	  Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	  for (int c = 0; c < channels; ++c) {
	    pos2[0] = pos1[0];
	    pos1 += Width1 * Height1;
	    pos2 += Width2 * Height2;
	  }
	}
      }
    }
    return;
  }
  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const Dtype w1lambda = w1r - w1;
      const Dtype w0lambda = Dtype(1.) - w1lambda;
      if (packed) {
	const Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
	Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
	for (int c = 0; c < channels; ++c) {
	  pos2[0] =
	    h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[channels * w1p]) + 
	    h1lambda * (w0lambda * pos1[channels * h1p * Width1] + w1lambda * pos1[channels * (h1p * Width1 + w1p)]);
	  pos1++;
	  pos2++;
	}
      }
      else {
	const Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	for (int c = 0; c < channels; ++c) {
	  pos2[0] =
	    h0lambda * (w0lambda * pos1[0]            + w1lambda * pos1[w1p]) + 
	    h1lambda * (w0lambda * pos1[h1p * Width1] + w1lambda * pos1[h1p * Width1 + w1p]);
	  pos1 += Width1 * Height1;
	  pos2 += Width2 * Height2;
	}
      }
    }
  }
}


// Backward (adjoint) operation 1 <- 2 (accumulates)
template <typename Dtype, bool packed>
void caffe_cpu_interp2_backward(const int channels,
    Dtype *data1, const int x1, const int y1, const int height1, const int width1, const int Height1, const int Width1,
    const Dtype *data2, const int x2, const int y2, const int height2, const int width2, const int Height2, const int Width2) {
  CHECK(x1 >= 0 && y1 >= 0 && height1 > 0 && width1 > 0 && x2 >= 0 && y2 >= 0 && height2 > 0 && width2 > 0);
  CHECK(Width1 >= width1 + x1 && Height1 >= height1 + y1 && Width2 >= width2 + x2 && Height2 >= height2 + y2);
  // special case: same-size matching grids
  if (height1 == height2 && width1 == width2) {
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = h2;
      for (int w2 = 0; w2 < width2; ++w2) {
	const int w1 = w2;
	if (packed) {
	  Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
	  const Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
	  for (int c = 0; c < channels; ++c) {
	    pos1[0] += pos2[0];
	    pos1++;
	    pos2++;
	  }
	}
	else {
	  Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	  const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	  for (int c = 0; c < channels; ++c) {
	    pos1[0] += pos2[0];
	    pos1 += Width1 * Height1;
	    pos2 += Width2 * Height2;
	  }
	}
      }
    }
    return;
  }
  const float rheight = (height2 > 1) ? static_cast<float>(height1 - 1) / (height2 - 1) : 0.f;
  const float rwidth = (width2 > 1) ? static_cast<float>(width1 - 1) / (width2 - 1) : 0.f;
  for (int h2 = 0; h2 < height2; ++h2) {
    const float h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const Dtype h1lambda = h1r - h1;
    const Dtype h0lambda = Dtype(1.) - h1lambda;
    for (int w2 = 0; w2 < width2; ++w2) {
      const float w1r = rwidth * w2;
      const int w1 = w1r;
      const int w1p = (w1 < width1 - 1) ? 1 : 0;
      const Dtype w1lambda = w1r - w1;
      const Dtype w0lambda = Dtype(1.) - w1lambda;
      if (packed) {
	Dtype* pos1 = &data1[channels * ((y1 + h1) * Width1 + (x1 + w1))];
	const Dtype* pos2 = &data2[channels * ((y2 + h2) * Width2 + (x2 + w2))];
	for (int c = 0; c < channels; ++c) {
	  pos1[0] += h0lambda * w0lambda * pos2[0];
	  pos1[channels * w1p] += h0lambda * w1lambda * pos2[0];
	  pos1[channels * h1p * Width1] += h1lambda * w0lambda * pos2[0];
	  pos1[channels * (h1p * Width1 + w1p)] += h1lambda * w1lambda * pos2[0];
	  pos1++;
	  pos2++;
	}
      }
      else {
	Dtype* pos1 = &data1[(y1 + h1) * Width1 + (x1 + w1)];
	const Dtype* pos2 = &data2[(y2 + h2) * Width2 + (x2 + w2)];
	for (int c = 0; c < channels; ++c) {
	  pos1[0] += h0lambda * w0lambda * pos2[0];
	  pos1[w1p] += h0lambda * w1lambda * pos2[0];
	  pos1[h1p * Width1] += h1lambda * w0lambda * pos2[0];
	  pos1[h1p * Width1 + w1p] += h1lambda * w1lambda * pos2[0];
	  pos1 += Width1 * Height1;
	  pos2 += Width2 * Height2;
	}
      }
    }
  }
}

// Create Gaussian pyramid of an image. Assume output space is pre-allocated.
// IN : [channels height width]
template <typename Dtype, bool packed>
void caffe_cpu_pyramid2(const int channels,
    const Dtype *data, const int height, const int width,
    Dtype *data_pyr, const int levels) {
  CHECK(height > 0 && width > 0 && levels >= 0);
  int height1 = height, width1 = width;
  int height2 = height, width2 = width;
  const Dtype *data1 = data;
  Dtype *data2 = data_pyr;
  for (int l = 0; l < levels; ++l) {
    height2 /= 2;
    width2 /= 2;
    if (height2 == 0 || width2 == 0) {
      break;
    }
    for (int h2 = 0; h2 < height2; ++h2) {
      const int h1 = 2 * h2;
      for (int w2 = 0; w2 < width2; ++w2) {
	const int w1 = 2 * w2;
	if (packed) {
	  const Dtype* pos1 = &data1[channels * (h1 * width1 + w1)];
	  Dtype* pos2 = &data2[channels * (h2 * width2 + w2)];
	  for (int c = 0; c < channels; ++c) {
	    pos2[0] =  static_cast<Dtype>(.25) *
	      (pos1[0]                 + pos1[channels] + 
	       pos1[channels * width1] + pos1[channels * (width1 + 1)]);
	    pos1++;
	    pos2++;
	  }
	}
	else {
	  const Dtype* pos1 = &data1[h1 * width1 + w1];
	  Dtype* pos2 = &data2[h2 * width2 + w2];
	  for (int c = 0; c < channels; ++c) {
	    pos2[0] =  static_cast<Dtype>(.25) *
	      (pos1[0]      + pos1[1] + 
	       pos1[width1] + pos1[width1 + 1]);
	    pos1 += width1 * height1;
	    pos2 += width2 * height2;
	  }
	}
      }
    }
    data1 = data2;
    height1 = height2;
    width1 = width2;
    data2 += channels * height2 * width2;
  }
}

  /*
template <typename Dtype, bool packed>
void caffe_cpu_mosaic(const int channels,
    const Dtype *data1, const MosaicParameter mosaic_params1,
    const Dtype *data_pyr, const int levels,
          Dtype *data2, const MosaicParameter mosaic_params2) {
  const int num1 = mosaic_params1.rects_size();
  const int num2 = mosaic_params2.rects_size();
  CHECK(num1 == num2 || (num1 == 1 && num2 > 1) || (num2 == 1 && num1 > 1));
  const int num = std::max(num1, num2);
  for (int i = 0; i < num; ++i) {
    const Rect rect1 = mosaic_params1.rects((i < num1) ? i : 0);
    const Rect rect2 = mosaic_params2.rects((i < num2) ? i : 0);
    int level = log2(sqrt((float)rect1.height() * rect1.width() / rect2.height() / rect2.width()));
    level = std::max(0, std::min(levels, level));
    if (data_pyr == 0 || level == 0) {
      caffe_cpu_interp2<Dtype,packed>(channels,
	  data1, rect1.x(), rect1.y(), rect1.height(), rect1.width(), mosaic_params1.height(), mosaic_params1.width(),
	  data2, rect2.x(), rect2.y(), rect2.height(), rect2.width(), mosaic_params2.height(), mosaic_params2.width());
    }
    else {
      const Dtype *data_pyr_l = data_pyr;
      int factor = 2;
      for (int l = 1; l < level; ++l) {
	data_pyr_l += channels * (mosaic_params1.height() / factor) * (mosaic_params1.width() / factor);
	factor *= 2;
      }
      caffe_cpu_interp2<Dtype,packed>(channels,
	  data_pyr_l, rect1.x() / factor, rect1.y() / factor, rect1.height() / factor, rect1.width() / factor, mosaic_params1.height() / factor, mosaic_params1.width() / factor,
	  data2, rect2.x(), rect2.y(), rect2.height(), rect2.width(), mosaic_params2.height(), mosaic_params2.width());      
    }
  }
}

template <typename Dtype, bool packed>
void caffe_gpu_mosaic(const int channels,
    const Dtype *data1, const MosaicParameter mosaic_params1,
    const Dtype *data_pyr, const int levels,
          Dtype *data2, const MosaicParameter mosaic_params2) {
  const int num1 = mosaic_params1.rects_size();
  const int num2 = mosaic_params2.rects_size();
  CHECK(num1 == num2 || (num1 == 1 && num2 > 1) || (num2 == 1 && num1 > 1));
  const int num = std::max(num1, num2);
  for (int i = 0; i < num; ++i) {
    const Rect rect1 = mosaic_params1.rects((i < num1) ? i : 0);
    const Rect rect2 = mosaic_params2.rects((i < num2) ? i : 0);
    int level = log2(sqrt((float)rect1.height() * rect1.width() / rect2.height() / rect2.width()));
    level = std::max(0, std::min(levels, level));
    if (data_pyr == 0 || level == 0) {
      caffe_gpu_interp2<Dtype,packed>(channels,
	  data1, rect1.x(), rect1.y(), rect1.height(), rect1.width(), mosaic_params1.height(), mosaic_params1.width(),
	  data2, rect2.x(), rect2.y(), rect2.height(), rect2.width(), mosaic_params2.height(), mosaic_params2.width());
    }
    else {
      const Dtype *data_pyr_l = data_pyr;
      int factor = 2;
      for (int l = 1; l < level; ++l) {
	data_pyr_l += channels * (mosaic_params1.height() / factor) * (mosaic_params1.width() / factor);
	factor *= 2;
      }
      caffe_gpu_interp2<Dtype,packed>(channels,
	  data_pyr_l, rect1.x() / factor, rect1.y() / factor, rect1.height() / factor, rect1.width() / factor, mosaic_params1.height() / factor, mosaic_params1.width() / factor,
	  data2, rect2.x(), rect2.y(), rect2.height(), rect2.width(), mosaic_params2.height(), mosaic_params2.width());      
    }
  }
}

  */

// Explicit instances
template void caffe_cpu_interp2<float,false>(const int, const float *, const int, const int, const int, const int, const int, const int, float *, const int, const int, const int, const int, const int, const int);
template void caffe_cpu_interp2<float,true>(const int, const float *, const int, const int, const int, const int, const int, const int, float *, const int, const int, const int, const int, const int, const int);
template void caffe_cpu_interp2<double,false>(const int, const double *, const int, const int, const int, const int, const int, const int, double *, const int, const int, const int, const int, const int, const int);
template void caffe_cpu_interp2<double,true>(const int, const double *, const int, const int, const int, const int, const int, const int, double *, const int, const int, const int, const int, const int, const int);

template void caffe_cpu_interp2_backward<float,false>(const int, float *, const int, const int, const int, const int, const int, const int, const float *, const int, const int, const int, const int, const int, const int);
template void caffe_cpu_interp2_backward<double,false>(const int, double *, const int, const int, const int, const int, const int, const int, const double *, const int, const int, const int, const int, const int, const int);

template void caffe_cpu_pyramid2<float,false>(const int, const float *, const int, const int, float *, const int);
template void caffe_cpu_pyramid2<float,true>(const int, const float *, const int, const int, float *, const int);
template void caffe_cpu_pyramid2<double,false>(const int, const double *, const int, const int, double *, const int);
template void caffe_cpu_pyramid2<double,true>(const int, const double *, const int, const int, double *, const int);

  /*
template void caffe_cpu_mosaic<float,false>(const int, const float *, const MosaicParameter, const float *, const int, float *, const MosaicParameter);
template void caffe_cpu_mosaic<float,true>(const int, const float *, const MosaicParameter, const float *, const int, float *, const MosaicParameter);
template void caffe_cpu_mosaic<double,false>(const int, const double *, const MosaicParameter, const double *, const int, double *, const MosaicParameter);
template void caffe_cpu_mosaic<double,true>(const int, const double *, const MosaicParameter, const double *, const int, double *, const MosaicParameter);

template void caffe_gpu_mosaic<float,false>(const int, const float *, const MosaicParameter, const float *, const int, float *, const MosaicParameter);
template void caffe_gpu_mosaic<float,true>(const int, const float *, const MosaicParameter, const float *, const int, float *, const MosaicParameter);
template void caffe_gpu_mosaic<double,false>(const int, const double *, const MosaicParameter, const double *, const int, double *, const MosaicParameter);
template void caffe_gpu_mosaic<double,true>(const int, const double *, const MosaicParameter, const double *, const int, double *, const MosaicParameter);
  */

}  // namespace caffe
