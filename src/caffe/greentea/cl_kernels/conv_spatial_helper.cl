#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(copyImage, Dtype)
    (__global Dtype* image_data,
     int_tp image_offset,
     const int_tp channels, const int_tp height, const int_tp width,
     const int_tp adjustedHeight, const int_tp adjustedWidth,
     const int_tp pad_h, const int_tp pad_w,
     __global Dtype* output_image, const int_tp output_offset) {

  uint_tp sX = get_global_id(0);
  uint_tp sY = get_global_id(1);
  uint_tp sZ = get_global_id(2);

  int_tp in_y = sY - pad_h;
  int_tp in_x = sX - pad_w;

  if((in_y >= 0 && in_y < height && in_x >= 0 && in_x < width))
  output_image[output_offset + sZ*adjustedHeight*adjustedWidth + sY*adjustedWidth +sX] = image_data[image_offset + sZ*height*width + in_y*width + in_x];
  else
  output_image[output_offset + sZ*adjustedHeight*adjustedWidth + sY*adjustedWidth +sX] = 0;
}

__kernel void TEMPLATE(copyWeightsSwizzled, Dtype)
    (__global Dtype* weightIn,
     __global Dtype* weightOut,
     const int_tp kernel_w,
     const int_tp kernel_h,
     const int_tp channels,
     const int_tp outputs,
     const int_tp swizzleFactor) {

  uint_tp sX = get_global_id(0);

  //Original location

  //Output location
  int_tp outputSublayer = channels / swizzleFactor;
  int_tp outputSublayerIndex = channels % swizzleFactor;

  int_tp filter = sX / (kernel_w*kernel_h*channels);
  int_tp kernel_X = sX % kernel_w;
  int_tp kernel_Y = (sX / kernel_w) % kernel_h;
  int_tp kernel_C = (sX / (kernel_w * kernel_h)) % channels;

  int_tp FP = filter / swizzleFactor;
  int_tp F1 = filter % swizzleFactor;

  weightOut[FP*(kernel_w*kernel_h*channels*swizzleFactor) + kernel_C*(kernel_w*kernel_h*swizzleFactor) + kernel_Y*(kernel_w*swizzleFactor) + kernel_X*swizzleFactor + F1]
  = weightIn[filter*(kernel_w*kernel_h*channels) + kernel_C*(kernel_w*kernel_h) + kernel_Y*kernel_w + kernel_X];
}


