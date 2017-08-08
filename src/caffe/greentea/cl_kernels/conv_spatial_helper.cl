#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(copyImage, Dtype)
    (__global Dtype* image_data,
     int_tp image_offset,
     const int_tp channels, const int_tp height, const int_tp width,
     const int_tp adjustedHeight, const int_tp adjustedWidth,
     const int_tp pad_h, const int_tp pad_w,
     __global Dtype* output_image,
     const int_tp output_offset,
     const int_tp batch_size) {

  uint_tp sX = get_global_id(0);
  uint_tp sY = get_global_id(1);
  uint_tp sZ = get_global_id(2);

  int_tp in_y = sY - pad_h;
  int_tp in_x = sX - pad_w;

  int_tp batch_offset = 0;
  int_tp adjusted_batch_offset = 0;
  for(uint_tp batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    int_tp dst_offset = adjusted_batch_offset + output_offset + sZ*adjustedHeight*adjustedWidth + sY*adjustedWidth +sX;
    int_tp src_offset = batch_offset + image_offset + sZ*height*width + in_y*width + in_x;
    if((in_y >= 0 && in_y < height && in_x >= 0 && in_x < width))
      output_image[dst_offset] = image_data[src_offset];
    else
      output_image[dst_offset] = 0;
    batch_offset += height * width * channels;
    adjusted_batch_offset += adjustedHeight * adjustedWidth * channels;
  }
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

/*------------------------------------------------------------------------
  Below is winograd functions for weights transform.*/
 
#if TYPE != TYPE_DOUBLE
#if TYPE == TYPE_HALF
#define WRITE_IMAGE write_imageh
#else
#define WRITE_IMAGE write_imagef
#endif
#define KERNEL_SIZE 9

/* For the filter g located at FILTERS[k][c], computes the transformation
 * u = G * g * G^T. Then, catters each matrix u into the output U. 
 * G has dimensions (ALPHA,r)*/
__kernel void TEMPLATE(filter_transform_4x4_v0,Dtype)(
        __global Dtype *filters,
        __write_only image2d_t U,
        int_tp total_input_channels,
        int_tp total_output_channels)
{
  // size_t id = get_global_id(0);
  /* Filter transform matrix. 
    G = {1/4, 0.0, 0.0,
         -1.0/6.0, -1.0/6.0,-1.0/6.0,
         -1.0/6.0, 1.0/6.0, -1.0/6.0,
         1/24, 1/12, 1.0/6.0,
         1/24,-1/12, 1.0/6.0,
         0.0,  0.0,  1};
  */
  //const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int in_channel = get_global_id(0);
  int out_channel = get_global_id(1);

  Dtype g[18];
  __global Dtype* f = filters + in_channel * KERNEL_SIZE + out_channel * total_input_channels *KERNEL_SIZE;
  int2 coordU = (int2)(in_channel*9, out_channel);
  /* Filter transform matrix. */
  /*
    G = {1/4, 0.0, 0.0,
         -1.0/6.0, -1.0/6.0,-1.0/6.0,
         -1.0/6.0, 1.0/6.0, -1.0/6.0,
         1/24, 1/12, 1.0/6.0,
         1/24,-1/12, 1.0/6.0,
         0.0,  0.0,  1};

  */
  /*
   f= { f0, f1, f2,
        f3, f4, f5,
        f6, f7, f8 }
  */
  /*

  g = { g0, g1, g2,
        g3, g4, g5,
        g6, g7, g8,
        g9, g10,g11,
        g12,g13,g14,
        g15,g16,g17};
   */
   g[0] = 0.25f*f[0];
   g[1] = 0.25f*f[1];
   g[2] = 0.25f*f[2];

   g[3] = -1.0/6.0*f[0]-1.0/6.0*f[3]-1.0/6.0*f[6];
   g[4] = -1.0/6.0*f[1]-1.0/6.0*f[4]-1.0/6.0*f[7];
   g[5] = -1.0/6.0*f[2]-1.0/6.0*f[5]-1.0/6.0*f[8];

   g[6] = g[3] + f[3]/3.0;
   g[7] = g[4] + f[4]/3.0;
   g[8] = g[5] + f[5]/3.0;

   g[9] = f[0]/24.0 + f[3]/12.0 + f[6]/6.0;
   g[10]= f[1]/24.0 + f[4]/12.0 + f[7]/6.0;
   g[11]= f[2]/24.0 + f[5]/12.0 + f[8]/6.0;

   g[12]=g[9]-f[3]/6.0;
   g[13]=g[10]-f[4]/6.0;
   g[14]=g[11]-f[5]/6.0;

   g[15]=f[6];
   g[16]=f[7];
   g[17]=f[8];


  /*U[K][C][ALPHA][ALPHA]
  g = { g0, g1, g2,
        g3, g4, g5,
        g6, g7, g8,
        g9, g10,g11,
        g12,g13,g14,
        g15,g16,g17};
  

    G^T = {1/4, -1.0/6.0, -1.0/6.0, 1/24, 1/24, 0,
           0,   -1.0/6.0,  1.0/6.0, 1/12,-1/12, 0,
           0,   -1.0/6.0, -1.0/6.0, 1.0/6.0,  1/6,  1};

  */

  WRITE_IMAGE(U, coordU, (Dtype4)(g[0]*0.25, -1.0/6.0*g[0]-1.0/6.0*g[1]-1.0/6.0*g[2], -1.0/6.0*g[0]+1.0/6.0*g[1]-1.0/6.0*g[2], g[0]/24+g[1]/12+g[2]/6));
  coordU.x +=1;
  WRITE_IMAGE(U, coordU, (Dtype4)(g[0]/24-g[1]/12+g[2]/6, g[2], g[3]*0.25, -1.0/6.0*g[3]-1.0/6.0*g[4]-1.0/6.0*g[5]));  
  coordU.x +=1;
  WRITE_IMAGE(U, coordU, (Dtype4)(-1.0/6.0*g[3]+1.0/6.0*g[4]-1.0/6.0*g[5], g[3]/24+g[4]/12+g[5]/6, g[3]/24-g[4]/12+g[5]/6, g[5]));
  coordU.x +=1;    
  WRITE_IMAGE(U, coordU, (Dtype4)(g[6]*0.25f, -1.0/6.0*g[6]-1.0/6.0*g[7]-1.0/6.0*g[8], -1.0/6.0*g[6]+1.0/6.0*g[7]-1.0/6.0*g[8], g[6]/24+g[7]/12+g[8]/6)); 
  coordU.x +=1;  
  WRITE_IMAGE(U, coordU, (Dtype4)(g[6]/24-g[7]/12+g[8]/6, g[8], g[9]*0.25, -1.0/6.0*g[9]-1.0/6.0*g[10]-1.0/6.0*g[11]));    
  coordU.x +=1;  
  WRITE_IMAGE(U, coordU, (Dtype4)(-1.0/6.0*g[9]+1.0/6.0*g[10]-1.0/6.0*g[11], g[9]/24+g[10]/12+g[11]/6, g[9]/24-g[10]/12+g[11]/6, g[11]));
  coordU.x +=1;  
  WRITE_IMAGE(U, coordU, (Dtype4)(g[12]*0.25f, -1.0/6.0*g[12]-1.0/6.0*g[13]-1.0/6.0*g[14], -1.0/6.0*g[12]+1.0/6.0*g[13]-1.0/6.0*g[14], g[12]/24+g[13]/12+g[14]/6)); 
  coordU.x +=1;  
  WRITE_IMAGE(U, coordU, (Dtype4)(g[12]/24-g[13]/12+g[14]/6, g[14], g[15]*0.25, -1.0/6.0*g[15]-1.0/6.0*g[16]-1.0/6.0*g[17]));   
  coordU.x +=1;  
  WRITE_IMAGE(U, coordU, (Dtype4)(-1.0/6.0*g[15]+1.0/6.0*g[16]-1.0/6.0*g[17], g[15]/24+g[16]/12+g[17]/6, g[15]/24-g[16]/12+g[17]/6, g[17]));
}
/*The v1 version for input channel size > 256*/
__kernel void TEMPLATE(filter_transform_4x4_v1,Dtype)(
        __global Dtype *filters,
        __write_only image2d_t U,
        int_tp total_input_channels,
        int_tp total_output_channels)
{
  // size_t id = get_global_id(0);
  /* Filter transform matrix. 
    G = {1/4, 0.0, 0.0,
         -1.0/6.0, -1.0/6.0,-1.0/6.0,
         -1.0/6.0, 1.0/6.0, -1.0/6.0,
         1/24, 1/12, 1.0/6.0,
         1/24,-1/12, 1.0/6.0,
         0.0,  0.0,  1};
  */
  //const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
  int in_channel = get_global_id(0);
  int out_channel = get_global_id(1);
  int slice = total_input_channels/2;
  float g[18];
  __global float* f = filters + in_channel * KERNEL_SIZE + out_channel * total_input_channels *KERNEL_SIZE;
  int2 coordU = (int2)((in_channel%slice)*9, (in_channel/slice)*total_output_channels+out_channel);
  /* Filter transform matrix. */
  /*
    G = {1/4, 0.0, 0.0,
         -1.0/6.0, -1.0/6.0,-1.0/6.0,
         -1.0/6.0, 1.0/6.0, -1.0/6.0,
         1/24, 1/12, 1.0/6.0,
         1/24,-1/12, 1.0/6.0,
         0.0,  0.0,  1};

  */
  /*
   f= { f0, f1, f2,
        f3, f4, f5,
        f6, f7, f8 }
  */
  /*

  g = { g0, g1, g2,
        g3, g4, g5,
        g6, g7, g8,
        g9, g10,g11,
        g12,g13,g14,
        g15,g16,g17};
   */
   g[0] = 0.25f*f[0];
   g[1] = 0.25f*f[1];
   g[2] = 0.25f*f[2];

   g[3] = -1.0/6.0*f[0]-1.0/6.0*f[3]-1.0/6.0*f[6];
   g[4] = -1.0/6.0*f[1]-1.0/6.0*f[4]-1.0/6.0*f[7];
   g[5] = -1.0/6.0*f[2]-1.0/6.0*f[5]-1.0/6.0*f[8];

   g[6] = g[3] + f[3]/3.0;
   g[7] = g[4] + f[4]/3.0;
   g[8] = g[5] + f[5]/3.0;

   g[9] = f[0]/24.0 + f[3]/12.0 + f[6]/6.0;
   g[10]= f[1]/24.0 + f[4]/12.0 + f[7]/6.0;
   g[11]= f[2]/24.0 + f[5]/12.0 + f[8]/6.0;

   g[12]=g[9]-f[3]/6.0;
   g[13]=g[10]-f[4]/6.0;
   g[14]=g[11]-f[5]/6.0;

   g[15]=f[6];
   g[16]=f[7];
   g[17]=f[8];


  /*U[K][C][ALPHA][ALPHA]
  g = { g0, g1, g2,
        g3, g4, g5,
        g6, g7, g8,
        g9, g10,g11,
        g12,g13,g14,
        g15,g16,g17};
  

    G^T = {1/4, -1.0/6.0, -1.0/6.0, 1/24, 1/24, 0,
           0,   -1.0/6.0,  1.0/6.0, 1/12,-1/12, 0,
           0,   -1.0/6.0, -1.0/6.0, 1.0/6.0,  1/6,  1};

  */

  WRITE_IMAGE(U, coordU, (Dtype4)(g[0]*0.25, -1./6.*g[0]-1.0/6.0*g[1]-1.0/6.0*g[2], -1.0/6.0*g[0]+1.0/6.0*g[1]-1.0/6.0*g[2], g[0]/24.0+g[1]/12.0+g[2]/6.0));
  coordU.x +=1;
  WRITE_IMAGE(U, coordU, (Dtype4)(g[0]/24.0-g[1]/12.0+g[2]/6.0, g[2], g[3]*0.25, -1.0/6.0*g[3]-1.0/6.0*g[4]-1.0/6.0*g[5]));  
  coordU.x +=1;
  WRITE_IMAGE(U, coordU, (Dtype4)(-1.0/6.0*g[3]+1.0/6.0*g[4]-1.0/6.0*g[5], g[3]/24+g[4]/12+g[5]/6, g[3]/24-g[4]/12+g[5]/6.0, g[5]));
  coordU.x +=1;    
  WRITE_IMAGE(U, coordU, (Dtype4)(g[6]*0.25f, -1.0/6.0*g[6]-1.0/6.0*g[7]-1.0/6.0*g[8], -1.0/6.0*g[6]+1.0/6.0*g[7]-1.0/6.0*g[8], g[6]/24+g[7]/12+g[8]/6)); 
  coordU.x +=1;  
  WRITE_IMAGE(U, coordU, (Dtype4)(g[6]/24-g[7]/12+g[8]/6, g[8], g[9]*0.25f, -1.0/6.0*g[9]-1.0/6.0*g[10]-1.0/6.0*g[11]));    
  coordU.x +=1;  
  WRITE_IMAGE(U, coordU, (Dtype4)(-1.0/6.0*g[9]+1.0/6.0*g[10]-1.0/6.0*g[11], g[9]/24+g[10]/12+g[11]/6, g[9]/24-g[10]/12+g[11]/6, g[11]));
  coordU.x +=1;  
  WRITE_IMAGE(U, coordU, (Dtype4)(g[12]*0.25f, -1.0/6.0*g[12]-1.0/6.0*g[13]-1.0/6.0*g[14], -1.0/6.0*g[12]+1.0/6.0*g[13]-1.0/6.0*g[14], g[12]/24+g[13]/12+g[14]/6)); 
  coordU.x +=1;  
  WRITE_IMAGE(U, coordU, (Dtype4)(g[12]/24-g[13]/12+g[14]/6, g[14], g[15]*0.25f, -1.0/6.0*g[15]-1.0/6.0*g[16]-1.0/6.0*g[17]));   
  coordU.x +=1;  
  WRITE_IMAGE(U, coordU, (Dtype4)(-1.0/6.0*g[15]+1.0/6.0*g[16]-1.0/6.0*g[17], g[15]/24+g[16]/12+g[17]/6, g[15]/24-g[16]/12+g[17]/6, g[17]));
}
#undef WRITE_IMAGE
#undef KERNEL_SIZE
#endif


