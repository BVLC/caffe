#ifndef __OPENCL_VERSION__
#include "header.cl"
#endif

__kernel void TEMPLATE(conv_layer_spatial_phony,Dtype)(Dtype arg) {
  Dtype out = arg;
}

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)
#define LOOP0(VAR, STMT)
#define LOOP1(VAR, STMT) (STMT); (VAR)++;
#define LOOP2(VAR, STMT) LOOP1(VAR, STMT); (STMT); (VAR)++;
#define LOOP3(VAR, STMT) LOOP2(VAR, STMT); (STMT); (VAR)++;
#define LOOP4(VAR, STMT) LOOP3(VAR, STMT); (STMT); (VAR)++;
#define LOOP5(VAR, STMT) LOOP4(VAR, STMT); (STMT); (VAR)++;
#define LOOP6(VAR, STMT) LOOP5(VAR, STMT); (STMT); (VAR)++;
#define LOOP7(VAR, STMT) LOOP6(VAR, STMT); (STMT); (VAR)++;
#define LOOP8(VAR, STMT) LOOP7(VAR, STMT); (STMT); (VAR)++;
#define LOOP9(VAR, STMT) LOOP8(VAR, STMT); (STMT); (VAR)++;
#define LOOP10(VAR, STMT) LOOP9(VAR, STMT); (STMT); (VAR)++;
#define LOOP11(VAR, STMT) LOOP10(VAR, STMT); (STMT); (VAR)++;
#define LOOP12(VAR, STMT) LOOP11(VAR, STMT); (STMT); (VAR)++;
#define LOOP13(VAR, STMT) LOOP12(VAR, STMT); (STMT); (VAR)++;
#define LOOP14(VAR, STMT) LOOP13(VAR, STMT); (STMT); (VAR)++;
#define LOOP15(VAR, STMT) LOOP14(VAR, STMT); (STMT); (VAR)++;
#define LOOP16(VAR, STMT) LOOP15(VAR, STMT); (STMT); (VAR)++;
#define LOOP(N, VAR, STMT) CAT(LOOP, N)((VAR), (STMT))

#ifdef MULTI
__kernel void CFMulti(__global Dtype* image_data, int_tp image_offset,
    __global Dtype* kernel_data, int_tp kernel_offset,
    __global Dtype* bias,const int_tp bias_offset,
    __global Dtype* convolved_image,const int_tp convolved_image_offset,
    const ushort WIDTH,
    const ushort HEIGHT,
    const ushort OUTPUT_W,
    const ushort OUTPUT_H) {

  const int_tp outputX = get_global_id(0);
  const int_tp outputY = get_global_id(1);
  const int_tp kernelNum = get_global_id(2)*ZPAR;
  if(outputX < OUTPUT_W && outputY < OUTPUT_H)
  {
    Dtype sum[ZPAR];
    Dtype4 vectorSum[ZPAR];
    for(int_tp kern =0; kern < ZPAR; kern++)
    {
      sum[kern] = 0.0f;
      vectorSum[kern] = (0.0f,0.0f,0.0f,0.0f);
    }

    const int_tp currentKernelOffset = kernel_offset + kernelNum*KERNEL_H*KERNEL_W*CHANNELS;
    const int_tp biasIndex=bias_offset + kernelNum;
    const int_tp local_image_offset = outputY*STRIDE_H*WIDTH + outputX*STRIDE_W;
    const int_tp imageSize = WIDTH*HEIGHT;
    const int_tp float4Reads = KERNEL_W / 4;
    const int_tp floatReads = KERNEL_W % 4;
    Dtype4 imageCache;

    __global Dtype* image_dataPtrFloat = (image_data + (image_offset + local_image_offset));
    __global Dtype* kernel_dataPtrFloat = (kernel_data + (currentKernelOffset));

    for(int_tp c = 0; c < CHANNELS; c++)
    {
      for(int_tp y = 0; y < KERNEL_H; y++)
      {

        for(int_tp x=0; x< float4Reads; x++)
        {
          imageCache = ((__global Dtype4*)image_dataPtrFloat)[x];
          for(int_tp kern =0; kern < ZPAR; kern++)
          {
            vectorSum[kern] += imageCache*((__global Dtype4*)&(kernel_dataPtrFloat[kern*KERNEL_H*KERNEL_W*CHANNELS]))[x];
          }
        }

        if(floatReads == 1)
        {
          imageCache = ((__global Dtype4*)image_dataPtrFloat)[float4Reads];
          for(int_tp kern =0; kern < ZPAR; kern++)
          vectorSum[kern].s0 += ( imageCache * ( (__global Dtype4*) &(kernel_dataPtrFloat[kern*KERNEL_H*KERNEL_W*CHANNELS]) )[float4Reads] ).s0;
        }
        else if(floatReads == 2)
        {
          imageCache = ((__global Dtype4*)image_dataPtrFloat)[float4Reads];
          for(int_tp kern =0; kern < ZPAR; kern++)
          vectorSum[kern].s01 += (imageCache*((__global Dtype4*)&(kernel_dataPtrFloat[kern*KERNEL_H*KERNEL_W*CHANNELS]))[float4Reads]).s01;
        }
        else if(floatReads == 3)
        {
          imageCache = ((__global Dtype4*)image_dataPtrFloat)[float4Reads];
          for(int_tp kern =0; kern < ZPAR; kern++)
          vectorSum[kern].s012 += (imageCache*((__global Dtype4*)&(kernel_dataPtrFloat[kern*KERNEL_H*KERNEL_W*CHANNELS]))[float4Reads]).s012;
        }

        image_dataPtrFloat += WIDTH;
        kernel_dataPtrFloat += KERNEL_W;
      }
      image_dataPtrFloat += imageSize - WIDTH*KERNEL_H;
    }
    for(int_tp kern =0; kern < ZPAR; kern++)
    sum[kern] = vectorSum[kern].x + vectorSum[kern].y + vectorSum[kern].z + vectorSum[kern].w;

    if(APPLY_BIAS == 1)
    {
      for(int_tp kern = 0; kern < ZPAR; kern++)
      if(kernelNum+kern < OUTPUT_Z)
      convolved_image[convolved_image_offset + (kernelNum+kern)*OUTPUT_H*OUTPUT_W + outputY*OUTPUT_W + outputX] =
      sum[kern] + bias[biasIndex +kern];
    }
    else
    for(int_tp kern = 0; kern < ZPAR; kern++)
    if(kernelNum+kern < OUTPUT_Z)
    convolved_image[convolved_image_offset + (kernelNum+kern)*OUTPUT_H*OUTPUT_W + outputY*OUTPUT_W + outputX] = sum[kern];
  }
}

#endif


#ifdef MULTI_11
__kernel void CFMulti_11_11_4(__global Dtype* image_data, int_tp image_offset,
    __global Dtype* kernel_data, int_tp kernel_offset,
    __global Dtype* bias,const int_tp bias_offset,
    __global Dtype* convolved_image,const int_tp convolved_image_offset,
    const ushort WIDTH,
    const ushort HEIGHT,
    const ushort OUTPUT_W,
    const ushort OUTPUT_H) {

  int_tp outputX = get_global_id(0)*XPAR;
  int_tp outputY = get_global_id(1)*YPAR;
  int_tp kernelNum = get_global_id(2)*ZPAR;
  if(outputX < OUTPUT_W && outputY < OUTPUT_H)
  {
    Dtype sum[XPAR*YPAR*ZPAR];
    for(int_tp kern =0; kern < XPAR*YPAR*ZPAR; kern++)
    {
      sum[kern] = 0.0f;
    }

    int_tp currentKernelOffset = kernel_offset + kernelNum*KERNELSIZE*CHANNELS;
    int_tp biasIndex=bias_offset + kernelNum;
    int_tp local_image_offset = outputY*STRIDE_H*WIDTH + outputX*STRIDE_W;
    int_tp imageSize = WIDTH*HEIGHT;
    int_tp index;

    __global Dtype* image_dataPtrFloat = (image_data + (image_offset + local_image_offset));
    __global Dtype* kernel_dataPtrFloat = (kernel_data + (currentKernelOffset));

    Dtype16 imageCache;
    Dtype8 imageCacheR;
    Dtype8 kernelCache;
    Dtype4 kernelCacheR;

    for(int_tp c = 0; c < CHANNELS; c++)
    {
      for(int_tp y = 0; y < 11; y++)
      {
        imageCache = ((__global Dtype16*)image_dataPtrFloat)[0];
        imageCacheR =((__global Dtype8*)image_dataPtrFloat)[2];

        for(int_tp kern =0; kern < ZPAR; kern++)
        {
          kernelCache = ((__global Dtype8*)&(kernel_dataPtrFloat[kern*KERNELSIZE*CHANNELS]))[0];
          kernelCacheR = ((__global Dtype4*)&(kernel_dataPtrFloat[kern*KERNELSIZE*CHANNELS]))[2];

          index = kern*XPAR;
          sum[index + 0] += dot(imageCache.S0123,kernelCache.S0123);
          sum[index + 1] += dot(imageCache.S4567,kernelCache.S0123);
          sum[index + 2] += dot(imageCache.S89AB,kernelCache.S0123);
          sum[index + 3] += dot(imageCache.SCDEF,kernelCache.S0123);

          sum[index + 0] += dot(imageCache.S4567,kernelCache.S4567);
          sum[index + 1] += dot(imageCache.S89AB,kernelCache.S4567);
          sum[index + 2] += dot(imageCache.SCDEF,kernelCache.S4567);
          sum[index + 3] += dot(imageCacheR.S0123,kernelCache.S4567);

          sum[index + 0] += dot(imageCache.S89A,kernelCacheR.S012);
          sum[index + 1] += dot(imageCache.SCDE,kernelCacheR.S012);
          sum[index + 2] += dot(imageCacheR.S012,kernelCacheR.S012);
          sum[index + 3] += dot(imageCacheR.S456,kernelCacheR.S012);
        }

        image_dataPtrFloat += WIDTH;
        kernel_dataPtrFloat += KERNEL_W;
      }
      image_dataPtrFloat += imageSize - WIDTH*KERNEL_H;
    }

    if(APPLY_BIAS == 1)
    {
      for(int_tp kern = 0; kern < ZPAR; kern++)
      {
        for(int_tp wi =0; wi < XPAR; wi++)
        if(kernelNum+kern < OUTPUT_Z && outputX + wi < OUTPUT_W)
        convolved_image[convolved_image_offset + (kernelNum+kern)*OUTPUT_H*OUTPUT_W + outputY*OUTPUT_W + outputX + wi] =
        sum[kern*XPAR + wi] + bias[biasIndex +kern];
      }
    }
    else
    for(int_tp kern = 0; kern < ZPAR; kern++)
    for(int_tp wi =0; wi < XPAR; wi++)
    if(kernelNum+kern < OUTPUT_Z && outputX + wi < OUTPUT_W)
    convolved_image[convolved_image_offset + (kernelNum+kern)*OUTPUT_H*OUTPUT_W + outputY*OUTPUT_W + outputX + wi] = sum[kern*XPAR + wi];
  }
}

#endif

#ifdef MULTI_GEN
__kernel void CFMulti_6(__global const Dtype* restrict image_data, const int_tp image_offset,
    __global const Dtype* restrict kernel_data, const int_tp kernel_offset,
    __global const Dtype* restrict bias,const int_tp bias_offset,
    __global Dtype* restrict convolved_image,const int_tp convolved_image_offset,
    const ushort WIDTH,
    const ushort HEIGHT,
    const ushort OUTPUT_W,
    const ushort OUTPUT_H) {

  const int_tp outputX = get_global_id(0)*XPAR;
  const int_tp outputY = get_global_id(1)*YPAR;
  const int_tp kernelNum = get_global_id(2)*ZPAR;

  if(outputX < OUTPUT_W && outputY < OUTPUT_H)
  {
    Dtype sum[XPAR*YPAR*ZPAR];
    for(uint_tp kern = 0; kern < XPAR*YPAR*ZPAR; kern++)
    sum[kern] = 0.0f;

    const int_tp currentKernelOffset = kernel_offset + kernelNum*KERNELSIZE*CHANNELS;
    const int_tp biasIndex=bias_offset + kernelNum;
    const int_tp local_image_offset = outputY*STRIDE_H*WIDTH + outputX*STRIDE_W;
    const int_tp imageSize = WIDTH*HEIGHT;
    int_tp index;

    __global const Dtype* image_dataPtrFloat[2];
    image_dataPtrFloat[0] = (image_data + (image_offset + local_image_offset));
    image_dataPtrFloat[1] = image_dataPtrFloat[0];
    __global const Dtype* kernel_dataPtrFloat = (kernel_data + (currentKernelOffset));

    DTImage imageCache[YPAR];
    DTKernel kernelCache;
    Dtype4 temp;

    for(uint_tp c = 0; c < CHANNELS; c++)
    {
      imageCache[0] = ((__global DTImage*)image_dataPtrFloat[1])[0];
      for(uint_tp preload = 1; preload < YPAR; preload++)
      {
        image_dataPtrFloat[1] += WIDTH;
        imageCache[preload] = ((__global DTImage*)image_dataPtrFloat[1])[0];
      }

      int_tp y =0;
      LOOP(KERNEL_H, y,
          {
            int_tp kern=0;
            LOOP(ZPAR, kern,
                {
                  kernelCache = ((__global DTKernel*)&(kernel_dataPtrFloat[kern*KERNELSIZE*CHANNELS]))[0];
                  index = kern*XPAR*YPAR;

                  for(uint_tp y_par = 0; y_par < YPAR; y_par++)
                  {
                    temp = floatDotV4(imageCache[y_par],kernelCache);
                    sum[index + y_par*XPAR + 0] += temp.s0;
                    sum[index + y_par*XPAR + 1] += temp.s1;
                    sum[index + y_par*XPAR + 2] += temp.s2;
                    sum[index + y_par*XPAR + 3] += temp.s3;
                  }
                });

            kernel_dataPtrFloat += KERNEL_W;

            for(uint_tp rotateData = 0; rotateData < YPAR - 1; rotateData++)
            imageCache[rotateData] = imageCache[rotateData + 1];

            image_dataPtrFloat[1] += WIDTH;
            imageCache[YPAR - 1] = ((__global DTImage*)image_dataPtrFloat[1])[0];
          });

      image_dataPtrFloat[0] += imageSize;
      image_dataPtrFloat[1] = image_dataPtrFloat[0];
    }

    if(APPLY_BIAS == 1)
    {
      for(uint_tp kern = 0; kern < ZPAR; kern++)
      {
        for(uint_tp hi =0; hi < YPAR; hi++)
        for(uint_tp wi =0; wi < XPAR; wi++)
        if(kernelNum+kern < OUTPUT_Z && outputX + wi < OUTPUT_W && outputY + hi < OUTPUT_H)
        convolved_image[convolved_image_offset + (kernelNum+kern)*OUTPUT_H*OUTPUT_W + (outputY +hi)*OUTPUT_W + outputX + wi] =
        sum[kern*XPAR*YPAR + XPAR*hi + wi] + bias[biasIndex +kern];
      }
    }
    else
    for(uint_tp kern = 0; kern < ZPAR; kern++)
    for(uint_tp hi =0; hi < YPAR; hi++)
    for(uint_tp wi =0; wi < XPAR; wi++)
    if(kernelNum+kern < OUTPUT_Z && outputX + wi < OUTPUT_W && outputY + hi < OUTPUT_H)
    convolved_image[convolved_image_offset + (kernelNum+kern)*OUTPUT_H*OUTPUT_W + (outputY + hi)*OUTPUT_W + outputX + wi] = sum[kern*XPAR*YPAR +XPAR*hi +wi];
  }
}
#endif

//Begin IDLF kernels below here
#ifdef IDLF

#define activation_function(x) (x)

#if 0
#define _IW INPUT_WIDTH
#define _IH INPUT_HEIGHT
#define _OW OUTPUT_WIDTH
#define _OH OUTPUT_HEIGHT
#endif

#define _ID INPUT_DEPTH
#define _OD NUM_FILTERS

#define FILTER_DEPTH INPUT_DEPTH
#define NUM_INPUT INPUT_DEPTH
#define NUM_OUTPUT NUM_FILTERS

#define KERNEL FILTER_WIDTH
// convolution stride, same for x and y
#define K_STRIDE STRIDEX

#ifndef IWPAD
#define IWPAD 0
#endif

#ifndef IHPAD
#define IHPAD 0
#endif

#define OUT_BLOCK_SIZE (OUT_BLOCK_WIDTH*OUT_BLOCK_HEIGHT)

#ifndef MASTER_OUT_BLOCK_WIDTH
#define MASTER_OUT_BLOCK_WIDTH OUT_BLOCK_WIDTH
#endif
#ifndef MASTER_OUT_BLOCK_HEIGHT
#define MASTER_OUT_BLOCK_HEIGHT OUT_BLOCK_HEIGHT
#endif

// Each work-item computes a 4x6 region of one output map.
// Each work-group (which will be mapped to 1 SIMD16 EU thread) will compute 16 different feature maps, but each feature map is for the same 4x6 region of the imput image.
// NDRange:  (_OW+pad)/ OUT_BLOCK_WIDTH, (_OH+pad)/OUT_BLOCK_HEIGHT, _OD/OUT_BLOCK_DEPTH

//#define SIMD_SIZE 16
// NOTE: this reqd_work_group_size does not guarantee that SIMD16 mode will be used, the compiler could choose to use two SIMD8 threads, and if that happens the code will break.
#ifdef SIMD16


#define TILE_X ((OUT_BLOCK_WIDTH - 1) * STRIDEX + KERNEL)
#define TILE_Y ((OUT_BLOCK_HEIGHT - 1) * STRIDEY + KERNEL)

#if (TILE_X % 4) != 0
__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))
kernel void
convolve_simd16(  // __global float *inputs, __global float* weights, __global float* outputs
    __global float* inputs_base,
    filter_qualifier float* weights_base,
    __global float* biases_base,
    __global float* outputs_base,
    const ushort _IW,
    const ushort _IH,
    const ushort _OW,
    const ushort _OH)
{
  __global float* outputs = outputs_base;
  __global float* inputs = inputs_base;
  filter_qualifier float* weights = weights_base;
  __global float* biases = biases_base;

  uint_tp oc = get_global_id(0) * MASTER_OUT_BLOCK_WIDTH;  // oc = Output Column
  uint_tp or = get_global_id(1) * MASTER_OUT_BLOCK_HEIGHT;// or = Output Row
  uint_tp fm = get_global_id(2);// fm = Feature Map = od = Output Depth
  uint_tp fmg = get_group_id(2);
  uint_tp lid = get_local_id(2);

  float in[IN_BUFFER_SIZE];// load 11x16 block of input data, really only need 11x15 for 4x6 outputs, but keep it simple.
  //float out[24]; // 4x6 block of outputs that is SIMD_SIZE deep (along the Feature Map dimension).
  float out[OUT_BLOCK_SIZE];

  uint_tp in_addr;

  // find weights adress of given neuron (lid is index)
  uint_tp weight_addr = (fmg % (_OD/SIMD_SIZE)) * INPUT_DEPTH * KERNEL * KERNEL * SIMD_SIZE + lid;

  for(int_tp i=0;i<OUT_BLOCK_SIZE;i++) {
    out[i]=0.0f;
  }

  uint_tp num_in_batch = fm / _OD;

  uint_tp input_batch_offset = num_in_batch * (_IH + IHPAD) * (_IW + IWPAD) * TOTAL_INPUT_DEPTH_SIZE;
  for(int_tp kd = 0; kd < _ID; kd++)
  {
    in_addr = input_batch_offset + (kd + INPUT_START_Z) * (_IH + IHPAD) * (_IW + IWPAD) + (or*K_STRIDE + INPUT_START_Y) * (_IW + IWPAD) + (oc*K_STRIDE + INPUT_START_X) + lid;

    // read 11x16 input block into registers.
    for(uint_tp reg = 0; reg < IN_BUFFER_SIZE; reg++) {
      in[reg] = inputs[in_addr];    // read 16 elements
      in_addr += (_IW + IWPAD);// move to next row down
    }

// PREF could be 4 or 8, could not be other values.
#define WEIGHT_PREF 8
    union {
      float w[WEIGHT_PREF];
      uint8 ui8;
    } weight_buf;
    int_tp w_idx=0;

    weight_buf.ui8 = intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);
    uint_tp orig_weight_addr = weight_addr;
    weight_addr += SIMD_SIZE * WEIGHT_PREF;

    int_tp kr = 0;  // kr = Kernel Row
    LOOP(KERNEL, kr,// LOOP is a macro that unrolls the loop.
        {
          int_tp kc = 0;  // kc = Kernel Column
          LOOP(KERNEL, kc,
              {
                for(int_tp br=0; br < OUT_BLOCK_HEIGHT; br++) {
                  for(int_tp bc=0; bc < OUT_BLOCK_WIDTH; bc++) {
                    float input = intel_sub_group_shuffle( in[br * K_STRIDE + kr], bc * K_STRIDE + kc);
                    out[br * OUT_BLOCK_WIDTH + bc] = mad(weight_buf.w[w_idx % WEIGHT_PREF], input, out[br * OUT_BLOCK_WIDTH + bc]);
                  }
                }
                // We assume KERNEL_W is equal to KERNEL_H here.
                if ((w_idx + 1) % WEIGHT_PREF == 0
                    #if KERNEL*KERNEL % 8 != 0
                    && ((w_idx + 1) <= (KERNEL * KERNEL - WEIGHT_PREF))
                    #endif
                    ) {
                  weight_buf.ui8 = intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);
                  weight_addr += SIMD_SIZE * WEIGHT_PREF;  // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
                }
              #if KERNEL*KERNEL % 8 == 0
                // need to do nothing
              #else
                else if ((w_idx + 1) %  WEIGHT_PREF == 0 && ((w_idx + 1) > (KERNEL * KERNEL - WEIGHT_PREF)))
                #if KERNEL*KERNEL % 8 == 1
                  weight_buf.w[0] = weights[weight_addr];
                #elif KERNEL*KERNEL % 4 == 0
                  weight_buf.ui8.s0123 = intel_sub_group_block_read4((__global uint *)&weights[weight_addr]);
                #else
                // should never be here if kernel_w equal to kernel_h. just in case.
                #error unsupported kernel size.
                #endif
              #endif
                ++w_idx;
              });
        });
    weight_addr = orig_weight_addr + KERNEL * KERNEL * SIMD_SIZE;

  }

#ifdef IMAGE_AS_OUTPUT
  // TODO: no ULT for that one yet!
  uint_tp out_addr = ( num_in_batch * TOTAL_OUTPUT_DEPTH + (fm % _OD)) * (_OW + OWPAD) * (_OH + OHPAD);// out_addr indexes into start of 16 feature maps.
#else
  // we need this address calculation for outputs because we support views and batching
  uint_tp out_addr = OUT_BUFF_OFFSET + ( num_in_batch * TOTAL_OUTPUT_DEPTH + (fm % _OD) ) * (_OW + OWPAD) * (_OH + OHPAD);
#endif

  out_addr += or * (_OW + OWPAD) + oc;  // offset for the 4x3 block that this workitem is working on;

  // we need this address calculation for biases because we support views and batching
  float bias = biases[(fm) % _OD ];
#ifndef WRITE_PADDED_VALUES
  if(get_global_id(0) != (get_global_size(0)-1) &&
      get_global_id(1) != (get_global_size(1)-1) )
  {
#endif
    for(uint_tp r = 0; r < OUT_BLOCK_HEIGHT; r++) {
      for(uint_tp c = 0; c < OUT_BLOCK_WIDTH; c++) {
        // this does a scattered write to 16 different feature maps, so that data within one map is contiguous, thus ready for input to next layer.
#ifdef IMAGE_AS_OUTPUT
        write_imagef(outputs,(int2)(out_addr + r * (_OW + OWPAD) + c,num_in_batch),activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]));
#else
        outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
#endif
      }
    }
#ifndef WRITE_PADDED_VALUES
  } else if ( get_global_id(1) != (get_global_size(1)-1) )
  {
    for(uint_tp r = 0; r < OUT_BLOCK_HEIGHT; r++) {
      for(uint_tp c = 0; c < LAST_BLOCK_WIDTH; c++) {
#ifdef IMAGE_AS_OUTPUT
        write_imagef(outputs,(int2)(out_addr + r * (_OW + OWPAD) + c,num_in_batch),activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]));
#else
        outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
#endif
      }
    }
  }
  else if ( get_global_id(0) != (get_global_size(0)-1) )
  {
    for(uint_tp r = 0; r < LAST_BLOCK_HEIGHT; r++) {
      for(uint_tp c = 0; c < OUT_BLOCK_WIDTH; c++) {
#ifdef IMAGE_AS_OUTPUT
        write_imagef(outputs,(int2)(out_addr + r * (_OW + OWPAD) + c,num_in_batch),activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]));
#else
        outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
#endif
      }
    }
  }
  else
  {
    for(uint_tp r = 0; r < LAST_BLOCK_HEIGHT; r++) {
      for(uint_tp c = 0; c < LAST_BLOCK_WIDTH; c++) {
#ifdef IMAGE_AS_OUTPUT
        write_imagef(outputs,(int2)(c,r*(_OW + OWPAD)),activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]));
#else
        outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
#endif
      }
    }
  }
#endif //#ifndef WRITE_PADDED_VALUES
}
#endif

#if TILE_X % 4 == 0
#define TILE_Y_STRIDE (64 / TILE_X)
#define INVEC_NUM ((TILE_Y + TILE_Y_STRIDE - 1) / TILE_Y_STRIDE)
__attribute__((reqd_work_group_size(1, 1, SIMD_SIZE)))
kernel void
convolve_simd16(  // __global float *inputs, __global float* weights, __global float* outputs
    __global float* inputs_base,
    filter_qualifier float* weights_base,
    __global float* biases_base,
    __global float* outputs_base,
    const ushort _IW,
    const ushort _IH,
    const ushort _OW,
    const ushort _OH)
{
  __global float* outputs = outputs_base;
  __global float* inputs = inputs_base;
  filter_qualifier float* weights = weights_base;
  __global float* biases = biases_base;

  uint_tp oc = get_global_id(0) * MASTER_OUT_BLOCK_WIDTH;  // oc = Output Column
  uint_tp or = get_global_id(1) * MASTER_OUT_BLOCK_HEIGHT;// or = Output Row
  uint_tp fm = get_global_id(2);// fm = Feature Map = od = Output Depth
  uint_tp fmg = get_group_id(2);
  uint_tp lid = get_local_id(2);

  float out[OUT_BLOCK_SIZE];

  uint_tp in_addr;

  // find weights adress of given neuron (lid is index)
  uint_tp weight_addr = (fmg % (_OD/SIMD_SIZE)) * INPUT_DEPTH * KERNEL * KERNEL * SIMD_SIZE + lid;

  for(int_tp i=0;i<OUT_BLOCK_SIZE;i++) {
    out[i]=0.0f;
  }

  uint_tp num_in_batch = ( fm ) / _OD;

  uint_tp input_batch_offset = num_in_batch * (_IH + IHPAD) * (_IW + IWPAD) * TOTAL_INPUT_DEPTH_SIZE;

  in_addr = input_batch_offset + INPUT_START_Z * (_IH + IHPAD) * (_IW + IWPAD) + (or*STRIDEY + INPUT_START_Y) * (_IW + IWPAD) + (oc*STRIDEX + INPUT_START_X)
            + ( lid / ( TILE_X / 4 ) ) * (_IW + IWPAD) * STRIDEY             // y tile offset
            + ( lid % ( TILE_X / 4 ) ) * 4 * STRIDEX;                        // x tile offset

  for(int_tp kd = 0; kd < _ID; kd++)
  {
    union {
      float4 in_vec[INVEC_NUM];
      float in_array[INVEC_NUM * 4];
    } in_buf;
    uint_tp in_offset = in_addr;
    int_tp reg = 0;
#if INVEC_NUM == 1
    LOOP(1, reg,
#elif INVEC_NUM == 2
    LOOP(2, reg,
#elif INVEC_NUM == 3
    LOOP(3, reg,
#elif INVEC_NUM == 4
    LOOP(4, reg,
#else
    #error too large invec_num.
#endif
      {
        in_buf.in_vec[reg] = *(global float4*)(inputs + in_offset);    // read 16 elements
        in_offset += (_IW + IWPAD) * TILE_Y_STRIDE;
      });
    in_addr += (_IH + IHPAD) * (_IW + IWPAD);

// PREF could be 4 or 8, could not be other values.
#define WEIGHT_PREF 8
    union {
      float w[WEIGHT_PREF];
      uint8 ui8;
    } weight_buf;
    int_tp w_idx=0;

    weight_buf.ui8 = intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);
    uint_tp orig_weight_addr = weight_addr;
    weight_addr += SIMD_SIZE * WEIGHT_PREF;

#define BLOCK_IN(n) sub_group_broadcast( in_buf.in_array[((n)%4) + ((n) / (TILE_Y_STRIDE * TILE_X)) * 4], (((n) % (TILE_Y_STRIDE * TILE_X))/4))

    int_tp kr = 0;  // kr = Kernel Row
    LOOP(KERNEL, kr,// LOOP is a macro that unrolls the loop.
        {
          int_tp kc = 0;  // kc = Kernel Column
          LOOP(KERNEL, kc,
              {
                for(int_tp br=0; br < OUT_BLOCK_HEIGHT; br++) {
                  for(int_tp bc=0; bc < OUT_BLOCK_WIDTH; bc++) {
                    float input = BLOCK_IN((br * STRIDEY + kr) * TILE_X + bc * STRIDEX + kc);//intel_sub_group_shuffle( in[br * K_STRIDE + kr], bc * K_STRIDE + kc);
                    out[br * OUT_BLOCK_WIDTH + bc] = mad(weight_buf.w[w_idx % WEIGHT_PREF], input, out[br * OUT_BLOCK_WIDTH + bc]);
                  }
                }
                // We assume KERNEL_W is equal to KERNEL_H here.
                if ((w_idx + 1) % WEIGHT_PREF == 0
                #if KERNEL*KERNEL % 8 != 0
                && ((w_idx + 1) <= (KERNEL * KERNEL - WEIGHT_PREF))
                #endif
                    ) {
                  weight_buf.ui8 = intel_sub_group_block_read8((__global uint *)&weights[weight_addr]);
                  weight_addr += SIMD_SIZE * WEIGHT_PREF;  // weights must be stored in just the right SIMD swizzled format for this to work, see host code for details.
                }
              #if KERNEL*KERNEL % 8 == 0
                // need to do nothing
              #else
                else if ((w_idx + 1) %  WEIGHT_PREF == 0 && ((w_idx + 1) > (KERNEL * KERNEL - WEIGHT_PREF)))
                #if KERNEL*KERNEL % 8 == 1
                  weight_buf.w[0] = weights[weight_addr];
                #elif KERNEL*KERNEL % 4 == 0
                  weight_buf.ui8.s0123 = intel_sub_group_block_read4((__global uint *)&weights[weight_addr]);
                #else
                // should never be here if kernel_w equal to kernel_h. just in case.
                #error unsupported kernel size.
                #endif
              #endif
                ++w_idx;
              });
        });
    weight_addr = orig_weight_addr + KERNEL * KERNEL * SIMD_SIZE;

  }

  // we need this address calculation for outputs because we support views and batching
  uint_tp out_addr = OUT_BUFF_OFFSET + ( num_in_batch * TOTAL_OUTPUT_DEPTH + (fm % _OD) ) * (_OW + OWPAD) * (_OH + OHPAD);

  out_addr += or * (_OW + OWPAD) + oc;  // offset for the 4x3 block that this workitem is working on;

  // we need this address calculation for biases because we support views and batching
  float bias = biases[(fm) % _OD ];
#ifndef WRITE_PADDED_VALUES
  if(get_global_id(0) != (get_global_size(0)-1) &&
      get_global_id(1) != (get_global_size(1)-1) )
  {
#endif
    for(uint_tp r = 0; r < OUT_BLOCK_HEIGHT; r++) {
      for(uint_tp c = 0; c < OUT_BLOCK_WIDTH; c++) {
        // this does a scattered write to 16 different feature maps, so that data within one map is contiguous, thus ready for input to next layer.
        outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
      }
    }
#ifndef WRITE_PADDED_VALUES
  } else if ( get_global_id(1) != (get_global_size(1)-1) )
  {
    for(uint_tp r = 0; r < OUT_BLOCK_HEIGHT; r++) {
      for(uint_tp c = 0; c < LAST_BLOCK_WIDTH; c++) {
        outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
      }
    }
  }
  else if ( get_global_id(0) != (get_global_size(0)-1) )
  {
    for(uint_tp r = 0; r < LAST_BLOCK_HEIGHT; r++) {
      for(uint_tp c = 0; c < OUT_BLOCK_WIDTH; c++) {
        outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
      }
    }
  }
  else
  {
    for(uint_tp r = 0; r < LAST_BLOCK_HEIGHT; r++) {
      for(uint_tp c = 0; c < LAST_BLOCK_WIDTH; c++) {
        outputs[out_addr + r * (_OW + OWPAD) + c] = activation_function(bias + out[r * OUT_BLOCK_WIDTH + c]);
      }
    }
  }
#endif //#ifndef WRITE_PADDED_VALUES
}
#endif // Stride > 2
#endif

#endif

/*******************************************************************************
Copyright © 2016, Intel Corporation

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
******************************************************************************/
#ifdef Conv_Interleaved
typedef struct float1 { float s0; } float1;
typedef struct float5 { float s0; float s1; float s2; float s3; float s4; } float5;
typedef struct float6 { float s0; float s1; float s2; float s3; float s4; float s5; } float6;
typedef struct float7 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; } float7;
typedef struct float9 { float s0; float s1; float s2; float s3; float s4; float s5; float s6; float s7; float s8; } float9;
typedef struct float10 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9;} float10;
typedef struct float11 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa;} float11;
typedef struct float12 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; } float12;
typedef struct float13 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc;} float13;
typedef struct float14 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; } float14;
typedef struct float15 { float s0; float s1; float s2; float s3; float s4; float s5;
                         float s6; float s7; float s8; float s9; float sa; float sb; float sc; float sd; float se; } float15;
typedef struct float0 { float s0; } float0; //never used but makes compiler happy.
#endif



#ifdef GEMM_LIKE_CONV_32_1
//////////////////////////////////////////////////////////////////////////////
// Conv_Interleaved_32_1
//
// Convolution: each workitem computes 1 patch x 32 filters worth of output
// data.  Kernel's inner loop works on a single tile consisting of one
// row from each patch and the filter data corresponding to that row.  Filter
// matrix is interleaved to reduce GRF bank conflicts.  Patches are walked
// by rows and then by slices.  Relies on sub_group extension for block
// reads and SIMD broadcast.

#define TILE_M          1
#define TILE_K          KERNEL_WIDTH
#define TILE_N          32

#ifndef __BEIGNET__
__attribute__((intel_reqd_sub_group_size(8)))
#endif
__kernel void Conv_Interleaved(
    const __global float *src0,
    const __global float *src1,
    const __global float *biases,
    __global float *dst)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);
    const int global_z = get_global_id(2);
    int interleaved_y;
    int kernel_y;
    int kernel_idx;

    // Result ctile (*dst) is M rows x N columns
    // LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.
    float8  blockC00 = 0.f;
    float8  blockC10 = 0.f;
    float8  blockC20 = 0.f;
    float8  blockC30 = 0.f;

    // Src0 (patch input) is directly used as atile.
    // Each work item points to the start of a different patch.
    // atile is M rows x K columns.
    const __global float *src0_read = src0
     + ALIGNED_INPUT_SIZE * global_z                            // batch offset
     + ( ( global_y / OUT_WIDTH ) * ROW_PITCH * STRIDE_Y )      // y offset
     + ( ( global_y % OUT_WIDTH ) * STRIDE_X );                 // x offset

    // Src1 (filter) is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    const __global float *src1_read = src1 + ( global_x * TILE_N  * 2);

#define DOT_PRODUCT_8( _result, _rowA, colB )    \
    {   \
        _result.s0 = mad( _rowA, sub_group_broadcast( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, sub_group_broadcast( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, sub_group_broadcast( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, sub_group_broadcast( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, sub_group_broadcast( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, sub_group_broadcast( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, sub_group_broadcast( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, sub_group_broadcast( colB, 7 ), _result.s7 );  \
    }
    typedef CAT( float, KERNEL_WIDTH ) float_t;

    // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
    // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch
    // and KERNEL_WIDTH/2 rows of interleaved filter.
    int patch_depth = 0;
    do
    {
        int patch_row = 0;
        do
        {
            // Load atile and btile.
            // Kernel data is partially interleaved.  Every 2 rows are interleaved at float8 granularity.
            // The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non
            // interleaved row is padded with zero to ensure same size as interleaved rows. This
            // interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the
            // kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.
            // (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..
            // (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...
            // (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
            // ...
            const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;
            float_t blockA00 = ( (const __global float_t*)src0_read )[  0  ]; src0_read += ROW_PITCH;
            float*  pblockA00 = (float*)(&blockA00);

            float blockB00[KERNEL_WIDTH*4];
            float8* p8BlockB00 = (float8*)blockB00;
            float4* p4BlockB00 = (float4*)blockB00;
            float*  pBlockB00 =  (float* )blockB00;

            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                p8BlockB00[interleaved_y] = as_float8( intel_sub_group_block_read8( (const __global uint*)src1_read ) );
                src1_read += WIDTH1 * 2;
            } )
            if ( kernel_width_is_odd )
            {
                p4BlockB00[KERNEL_WIDTH - 1] = as_float4( intel_sub_group_block_read4( (const __global uint*)src1_read ) );
                src1_read += WIDTH1 * 2;
            }

            // Perform MADs
            kernel_idx = 0;
            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
            } )
            if ( kernel_width_is_odd )
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
            }
        }

        //while( ++patch_row < 1 ); //debug
        while( ++patch_row < KERNEL_HEIGHT );

        src0_read += SLICE_PITCH - ( KERNEL_HEIGHT * ROW_PITCH ); // reset to start of next slice of patch
    }
    //while ( ++patch_depth < 1 );  //debug
    while ( ++patch_depth < INPUT_DEPTH );

    // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
    // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
    __global float *out = dst
     + global_z * OUT_PITCH_Z                                                   // batch offset
     + ( group_x * TILE_N ) * OUT_PITCH_Y                                       // channel offset
     + ( ( global_y * TILE_M ) / OUT_WIDTH + OUT_PADDING_HEIGHT) * OUT_PITCH_X  // y offset
     + ( ( global_y * TILE_M ) % OUT_WIDTH ) + OUT_PADDING_LEFT;               // x offset
    float bias[4];
    float4 *bias_vec;
    bias_vec = (float4*)bias;
    *bias_vec = as_float4(intel_sub_group_block_read4((__global uint *)biases + group_x * TILE_N));
    if ( global_y * TILE_M < OUT_WIDTH * OUT_HEIGHT )
    {
        if ( ( OUT_DEPTH % TILE_N ) == 0 )
        {
            for ( int i = 0; i < 8; i++ )
            {
                out[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                out[( 8+i) * OUT_PITCH_Y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);
                out[(16+i) * OUT_PITCH_Y] = blockC20[i] + intel_sub_group_shuffle(bias[2], i);
                out[(24+i) * OUT_PITCH_Y] = blockC30[i] + intel_sub_group_shuffle(bias[3], i);
            }
        }
        else
        {
            if ( ( global_x + 1 ) < get_global_size(0) )
            {
                for ( int i = 0; i < 8; i++ )
                {
                    out[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                    out[( 8+i) * OUT_PITCH_Y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);
                    out[(16+i) * OUT_PITCH_Y] = blockC20[i] + intel_sub_group_shuffle(bias[2], i);
                    out[(24+i) * OUT_PITCH_Y] = blockC30[i] + intel_sub_group_shuffle(bias[3], i);
                }
            }
            else
            {
                if ( ( OUT_DEPTH % TILE_N ) >= 24 )
                {
                    for (int i = 0; i < 8; i++)
                    {
                        out[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                        out[( 8+i) * OUT_PITCH_Y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);
                        out[(16+i) * OUT_PITCH_Y] = blockC20[i] + intel_sub_group_shuffle(bias[2], i);
                    }

                    // Remaining channels
                    for (int i = 0; i < OUT_DEPTH % 24; i++)
                    {
                        out[(24+i) * OUT_PITCH_Y] = blockC30[i] + intel_sub_group_shuffle(bias[3], i);
                    }
                }
                else if ( ( OUT_DEPTH % TILE_N ) >= 16 )
                {
                    for (int i = 0; i < 8; i++)
                    {
                        out[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                        out[( 8+i) * OUT_PITCH_Y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);
                    }

                    for (int i = 0; i < OUT_DEPTH % 16; i++)
                    {
                        out[(16+i) * OUT_PITCH_Y] = blockC20[i] + intel_sub_group_shuffle(bias[2], i);
                    }
                }
                else if ( ( OUT_DEPTH % TILE_N ) >= 8 )
                {
                    for (int i = 0; i < 8; i++)
                    {
                        out[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                    }

                    for (int i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out[(8+i) * OUT_PITCH_Y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);
                    }
                }
                else
                {
                    for (int i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                    }
                }
            }

        }
    }
}
#endif

#ifdef GEMM_LIKE_CONV_32_2
//////////////////////////////////////////////////////////////////////////////
// Conv_Interleaved_32_2
//
// Convolution: each workitem computes 2 patches x 32 filters worth of output
// data.  Kernel's inner loop works on a single tile consisting of one
// row from each patch and the filter data corresponding to that row.  Filter
// matrix is interleaved to reduce GRF bank conflicts.  Patches are walked
// by rows and then by slices.  Relies on sub_group extension for block
// reads and SIMD broadcast.
#define TILE_M          2
#define TILE_K          KERNEL_WIDTH
#define TILE_N          32

#ifndef __BEIGNET__
__attribute__((intel_reqd_sub_group_size(8)))
#endif
__kernel void Conv_Interleaved(
    const __global float *src0,
    const __global float *src1,
    const __global float *biases,
    __global float *dst)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);
    const int global_z = get_global_id(2);
    int interleaved_y;
    int kernel_y;
    int kernel_idx;

    // Result ctile (*dst) is M rows x N columns
    // LWG size is 1x8.  Thus each thread calculates 8*M rows x N cols of ctile.
    float8  blockC00 = 0.f;
    float8  blockC10 = 0.f;
    float8  blockC20 = 0.f;
    float8  blockC30 = 0.f;
    float8  blockC01 = 0.f;
    float8  blockC11 = 0.f;
    float8  blockC21 = 0.f;
    float8  blockC31 = 0.f;

    // Src0 (patch input) is directly used as atile.
    // Each work item points to the start of a different patch.
    // atile is M rows x K columns.
    const __global float *src0_read0 = src0
     + ALIGNED_INPUT_SIZE * global_z                                            // batch offset
     + ( ( ( global_y * TILE_M + 0 ) / OUT_WIDTH ) * ROW_PITCH * STRIDE_Y )   // y offset
     + ( ( ( global_y * TILE_M + 0 ) % OUT_WIDTH ) * STRIDE_X );                // x offset
    const __global float *src0_read1 = src0
     + ALIGNED_INPUT_SIZE * global_z                                            // batch offset
     + ( ( ( global_y * TILE_M + 1 ) / OUT_WIDTH ) * ROW_PITCH * STRIDE_Y )   // y offset
     + ( ( ( global_y * TILE_M + 1 ) % OUT_WIDTH ) * STRIDE_X );                // x offset

    // Src1 (filter) is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    const __global float *src1_read = src1 + ( global_x * TILE_N * 2);

#define DOT_PRODUCT_8( _result, _rowA, colB )    \
    {   \
        _result.s0 = mad( _rowA, sub_group_broadcast( colB, 0 ), _result.s0 );  \
        _result.s1 = mad( _rowA, sub_group_broadcast( colB, 1 ), _result.s1 );  \
        _result.s2 = mad( _rowA, sub_group_broadcast( colB, 2 ), _result.s2 );  \
        _result.s3 = mad( _rowA, sub_group_broadcast( colB, 3 ), _result.s3 );  \
        _result.s4 = mad( _rowA, sub_group_broadcast( colB, 4 ), _result.s4 );  \
        _result.s5 = mad( _rowA, sub_group_broadcast( colB, 5 ), _result.s5 );  \
        _result.s6 = mad( _rowA, sub_group_broadcast( colB, 6 ), _result.s6 );  \
        _result.s7 = mad( _rowA, sub_group_broadcast( colB, 7 ), _result.s7 );  \
    }
    typedef CAT( float, KERNEL_WIDTH ) float_t;

    // Walk DOWN src0 (patch 0, 1, 2, ...) and DOWN src1.
    // Inner loop loads and FMADs one row (KERNEL_WIDTH) of each input patch
    // and KERNEL_WIDTH/2 rows of interleaved filter.
    int patch_depth = 0;
    do
    {
        int patch_row = 0;
        do
        {
            // Load atile and btile.
            // Kernel data is partially interleaved.  Every 2 rows are interleaved at float8 granularity.
            // The exception is that if KERNEL_WIDTH is odd the last row is not interleaved.  The non
            // interleaved row is padded with zero to ensure same size as interleaved rows. This
            // interleaving is done to ensure 0% GDR bank conflicts.  For example, this is how the
            // kernel data would be arranged before/after interleaving for KERNEL_WIDTH=3.
            // (0, 0) (8, 0) (16, 0) (24, 0) ...       (0, 0) (0, 1) (8, 0) (0, 1) (16, 0) (0, 1) (24, 0) ..
            // (0, 1) (8, 1) (16, 1) (24, 1) ... =>    (0, 2) (8, 2) (16, 2) (24, 2) ...
            // (0, 2) (8, 2) (16, 2) (24, 2) ...       ...
            // ...
            const bool kernel_width_is_odd = KERNEL_WIDTH % 2 == 1;
            float_t blockA00 = ( (const __global float_t*)src0_read0 )[  0  ]; src0_read0 += ROW_PITCH;
            float_t blockA01 = ( (const __global float_t*)src0_read1 )[  0  ]; src0_read1 += ROW_PITCH;
            float*  pblockA00 = (float*)(&blockA00);
            float*  pblockA01 = (float*)(&blockA01);

            float blockB00[KERNEL_WIDTH*4];
            float8* p8BlockB00 = (float8*)blockB00;
            float4* p4BlockB00 = (float4*)blockB00;
            float*  pBlockB00 =  (float* )blockB00;

            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                p8BlockB00[interleaved_y] = as_float8( intel_sub_group_block_read8( (const __global uint*)src1_read ) );
                src1_read += WIDTH1 * 2;
            } )
            if ( kernel_width_is_odd )
            {
                p4BlockB00[KERNEL_WIDTH - 1] = as_float4( intel_sub_group_block_read4( (const __global uint*)src1_read ) );
                src1_read += WIDTH1 * 2;
            }

            // Perform MADs
            kernel_idx = 0;
            interleaved_y = 0;
            LOOP(KERNEL_WIDTH_DIV2, interleaved_y,
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y    ], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y    ], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y + 1], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y + 1], pBlockB00[kernel_idx] ); kernel_idx++;
            } )
            if ( kernel_width_is_odd )
            {
                kernel_y = interleaved_y * 2;
                DOT_PRODUCT_8( blockC00, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC01, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC10, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC11, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC20, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC21, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
                DOT_PRODUCT_8( blockC30, pblockA00[kernel_y], pBlockB00[kernel_idx] );
                DOT_PRODUCT_8( blockC31, pblockA01[kernel_y], pBlockB00[kernel_idx] ); kernel_idx++;
            }
        }

        //while( ++patch_row < 1 ); //debug
        while( ++patch_row < KERNEL_HEIGHT );

        src0_read0 += SLICE_PITCH - ( KERNEL_HEIGHT * ROW_PITCH ); // reset to start of next slice of patch
        src0_read1 += SLICE_PITCH - ( KERNEL_HEIGHT * ROW_PITCH ); // reset to start of next slice of patch
    }
    //while ( ++patch_depth < 1 );  //debug
    while ( ++patch_depth < INPUT_DEPTH );

    // Dst resembles a cube of width x height x (output channel * batches).  Each tile writes:
    // (SIMD * TILE_M) x 1 x TILE_N.  Partial writes most likely generated if padding used.
    __global float *out0 = dst
     + global_z * OUT_PITCH_Z                                                       // batch offset
     + ( group_x * TILE_N ) * OUT_PITCH_Y                                           // channel offset
     + ( ( global_y * TILE_M + 0 ) / OUT_WIDTH + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset
     + ( ( global_y * TILE_M + 0 ) % OUT_WIDTH ) + OUT_PADDING_LEFT;               // x offset
    __global float *out1 = dst
     + global_z * OUT_PITCH_Z                                                       // batch offset
     + ( group_x * TILE_N ) * OUT_PITCH_Y                                           // channel offset
     + ( ( global_y * TILE_M + 1 ) / OUT_WIDTH + OUT_PADDING_HEIGHT ) * OUT_PITCH_X // y offset
     + ( ( global_y * TILE_M + 1 ) % OUT_WIDTH ) + OUT_PADDING_LEFT;               // x offset
    float bias[4];
    float4 *bias_vec;
    bias_vec = (float4*)bias;
    *bias_vec = as_float4(intel_sub_group_block_read4((__global uint *)biases + group_x * TILE_N));


    if( global_y * TILE_M < OUT_WIDTH * OUT_HEIGHT )
    {
        if ( ( OUT_DEPTH % TILE_N ) == 0 )
        {
            for( int i = 0; i < 8; i++ )
            {
                out0[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                out0[( 8+i) * OUT_PITCH_Y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);
                out0[(16+i) * OUT_PITCH_Y] = blockC20[i] + intel_sub_group_shuffle(bias[2], i);
                out0[(24+i) * OUT_PITCH_Y] = blockC30[i] + intel_sub_group_shuffle(bias[3], i);
            }
        }
        else
        {
            if ( ( global_x + 1 ) < get_global_size(0) )
            {
                for ( int i = 0; i < 8; i++ )
                {
                    out0[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                    out0[( 8+i) * OUT_PITCH_Y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);
                    out0[(16+i) * OUT_PITCH_Y] = blockC20[i] + intel_sub_group_shuffle(bias[2], i);
                    out0[(24+i) * OUT_PITCH_Y] = blockC30[i] + intel_sub_group_shuffle(bias[3], i);
                }
            }
            else
            {
                if ( ( OUT_DEPTH % TILE_N ) >= 24 )
                {
                    for (int i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                        out0[( 8+i) * OUT_PITCH_Y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);
                        out0[(16+i) * OUT_PITCH_Y] = blockC20[i] + intel_sub_group_shuffle(bias[2], i);
                    }

                    // remaining output channels
                    for (int i = 0; i < OUT_DEPTH % 24; i++)
                    {
                        out0[(24+i) * OUT_PITCH_Y] = blockC30[i] + intel_sub_group_shuffle(bias[3], i);
                    }
                }
                else if ( ( OUT_DEPTH % TILE_N ) >= 16 )
                {
                    for (int i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                        out0[( 8+i) * OUT_PITCH_Y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);
                    }

                    for (int i = 0; i < OUT_DEPTH % 16; i++)
                    {
                        out0[(16+i) * OUT_PITCH_Y] = blockC20[i] + intel_sub_group_shuffle(bias[2], i);
                    }
                }
                else if ( ( OUT_DEPTH % TILE_N ) >= 8 )
                {
                    for (int i = 0; i < 8; i++)
                    {
                        out0[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                    }

                    for (int i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out0[(8+i) * OUT_PITCH_Y] = blockC10[i] + intel_sub_group_shuffle(bias[1], i);
                    }
                }
                else
                {
                    for (int i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out0[( 0+i) * OUT_PITCH_Y] = blockC00[i] + intel_sub_group_shuffle(bias[0], i);
                    }
                }
            }
        }
    }

    if( global_y * TILE_M + 1 < OUT_WIDTH * OUT_HEIGHT )
    {
        if ( ( OUT_DEPTH % TILE_N ) == 0 )
        {
            for( int i = 0; i < 8; i++ )
            {
                out1[( 0+i) * OUT_PITCH_Y] = blockC01[i] + intel_sub_group_shuffle(bias[0], i);
                out1[( 8+i) * OUT_PITCH_Y] = blockC11[i] + intel_sub_group_shuffle(bias[1], i);
                out1[(16+i) * OUT_PITCH_Y] = blockC21[i] + intel_sub_group_shuffle(bias[2], i);
                out1[(24+i) * OUT_PITCH_Y] = blockC31[i] + intel_sub_group_shuffle(bias[3], i);
            }
        }
        else
        {
            if ( ( global_x + 1 ) < get_global_size(0) )
            {
                for ( int i = 0; i < 8; i++ )
                {
                    out1[( 0+i) * OUT_PITCH_Y] = blockC01[i] + intel_sub_group_shuffle(bias[0], i);
                    out1[( 8+i) * OUT_PITCH_Y] = blockC11[i] + intel_sub_group_shuffle(bias[1], i);
                    out1[(16+i) * OUT_PITCH_Y] = blockC21[i] + intel_sub_group_shuffle(bias[2], i);
                    out1[(24+i) * OUT_PITCH_Y] = blockC31[i] + intel_sub_group_shuffle(bias[3], i);
                }
            }
            else
            {
                if ( ( OUT_DEPTH % TILE_N ) >= 24 )
                {
                    for (int i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUT_PITCH_Y] = blockC01[i] + intel_sub_group_shuffle(bias[0], i);
                        out1[( 8+i) * OUT_PITCH_Y] = blockC11[i] + intel_sub_group_shuffle(bias[1], i);
                        out1[(16+i) * OUT_PITCH_Y] = blockC21[i] + intel_sub_group_shuffle(bias[2], i);
                    }

                    // Remaining channels
                    for (int i = 0; i < OUT_DEPTH % 24; i++)
                    {
                        out1[(24+i) * OUT_PITCH_Y] = blockC31[i] + intel_sub_group_shuffle(bias[3], i);
                    }
                }
                else if ( ( OUT_DEPTH % TILE_N ) >= 16 )
                {
                    for (int i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUT_PITCH_Y] = blockC01[i] + intel_sub_group_shuffle(bias[0], i);
                        out1[( 8+i) * OUT_PITCH_Y] = blockC11[i] + intel_sub_group_shuffle(bias[1], i);
                    }

                    for (int i = 0; i < OUT_DEPTH % 16; i++)
                    {
                        out1[(16+i) * OUT_PITCH_Y] = blockC21[i] + intel_sub_group_shuffle(bias[2], i);
                    }
                }
                else if ( ( OUT_DEPTH % TILE_N ) >= 8 )
                {
                    for (int i = 0; i < 8; i++)
                    {
                        out1[( 0+i) * OUT_PITCH_Y] = blockC01[i] + intel_sub_group_shuffle(bias[0], i);
                    }

                    for (int i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out1[(8+i) * OUT_PITCH_Y] = blockC11[i] + intel_sub_group_shuffle(bias[1], i);
                    }
                }
                else
                {
                    for (int i = 0; i < OUT_DEPTH % 8; i++)
                    {
                        out1[( 0+i) * OUT_PITCH_Y] = blockC01[i] + intel_sub_group_shuffle(bias[0], i);
                    }
                }
            }
        }
    }
}
#endif
