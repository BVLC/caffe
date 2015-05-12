#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#define MAX_SPATIAL_AXES 10
namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype* bottom_data,
    int num, int channels,int num_axes, const int* im_shape, const int* pooled_shape,
    const int* kernel_shape, const int* stride, const int* pad, Dtype* top_data,
    int* mask, Dtype* top_mask) {
    
    int pool_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int im_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int starts[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int ends[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
  CUDA_KERNEL_LOOP(index, nthreads) {

    // ind2sub
    int k = index;
    for (int i = num_axes-1; i >=0 ;--i) {
      pool_loc[i] = k % pooled_shape[i];
      k /= pooled_shape[i];
    }
    int c = k % channels;
    int n = k / channels;
    // Get starts and calc size (sub2ind)
    int start, end;
    int im_size = 1;
    for (int i = 0; i < num_axes; ++i) { 
      starts[i] = pool_loc[i]*stride[i] - pad[i];
      ends[i] = min(starts[i]+kernel_shape[i], im_shape[i+1]);
      starts[i] = max(starts[i],0);
      ends[i] = min(ends[i],im_shape[i+1]);
      if (i!=0) {
        start = start*im_shape[i+1]+starts[i];
        end = end*im_shape[i+1]+ends[i];
      } else {
        start = starts[0];
        end = ends[0];
      }
      im_size *= im_shape[i+1];
    }
    // get input kernel and compute pool
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data += (n * channels + c) * im_size;
    if (num_axes ==2 ){
      for (int h = starts[0]; h < ends[0]; ++h){
        for (int w = starts[1]; w < ends[1]; ++w){
          const int bottom_index = h * im_shape[2] + w;
          if ( bottom_data[bottom_index] > maxval ) {
              maxidx = bottom_index;
              maxval = bottom_data[maxidx];
          }
        }
      }
    } else if (num_axes == 3) {
        for (int h = starts[0]; h < ends[0]; ++h){
          for (int w = starts[1]; w < ends[1]; ++w){
            for (int z = starts[2]; z < ends[2]; ++z){
              const int bottom_index = (h * im_shape[2] + w)*im_shape[3]+z;
              if( bottom_data[bottom_index] > maxval) {
                  maxidx = bottom_index;
                  maxval = bottom_data[maxidx];
              }
            }
          }
        }
    } else {
      for (int input_index = start; input_index < end; ++input_index){
        bool in_range = true;
        // ind2sub
        int m = input_index;
        for (int j = num_axes-1;j>=0;--j) {
          im_loc[j] = m % im_shape[j+1];
          m /= im_shape[j+1];
          in_range &= (m<im_shape[j+1])&&(im_loc[j] >= starts[j]) && (im_loc[j] < ends[j]);
          if (!in_range) { break; }
        }
        if (in_range){
          if (bottom_data[input_index] > maxval) {
            maxval = bottom_data[input_index];
            maxidx = input_index;
          }
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    } 
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads, const Dtype* bottom_data,
    int num, int channels,int num_axes, const int* im_shape, const int* pooled_shape,
    const int* kernel_shape, const int* stride, const int* pad, Dtype* top_data) {
    int pool_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int im_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int starts[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int ends[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
  CUDA_KERNEL_LOOP(index, nthreads) {
    // ind2sub
    int k = index;
    for (int i = num_axes-1; i >=0 ;--i) {
      pool_loc[i] = k % pooled_shape[i];
      k /= pooled_shape[i];
    }
    int c = k % channels;
    int n = k / channels;

    // Get starts and calc size (sub2ind)
    int pool_size = 1;
    int start,end;
    int im_size = 1;

    for (int i = 0; i < num_axes; ++i) {
      starts[i] = pool_loc[i]*stride[i] - pad[i];
      ends[i] = min(starts[i]+kernel_shape[i], im_shape[i+1] + pad[i]);
      int s1 = ends[i];
      int s2 = starts[i];
      pool_size *= (s1 - s2);
      starts[i] = max(starts[i],0);
      if (i!=0) {
        start = start*im_shape[i+1]+starts[i];
        end = end*im_shape[i+1]+ends[i];
      } else {
        start = starts[0];
        end = ends[0];
      }
      ends[i] = min(ends[i],im_shape[i+1]);
      im_size *= im_shape[i+1];

    }

    Dtype aveval = 0;
    bottom_data += (n * channels + c) * im_size;

    if (num_axes==2) { 
      for (int h = starts[0]; h < ends[0]; ++h){
        for (int w = starts[1]; w < ends[1]; ++w){
          const int bottom_index = h * im_shape[2] + w;
          aveval += bottom_data[bottom_index];
        }
      }
    } else if (num_axes == 3 ) {
      for (int h = starts[0]; h < ends[0]; ++h){
        for (int w = starts[1]; w < ends[1]; ++w){
          for (int z = starts[2]; z < ends[2]; ++z){
            const int bottom_index = (h * im_shape[2] + w)*im_shape[3]+z;
            aveval += bottom_data[bottom_index];
          }
        }
      }
    } else {
      for (int input_index = start; input_index < end; ++input_index){
        bool in_range = true;
        // ind2sub
        int m = input_index;
        for (int j = num_axes-1;j>=0;--j) {
          im_loc[j] = m % im_shape[j+1];
          m /= im_shape[j+1];
          in_range &= (m<im_shape[j+1])&&(im_loc[j] >= starts[j]) && (im_loc[j] < ends[j]);
          if (!in_range) { break; }
        }
        if (in_range){
          aveval += bottom_data[input_index];
        } 
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads, const Dtype* bottom_data,
    int num, int channels,int num_axes, const int* im_shape, const int* pooled_shape,
    const int* kernel_shape, const int* stride, Dtype* rand_idx, Dtype* top_data) {
    int pool_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int input_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int starts[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int stops[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
  CUDA_KERNEL_LOOP(index, nthreads) {
    // ind2sub
    int k = index;
    for (int i = num_axes-1; i >=0 ;--i) {
      pool_loc[i] = k % pooled_shape[i];
      k /= pooled_shape[i];
    }
    int c = k % channels;
    int n = k / channels;

    // Get starts and calc size (sub2ind)
    int pool_size = 1;
    int start = 1;
    int stop = 1;
    int im_size = 1;
    for (int i = 0; i < num_axes; ++i) {
      starts[i] = pool_loc[i]*stride[i];
      stops[i] = min(starts[i]+kernel_shape[i] , im_shape[i+1]);
      int s1 = stops[i];
      int s2 = starts[i];
      pool_size *= (s1 - s2);
      im_size *= im_shape[i+1];
      if (i!=0) {
        start = start*im_shape[i+1]+starts[i];
        stop = stop*im_shape[i+1]+stops[i];
      } else {
        start = starts[0];
        stop = stops[0];
      }
    }
    Dtype cumsum = 0.;
    bottom_data += (n * channels + c) * im_size;
    // First pass: get sum
    for (int input_index = start; input_index < stop; ++input_index){
      bool in_range = true;
      // ind2sub
      int m = input_index;
      for (int j = num_axes-1;j>=0;--j) {
        input_loc[j] = m % im_shape[j+1];
        m /= im_shape[j+1];
        in_range &= (m<im_shape[j+1])&&(input_loc[j] >= starts[j]) && (input_loc[j] < stops[j]);
        if (!in_range) { break;}
      }
      if (in_range){
        cumsum += bottom_data[input_index];
      }
    }
    float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int input_index = start; input_index < stop; ++input_index){
      bool in_range = true;
      // ind2sub
      int m = input_index;
      for (int j = num_axes-1;j>=0;--j) {
        input_loc[j] = m % im_shape[j+1];
        m /= im_shape[j+1];
       in_range &= (m<im_shape[j+1])&&(input_loc[j] >= starts[j]) && (input_loc[j] < stops[j]);
        if (!in_range) { break;}
      }
      if (in_range){
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * im_size) + input_index;
          top_data[index] = bottom_data[input_index];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads, const Dtype* bottom_data,
    int num, int channels, int num_axes, const int* im_shape, const int* pooled_shape,
    const int* kernel_shape, const int* stride, Dtype* top_data) {
    int pool_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int input_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int starts[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int stops[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
  CUDA_KERNEL_LOOP(index, nthreads) {
    // ind2sub
    int k = index;
    for (int i = num_axes-1; i >=0 ;--i) {
      pool_loc[i] = k % pooled_shape[i];
      k /= pooled_shape[i];
    }
    int c = k % channels;
    int n = k / channels;

    // Get starts and calc size (sub2ind)
    int pool_size = 1;
    int start = 1;
    int stop = 1;
    int im_size = 1;
    for (int i = 0; i < num_axes; ++i) {
      starts[i] = pool_loc[i]*stride[i];
      stops[i] = min(starts[i]+kernel_shape[i] , im_shape[i+1]);
      int s1 = stops[i];
      int s2 = starts[i];
      pool_size *= (s1 - s2); 
      starts[i] = max(starts[i],0);
      im_size *= im_shape[i+1];
      if (i!=0) {
        start = start*im_shape[i+1]+starts[i];
        stop = stop*im_shape[i+1]+stops[i];
      } else {
        start = starts[0];
        stop = stops[0];
      }
    }
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    bottom_data += (n * channels + c) * im_size;
    // First pass: get sum
    for (int input_index = start; input_index < stop; ++input_index){
      bool in_range = true;
      // ind2sub
      int m = input_index;
      for (int j = num_axes-1;j>=0;--j) {
        input_loc[j] = m % im_shape[j+1];
        m /= im_shape[j+1];
        in_range &= (m<im_shape[j+1])&&(input_loc[j] >= starts[j]) && (input_loc[j] < stops[j]);
        if (!in_range) { break;}
      }
      if (in_range){
        cumsum += bottom_data[input_index];
        cumvalues += bottom_data[input_index] * bottom_data[input_index];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
  // printf("Forward_gpuMAX\n");
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, num_, channels_, num_spatial_axes_,
        input_shape_.gpu_data(), output_shape_.gpu_data(), kernel_shape_.gpu_data(),
        stride_.gpu_data(), pad_.gpu_data(), top_data,
        mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data,  num_, channels_, num_spatial_axes_,
        input_shape_.gpu_data(), output_shape_.gpu_data(), kernel_shape_.gpu_data(),
        stride_.gpu_data(), pad_.gpu_data(), top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data,  num_, channels_, num_spatial_axes_,
        input_shape_.gpu_data(), output_shape_.gpu_data(), kernel_shape_.gpu_data(),
        stride_.gpu_data(),
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data,  num_, channels_, num_spatial_axes_,
        input_shape_.gpu_data(), output_shape_.gpu_data(), kernel_shape_.gpu_data(),
        stride_.gpu_data(), top_data);
    }
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* top_diff,
    const int* mask, const Dtype* top_mask, int num, 
    int channels, int num_axes, const int* im_shape, const int* pooled_shape,
    const int* kernel_shape, const int* stride, const int* pad,
    Dtype* bottom_diff) {
    int pool_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int im_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int starts[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int ends[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)

  CUDA_KERNEL_LOOP(index, nthreads) {
    int k = index;
    for (int i = num_axes-1; i >=0 ;--i) {
      im_loc[i] = k % im_shape[i+1];
      k /= im_shape[i+1];
    }
    int c = k % channels;
    int n = k / channels;
    int shift_size = 1;
    int start, end;
    for (int i = 0; i < num_axes; ++i) {
      starts[i] = (im_loc[i]+pad[i] < kernel_shape[i]) ? 0 : (im_loc[i] + pad[i] - kernel_shape[i]) / stride[i] + 1;
      ends[i] = min((im_loc[i]+pad[i]) / stride[i] + 1, pooled_shape[i]);
      shift_size *= pooled_shape[i];
      if (num_axes > 0) {
        if (i!=0) {
          start = start*pooled_shape[i]+starts[i];
          end = end*pooled_shape[i]+ends[i];
        } else {
          start = starts[0];
          end = ends[0];
      }
      }
    }
    Dtype gradient = 0;
    bool use_top_mask = !(mask);
    int offset = (n * channels + c) * shift_size;
    top_diff += offset;
    mask += offset;
    top_mask += offset;
    if (num_axes == 2){
      for (int ph = starts[0]; ph < ends[0]; ++ph) {
        for (int pw = starts[1]; pw < ends[1]; ++pw) {
          const int top_index = ph * pooled_shape[1] + pw;
          const int im_index = im_loc[0] * im_shape[2] + im_loc[1];
          const int index_match = 
                use_top_mask ? top_mask[top_index] : mask[top_index];
          if (index_match == im_index)
              gradient += top_diff[top_index];
        }
      }
    } else if (num_axes == 3) {
          for (int ph = starts[0]; ph < ends[0]; ++ph) {
            for (int pw = starts[1]; pw < ends[1]; ++pw) {
              for (int pz = starts[2]; pz < ends[2]; ++pz) {
                const int top_index = (ph * pooled_shape[1] + pw)*pooled_shape[2]+pz;
                const int im_index = (im_loc[0] * im_shape[2] + im_loc[1])*im_shape[3]+im_loc[2];
                const int index_match =
                      use_top_mask ? top_mask[top_index] : mask[top_index];
                if (index_match == im_index)
                    gradient += top_diff[top_index];
              }
            }
          }
    } else {
        // ND is slower...
        for (int pool_index = start; pool_index < end; ++pool_index){
          bool in_range = true;
          // ind2sub
          int m = pool_index;
          for (int j = num_axes-1;j>=0;--j) {
            pool_loc[j] = m % pooled_shape[j];
            m /= pooled_shape[j];
            in_range &= (m<pooled_shape[j])&&(pool_loc[j] >= starts[j]) && (pool_loc[j] < ends[j]);
            //if (!in_range) { break;}
          }
          if (in_range){
            int im_index = im_loc[0];
            for (int i=1; i< num_axes; ++i){
              im_index = im_index*im_shape[i+1]+im_loc[i];
            }
            const int index_match =
                        use_top_mask ? top_mask[pool_index] : mask[pool_index];
            if (index_match == im_index)
                gradient += top_diff[pool_index];
          }
        }
    }
    bottom_diff[index] = gradient;

  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* top_diff,
    int num, int channels, int num_axes, const int* im_shape, const int* pooled_shape,
    const int* kernel_shape, const int* stride, const int* pad,
    Dtype* bottom_diff) {
    int pool_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int im_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int starts[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int ends[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
  CUDA_KERNEL_LOOP(index, nthreads) {
    // ind2sub
    int k = index;
    for (int i = num_axes-1; i >=0 ;--i) {
      im_loc[i] = (k % im_shape[i+1])+pad[i];
      k /= im_shape[i+1];
    }
    int c = k % channels;
    int n = k / channels;
    int shift_size = 1;
    int start = 1;
    int stop = 1;
    for (int i = 0; i < num_axes; ++i) {
      starts[i] = (im_loc[i] < kernel_shape[i]) ? 0 : (im_loc[i] - kernel_shape[i]) / stride[i] + 1;
      ends[i] = min(im_loc[i] / stride[i] + 1, pooled_shape[i]);
      shift_size *= pooled_shape[i];
      if (num_axes > 3){
        // obtain the stop index for ND calculation.
        if (i!=0) {
          start = start*pooled_shape[i]+starts[i];
          stop = stop*pooled_shape[i]+ends[i];
        } else {
          start = starts[0];
          stop = ends[0];
        }
      }
    }
    int pool_size;
    Dtype gradient = 0;
    top_diff += (n * channels + c) * shift_size;
    if (num_axes == 2) {
      for (int ph = starts[0]; ph < ends[0]; ++ph) {
        for (int pw = starts[1]; pw < ends[1]; ++pw) {
          int hstart = ph * stride[0] - pad[0];
          int wstart = pw * stride[1] - pad[1];
          int hend = min(hstart + kernel_shape[0], im_shape[1] + pad[0]);
          int wend = min(wstart + kernel_shape[1], im_shape[2] + pad[1]);
          pool_size = (hend - hstart) * (wend - wstart);
          gradient += top_diff[ph * pooled_shape[1] + pw] / pool_size;
        }
      }
    } else if (num_axes == 3) {
      for (int ph = starts[0]; ph < ends[0]; ++ph) {
        for (int pw = starts[1]; pw < ends[1]; ++pw) {
          for (int pz = starts[2]; pz < ends[2]; ++pz) {
            int hstart = ph * stride[0] - pad[0];
            int wstart = pw * stride[1] - pad[1];
            int zstart = pz * stride[2] - pad[2];
            int hend = min(hstart + kernel_shape[0], im_shape[1] + pad[0]);
            int wend = min(wstart + kernel_shape[1], im_shape[2] + pad[1]);
            int zend = min(wstart + kernel_shape[2], im_shape[3] + pad[2]);
            pool_size = (hend - hstart) * (wend - wstart)*(zend - zstart);
            gradient += top_diff[(ph * pooled_shape[1] + pw)*pooled_shape[2]+pz] / pool_size;
          }
        }
      }
    } else {
      // ND loop (much slower)
        for (int pool_index = start; pool_index < stop; ++pool_index){
          bool in_range = true;
          // ind2sub
          int m = pool_index;
          for (int j = num_axes-1;j>=0;--j) {
            pool_loc[j] = m % pooled_shape[j];
            m /= pooled_shape[j];
            in_range &= (m<pooled_shape[j])&&(pool_loc[j] >= starts[j]) && (pool_loc[j] < ends[j]);
            if (!in_range) { break;}
          }
          if (in_range){
            pool_size = 1;
            for (int i=0; i< num_axes; ++i){
              int pstart = pool_loc[i]*stride[i]-pad[i];
              int pend = min(pstart + kernel_shape[i], im_shape[i+1] + pad[i]);
              pool_size *= (pend - pstart); 
            }
            gradient += top_diff[pool_index] / pool_size;
          }
        }
    }
 bottom_diff[index] = gradient;

  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* rand_idx, const Dtype* top_diff,
    int num, int channels, int num_axes, const int* im_shape, const int* pooled_shape,
    const int* kernel_shape, const int* stride, Dtype* bottom_diff) {
    int pool_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int im_loc[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int starts[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
    int ends[MAX_SPATIAL_AXES];  // NOLINT(runtime/arrays)
  CUDA_KERNEL_LOOP(index, nthreads) {
    // ind2sub
    int k = index;
    for (int i = num_axes-1; i >=0 ;--i) {
      im_loc[i] = (k % im_shape[i+1]);
      k /= im_shape[i+1];
    }
    int c = k % channels;
    int n = k / channels;
    int shift_size = 1;
    int start, end;
    for (int i = 0; i < num_axes; ++i) {
      starts[i] = (im_loc[i] < kernel_shape[i]) ? 0 : (im_loc[i] - kernel_shape[i]) / stride[i] + 1;
      ends[i] = min((im_loc[i]) / stride[i] + 1, pooled_shape[i]);
      shift_size *= pooled_shape[i];
      if (num_axes > 3) {
        if (i!=0) {
          start = start*pooled_shape[i]+starts[i];
          end = end*pooled_shape[i]+ends[i];
        } else {
          start = starts[0];
          end = ends[0];
        }
      }
    }
    Dtype gradient = 0;
    rand_idx += (n * channels + c) * shift_size;
    top_diff += (n * channels + c) * shift_size;
    if (num_axes == 2){
      for (int ph = starts[0]; ph < ends[0]; ++ph) {
        for (int pw = starts[1]; pw < ends[1]; ++pw) {
          gradient += top_diff[ph * pooled_shape[1] + pw] *
            (index == static_cast<int>(rand_idx[ph * pooled_shape[1] + pw]));
        }
      }
    } else if (num_axes == 3){
      for (int ph = starts[0]; ph < ends[0]; ++ph) {
        for (int pw = starts[1]; pw < ends[1]; ++pw) {
          for (int pz = starts[2]; pz < ends[2]; ++pz) {
            gradient += top_diff[(ph * pooled_shape[1] + pw)*pooled_shape[2]+pz] *
              (index == static_cast<int>(rand_idx[(ph * pooled_shape[1] + pw)*pooled_shape[2]+pz]));
          }
        }
      }
    } else {
        for (int pool_index = start; pool_index < end; ++pool_index){
          bool in_range = true;
          // ind2sub
          int m = pool_index;
          for (int j = num_axes-1;j>=0;--j) {
            pool_loc[j] = m % pooled_shape[j];
            m /= pooled_shape[j];
            in_range &= (m<pooled_shape[j])&&(pool_loc[j] >= starts[j]) && (pool_loc[j] < ends[j]);
            if (!in_range) { break;}
          }
          if (in_range){
             gradient += top_diff[pool_index] *
              (index == static_cast<int>(rand_idx[pool_index]));
          }
        }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  const int* kernel_shape = kernel_shape_.gpu_data();
  int top_num = top[0]->count(0, channel_axis_);
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top_num, channels_, num_spatial_axes_,
        input_shape_.gpu_data(), output_shape_.gpu_data(), kernel_shape_.gpu_data(),
          stride_.gpu_data(), pad_.gpu_data(),
        bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top_num, channels_, num_spatial_axes_,
        input_shape_.gpu_data(), output_shape_.gpu_data(), kernel_shape_.gpu_data(),
          stride_.gpu_data(), pad_.gpu_data(), bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top_num, channels_, num_spatial_axes_,
        input_shape_.gpu_data(), output_shape_.gpu_data(), kernel_shape_.gpu_data(),
          stride_.gpu_data(),
        bottom_diff);
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
