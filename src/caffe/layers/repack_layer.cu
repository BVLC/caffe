#include <vector>
#include "caffe/vision_layers.hpp"

namespace caffe {

__device__ void computeCoord(int * x, int * y, int * i, int * j, int * c,
                             int * n, int index, int u_n, int u_c, int p_h,
                             int p_w, int s_w) {
  // Get the coordinates
  *x = index;
  *i = *x % p_w;  // width dimension
  *x /= p_w;
  *j = *x % p_h;  // height dimension
  *x /= p_h;
  *c = *x % u_c;  // channel dimension
  *x /= u_c;
  *n = *x % u_n;  // num dimension
  *x /= u_n;

  // Get the packing indices
  *y = *x % s_w;
  *x /= s_w;
}

template <typename Dtype>
__global__ void pack_gpu_kernel(const Dtype* unpacked, int u_n, int u_c,
                                int u_h, int u_w, Dtype* packed, int s_h,
                                int s_w) {
  const int p_n = s_h*s_w*u_n, p_w = (u_w-1)/s_w+1, p_h = (u_h-1)/s_h+1;
  const int N = p_n*u_c*p_h*p_w;
  CUDA_KERNEL_LOOP(index, N) {
    int x, y, i, j, c, n;
    computeCoord(&x, &y, &i, &j, &c, &n, index, u_n, u_c, p_h, p_w, s_w);
    if (j*s_h+y < u_h && i*s_w+x < u_w )
      packed[index] = unpacked[((n*u_c+c)*u_h+j*s_h+y)*u_w+i*s_w+x];
    else
      packed[index] = 0;
  }
}

template <typename Dtype>
void pack_gpu(const Dtype* unpacked, int u_n, int u_c, int u_h, int u_w,
              Dtype* packed, int s_h, int s_w) {
  const int p_n = s_h*s_w*u_n, p_w = (u_w-1)/s_w+1, p_h = (u_h-1)/s_h+1;
  const int N = p_n*u_c*p_h*p_w;
  /* NOLINT_NEXT_LINE(whitespace/operators) */
  pack_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      unpacked, u_n, u_c, u_h, u_w, packed, s_h, s_w);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void unpack_gpu_kernel(const Dtype* packed, Dtype* unpacked,
                                  int u_n, int u_c, int u_h, int u_w,
                                  int s_h, int s_w) {
  const int p_n = s_h*s_w*u_n, p_w = (u_w-1)/s_w+1, p_h = (u_h-1)/s_h+1;
  const int N = p_n*u_c*p_h*p_w;
  CUDA_KERNEL_LOOP(index, N) {
    int x, y, i, j, c, n;
    computeCoord(x, y, i, j, c, n, index, u_n, u_c, p_h, p_w, s_w);
    // TODO: This is not all that memory friendly!
    if (j*s_h+y < u_h && i*s_w+x < u_w )
      unpacked[((n*u_c+c)*u_h+j*s_h+y)*u_w+i*s_w+x] = packed[index];
  }
}

template <typename Dtype>
void unpack_gpu(const Dtype* packed, Dtype* unpacked, int u_n, int u_c,
                int u_h, int u_w, int s_h, int s_w) {
  int p_n = s_h*s_w*u_n, p_w = (u_w-1)/s_w+1, p_h = (u_h-1)/s_h+1;
  const int N = p_n*u_c*p_h*p_w;
  /* NOLINT_NEXT_LINE(whitespace/operators) */
  unpack_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      packed, unpacked, u_n, u_c, u_h, u_w, s_h, s_w);
  CUDA_POST_KERNEL_CHECK;
}

template<typename Dtype>
void RepackLayer< Dtype >::Forward_gpu(const vector< Blob< Dtype >* >& bottom,
                                       const vector< Blob< Dtype >* >& top) {
  if ( operation_ == RepackParameter_Operation_PACK_IMAGE )
    pack_gpu(bottom[0]->gpu_data(), bottom[0]->num(), bottom[0]->channels(),
             bottom[0]->height(), bottom[0]->width(),
             top[0]->mutable_gpu_data(), stride_h_, stride_w_);
  else
    unpack_gpu(bottom[0]->gpu_data(), top[0]->mutable_gpu_data(), top[0]->num(),
               top[0]->channels(), top[0]->height(), top[0]->width(),
               stride_h_, stride_w_);
}
template<typename Dtype>
void RepackLayer< Dtype >::Backward_gpu(const vector< Blob<Dtype>* >& top,
                                        const vector<bool>& propagate_down,
                                        const vector< Blob<Dtype>* >& bottom) {
  if (!propagate_down[0]) return;
  if ( operation_ == RepackParameter_Operation_UNPACK_IMAGE )
    pack_gpu(top[0]->gpu_diff(), top[0]->num(), top[0]->channels(),
             top[0]->height(), top[0]->width(),
             bottom[0]->mutable_gpu_diff(), stride_h_, stride_w_);
  else
    unpack_gpu(top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(),
               bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(),
               bottom[0]->width(), stride_h_, stride_w_);
}

INSTANTIATE_LAYER_GPU_FUNCS(RepackLayer);

}  // namespace caffe
