#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

namespace caffe {
template<typename Dtype>
void RepackLayer< Dtype >::LayerSetUp(const vector< Blob< Dtype >* >& bottom,
                                      const vector< Blob< Dtype >* >& top) {
  RepackParameter repack_param = this->layer_param_.repack_param();

  if (!repack_param.has_stride_h()) {
    stride_h_ = stride_w_ = repack_param.stride();
  } else {
    stride_h_ = repack_param.stride_h();
    stride_w_ = repack_param.stride_w();
  }
  operation_ = repack_param.operation();
}
template<typename Dtype>
void RepackLayer< Dtype >::Reshape(const vector< Blob< Dtype >* >& bottom,
                                   const vector< Blob< Dtype >* >& top) {
  if ( operation_ == RepackParameter_Operation_PACK_IMAGE ) {
    int nW = (bottom[0]->width()-1)/stride_w_+1;
    int nH = (bottom[0]->height()-1)/stride_h_+1;
    int nN = stride_w_*stride_h_*bottom[0]->num();
    top[0]->Reshape(nN, bottom[0]->channels(), nH, nW);
  } else if ( operation_ == RepackParameter_Operation_UNPACK_IMAGE ) {
    int nW = bottom[0]->width()*stride_w_;
    int nH = bottom[0]->height()*stride_h_;
    int nN = bottom[0]->num()/(stride_w_*stride_h_);
    top[0]->Reshape(nN, bottom[0]->channels(), nH, nW);
  } else { LOG(ERROR) << "Unknown repacking operation!"; }
}
template<typename Dtype>
inline void pack_cpu(const Dtype* input, int in_n, int in_c, int in_h, int in_w,
                     Dtype* output, int s_h, int s_w) {
  int out_w = (in_w-1)/s_w+1, out_h = (in_h-1)/s_h+1;
  for ( int y = 0, o = 0; y < s_h; y++ )
    for ( int x = 0; x < s_w; x++ )
      for ( int n = 0; n < in_n; n++ )
        for ( int c = 0; c < in_c; c++ )
          for ( int j = 0; j < out_h; j++ )
            for ( int i = 0; i < out_w; i++, o++ )
              if (j*s_h+y < in_h && i*s_w+x < in_w )
                output[o] = input[((n*in_c+c)*in_h+j*s_h+y)*in_w+i*s_w+x];
              else
                output[o] = 0;
}
template<typename Dtype>
inline void unpack_cpu(const Dtype* input, Dtype* output, int out_n, int out_c,
                       int out_h, int out_w, int s_h, int s_w) {
  int in_w = (out_w-1)/s_w+1, in_h = (out_h-1)/s_h+1;
  for ( int y = 0, o = 0; y < s_h; y++ )
    for ( int x = 0; x < s_w; x++ )
      for ( int n = 0; n < out_n; n++ )
        for ( int c = 0; c < out_c; c++ )
          for ( int j = 0; j < in_h; j++ )
            for ( int i = 0; i < in_w; i++, o++ )
              if (j*s_h+y < out_h && i*s_w+x < out_w )
                output[((n*out_c+c)*out_h+j*s_h+y)*out_w+i*s_w+x] = input[o];
}
template<typename Dtype>
void RepackLayer< Dtype >::Forward_cpu(const vector< Blob< Dtype >* >& bottom,
                                       const vector< Blob< Dtype >* >& top) {
  if ( operation_ == RepackParameter_Operation_PACK_IMAGE )
    pack_cpu(bottom[0]->cpu_data(), bottom[0]->num(), bottom[0]->channels(),
             bottom[0]->height(), bottom[0]->width(),
             top[0]->mutable_cpu_data(), stride_h_, stride_w_);
  else
    unpack_cpu(bottom[0]->cpu_data(), top[0]->mutable_cpu_data(), top[0]->num(),
               top[0]->channels(), top[0]->height(), top[0]->width(),
               stride_h_, stride_w_);
}
template<typename Dtype>
void RepackLayer< Dtype >::Backward_cpu(const vector< Blob<Dtype>* >& top,
                                        const vector<bool>& propagate_down,
                                        const vector< Blob<Dtype>* >& bottom) {
  if (!propagate_down[0]) return;
  if ( operation_ == RepackParameter_Operation_UNPACK_IMAGE )
    pack_cpu(top[0]->cpu_diff(), top[0]->num(), top[0]->channels(),
             top[0]->height(), top[0]->width(),
             bottom[0]->mutable_cpu_diff(), stride_h_, stride_w_);
  else
    unpack_cpu(top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff(),
               bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(),
               bottom[0]->width(), stride_h_, stride_w_);
}



#ifdef CPU_ONLY
STUB_GPU(RepackLayer);
#endif
INSTANTIATE_CLASS(RepackLayer);
REGISTER_LAYER_CLASS(Repack);

}  // namespace caffe
