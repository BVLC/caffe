#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  slice_ = this->layer_param_.softmax_param().slice();
  top[0]->ReshapeLike(*bottom[0]);
  
  int channels =  bottom[0]->shape(softmax_axis_);  
  CHECK_EQ(channels%slice_, 0)
		<< "I canali devono essere divisibili per slice_";
    
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_)/slice_);
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
		
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = (bottom[0]->count() / outer_num_);
  
  CHECK_EQ(channels%slice_, 0)
		<< "I canali devono essere divisibili per slice_";
  // We want to softmax to slice_ slices of bottom[0]
  // each slice is made fo step elements
  int step = channels/slice_;
  

  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  
    
  for (int i = 0; i < outer_num_; ++i) {
    
    bottom_data = bottom[0]->cpu_data() + (i*dim);
    
    for(int fp = 0; fp < slice_; ++fp) {
      bottom_data +=  (fp * step) * (inner_num_);
      // initialize scale_data to the first plane
      caffe_copy(inner_num_, bottom_data, scale_data);
      for (int j = 0; j < step; j++) {
        for (int k = 0; k < inner_num_; k++) {
          scale_data[k] = std::max(scale_data[k],
              bottom_data[j * inner_num_ + k]);
              
        }
      }
      // subtraction
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, step, inner_num_,
          1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
      // exponentiation
      caffe_exp<Dtype>(dim/slice_, top_data, top_data);
      // sum after exp
      caffe_cpu_gemv<Dtype>(CblasTrans, step, inner_num_, 1.,
          top_data, sum_multiplier_.cpu_data(), 0., scale_data);
      // division
      for (int j = 0; j < step; j++) {
        caffe_div(inner_num_, top_data, scale_data, top_data);
        top_data += inner_num_;
      }
    }
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxLayer);
#endif

INSTANTIATE_CLASS(SoftmaxLayer);

}  // namespace caffe
