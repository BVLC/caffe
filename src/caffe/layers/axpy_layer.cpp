#include "caffe/layers/axpy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AxpyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Reshape_const(bottom,top);
}

template <typename Dtype>
void AxpyLayer<Dtype>::Reshape_const(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) const {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1));
  if (bottom[0]->num_axes() == 4) {
    CHECK_EQ(bottom[0]->shape(2), 1);
    CHECK_EQ(bottom[0]->shape(3), 1);
  }
  CHECK(bottom[1]->shape() == bottom[2]->shape());  
  top[0]->ReshapeLike(*bottom[1]);
}

template <typename Dtype>
void AxpyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
    const vector<Blob<Dtype>*>& top) { 
  Forward_const_cpu(bottom,top);
}

template <typename Dtype>
void AxpyLayer<Dtype>::Forward_const_cpu(const vector<Blob<Dtype>*>& bottom,  
    const vector<Blob<Dtype>*>& top) const { 
  int channel_dim = bottom[1]->channels();
  int spatial_dim = bottom[1]->count(2);
  const Dtype* scale_data = bottom[0]->cpu_data();
  const Dtype* x_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_copy(bottom[2]->count(), bottom[2]->cpu_data(), top_data);
  for (int n = 0; n < bottom[1]->num(); ++n) {
    for (int c = 0; c < channel_dim; ++c) {
      int scale_offset = n * channel_dim + c;
      caffe_axpy(spatial_dim, scale_data[scale_offset], 
          x_data + scale_offset * spatial_dim, 
          top_data + scale_offset * spatial_dim);  
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(AxpyLayer);
#endif

INSTANTIATE_CLASS(AxpyLayer);
REGISTER_LAYER_CLASS(Axpy);

} // namespace
