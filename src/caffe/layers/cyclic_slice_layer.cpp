#include <vector>

#include "caffe/layers/cyclic_slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CyclicSliceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
    "allow in-place computation.";
  CHECK_EQ(bottom[0]->height(), bottom[0]->width()) <<
    "feature maps must be square";
}

template <typename Dtype>
void CyclicSliceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> shape = bottom[0]->shape();
  shape[0] *= 4;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void CyclicSliceLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->shape(0);
  const int channels = bottom[0]->shape(1);
  const int height = bottom[0]->shape(2);
  const int width = bottom[0]->shape(3);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype tmp;
  int bottom_outer_index, top_outer_index_0, top_outer_index_1,
    top_outer_index_2, top_outer_index_3,
    inner_index_a, inner_index_b, inner_index_c, inner_index_d;
  for (int i = 0; i < num; ++i) {
    for (int c = 0; c < channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          bottom_outer_index = (i*channels + c)*height*width;
          top_outer_index_0 = (4*i*channels + c)*height*width;
          top_outer_index_1 = ((4*i+1)*channels + c)*height*width;
          top_outer_index_2 = ((4*i+2)*channels + c)*height*width;
          top_outer_index_3 = ((4*i+3)*channels + c)*height*width;
          // a b c d counter clock wise
          inner_index_a = h*width + w;
          inner_index_b = w*width + (width-1-h);
          inner_index_c = (height-1-h)*width + (width-1-w);
          inner_index_d = (height-1-w)*width + h;
          // assign values
          tmp = bottom_data[bottom_outer_index+inner_index_a];
          top_data[top_outer_index_0+inner_index_a] = tmp;
          top_data[top_outer_index_1+inner_index_b] = tmp;
          top_data[top_outer_index_2+inner_index_c] = tmp;
          top_data[top_outer_index_3+inner_index_d] = tmp;
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(CyclicSliceLayer, Forward);
#endif

INSTANTIATE_CLASS(CyclicSliceLayer);
REGISTER_LAYER_CLASS(CyclicSlice);

}  // namespace caffe
