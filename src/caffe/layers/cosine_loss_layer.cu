#include <vector>

#include "caffe/layers/cosine_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CosineLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* inp_data = bottom[0]->gpu_data();
  const Dtype* label_data = bottom[1]->gpu_data();
  Dtype* len_inp_data = len_inp_.mutable_cpu_data();
  Dtype* len_label_data = len_label_.mutable_cpu_data();
  Dtype* dots_data = dots_.mutable_cpu_data();
  const int channels = bottom[0]->shape(cosine_axis_);
  const int dim = bottom[0]->count() / outer_num_;

  // compute lengths and dot products for the batch
  // as the loss is the mean over all vector angles, we can determine the
  // loss part of the current vector comparison here also
  Dtype loss(0.0);
  Dtype len_inp, len_label, dot;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const Dtype* cur_inp_data = inp_data + (i*dim + j);
      const Dtype* cur_label_data = label_data + (i*dim + j);
      caffe_gpu_strided_nrm2(channels, cur_inp_data, inner_num_, &len_inp);
      caffe_gpu_strided_nrm2(channels, cur_label_data, inner_num_, &len_label);
      caffe_gpu_strided_dot(channels,
                            cur_inp_data, inner_num_,
                            cur_label_data, inner_num_,
                            &dot);
      len_inp_data[i*inner_num_ + j] = len_inp;
      len_label_data[i*inner_num_ + j] = len_label;
      dots_data[i*inner_num_ + j] = dot;
      loss += 1 - dot / (len_inp*len_label);
    }
  }

  Dtype normalizer = get_normalizer(normalization_, outer_num_*inner_num_);
  top[0]->mutable_cpu_data()[0] = loss / normalizer;
}

template <typename Dtype>
void CosineLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* inp_data = bottom[0]->gpu_data();
  const Dtype* label_data = bottom[1]->gpu_data();
  const Dtype* len_inp_data = len_inp_.cpu_data();
  const Dtype* len_label_data = len_label_.cpu_data();
  const Dtype* dots_data = dots_.cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int channels = bottom[0]->shape(cosine_axis_);
  const int dim = bottom[0]->count() / outer_num_;
  Dtype normalizer = get_normalizer(normalization_, outer_num_*inner_num_);
  const Dtype scale = top[0]->cpu_diff()[0] / normalizer;

  Dtype* bottom_diff_0 = bottom[0]->mutable_gpu_diff();
  Dtype* bottom_diff_1 = bottom[1]->mutable_gpu_diff();

  if (propagate_down[0]) {
    caffe_gpu_axpby(bottom[0]->count(),
                    Dtype(-1), label_data,
                    Dtype(0), bottom_diff_0);
  }
  if (propagate_down[1]) {
    caffe_gpu_axpby(bottom[1]->count(),
                    Dtype(-1), inp_data,
                    Dtype(0), bottom_diff_1);
  }

  Dtype s, c, len_inp, len_label, dot;
  const Dtype* cur_inp_data;
  const Dtype* cur_label_data;
  Dtype* cur_bottom_diff;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      len_inp = len_inp_data[i*inner_num_ + j];
      len_label = len_label_data[i*inner_num_ + j];
      dot = dots_data[i*inner_num_ + j];
      s = Dtype(1) / (len_inp * len_label);

      if (propagate_down[0]) {
        c = dot / (len_inp * len_inp);
        cur_bottom_diff = bottom_diff_0 + (i*dim + j);
        cur_inp_data = inp_data + (i*dim + j);
        caffe_gpu_strided_axpby(channels,
                                scale*s*c, cur_inp_data, inner_num_,
                                scale*s, cur_bottom_diff, inner_num_);
      }
      if (propagate_down[1]) {
        c = dot / (len_label * len_label);
        cur_bottom_diff = bottom_diff_1 + (i*dim + j);
        cur_label_data = label_data + (i*dim + j);
        caffe_gpu_strided_axpby(channels,
                                scale*s*c, cur_label_data, inner_num_,
                                scale*s, cur_bottom_diff, inner_num_);
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CosineLossLayer);

}  // namespace caffe
