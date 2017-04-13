#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/cosine_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CosineLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    normalization_ = this->layer_param_.loss_param().normalization();
}

template <typename Dtype>
void CosineLossLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "Inputs must have the same dimension.";

  int axis_id = this->layer_param_.cosine_loss_param().axis();
  cosine_axis_ = bottom[0]->CanonicalAxisIndex(axis_id);
  outer_num_ = bottom[0]->count(0, cosine_axis_);
  inner_num_ = bottom[0]->count(cosine_axis_ + 1);

  vector<int> shape_vec;
  shape_vec.push_back(outer_num_);
  shape_vec.push_back(inner_num_);
  dots_.Reshape(shape_vec);
  len_inp_.Reshape(shape_vec);
  len_label_.Reshape(shape_vec);
}

template <typename Dtype>
void CosineLossLayer<Dtype>::Forward_cpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* inp_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* len_inp_data = len_inp_.mutable_cpu_data();
  Dtype* len_label_data = len_label_.mutable_cpu_data();
  Dtype* dots_data = dots_.mutable_cpu_data();
  const int channels = bottom[0]->shape(cosine_axis_);
  const int dim = bottom[0]->count() / outer_num_;

  // compute lengths and dot products for the current batch
  // as the loss is the mean over all vector angles, we can determine the
  // loss part of the individual vector comparison here also
  Dtype loss(0.0);
  Dtype len_inp, len_label, dot;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; ++j) {
      const Dtype* cur_inp_data = inp_data + (i*dim + j);
      const Dtype* cur_label_data = label_data + (i*dim + j);
      len_inp = caffe_cpu_strided_nrm2(channels, cur_inp_data, inner_num_);
      len_label = caffe_cpu_strided_nrm2(channels, cur_label_data, inner_num_);
      dot = caffe_cpu_strided_dot(channels,
                                  inp_data + (i*dim + j), inner_num_,
                                  label_data + (i*dim + j), inner_num_);

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
void CosineLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* inp_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  const Dtype* len_inp_data = len_inp_.cpu_data();
  const Dtype* len_label_data = len_label_.cpu_data();
  const Dtype* dots_data = dots_.cpu_data();
  const int channels = bottom[0]->shape(cosine_axis_);
  const int dim = bottom[0]->count() / outer_num_;
  Dtype normalizer = get_normalizer(normalization_, outer_num_*inner_num_);
  const Dtype scale = top[0]->cpu_diff()[0] / normalizer;

  Dtype* bottom_diff_0 = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_diff_1 = bottom[1]->mutable_cpu_diff();

  if (propagate_down[0]) {
    caffe_cpu_axpby(bottom[0]->count(),
                    Dtype(-1), label_data,
                    Dtype(0), bottom_diff_0);
  }
  if (propagate_down[1]) {
    caffe_cpu_axpby(bottom[1]->count(),
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
        caffe_cpu_strided_axpby(channels,
                                scale*s*c, cur_inp_data, inner_num_,
                                scale*s, cur_bottom_diff, inner_num_);
      }
      if (propagate_down[1]) {
        c = dot / (len_label * len_label);
        cur_bottom_diff = bottom_diff_1 + (i*dim + j);
        cur_label_data = label_data + (i*dim + j);
        caffe_cpu_strided_axpby(channels,
                                scale*s*c, cur_label_data, inner_num_,
                                scale*s, cur_bottom_diff, inner_num_);
      }
    }
  }
}

template <typename Dtype>
Dtype CosineLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

#ifdef CPU_ONLY
STUB_GPU(CosineLossLayer);
#endif

INSTANTIATE_CLASS(CosineLossLayer);
REGISTER_LAYER_CLASS(CosineLoss);

}  // namespace caffe
