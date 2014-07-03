// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

using std::max;

template <typename Dtype>
Dtype TripletRankingHingeLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* query_data = bottom[0]->cpu_data();
  const Dtype* similar_sample_data = bottom[1]->cpu_data();
  const Dtype* dissimilar_sample_data = bottom[2]->cpu_data();
  Dtype* similar_sample_diff = bottom[1]->mutable_cpu_diff();
  Dtype* dissimilar_sample_diff = bottom[2]->mutable_cpu_diff();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;
  caffe_sub(count, query_data, similar_sample_data,
            similar_sample_diff);
  caffe_sub(count, query_data, dissimilar_sample_data,
            dissimilar_sample_diff);

  Dtype loss = 0;
  Dtype query_similar_distance_norm;
  Dtype query_dissimilar_distance_norm;
  switch (this->layer_param_.triplet_ranking_hinge_loss_param().norm()) {
  case TripletRankingHingeLossParameter_Norm_L1: {
    for (int i = 0; i < num; ++i) {
      query_similar_distance_norm = caffe_cpu_asum(
          dim, similar_sample_diff + bottom[1]->offset(i));
      query_dissimilar_distance_norm = caffe_cpu_asum(
          dim, dissimilar_sample_diff + bottom[2]->offset(i));
      loss += max(Dtype(0), query_similar_distance_norm -
                  query_dissimilar_distance_norm + 1);
    }
    break;
  }
  case TripletRankingHingeLossParameter_Norm_L2: {
    for (int i = 0; i < num; ++i) {
      query_similar_distance_norm = caffe_cpu_dot(
          dim, similar_sample_diff + bottom[1]->offset(i),
          similar_sample_diff + bottom[1]->offset(i));
      query_dissimilar_distance_norm = caffe_cpu_dot(
          dim, dissimilar_sample_diff + bottom[2]->offset(i),
          dissimilar_sample_diff + bottom[2]->offset(i));
      loss += max(Dtype(0), query_similar_distance_norm -
                  query_dissimilar_distance_norm + 1);
    }
    break;
  }
  default: {
    LOG(FATAL) << "Unknown TripletRankingHingeLoss norm " <<
        this->layer_param_.triplet_ranking_hinge_loss_param().norm();
  }
  }
  return loss / num;
}

template <typename Dtype>
void TripletRankingHingeLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
//    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* similar_sample_diff = (*bottom)[1]->mutable_cpu_diff();
    Dtype* dissimilar_sample_diff = (*bottom)[2]->mutable_cpu_diff();

    int num = (*bottom)[0]->num();
    int count = (*bottom)[0]->count();

    switch (this->layer_param_.triplet_ranking_hinge_loss_param().norm()) {
    case TripletRankingHingeLossParameter_Norm_L1: {
      caffe_cpu_sign(count, similar_sample_diff, similar_sample_diff);
      caffe_scal(count, Dtype(-1. / num), similar_sample_diff);
      caffe_cpu_sign(count, dissimilar_sample_diff, dissimilar_sample_diff);
      caffe_scal(count, Dtype(1. / num), similar_sample_diff);
      break;
    }
    case TripletRankingHingeLossParameter_Norm_L2: {
      caffe_scal(count, Dtype(-2. / num), similar_sample_diff);
      caffe_scal(count, Dtype(2. / num), dissimilar_sample_diff);
      break;
    }
    default: {
      LOG(FATAL) << "Unknown TripletRankingHingeLoss norm " <<
          this->layer_param_.triplet_ranking_hinge_loss_param().norm();
    }
    }
  }
}

INSTANTIATE_CLASS(TripletRankingHingeLossLayer);

}  // namespace caffe
