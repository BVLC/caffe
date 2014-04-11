// Copyright 2014 kloudkl@github

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void WARPLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
                                 vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2)<<
  "WARPLossLayer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "WARPLossLayer takes no output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) <<
      "The two input blobs should have the same number.";
  int dim = bottom[0]->count() / bottom[0]->num();
  rank_weights_.Reshape(1, dim, 1, 1);
  Dtype* rank_weights_data = rank_weights_.mutable_cpu_data();
  rank_weights_data[0] = 1. / (1 + 0);
  for (int i = 1; i < dim; ++i) {
    rank_weights_data[i] = rank_weights_data[i - 1] + 1. / (1 + i);
  }
};

template<typename Dtype>
Dtype WARPLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_labels = bottom[1]->cpu_data();
  const Dtype* rank_weights_data = rank_weights_.cpu_data();
  const int num_data = bottom[0]->num();
  const int dim = bottom[0]->count() / bottom[0]->num();
  const int num_bottom_labels = dim;
  const int max_num_trials = num_bottom_labels - 1;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  memset(bottom_diff, 0, sizeof(Dtype) * bottom[0]->count());
  Dtype loss = 0;
  Dtype score_margin;
  int random_label;
  int estimated_rank;
  int num_trials;
  int offset;
  Dtype rank_weight;
  for (int i = 0; i < num_data; ++i) {
//    printf("i %d\n", i);
    offset = i * dim;
    for (int j = 0; j < num_bottom_labels; ++j) {
//      printf("\t j %d\n", j);
      if (bottom_labels[offset + j] == 1) {
        // Since the real rank is too costly to compute when the number of
        //   labels is large, bottom_labels j's rank is estimated based on
        //   the score margin violation which is defined as
        //   1 - bottom_data[j] + bottom_data[a_negative_label] > 0.
        num_trials = 0;
        do {
          do { // sample with replacement, TODO: without replacement
            // TODO: more precise uniform random int generator
            random_label = rand()
                % (num_bottom_labels - 1/* num bottom_labels except j */);
            if (random_label >= j) {
              ++random_label;  // shift one to skip j
            }
//            printf("\t random_label %d\n", random_label);
            // sample until a negative bottom_labels is found
          } while (bottom_labels[offset + random_label] == 1);
          ++num_trials;
//          printf("\t num_trials %d\n", num_trials);
          score_margin = 1 - bottom_data[offset + j] +
              bottom_data[offset + random_label];
        } while (score_margin <= 0 & num_trials < max_num_trials);
        estimated_rank = floor(max_num_trials / num_trials);
        rank_weight = rank_weights_data[estimated_rank];
//        LOG(ERROR)<< "rank_weight " << rank_weight;
        for (int k = 0; k < num_bottom_labels; ++k) {
//          printf("\t\t k %d\n", k);
          if (bottom_labels[offset + k] == 0) {
            score_margin = 1 - bottom_data[offset + j] + bottom_data[offset + k];
//            LOG(ERROR)<< "score margin " << score_margin <<
//                ", loss " << (loss + rank_weight * score_margin);
            if (score_margin > 0) {
              loss += rank_weight * score_margin;
              bottom_diff[offset + j] -= rank_weight;
              bottom_diff[offset + k] += rank_weight;
            }
          }
        }  // for (int k = 0; k < num_bottom_labels; ++k) {
      }  // if (bottom_labels[j] == 1) {
    }  // for (int j = 0; j < num_bottom_labels; ++j) {
  }  //  for (int i = 0; i < num_data; ++i) {
  caffe_scal(bottom[0]->count(), Dtype(1) / num_data, bottom_diff);
  return loss / num_data;
}

template<typename Dtype>
Dtype WARPLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       vector<Blob<Dtype>*>* top) {
  return Forward_cpu(bottom, top);
}

INSTANTIATE_CLASS(WARPLossLayer);

}  // namespace caffe
