#include <algorithm>
#include <vector>
#include "caffe/layer.hpp"
#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
namespace caffe {
template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // number of triplet in a batch
  int num_negatives = this->layer_param_.triplet_loss_param().num_negatives();
  // dimension of each descriptor
  int dim = bottom[0]->count()/bottom[0]->num();
  CHECK_EQ(bottom[0]->channels(), dim);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  // In each set, we have:
  // the descriptor of reference sample, closest sample, and negative samples
  // number of sets in the whole batch
  int num_set = bottom[0]->num()/(2 + num_negatives);
  dist_sq_.Reshape(num_set, 1, 1, 1);
  diff_pos.Reshape(num_set, dim, 1, 1);
  dist_sq_pos.Reshape(num_set, 1, 1, 1);
  diff_neg.Reshape(num_set, dim, 1, 1);
  dist_sq_neg.Reshape(num_set, 1, 1, 1);
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
}
template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  Dtype losstype = this->layer_param_.triplet_loss_param().losstype();
  int num_negatives = this->layer_param_.triplet_loss_param().num_negatives();
  int use_pair = this->layer_param_.triplet_loss_param().use_pair();
  CHECK_EQ(bottom[0]->num()%(2 + num_negatives), 0);
  Dtype loss(0.0);
  int dim = bottom[0]->count()/bottom[0]->num();
  int num_set = bottom[0]->num()/(2 + num_negatives);
  if (losstype == 0) {
    for (int i = 0; i < num_set; ++i) {
      caffe_sub(
                dim,
                bottom[0]->cpu_data() +
                (2 + num_negatives)*i*dim,  // reference
                bottom[0]->cpu_data() +
                ((2 + num_negatives)*i + 1)*dim,  // positive
                diff_pos.mutable_cpu_data() + i*dim);  // reference-pose_close
      // Loss component calculated from reference and close one
      dist_sq_pos.mutable_cpu_data()[i] =
      caffe_cpu_dot(dim,
                    diff_pos.cpu_data() + i*dim,
                    diff_pos.cpu_data() + i*dim);
      // a b is a similar pair for pair wise
      // loss accumulated by the pair wise part
      if (use_pair == 1) {
        loss += dist_sq_pos.cpu_data()[i];
      }
      for (int triplet = 0; triplet < num_negatives; ++triplet) {
        // Triplet loss accumulation
        // a and negative[triplet] is a similar pair for triplet
        dist_sq_.mutable_cpu_data()[i] = dist_sq_pos.cpu_data()[i];
        // Loss component calculated from negative part
        caffe_sub(
                  dim,
                  bottom[0]->cpu_data() +
                  (2 + num_negatives)*i*dim,  // reference
                  bottom[0]->cpu_data() +
                  ((2 + num_negatives)*i + 2 + triplet)*dim,
                  diff_neg.mutable_cpu_data() + i*dim);  // reference-negative
        dist_sq_neg.mutable_cpu_data()[i] =
        caffe_cpu_dot(dim,
                      diff_neg.cpu_data() + i*dim,
                      diff_neg.cpu_data() + i*dim);
        // a and negative[triplet] is a dissimilar pair for triplet
        dist_sq_.mutable_cpu_data()[i] -= dist_sq_neg.cpu_data()[i];
        // loss accumulated accumulated by the triplet part
        loss += std::max(margin + dist_sq_.cpu_data()[i], Dtype(0.0));
      }
    }
    loss = loss / static_cast<Dtype>(num_set) / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
  } else if (losstype == 1) {
    for (int i = 0; i < num_set; ++i) {
      caffe_sub(
                dim,
                bottom[0]->cpu_data() +
                (2 + num_negatives)*i*dim,  // reference
                bottom[0]->cpu_data() +
                ((2 + num_negatives)*i + 1)*dim,  // positive
                diff_pos.mutable_cpu_data() + i*dim);  // reference-pose_close
      // Loss component calculated from reference and close one
      dist_sq_pos.mutable_cpu_data()[i] =
      caffe_cpu_dot(dim,
                    diff_pos.cpu_data() + i*dim,
                    diff_pos.cpu_data() + i*dim);
      // a b is a similar pair for pair wise
      // loss accumulated by the pair wise part
      if (use_pair == 1) {
        loss += dist_sq_pos.cpu_data()[i];
      }
      for (int triplet = 0; triplet < num_negatives; ++triplet) {
        dist_sq_.mutable_cpu_data()[i] = dist_sq_pos.cpu_data()[i];
        dist_sq_.mutable_cpu_data()[i] += margin;
        // Loss component calculated from negative part
        caffe_sub(
                  dim,
                  bottom[0]->cpu_data() +
                  (2 + num_negatives)*i*dim,  // reference
                  bottom[0]->cpu_data() +
                  ((2 + num_negatives)*i + 2 + triplet)*dim,
                  diff_neg.mutable_cpu_data() + i*dim);  // reference-negative
        dist_sq_neg.mutable_cpu_data()[i] =
        caffe_cpu_dot(dim,
                      diff_neg.cpu_data() + i*dim,
                      diff_neg.cpu_data() + i*dim);
        // a and negative[triplet] is a dissimilar pair for triplet
        dist_sq_.mutable_cpu_data()[i] = 1 - \
        dist_sq_neg.cpu_data()[i] / dist_sq_.cpu_data()[i];
        // loss accumulated accumulated by the triplet part
        loss += std::max(dist_sq_.cpu_data()[i], Dtype(0.0));
      }
    }
    loss = loss / static_cast<Dtype>(num_set) / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
  } else if (losstype == 2) {
  for (int i = 0; i < num_set; ++i) {
    caffe_sub(
        dim,
        bottom[0]->cpu_data() +
        (2 + num_negatives)*i*dim,  // reference
        bottom[0]->cpu_data() +
        ((2 + num_negatives)*i + 1)*dim,  // positive
        diff_pos.mutable_cpu_data() + i*dim);  // reference-pose_close
    // Loss component calculated from reference and close one
    dist_sq_pos.mutable_cpu_data()[i] =
    caffe_cpu_dot(dim,
          diff_pos.cpu_data() + i*dim,
          diff_pos.cpu_data() + i*dim);
    // a b is a similar pair for pair wise
    // loss accumulated by the pair wise part
    if (use_pair == 1) {
    loss += dist_sq_pos.cpu_data()[i];
    }
    for (int triplet = 0; triplet < num_negatives; ++triplet) {
    dist_sq_.mutable_cpu_data()[i] = exp(dist_sq_pos.cpu_data()[i]);
    dist_sq_.mutable_cpu_data()[i] += margin;
    // Loss component calculated from negative part
    caffe_sub(
          dim,
          bottom[0]->cpu_data() +
          (2 + num_negatives)*i*dim,  // reference
          bottom[0]->cpu_data() +
          ((2 + num_negatives)*i + 2 + triplet)*dim,
          diff_neg.mutable_cpu_data() + i*dim);  // reference-negative
    dist_sq_neg.mutable_cpu_data()[i] =
    caffe_cpu_dot(dim,
            diff_neg.cpu_data() + i*dim,
            diff_neg.cpu_data() + i*dim);
    // a and negative[triplet] is a dissimilar pair for triplet
    dist_sq_.mutable_cpu_data()[i] = 1 - \
    exp(dist_sq_neg.cpu_data()[i]) / dist_sq_.cpu_data()[i];
    // loss accumulated accumulated by the triplet part
    loss += std::max(dist_sq_.cpu_data()[i], Dtype(0.0));
    }
  }
  loss = loss / static_cast<Dtype>(num_set) / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
  }
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  Dtype margin = this->layer_param_.triplet_loss_param().margin();
  Dtype losstype = this->layer_param_.triplet_loss_param().losstype();
  int num_negatives = this->layer_param_.triplet_loss_param().num_negatives();
  int use_pair = this->layer_param_.triplet_loss_param().use_pair();
  int dim = bottom[0]->count()/bottom[0]->num();
  int num_set = bottom[0]->num()/(2 + num_negatives);
  if (losstype == 0) {
    // BP for feat1(extracted from reference)
    for (int i = 0; i < 1; ++i) {
        if (propagate_down[0]) {
          const Dtype sign = 1;
          const Dtype alpha = sign * top[0]->cpu_diff()[0] /
          static_cast<Dtype>(num_set);
          for (int j = 0; j < num_set; ++j) {
            Dtype* bout = bottom[0]->mutable_cpu_diff();
            // the pair part
            if (use_pair == 1) {
              caffe_cpu_axpby(
          dim,
          alpha,
          diff_pos.cpu_data() + (j*dim),
          Dtype(0.0),
          bout + ((2 + num_negatives)*j + i)*dim);
            } else {
              caffe_cpu_axpby(
          dim,
          Dtype(0.0),
          diff_pos.cpu_data() + (j*dim),
          Dtype(0.0),
          bout + ((2 + num_negatives)*j + i)*dim);
            }
            // the num_negatives triplet part
            for (int triplet = 0; triplet < num_negatives; ++triplet) {
              caffe_sub(
                        dim,
                        bottom[0]->cpu_data() +
                        (2 + num_negatives)*j*dim,  // reference
                        bottom[0]->cpu_data() +
                        ((2 + num_negatives)*j + 2 + triplet)*dim,
                        diff_neg.mutable_cpu_data() +
                        j*dim);  // reference-negative
              // Triplet loss accumulation
              // a and negative[triplet] is a similar pair for triplet
              dist_sq_.mutable_cpu_data()[j] = dist_sq_pos.cpu_data()[j];
              dist_sq_neg.mutable_cpu_data()[j] =
              caffe_cpu_dot(dim,
                            diff_neg.cpu_data() + j*dim,
                            diff_neg.cpu_data() + j*dim);
              // a and negative[triplet] is a dissimilar pair for triplet
              dist_sq_.mutable_cpu_data()[j] -= dist_sq_neg.cpu_data()[j];
              // Loss component calculated from negative part
              if ((margin + dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
                // similar pair in triplet
                caffe_cpu_axpby(
          dim,
          alpha,
          diff_pos.cpu_data() + (j*dim),
          Dtype(1.0),
          bout + (2 + num_negatives)*j*dim);
                // dissimilar pair in triplet
                caffe_cpu_axpby(
          dim,
          -alpha,
          diff_neg.cpu_data() + (j*dim),
          Dtype(1.0),
          bout + ((2 + num_negatives)*j + i)*dim);
            }
          }
        }
      }
    }
    // BP for feat2(extracted from the closest sample)
    for (int i = 1; i < 2; ++i) {
      if (propagate_down[0]) {
        const Dtype sign = -1;
        const Dtype alpha = sign * top[0]->cpu_diff()[0] /
        static_cast<Dtype>(num_set);
        for (int j = 0; j < num_set; ++j) {
          Dtype* bout = bottom[0]->mutable_cpu_diff();
          // the pair part
          if (use_pair == 1) {
            caffe_cpu_axpby(
        dim,
        alpha,
        diff_pos.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
          } else {
            caffe_cpu_axpby(
        dim,
        Dtype(0.0),
        diff_pos.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
          }
          // the num_negatives triplet part
          for (int triplet = 0; triplet < num_negatives; ++triplet) {
            caffe_sub(
                      dim,
                      bottom[0]->cpu_data() +
                      (2 + num_negatives)*j*dim,  // reference
                      bottom[0]->cpu_data() +
                      ((2 + num_negatives)*j + 2 + triplet)*dim,
                      diff_neg.mutable_cpu_data() +
                      j*dim);  // reference-negative
            // Triplet loss accumulation
            // a and negative[triplet] is a similar pair for triplet
            dist_sq_.mutable_cpu_data()[j] = dist_sq_pos.cpu_data()[j];
            dist_sq_neg.mutable_cpu_data()[j] =
            caffe_cpu_dot(dim,
                          diff_neg.cpu_data() + j*dim,
                          diff_neg.cpu_data() + j*dim);
            // a and negative[triplet] is a dissimilar pair for triplet
            dist_sq_.mutable_cpu_data()[j] -= dist_sq_neg.cpu_data()[j];
            if ((margin + dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
              // similar pair in triplet
              caffe_cpu_axpby(
          dim,
          alpha,
          diff_pos.cpu_data() + (j*dim),
          Dtype(1.0),
          bout + ((2 + num_negatives)*j + i)*dim);
            }
          }
        }
      }
    }
    // BP for negative feature used in the num_negatives triplet part
    for (int i = 2; i < 2 + num_negatives; ++i) {
      if (propagate_down[0]) {
        const Dtype sign = 1;
        const Dtype alpha = sign * top[0]->cpu_diff()[0] /
        static_cast<Dtype>(num_set);
        for (int j = 0; j < num_set; ++j) {
          Dtype* bout = bottom[0]->mutable_cpu_diff();
          caffe_sub(
                    dim,
                    bottom[0]->cpu_data() +
                    (2 + num_negatives)*j*dim,  // reference
                    bottom[0]->cpu_data() +
                    ((2 + num_negatives)*j + i)*dim,
                    diff_neg.mutable_cpu_data() + j*dim);  // reference-negative
          // Triplet loss accumulation
          // a and negative[triplet] is a similar pair for triplet
          dist_sq_.mutable_cpu_data()[j] = dist_sq_pos.cpu_data()[j];
          dist_sq_neg.mutable_cpu_data()[j] =
          caffe_cpu_dot(dim,
                        diff_neg.cpu_data() + j*dim,
                        diff_neg.cpu_data() + j*dim);
          // a and negative[triplet] is a dissimilar pair for triplet
          dist_sq_.mutable_cpu_data()[j] -= dist_sq_neg.cpu_data()[j];
          if ((margin + dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
            // dissimilar pairs
            caffe_cpu_axpby(
        dim,
        alpha,
        diff_neg.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
          } else {
            caffe_set(dim, Dtype(0), bout + ((2 + num_negatives)*j + i)*dim);
          }
        }
      }
    }
  } else if (losstype == 1) {
    for (int i = 0; i < 1; ++i) {
      // BP for data1(feat1)
      if (propagate_down[0]) {
        const Dtype sign = 1;
        const Dtype alpha = sign * top[0]->cpu_diff()[0] /
        static_cast<Dtype>(num_set);
        for (int j = 0; j < num_set; ++j) {
          Dtype* bout = bottom[0]->mutable_cpu_diff();
          // the pair part
          if (use_pair == 1) {
            caffe_cpu_axpby(
        dim,
        alpha,
        diff_pos.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
          } else {
            caffe_cpu_axpby(
        dim,
        Dtype(0.0),
        diff_pos.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
          }
          // the num_negatives triplet part
          for (int triplet = 0; triplet < num_negatives; ++triplet) {
            dist_sq_.mutable_cpu_data()[j] = dist_sq_pos.mutable_cpu_data()[j];
            dist_sq_.mutable_cpu_data()[j] += margin;
            // Loss component calculated from negative part
            caffe_sub(
                      dim,
                      bottom[0]->cpu_data() +
                      (2 + num_negatives)*j*dim,  // reference
                      bottom[0]->cpu_data() +
                      ((2 + num_negatives)*j + 2 + triplet)*dim,
                      diff_neg.mutable_cpu_data() +
                      j*dim);  // reference-negative
            dist_sq_neg.mutable_cpu_data()[j] =
            caffe_cpu_dot(dim,
                          diff_neg.cpu_data() + j*dim,
                          diff_neg.cpu_data() + j*dim);
            // a and negative[triplet] is a dissimilar pair for triplet
            dist_sq_.mutable_cpu_data()[j] = 1 - \
            dist_sq_neg.cpu_data()[j] / dist_sq_.cpu_data()[j];
            // loss accumulated accumulated by the triplet part
            if ((dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
              caffe_cpu_axpby(
          dim,
          alpha*dist_sq_neg.cpu_data()[j]/
          ((dist_sq_pos.cpu_data()[j]+margin)*
           (dist_sq_pos.cpu_data()[j]+margin)),
          diff_pos.cpu_data() + (j*dim),
          Dtype(1.0),
          bout + ((2 + num_negatives)*j + i)*dim);
              caffe_cpu_axpby(
          dim,
          -alpha/(dist_sq_pos.mutable_cpu_data()[j]+margin),
          diff_neg.cpu_data() + (j*dim),
          Dtype(1.0),
          bout + ((2 + num_negatives)*j + i)*dim);
            }
          }
        }
      }
    }
    for (int i = 1; i < 2; ++i) {
      // BP for positive data(feat2)
      if (propagate_down[0]) {
        const Dtype sign = -1;
        const Dtype alpha = sign * top[0]->cpu_diff()[0] /
        static_cast<Dtype>(num_set);
        for (int j = 0; j < num_set; ++j) {
          Dtype* bout = bottom[0]->mutable_cpu_diff();
          // the pair part
          if (use_pair == 1) {
            caffe_cpu_axpby(
        dim,
        alpha,
        diff_pos.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
          } else {
            caffe_cpu_axpby(
        dim,
        Dtype(0.0),
        diff_pos.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
          }
          // the num_negatives triplet part
          for (int triplet = 0; triplet < num_negatives; ++triplet) {
            dist_sq_.mutable_cpu_data()[j] = dist_sq_pos.cpu_data()[j];
            dist_sq_.mutable_cpu_data()[j] += margin;
            // Loss component calculated from negative part
            caffe_sub(
                      dim,
                      bottom[0]->cpu_data() +
                      (2 + num_negatives)*j*dim,  // reference
                      bottom[0]->cpu_data() +
                      ((2 + num_negatives)*j + 2 + triplet)*dim,
                      diff_neg.mutable_cpu_data() +
                      j*dim);  // reference-negative
            dist_sq_neg.mutable_cpu_data()[j] =
            caffe_cpu_dot(dim,
                          diff_neg.cpu_data() + j*dim,
                          diff_neg.cpu_data() + j*dim);
            // a and negative[triplet] is a dissimilar pair for triplet
            dist_sq_.mutable_cpu_data()[j] = 1 - \
            dist_sq_neg.cpu_data()[j] / dist_sq_.cpu_data()[j];
            // loss accumulated accumulated by the triplet part
            if ((dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
              caffe_cpu_axpby(
          dim,
          alpha*dist_sq_neg.cpu_data()[j]/
            ((dist_sq_pos.cpu_data()[j]+margin)*
           (dist_sq_pos.cpu_data()[j]+margin)),
          diff_pos.cpu_data() + (j*dim),
          Dtype(1.0),
          bout + ((2 + num_negatives)*j + i)*dim);
            }
          }
        }
      }
    }
    for (int i = 2; i < 2 + num_negatives; ++i) {
      // BP for negative data(feat3)
      if (propagate_down[0]) {
        const Dtype sign = 1;
        const Dtype alpha = sign * top[0]->cpu_diff()[0] /
        static_cast<Dtype>(num_set);
        for (int j = 0; j < num_set; ++j) {
          Dtype* bout = bottom[0]->mutable_cpu_diff();
          dist_sq_.mutable_cpu_data()[j] = dist_sq_pos.cpu_data()[j];
          dist_sq_.mutable_cpu_data()[j] += margin;
          // Loss component calculated from negative part
          caffe_sub(
                    dim,
                    bottom[0]->cpu_data() + (2 + num_negatives)*j*dim,  // ref
                    bottom[0]->cpu_data() + ((2 + num_negatives)*j + i)*dim,
                    diff_neg.mutable_cpu_data() + j*dim);  // ref-negative
          dist_sq_neg.mutable_cpu_data()[j] =
          caffe_cpu_dot(dim,
                        diff_neg.cpu_data() + j*dim,
                        diff_neg.cpu_data() + j*dim);
          // a and negative[triplet] is a dissimilar pair for triplet
          dist_sq_.mutable_cpu_data()[j] = 1 - \
          dist_sq_neg.cpu_data()[j] / dist_sq_.cpu_data()[j];
          // loss accumulated accumulated by the triplet part
          if ((dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
            caffe_cpu_axpby(
        dim,
        alpha/(dist_sq_pos.cpu_data()[j] + margin),
        diff_neg.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
          } else {
            caffe_set(dim, Dtype(0), bout + ((2 + num_negatives)*j + i)*dim);
          }
        }
      }
    }
  } else if (losstype == 2) {
  for (int i = 0; i < 1; ++i) {
    // BP for data1(feat1)
    if (propagate_down[0]) {
    const Dtype sign = 1;
    const Dtype alpha = sign * top[0]->cpu_diff()[0] /
    static_cast<Dtype>(num_set);
    for (int j = 0; j < num_set; ++j) {
      Dtype* bout = bottom[0]->mutable_cpu_diff();
      // the pair part
      if (use_pair == 1) {
      caffe_cpu_axpby(
        dim,
        alpha,
        diff_pos.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
      } else {
      caffe_cpu_axpby(
        dim,
        Dtype(0.0),
        diff_pos.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
      }
      // the num_negatives triplet part
      for (int triplet = 0; triplet < num_negatives; ++triplet) {
      dist_sq_.mutable_cpu_data()[j] =
        exp(dist_sq_pos.cpu_data()[j]);
      dist_sq_.mutable_cpu_data()[j] += margin;
      // Loss component calculated from negative part
      caffe_sub(
        dim,
        bottom[0]->cpu_data()+(2 + num_negatives)*j*dim,  // reference
        bottom[0]->cpu_data()+((2 + num_negatives)*j + 2 + triplet)*dim,
          diff_neg.mutable_cpu_data() + j*dim);  // reference-negative
      dist_sq_neg.mutable_cpu_data()[j] =
      caffe_cpu_dot(dim,
        diff_neg.cpu_data() + j*dim,
        diff_neg.cpu_data() + j*dim);
      // a and negative[triplet] is a dissimilar pair for triplet
      dist_sq_.mutable_cpu_data()[j] = 1 - \
      exp(dist_sq_neg.cpu_data()[j]) / dist_sq_.cpu_data()[j];
      // loss accumulated accumulated by the triplet part
      if ((dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
        caffe_cpu_axpby(
          dim,
          alpha*
          Dtype(exp(dist_sq_neg.cpu_data()[j]))*
          Dtype(exp(dist_sq_pos.cpu_data()[j]))/
            (Dtype((exp(dist_sq_pos.cpu_data()[j]))+margin)*
              (Dtype(exp(dist_sq_pos.cpu_data()[j]))+margin)),
          diff_pos.cpu_data() + (j*dim),
          Dtype(1.0),
          bout + ((2 + num_negatives)*j + i)*dim);
        caffe_cpu_axpby(
          dim,
          -alpha*
          Dtype(exp(dist_sq_neg.cpu_data()[j]))/
          (Dtype(exp(dist_sq_pos.cpu_data()[j]))+margin),
          diff_neg.cpu_data() + (j*dim),
          Dtype(1.0),
          bout + ((2 + num_negatives)*j + i)*dim);
      }
      }
    }
    }
  }
  for (int i = 1; i < 2; ++i) {
    // BP for positive data(feat2)
    if (propagate_down[0]) {
    const Dtype sign = -1;
    const Dtype alpha = sign * top[0]->cpu_diff()[0] /
    static_cast<Dtype>(num_set);
    for (int j = 0; j < num_set; ++j) {
      Dtype* bout = bottom[0]->mutable_cpu_diff();
      // the pair part
      if (use_pair == 1) {
      caffe_cpu_axpby(
        dim,
        alpha,
        diff_pos.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
      } else {
      caffe_cpu_axpby(
        dim,
        Dtype(0.0),
        diff_pos.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
      }
      // the num_negatives triplet part
      for (int triplet = 0; triplet < num_negatives; ++triplet) {
      dist_sq_.mutable_cpu_data()[j] =
        exp(dist_sq_pos.cpu_data()[j]);
      dist_sq_.mutable_cpu_data()[j] += margin;
      // Loss component calculated from negative part
      caffe_sub(
        dim,
          bottom[0]->cpu_data()+(2+num_negatives)*j*dim,  // reference
          bottom[0]->cpu_data()+((2+num_negatives)*j+2+triplet)*dim,
        diff_neg.mutable_cpu_data()+j*dim);  // reference-negative
      dist_sq_neg.mutable_cpu_data()[j] =
      caffe_cpu_dot(dim,
        diff_neg.cpu_data() + j*dim,
          diff_neg.cpu_data() + j*dim);
      // a and negative[triplet] is a dissimilar pair for triplet
      dist_sq_.mutable_cpu_data()[j] = 1 - \
      exp(dist_sq_neg.cpu_data()[j]) / dist_sq_.cpu_data()[j];
      // loss accumulated accumulated by the triplet part
      if ((dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
        caffe_cpu_axpby(
          dim,
          alpha*
          Dtype(exp(dist_sq_neg.cpu_data()[j]))*
          Dtype(exp(dist_sq_pos.cpu_data()[j]))/
            ((Dtype(exp(dist_sq_pos.cpu_data()[j]))+margin)*
            (Dtype(exp(dist_sq_pos.cpu_data()[j]))+margin)),
          diff_pos.cpu_data() + (j*dim),
          Dtype(1.0),
          bout + ((2 + num_negatives)*j + i)*dim);
      }
      }
    }
    }
  }
  for (int i = 2; i < 2 + num_negatives; ++i) {
    // BP for negative data(feat3)
    if (propagate_down[0]) {
    const Dtype sign = 1;
    const Dtype alpha = sign * top[0]->cpu_diff()[0] /
    static_cast<Dtype>(num_set);
    for (int j = 0; j < num_set; ++j) {
      Dtype* bout = bottom[0]->mutable_cpu_diff();
      dist_sq_.mutable_cpu_data()[j] =
        exp(dist_sq_pos.cpu_data()[j]);
      dist_sq_.mutable_cpu_data()[j] += margin;
      // Loss component calculated from negative part
      caffe_sub(
        dim,
        bottom[0]->cpu_data() + (2 + num_negatives)*j*dim,  // ref
        bottom[0]->cpu_data() + ((2 + num_negatives)*j + i)*dim,
        diff_neg.mutable_cpu_data() + j*dim);  // ref-negative
      dist_sq_neg.mutable_cpu_data()[j] =
      caffe_cpu_dot(dim,
            diff_neg.cpu_data() + j*dim,
            diff_neg.cpu_data() + j*dim);
      // a and negative[triplet] is a dissimilar pair for triplet
      dist_sq_.mutable_cpu_data()[j] = 1 - \
      exp(dist_sq_neg.cpu_data()[j]) / dist_sq_.mutable_cpu_data()[j];
      // loss accumulated accumulated by the triplet part
      if ((dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
      caffe_cpu_axpby(
        dim,
        alpha*Dtype(exp(dist_sq_neg.cpu_data()[j]))/
          (Dtype(exp(dist_sq_pos.cpu_data()[j]))+margin),
        diff_neg.cpu_data() + (j*dim),
        Dtype(0.0),
        bout + ((2 + num_negatives)*j + i)*dim);
      } else {
      caffe_set(dim, Dtype(0), bout + ((2 + num_negatives)*j + i)*dim);
      }
    }
    }
  }
  }
}
#ifdef CPU_ONLY
    STUB_GPU(TripletLossLayer);
#endif
    INSTANTIATE_CLASS(TripletLossLayer);
    REGISTER_LAYER_CLASS(TripletLoss);
}  // namespace caffe
