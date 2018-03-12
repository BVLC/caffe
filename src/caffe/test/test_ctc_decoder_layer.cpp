#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/ctc_decoder_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class CTCDecoderLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CTCDecoderLayerTest()
      : T_(4),
        N_(2),
        num_labels_(3),
        blob_bottom_data_(new Blob<Dtype>(T_, N_, num_labels_, 1)),
        blob_bottom_seq_ind_(new Blob<Dtype>(T_, N_, 1, 1)),
        blob_bottom_target_seq_(new Blob<Dtype>(T_, N_, 1, 1)),
        blob_top_sequences_(new Blob<Dtype>()),
        blob_top_scores_(new Blob<Dtype>()),
        blob_top_accuracy_(new Blob<Dtype>()) {
    // Add blobs to the correct bottom/top lists
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_seq_ind_);
    blob_top_vec_.push_back(blob_top_sequences_);
    blob_top_vec_.push_back(blob_top_scores_);
  }

  virtual ~CTCDecoderLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_seq_ind_;
    delete blob_top_scores_;
  }

  void Reshape(int t, int n, int c) {
    T_ = t;
    N_ = n;
    num_labels_ = c;
    blob_bottom_data_->Reshape(T_, N_, num_labels_, 1);
    blob_bottom_seq_ind_->Reshape(T_, N_, 1, 1);
    blob_bottom_target_seq_->Reshape(T_, N_, 1, 1);
  }

  void AddAccuracyOutput() {
    blob_bottom_vec_.push_back(blob_bottom_target_seq_);
    blob_top_vec_.push_back(blob_top_accuracy_);
  }

  template <class T>
  vector<T> raw(const T data[], int size) {
    vector<T> o(size);
    caffe_copy(size, data, o.data());
    return o;
  }

  vector<Dtype> log(const vector<Dtype> &in) {
    vector<Dtype> o(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
      o[i] = std::log(in[i]);
    }
    return o;
  }

  Dtype sum(const vector<Dtype> &in) {
    return std::accumulate(in.begin(), in.end(), static_cast<Dtype>(0));
  }

  vector<Dtype> neg(const vector<Dtype> &in) {
    vector<Dtype> o(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
      o[i] = -in[i];
    }
    return o;
  }

  void TestGreedyDecoder(bool check_accuracy = true) {
    if (check_accuracy) {
      AddAccuracyOutput();
    }

    // Test two batch entries - best path decoder.
    const int max_time_steps = 6;
    const int depth = 4;

    // const int seq_len_0 = 4;
    const Dtype input_prob_matrix_0[max_time_steps * depth] =
      {1.0, 0.0, 0.0, 0.0,  // t=0
       0.0, 0.0, 0.4, 0.6,  // t=1
       0.0, 0.0, 0.4, 0.6,  // t=2
       0.0, 0.9, 0.1, 0.0,  // t=3
       0.0, 0.0, 0.0, 0.0,  // t=4 (ignored)
       0.0, 0.0, 0.0, 0.0   // t=5 (ignored)
      };

    const vector<Dtype> input_log_prob_matrix_0(
                log(raw(input_prob_matrix_0, max_time_steps * depth)));
    const Dtype prob_truth_0[depth] = {1.0, 0.6, 0.6, 0.9};
    const int label_len_0 = 2;
    const int correct_sequence_0[label_len_0] = {0, 1};

    // const int seq_len_1 = 5;
    const Dtype input_prob_matrix_1[max_time_steps * depth] =
      {0.1, 0.9, 0.0, 0.0,  // t=0
       0.0, 0.9, 0.1, 0.0,  // t=1
       0.0, 0.0, 0.1, 0.9,  // t=2
       0.0, 0.9, 0.1, 0.1,  // t=3
       0.9, 0.1, 0.0, 0.0,  // t=4
       0.0, 0.0, 0.0, 0.0   // t=5 (ignored)
      };

    const vector<Dtype> input_log_prob_matrix_1(
                log(raw(input_prob_matrix_1, max_time_steps * depth)));
    const Dtype prob_truth_1[depth] = {0.9, 0.9, 0.9, 0.9};
    const int label_len_1 = 3;
    const int correct_sequence_1[label_len_1] = {1, 1, 0};

    const Dtype log_prob_truth[2] = {
        sum(neg(log(raw(prob_truth_0, depth)))),
        sum(neg(log(raw(prob_truth_1, depth))))
    };

    Reshape(max_time_steps, 2, depth);

    // copy data
    Dtype* data = blob_bottom_data_->mutable_cpu_data();
    for (int t = 0; t < max_time_steps; ++t) {
      for (int c = 0; c < depth; ++c) {
        data[blob_bottom_data_->offset(t, 0, c)]
              = input_log_prob_matrix_0[t * depth + c];
        data[blob_bottom_data_->offset(t, 1, c)]
              = input_log_prob_matrix_1[t * depth + c];
      }
    }

    // set sequence indicators
    Dtype* seq_ind = blob_bottom_seq_ind_->mutable_cpu_data();
    caffe_set(blob_bottom_seq_ind_->count(), static_cast<Dtype>(1), seq_ind);
    // sequence 1:
    seq_ind[0 * 2 + 0] = seq_ind[4 * 2 + 0] = seq_ind[5 * 2 + 0] = 0;
    // sequence 2;
    seq_ind[0 * 2 + 1] = seq_ind[5 * 2 + 1] = 0;

    // target sequences
    if (check_accuracy) {
      Dtype* target_seq = blob_bottom_target_seq_->mutable_cpu_data();
      caffe_set(blob_bottom_target_seq_->count(),
                static_cast<Dtype>(-1), target_seq);
      for (int i = 0; i < label_len_0; ++i) {
        target_seq[blob_bottom_target_seq_->offset(i, 0)]
                = correct_sequence_0[i];
      }

      for (int i = 0; i < label_len_1; ++i) {
        target_seq[blob_bottom_target_seq_->offset(i, 1)]
                = correct_sequence_1[i];
      }
    }

    LayerParameter layer_param;
    CTCGreedyDecoderLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    layer.Forward(blob_bottom_vec_, blob_top_vec_);

    CHECK_EQ(layer.OutputSequences().size(), 2);

    const Dtype* scores = blob_top_scores_->cpu_data();

    // check n=0
    EXPECT_EQ(label_len_0, layer.OutputSequences()[0].size());
    for (int i = 0; i < label_len_0; ++i) {
        EXPECT_EQ(correct_sequence_0[i], layer.OutputSequences()[0][i]);
    }
    EXPECT_FLOAT_EQ(scores[0], log_prob_truth[0]);

    // check n=1
    EXPECT_EQ(label_len_1, layer.OutputSequences()[1].size());
    for (int i = 0; i < label_len_1; ++i) {
        EXPECT_EQ(correct_sequence_1[i], layer.OutputSequences()[1][i]);
    }
    EXPECT_FLOAT_EQ(scores[0], log_prob_truth[0]);

    if (check_accuracy) {
      const Dtype *acc = blob_top_accuracy_->cpu_data();
      // output must have a edit distance of 0,
      // the accuracy must therefore be 100%
      EXPECT_FLOAT_EQ(acc[0], 1);
    }
  }

  int T_;
  int N_;
  int num_labels_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_seq_ind_;
  Blob<Dtype>* const blob_bottom_target_seq_;
  Blob<Dtype>* const blob_top_sequences_;
  Blob<Dtype>* const blob_top_scores_;
  Blob<Dtype>* const blob_top_accuracy_;

  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CTCDecoderLayerTest, TestDtypesAndDevices);

TYPED_TEST(CTCDecoderLayerTest, TestGreedyDecoder) {
  this->TestGreedyDecoder(true);    // with acc test
  this->TestGreedyDecoder(false);   // without acc test
}


}  // namespace caffe
