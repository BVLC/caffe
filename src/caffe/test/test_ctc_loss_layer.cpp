#include <cmath>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/ctc_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CTCLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CTCLossLayerTest()
      : T_(4),
        N_(2),
        num_labels_(3),
        blob_bottom_data_(new Blob<Dtype>(T_, N_, num_labels_, 1)),
        blob_bottom_label_(new Blob<Dtype>(T_, N_, 1, 1)),
        blob_bottom_seq_ind_(new Blob<Dtype>(T_, N_, 1, 1)),
        blob_top_loss_(new Blob<Dtype>()) {
    // Add blobs to the correct bottom/top lists
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_seq_ind_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }

  virtual ~CTCLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_bottom_seq_ind_;
    delete blob_top_loss_;
  }

  void Reshape(int t, int n, int c) {
    T_ = t;
    N_ = n;
    num_labels_ = c;
    blob_bottom_data_->Reshape(T_, N_, num_labels_, 1);
    blob_bottom_label_->Reshape(T_, N_, 1, 1);
    blob_bottom_seq_ind_->Reshape(T_, N_, 1, 1);
  }

  void InitConstantCorrect() {
    // fill the values with constant values
    // The prediction is 100% correct so that the loss
    // must be 0

    FillerParameter filler_c1_param;
    filler_c1_param.set_value(1);
    ConstantFiller<Dtype> c1_filler(filler_c1_param);
    c1_filler.Fill(blob_bottom_seq_ind_);

    FillerParameter filler_c0_param;
    filler_c0_param.set_value(0);
    ConstantFiller<Dtype> c0_filler(filler_c0_param);
    c0_filler.Fill(blob_bottom_data_);

    FillerParameter filler_cn1_param;
    filler_cn1_param.set_value(-1);
    ConstantFiller<Dtype> cn1_filler(filler_cn1_param);
    cn1_filler.Fill(blob_bottom_label_);

    // sequence start (full size)
    for (int b = 0; b < N_; ++b) {
        blob_bottom_seq_ind_->mutable_cpu_data()[b] = 0;
    }

    const Dtype one = std::numeric_limits<Dtype>::max();

    // set label
    Dtype *label = blob_bottom_label_->mutable_cpu_data();
    label[blob_bottom_label_->offset(0, 0)] = 0;
    label[blob_bottom_label_->offset(1, 0)] = 1;

    label[blob_bottom_label_->offset(0, 1)] = 0;
    label[blob_bottom_label_->offset(1, 1)] = 1;

    // set probabilities
    // (100% correct, but second with one additional timestep)
    Dtype *data = blob_bottom_data_->mutable_cpu_data();
    data[blob_bottom_data_->offset(0, 0, 0)] = one;
    data[blob_bottom_data_->offset(1, 0, 1)] = one;
    data[blob_bottom_data_->offset(2, 0, num_labels_ - 1)] = one;
    data[blob_bottom_data_->offset(3, 0, num_labels_ - 1)] = one;

    data[blob_bottom_data_->offset(0, 1, 0)] = one;
    data[blob_bottom_data_->offset(1, 1, num_labels_ - 1)] = one;
    data[blob_bottom_data_->offset(2, 1, 1)] = one;
    data[blob_bottom_data_->offset(3, 1, num_labels_ - 1)] = one;


    // Check data for consistency
    CheckData();
  }

  void InitConstantWrong() {
    // fill the values with constant values
    // The prediction is wrong in one label

    FillerParameter filler_c1_param;
    filler_c1_param.set_value(1);
    ConstantFiller<Dtype> c1_filler(filler_c1_param);
    c1_filler.Fill(blob_bottom_seq_ind_);

    FillerParameter filler_c0_param;
    filler_c0_param.set_value(0);
    ConstantFiller<Dtype> c0_filler(filler_c0_param);
    c0_filler.Fill(blob_bottom_data_);

    FillerParameter filler_cn1_param;
    filler_cn1_param.set_value(-1);
    ConstantFiller<Dtype> cn1_filler(filler_cn1_param);
    cn1_filler.Fill(blob_bottom_label_);

    // sequence start (full size)
    for (int b = 0; b < N_; ++b) {
        blob_bottom_seq_ind_->mutable_cpu_data()[b] = 0;
    }

    const Dtype one = std::numeric_limits<Dtype>::max();

    // set label
    Dtype *label = blob_bottom_label_->mutable_cpu_data();
    label[blob_bottom_label_->offset(0, 0)] = 0;
    label[blob_bottom_label_->offset(1, 0)] = 1;

    label[blob_bottom_label_->offset(0, 1)] = 0;
    label[blob_bottom_label_->offset(1, 1)] = 1;

    // set probabilities
    // (flipped predictions compared to correct, last row 1<->0)
    Dtype *data = blob_bottom_data_->mutable_cpu_data();
    data[blob_bottom_data_->offset(0, 0, num_labels_ - 1)] = one;
    data[blob_bottom_data_->offset(1, 0, num_labels_ - 1)] = one;
    data[blob_bottom_data_->offset(2, 0, num_labels_ - 1)] = one;
    data[blob_bottom_data_->offset(3, 0, num_labels_ - 1)] = one;

    data[blob_bottom_data_->offset(0, 1, num_labels_ - 1)] = one;
    data[blob_bottom_data_->offset(1, 1, num_labels_ - 1)] = one;
    data[blob_bottom_data_->offset(2, 1, num_labels_ - 1)] = one;
    data[blob_bottom_data_->offset(3, 1, num_labels_ - 1)] = one;


    // Check data for consistency
    CheckData();
  }

  void InitConstantEqual(int sequence_length, int T = 5, int C = 10) {
    CHECK_LE(sequence_length, T);
    // fill the values with constant values
    // The prediction is wrong is equal in each position (everything is 0)

    Reshape(T, 1, C);

    FillerParameter filler_c1_param;
    filler_c1_param.set_value(1);
    ConstantFiller<Dtype> c1_filler(filler_c1_param);
    c1_filler.Fill(blob_bottom_seq_ind_);

    FillerParameter filler_c0_param;
    filler_c0_param.set_value(0);
    ConstantFiller<Dtype> c0_filler(filler_c0_param);
    c0_filler.Fill(blob_bottom_data_);

    FillerParameter filler_cn1_param;
    filler_cn1_param.set_value(-1);
    ConstantFiller<Dtype> cn1_filler(filler_cn1_param);
    cn1_filler.Fill(blob_bottom_label_);

    // sequence start (full size)
    for (int b = 0; b < N_; ++b) {
        blob_bottom_seq_ind_->mutable_cpu_data()[b] = 0;
    }

    const Dtype one = std::numeric_limits<Dtype>::max();

    // set label
    Dtype *label = blob_bottom_label_->mutable_cpu_data();
    int label_to_set = 0;
    for (int t = 0; t < sequence_length; ++t) {
      label[blob_bottom_label_->offset(t, 0)] = label_to_set;
      label_to_set = (label_to_set + 1) % (num_labels_ - 1);
    }

    // Check data for consistency
    CheckData();
  }

  void InitRandom() {
    // This will create random data and random labels for all blobs.

    // increase T_, N_, num_labels_
    Reshape(41, 29, 37);

    // Fill the data
    // =============================================================

    FillerParameter gfp;
    gfp.set_std(1);
    GaussianFiller<Dtype> gf(gfp);
    // random data
    gf.Fill(blob_bottom_data_);

    // Fill the sequence indicators
    // ==============================================================

    // arbitrary sequence length, at least 1
    vector<Dtype> seq_lengths(N_);
    caffe_rng_uniform<Dtype>(N_, 1.0, T_ - 0.001, seq_lengths.data());

    // 1. Fill with 1
    FillerParameter filler_c1_param;
    filler_c1_param.set_value(1);
    ConstantFiller<Dtype> c1_filler(filler_c1_param);
    c1_filler.Fill(blob_bottom_seq_ind_);

    // 2. sequence start (always at beginning = 0)
    // 3. sequence length (all further = 0)
    Dtype* seq_data = blob_bottom_seq_ind_->mutable_cpu_data();
    for (int b = 0; b < N_; ++b) {
      seq_data[b] = 0;
      for (int t = static_cast<int>(seq_lengths[b]); t < T_; ++t) {
        seq_data[blob_bottom_seq_ind_->offset(t, b)] = 0;
      }
    }

    // Fill the labels
    // ==============================================================

    // arbitrary labels
    FillerParameter lufp;
    lufp.set_min(0);
    lufp.set_max(num_labels_ - 1.0001);  // note that last label is blank
    UniformFiller<Dtype> luf(lufp);
    luf.Fill(blob_bottom_label_);

    // loop through all elements and set to integer values (e.g. 4.25 to 4)
    Dtype* label = blob_bottom_label_->mutable_cpu_data();
    const Dtype* end_label = label + blob_bottom_label_->count();
    for (; label < end_label; ++label) {
        *label = static_cast<Dtype>(static_cast<int>(*label));
    }

    // loop though all elements and set the label length to
    // 1 <= label length <= sequence length
    label = blob_bottom_label_->mutable_cpu_data();
    for (int n = 0; n < N_; ++n) {
      const int seq_len = static_cast<int>(seq_lengths[n]);
      const int label_len = caffe_rng_rand() % (seq_len) + 1;
      CHECK_LE(label_len, seq_len);
      CHECK_GE(label_len, 0);

      for (int t = label_len; t < T_; ++t) {
        label[blob_bottom_label_->offset(t, n)] = -1;
      }
    }

    // Check data for consistency
    // ==============================================================
    CheckData();
  }

  void CheckData() {
    // check label_length <= sequence length
    const Dtype* label_data = blob_bottom_label_->cpu_data();
    const Dtype* seq_data = blob_bottom_seq_ind_->cpu_data();
    for (int n = 0; n < N_; ++n) {
      Dtype seq_len = -1;
      Dtype lab_len = -1;
      for (int t = 0; t < T_; ++t) {
        const Dtype lab = label_data[blob_bottom_label_->offset(t, n)];

        // expect all following labels to be negative (not filled)
        if (lab_len >= 0.0) {
          EXPECT_LT(lab, 0.0);
        } else {
          // if first not filled label appears we know the label length
          if (lab < 0.0) {
            lab_len = t;
          }
        }
      }

      for (int t = 1; t < T_; ++t) {
        const Dtype seq
              = seq_data[blob_bottom_seq_ind_->offset(t, n)];

        // expect all following sequence indicators to be 0.0, in our test case
        if (seq_len >= 0.0) {
          EXPECT_DOUBLE_EQ(seq, 0.0);
        } else {
          // if another 0 appears we know the sequence length
          if (seq == 0.0) {
            seq_len = t;
          }
        }
      }

      // check if no end indicator was found, therefore the complete T_ is used
      if (lab_len < 0.0) {
        lab_len = T_;
      }
      if (seq_len < 0.0) {
        seq_len = T_;
      }

      EXPECT_GE(seq_len, 0);
      EXPECT_GE(lab_len, 0);
      EXPECT_LE(lab_len, seq_len);
    }
  }

  void TestForward() {
    InitConstantCorrect();  // constant and correct data (loss must be 0)

    LayerParameter layer_param;
    CTCLossLayer<Dtype> layer_1(layer_param);
    layer_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const Dtype loss = blob_top_loss_->cpu_data()[0];
    EXPECT_FLOAT_EQ(loss, 0.0);

    EXPECT_EQ(layer_1.SequenceLength().num(), N_);
    EXPECT_EQ(layer_1.SequenceLength().count(), N_);
    EXPECT_EQ(layer_1.SequenceLength().cpu_data()[0], 4);
    EXPECT_EQ(layer_1.SequenceLength().cpu_data()[1], 4);

    EXPECT_EQ(layer_1.LabelLength().num(), N_);
    EXPECT_EQ(layer_1.LabelLength().count(), N_);
    EXPECT_EQ(layer_1.LabelLength().cpu_data()[0], 2);
    EXPECT_EQ(layer_1.LabelLength().cpu_data()[1], 2);

    // check loss for all other t
    // (to check Graves Eq. (7.27) that holds for all t)
    for (int t = 1; t < T_; ++t) {
        layer_1.SetLossCalculationT(t);
        layer_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

        const Dtype loss_t = blob_top_loss_->cpu_data()[0];
        EXPECT_FLOAT_EQ(loss, loss_t);
    }

    // The gradients (deltas on all variables) must be 0 in this special case
    EXPECT_FLOAT_EQ(blob_bottom_data_->asum_diff(), 0);
  }

  void TestForwardWrong() {
    InitConstantWrong();
    LayerParameter layer_param;

    CTCLossLayer<Dtype> layer_1(layer_param);
    layer_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const Dtype loss = blob_top_loss_->cpu_data()[0];
    EXPECT_FLOAT_EQ(loss, std::numeric_limits<Dtype>::max());

    EXPECT_EQ(layer_1.SequenceLength().num(), N_);
    EXPECT_EQ(layer_1.SequenceLength().count(), N_);
    EXPECT_EQ(layer_1.SequenceLength().cpu_data()[0], 4);
    EXPECT_EQ(layer_1.SequenceLength().cpu_data()[1], 4);

    EXPECT_EQ(layer_1.LabelLength().num(), N_);
    EXPECT_EQ(layer_1.LabelLength().count(), N_);
    EXPECT_EQ(layer_1.LabelLength().cpu_data()[0], 2);
    EXPECT_EQ(layer_1.LabelLength().cpu_data()[1], 2);

    // check loss for all other t
    // (to check Graves Eq. (7.27) that holds for all t)
    for (int t = 1; t < T_; ++t) {
        layer_1.SetLossCalculationT(t);
        layer_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

        const Dtype loss_t = blob_top_loss_->cpu_data()[0];
        EXPECT_FLOAT_EQ(loss, loss_t);
    }


    vector<bool> prob_down(0, false);
    prob_down[0] = true;
    layer_1.Backward(this->blob_top_vec_, prob_down, this->blob_bottom_vec_);

    // expected output of gradient is softmax of input
    for (int i = 0; i < blob_bottom_data_->count(); ++i) {
        const Dtype d = blob_bottom_data_->cpu_data()[i];
        const Dtype dd = blob_bottom_data_->cpu_diff()[i];
        if (d == std::numeric_limits<Dtype>::max()) {
          EXPECT_FLOAT_EQ(dd, 1);
        } else {
          EXPECT_FLOAT_EQ(dd, 0);
        }
    }
  }

  // Returns value of Binomial Coefficient C(n, k)
  int binomialCoeff(int n, int k) {
    int res = 1;

    // Since C(n, k) = C(n, n-k)
    if ( k > n - k )
        k = n - k;

    // Calculate value of [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
    for (int i = 0; i < k; ++i) {
        res *= (n - i);
        res /= (i + 1);
    }

    return res;
  }

  void TestForwardEqual() {
    int sequence_length = 4;
    InitConstantEqual(sequence_length, 20, 5);
    LayerParameter layer_param;

    CTCLossLayer<Dtype> layer_1(layer_param);
    layer_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const Dtype loss = blob_top_loss_->cpu_data()[0];

    // the expected loss
    Dtype expected_loss = -log(binomialCoeff(T_ + sequence_length,
                                             2 * sequence_length)
                               / pow(num_labels_, T_));
    EXPECT_FLOAT_EQ(loss, expected_loss);

    // check loss for all other t
    // (to check Graves Eq. (7.27) that holds for all t)
    for (int t = 1; t < T_; ++t) {
        layer_1.SetLossCalculationT(t);
        layer_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

        const Dtype loss_t = blob_top_loss_->cpu_data()[0];
        EXPECT_FLOAT_EQ(loss, loss_t);
    }
  }

  void TestForwardRandom() {
    InitRandom();

    LayerParameter layer_param;
    CTCLossLayer<Dtype> layer_1(layer_param);
    layer_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    const Dtype loss = blob_top_loss_->cpu_data()[0];

    // check loss for all other t
    // (to check Graves Eq. (7.27) that holds for all t)
    for (int t = 1; t < T_; ++t) {
        layer_1.SetLossCalculationT(t);
        layer_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

        const Dtype loss_t = blob_top_loss_->cpu_data()[0];

        EXPECT_DOUBLE_EQ(loss, loss_t) << " at loss calculation for t = " << t;
    }
  }

  void TestGradient() {
    // Input and ground truth for gradient from Alex Graves' implementation
    // (taken from Tensorflow test)
    Reshape(5, 2, 6);

    const Dtype targets_0[5] = {0, 1, 2, 1, 0};
    const Dtype loss_log_prob_0 = -3.34211;

    const Dtype input_prob_matrix_0[5 * 6] =
      {0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553,
       0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436,
       0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688,
       0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533,
       0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107};
    vector<Dtype> input_log_prob_matrix_0(5 * 6);

    for (int i = 0; i < input_log_prob_matrix_0.size(); ++i) {
        input_log_prob_matrix_0[i] = log(input_prob_matrix_0[i]);
    }

    const Dtype gradient_log_prob_0[5 * 6] =
      {-0.366234, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553,
       0.111121, -0.411608, 0.278779, 0.0055756, 0.00569609, 0.010436,
       0.0357786, 0.633813, -0.678582, 0.00249248, 0.00272882, 0.0037688,
       0.0663296, -0.356151, 0.280111, 0.00283995, 0.0035545, 0.00331533,
       -0.541765, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107};

    const Dtype targets_1[4] = {0, 1, 1, 0};
    const Dtype loss_log_prob_1 = -5.42262;

    const Dtype input_prob_matrix_1[5 * 6] =
      {0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508,
       0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549,
       0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456,
       0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345,
       0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046};

    vector<Dtype> input_log_prob_matrix_1(5 * 6);

    for (int i = 0; i < input_log_prob_matrix_1.size(); ++i) {
        input_log_prob_matrix_1[i] = log(input_prob_matrix_1[i]);
    }

    const Dtype gradient_log_prob_1[5 * 6] =
      {-0.69824, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508,
       0.24082, -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549,
       0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, -0.797544,
       0.280884, -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345,
       -0.576714, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046};

    Dtype *data = blob_bottom_data_->mutable_cpu_data();
    for (int t = 0; t < T_; ++t) {
      for (int c = 0; c < num_labels_; ++c) {
        data[blob_bottom_data_->offset(t, 0, c)]
              = input_log_prob_matrix_0[t * num_labels_ + c];
        data[blob_bottom_data_->offset(t, 1, c)]
              = input_log_prob_matrix_1[t * num_labels_ + c];
      }
    }

    FillerParameter filler_c1_param;
    filler_c1_param.set_value(1);
    ConstantFiller<Dtype> c1_filler(filler_c1_param);
    c1_filler.Fill(blob_bottom_seq_ind_);
    for (int n = 0; n < N_; ++n) {
      blob_bottom_seq_ind_->mutable_cpu_data()[n] = 0;
    }

    FillerParameter filler_cn1_param;
    filler_cn1_param.set_value(-1);
    ConstantFiller<Dtype> cn1_filler(filler_cn1_param);
    cn1_filler.Fill(blob_bottom_label_);
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int t = 0; t < 5; ++t) {
      label_data[blob_bottom_label_->offset(t, 0)] = targets_0[t];
    }

    for (int t = 0; t < 4; ++t) {
      label_data[blob_bottom_label_->offset(t, 1)] = targets_1[t];
    }

    LayerParameter layer_param;
    CTCLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    // forward AND backward pass
    const Dtype loss_weight_1 =
        layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);


    const Dtype loss = blob_top_loss_->cpu_data()[0];
    EXPECT_LE(abs((-loss) - (loss_log_prob_0 + loss_log_prob_1) / 2),
              0.000001);

    // check loss for all other t
    // (to check Graves Eq. (7.27) that holds for all t)
    for (int t = 1; t < T_; ++t) {
        layer.SetLossCalculationT(t);
        layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

        const Dtype loss_t = blob_top_loss_->cpu_data()[0];

        EXPECT_FLOAT_EQ(loss, loss_t)
              << " at loss calculation for t = " << t;
    }

    vector<bool> prob_down(0, false);
    prob_down[0] = true;
    layer.Backward(this->blob_top_vec_, prob_down, this->blob_bottom_vec_);

    const Dtype *diff = blob_bottom_data_->cpu_diff();
    for (int t = 0; t < T_; ++t) {
      for (int c = 0; c < num_labels_; ++c) {
        EXPECT_LE(std::abs(diff[blob_bottom_data_->offset(t, 0, c)]
                  - gradient_log_prob_0[t * num_labels_ + c]),
                0.000001);
        EXPECT_LE(std::abs(diff[blob_bottom_data_->offset(t, 1, c)]
                  - gradient_log_prob_1[t * num_labels_ + c]),
                0.000001);
      }
    }
  }

  int T_;
  int N_;
  int num_labels_;
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_seq_ind_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CTCLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(CTCLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(CTCLossLayerTest, TestForwardWrong) {
  this->TestForwardWrong();
}

TYPED_TEST(CTCLossLayerTest, TestForwardEqual) {
  this->TestForwardEqual();
}

TYPED_TEST(CTCLossLayerTest, TestForwardRandom) {
  this->TestForwardRandom();
}

TYPED_TEST(CTCLossLayerTest, TestGradient) {
  this->TestGradient();
}


}  // namespace caffe
