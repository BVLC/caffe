#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::ostringstream;

namespace caffe {

template <typename TypeParam>
class GradientBasedSolverTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GradientBasedSolverTest() :
      seed_(1701), num_(5), channels_(3), height_(10), width_(10) {}

  shared_ptr<SGDSolver<Dtype> > solver_;
  int seed_;
  int num_, channels_, height_, width_;
  Dtype delta_;  // Stability constant for AdaGrad.

  virtual SolverParameter_SolverType solver_type() = 0;
  virtual void InitSolver(const SolverParameter& param) = 0;

  virtual void InitSolverFromProtoString(const string& proto) {
    SolverParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    // Disable saving a final snapshot so the tests don't pollute the user's
    // working directory with useless snapshots.
    param.set_snapshot_after_train(false);
    // Set the solver_mode according to current Caffe::mode.
    switch (Caffe::mode()) {
      case Caffe::CPU:
        param.set_solver_mode(SolverParameter_SolverMode_CPU);
        break;
      case Caffe::GPU:
        param.set_solver_mode(SolverParameter_SolverMode_GPU);
        break;
      default:
        LOG(FATAL) << "Unknown Caffe mode: " << Caffe::mode();
    }
    InitSolver(param);
    delta_ = (solver_type() == SolverParameter_SolverType_ADAGRAD) ?
         param.delta() : 0;
  }

  void RunLeastSquaresSolver(const Dtype learning_rate,
      const Dtype weight_decay, const Dtype momentum, const int num_iters) {
    ostringstream proto;
    proto <<
       "max_iter: " << num_iters << " "
       "base_lr: " << learning_rate << " "
       "lr_policy: 'fixed' "
       "net_param { "
       "  name: 'TestNetwork' "
       "  layers: { "
       "    name: 'data' "
       "    type: DUMMY_DATA "
       "    dummy_data_param { "
       "      num: " << num_ << " "
       "      channels: " << channels_ << " "
       "      height: " << height_ << " "
       "      width: " << width_ << " "
       "      channels: 1 "
       "      height: 1 "
       "      width: 1 "
       "      data_filler { "
       "        type: 'gaussian' "
       "        std: 1.0 "
       "      } "
       "    } "
       "    top: 'data' "
       "    top: 'targets' "
       "  } "
       "  layers: { "
       "    name: 'innerprod' "
       "    type: INNER_PRODUCT "
       "    inner_product_param { "
       "      num_output: 1 "
       "      weight_filler { "
       "        type: 'gaussian' "
       "        std: 1.0 "
       "      } "
       "      bias_filler { "
       "        type: 'gaussian' "
       "        std: 1.0 "
       "      } "
       "    } "
       "    bottom: 'data' "
       "    top: 'innerprod' "
       "  } "
       "  layers: { "
       "    name: 'loss' "
       "    type: EUCLIDEAN_LOSS "
       "    bottom: 'innerprod' "
       "    bottom: 'targets' "
       "  } "
       "} ";
    if (weight_decay != 0) {
      proto << "weight_decay: " << weight_decay << " ";
    }
    if (momentum != 0) {
      proto << "momentum: " << momentum << " ";
    }
    Caffe::set_random_seed(this->seed_);
    this->InitSolverFromProtoString(proto.str());
    this->solver_->Solve();
  }

  // Compute an update value given the current state of the train net,
  // using the analytical formula for the least squares gradient.
  // updated_params will store the updated weight and bias results,
  // using the blobs' diffs to hold the update values themselves.
  void ComputeLeastSquaresUpdate(const Dtype learning_rate,
      const Dtype weight_decay, const Dtype momentum,
      vector<shared_ptr<Blob<Dtype> > >* updated_params) {
    const int N = num_;
    const int D = channels_ * height_ * width_;

    // Run a forward pass, and manually compute the update values from the
    // result.
    Net<Dtype>& net = *this->solver_->net();
    vector<Blob<Dtype>*> empty_bottom_vec;
    net.Forward(empty_bottom_vec);
    ASSERT_TRUE(net.has_blob("data"));
    const Blob<Dtype>& data = *net.blob_by_name("data");
    ASSERT_TRUE(net.has_blob("targets"));
    const Blob<Dtype>& targets = *net.blob_by_name("targets");
    ASSERT_TRUE(net.has_layer("innerprod"));
    const vector<shared_ptr<Blob<Dtype> > >& param_blobs =
        net.layer_by_name("innerprod")->blobs();
    const int num_param_blobs = 2;
    ASSERT_EQ(num_param_blobs, param_blobs.size());
    const Blob<Dtype>& weights = *param_blobs[0];
    const Blob<Dtype>& bias = *param_blobs[1];
    ASSERT_EQ(D * N, data.count());
    ASSERT_EQ(N, targets.count());
    ASSERT_EQ(D, weights.count());
    ASSERT_EQ(1, bias.count());

    updated_params->clear();
    updated_params->resize(num_param_blobs);
    for (int i = 0; i < num_param_blobs; ++i) {
      (*updated_params)[i].reset(new Blob<Dtype>());
    }
    Blob<Dtype>& updated_weights = *(*updated_params)[0];
    updated_weights.ReshapeLike(weights);
    Blob<Dtype>& updated_bias = *(*updated_params)[1];
    updated_bias.ReshapeLike(bias);

    for (int i = 0; i <= D; ++i) {
      // Compute the derivative with respect to the ith weight (i.e., the ith
      // element of the gradient).
      Dtype grad = 0;
      for (int j = 0; j <= D; ++j) {
        // Compute element (i, j) of X^T * X.
        Dtype element = 0;
        for (int k = 0; k < N; ++k) {
          // (i, k) in X^T (== (k, i) in X) times (k, j) in X.
          const Dtype element_i = (i == D) ? 1 : data.cpu_data()[k * D + i];
          const Dtype element_j = (j == D) ? 1 : data.cpu_data()[k * D + j];
          element += element_i * element_j;
        }
        if (j == D) {
          grad += element * bias.cpu_data()[0];
        } else {
          grad += element * weights.cpu_data()[j];
        }
      }
      for (int k = 0; k < N; ++k) {
        const Dtype element_i = (i == D) ? 1 : data.cpu_data()[k * D + i];
        grad -= element_i * targets.cpu_data()[k];
      }
      // Scale the gradient over the N samples.
      grad /= N;
      // Add the weight decay to the gradient.
      grad += weight_decay *
          ((i == D) ? bias.cpu_data()[0] : weights.cpu_data()[i]);
      // Finally, compute update.
      const vector<shared_ptr<Blob<Dtype> > >& history = solver_->history();
      ASSERT_EQ(2, history.size());  // 1 blob for weights, 1 for bias
      Dtype update_value = learning_rate * grad;
      const Dtype history_value = (i == D) ?
            history[1]->cpu_data()[0] : history[0]->cpu_data()[i];
      const Dtype temp = momentum * history_value;
      switch (solver_type()) {
      case SolverParameter_SolverType_SGD:
        update_value += temp;
        break;
      case SolverParameter_SolverType_NESTEROV:
        update_value += temp;
        // step back then over-step
        update_value = (1 + momentum) * update_value - temp;
        break;
      case SolverParameter_SolverType_ADAGRAD:
        update_value /= std::sqrt(history_value + grad * grad) + delta_;
        break;
      default:
        LOG(FATAL) << "Unknown solver type: " << solver_type();
      }
      if (i == D) {
        updated_bias.mutable_cpu_diff()[0] = update_value;
        updated_bias.mutable_cpu_data()[0] = bias.cpu_data()[0] - update_value;
      } else {
        updated_weights.mutable_cpu_diff()[i] = update_value;
        updated_weights.mutable_cpu_data()[i] =
            weights.cpu_data()[i] - update_value;
      }
    }
  }

  void CheckLeastSquaresUpdate(
      const vector<shared_ptr<Blob<Dtype> > >& updated_params) {
    const int D = channels_ * height_ * width_;

    const Blob<Dtype>& updated_weights = *updated_params[0];
    const Blob<Dtype>& updated_bias = *updated_params[1];

    Net<Dtype>& net = *this->solver_->net();
    ASSERT_TRUE(net.has_layer("innerprod"));
    const vector<shared_ptr<Blob<Dtype> > >& param_blobs =
        net.layer_by_name("innerprod")->blobs();
    ASSERT_EQ(2, param_blobs.size());
    const Blob<Dtype>& solver_updated_weights = *param_blobs[0];
    ASSERT_EQ(D, solver_updated_weights.count());
    const double kPrecision = 1e-2;
    const double kMinPrecision = 1e-7;
    for (int i = 0; i < D; ++i) {
      const Dtype expected_updated_weight = updated_weights.cpu_data()[i];
      const Dtype solver_updated_weight = solver_updated_weights.cpu_data()[i];
      const Dtype error_margin = std::max(kMinPrecision, kPrecision *
          std::min(fabs(expected_updated_weight), fabs(solver_updated_weight)));
      EXPECT_NEAR(expected_updated_weight, solver_updated_weight, error_margin);
    }
    const Blob<Dtype>& solver_updated_bias_blob = *param_blobs[1];
    ASSERT_EQ(1, solver_updated_bias_blob.count());
    const Dtype expected_updated_bias = updated_bias.cpu_data()[0];
    const Dtype solver_updated_bias = solver_updated_bias_blob.cpu_data()[0];
    const Dtype error_margin = std::max(kMinPrecision, kPrecision *
          std::min(fabs(expected_updated_bias), fabs(solver_updated_bias)));
    EXPECT_NEAR(expected_updated_bias, solver_updated_bias, error_margin);

    // Check the solver's history -- should contain the previous update value.
    if (solver_type() == SolverParameter_SolverType_SGD) {
      const vector<shared_ptr<Blob<Dtype> > >& history = solver_->history();
      ASSERT_EQ(2, history.size());
      for (int i = 0; i < D; ++i) {
        const Dtype expected_history = updated_weights.cpu_diff()[i];
        const Dtype solver_history = history[0]->cpu_data()[i];
        const Dtype error_margin_hist = std::max(kMinPrecision, kPrecision *
            std::min(fabs(expected_history), fabs(solver_history)));
        EXPECT_NEAR(expected_history, solver_history, error_margin_hist);
      }
      const Dtype expected_history = updated_bias.cpu_diff()[0];
      const Dtype solver_history = history[1]->cpu_data()[0];
      const Dtype error_margin_hist = std::max(kMinPrecision, kPrecision *
          std::min(fabs(expected_history), fabs(solver_history)));
      EXPECT_NEAR(expected_history, solver_history, error_margin_hist);
    }
  }

  // Test that the correct update is computed for a regularized least squares
  // problem:
  //
  //            E = (1/(2n)) || X w - y ||^2 + (lambda / 2) || w ||^2
  //   \nabla_w E = (1/n) (X^T X w - X^T y) + lambda * w
  //
  // X \in R^{n x (d+1)} (each example is a row, (d+1)th element is always 1)
  // w \in R^{(d+1) x 1} ((d+1)th element is the bias)
  // y \in R^{n x 1}
  // lambda is weight_decay
  //
  // TestLeastSquaresUpdate works "inductively", assuming that the solver
  // correctly updates the net K (= iter_to_check) times, then given the history
  // from the Kth update, we compute the (K+1)th update and check that it
  // matches the solver's (K+1)th update.
  void TestLeastSquaresUpdate(const Dtype learning_rate = 1.0,
      const Dtype weight_decay = 0.0, const Dtype momentum = 0.0,
      const int iter_to_check = 0) {
    // Initialize the solver and run K (= iter_to_check) solver iterations.
    RunLeastSquaresSolver(learning_rate, weight_decay, momentum, iter_to_check);

    // Compute the (K+1)th update using the analytic least squares gradient.
    vector<shared_ptr<Blob<Dtype> > > updated_params;
    ComputeLeastSquaresUpdate(learning_rate, weight_decay, momentum,
                              &updated_params);

    // Reinitialize the solver and run K+1 solver iterations.
    RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
                          iter_to_check + 1);

    // Check that the solver's solution matches ours.
    CheckLeastSquaresUpdate(updated_params);
  }
};


template <typename TypeParam>
class SGDSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new SGDSolver<Dtype>(param));
  }

  virtual SolverParameter_SolverType solver_type() {
    return SolverParameter_SolverType_SGD;
  }
};

TYPED_TEST_CASE(SGDSolverTest, TestDtypesAndDevices);

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdate) {
  typedef typename TypeParam::Dtype Dtype;
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateLROneTenth) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 1.0;
  const Dtype kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithMomentum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 1.0;
  const Dtype kWeightDecay = 0.0;
  const Dtype kMomentum = 0.5;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithMomentumMultiIter) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 1.0;
  const Dtype kWeightDecay = 0.0;
  const Dtype kMomentum = 0.5;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.1;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}


template <typename TypeParam>
class AdaGradSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new AdaGradSolver<Dtype>(param));
  }
  virtual SolverParameter_SolverType solver_type() {
    return SolverParameter_SolverType_ADAGRAD;
  }
};

TYPED_TEST_CASE(AdaGradSolverTest, TestDtypesAndDevices);

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdate) {
  typedef typename TypeParam::Dtype Dtype;
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateLROneTenth) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 1.0;
  const Dtype kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.1;
  const Dtype kMomentum = 0.0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}


template <typename TypeParam>
class NesterovSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new NesterovSolver<Dtype>(param));
  }
  virtual SolverParameter_SolverType solver_type() {
    return SolverParameter_SolverType_NESTEROV;
  }
};

TYPED_TEST_CASE(NesterovSolverTest, TestDtypesAndDevices);

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdate) {
  typedef typename TypeParam::Dtype Dtype;
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateLROneTenth) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 1.0;
  const Dtype kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithMomentum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 1.0;
  const Dtype kWeightDecay = 0.0;
  const Dtype kMomentum = 0.5;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestLeastSquaresUpdateWithMomentumMultiIter) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 1.0;
  const Dtype kWeightDecay = 0.0;
  const Dtype kMomentum = 0.5;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.1;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

}  // namespace caffe
