#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::ostringstream;

namespace caffe {

using google::protobuf::TextFormat;

template <typename TypeParam>
class SolverTraceTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SolverTraceTest() :
      seed_(1701), num_(4), channels_(3), height_(10), width_(10),
      share_(false) {
        input_file_ = new string(
        CMAKE_SOURCE_DIR "caffe/test/test_data/solver_data_list.txt" CMAKE_EXT);
      }
  ~SolverTraceTest() {
    delete input_file_;
  }

  string snapshot_prefix_;
  shared_ptr<SGDSolver<Dtype> > solver_;
  shared_ptr<P2PSync<Dtype> > sync_;
  int seed_;
  // Dimensions are determined by generate_sample_data.py
  // TODO this is brittle and the hdf5 file should be checked instead.
  int num_, channels_, height_, width_;
  bool share_;
  Dtype delta_;  // Stability constant for RMSProp, AdaGrad, AdaDelta and Adam

  // Test data: check out generate_sample_data.py in the same directory.
  string* input_file_;

  virtual void InitSolver(const SolverParameter& param) = 0;

  virtual void InitSolverFromProtoString(const string& proto) {
    SolverParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
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
    delta_ = param.delta();
  }

  std::pair<string, string> RunLeastSquaresSolver(const Dtype learning_rate,
      const Dtype weight_decay, const Dtype momentum, const int num_iters,
      const int iter_size = 1, const int devices = 1,
      const bool snapshot = false, const char* from_snapshot = NULL,
      const char* from_trace = NULL, string extra_proto="") {
    ostringstream proto;
    int device_id = 0;
#ifndef CPU_ONLY
    if (Caffe::mode() == Caffe::GPU) {
      CUDA_CHECK(cudaGetDevice(&device_id));
    }
#endif
    proto <<
       "snapshot_after_train: " << snapshot << " "
       "max_iter: " << num_iters << " "
       "base_lr: " << learning_rate << " "
       "lr_policy: 'fixed' "
       "iter_size: " << iter_size << " "
       "device_id: " << device_id << " "
       "net_param { "
       "  name: 'TestNetwork' "
       "  layer { "
       "    name: 'data' "
       "    type: 'HDF5Data' "
       "    hdf5_data_param { "
       "      source: '" << *(this->input_file_) << "' "
       "      batch_size: " << num_ / iter_size << " "
       "    } "
       "    top: 'data' "
       "    top: 'targets' "
       "  } ";
    if (share_) {
      proto <<
         "  layer { "
         "    name: 'slice' "
         "    type: 'Slice' "
         "    bottom: 'data' "
         "    top: 'data1' "
         "    top: 'data2' "
         "    slice_param { "
         "      axis: 0 "
         "    } "
         "  } ";
    }
    proto <<
       "  layer { "
       "    name: 'innerprod' "
       "    type: 'InnerProduct' "
       "    param { name: 'weights' } "
       "    param { name: 'bias' } "
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
       "    bottom: '" << string(share_ ? "data1": "data") << "' "
       "    top: '" << string(share_ ? "innerprod1": "innerprod") << "' "
       "  } ";
    if (share_) {
      proto <<
         "  layer { "
         "    name: 'innerprod2' "
         "    type: 'InnerProduct' "
         "    param { name: 'weights' } "
         "    param { name: 'bias' } "
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
         "    bottom: 'data2' "
         "    top: 'innerprod2' "
         "  } "
         "  layer { "
         "    name: 'concat' "
         "    type: 'Concat' "
         "    bottom: 'innerprod1' "
         "    bottom: 'innerprod2' "
         "    top: 'innerprod' "
         "    concat_param { "
         "      axis: 0 "
         "    } "
         "  } ";
    }
    proto <<
       "  layer { "
       "    name: 'loss' "
       "    type: 'EuclideanLoss' "
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
    proto << "snapshot_prefix: '" << snapshot_prefix_ << "/' ";
    if (snapshot) {
      proto << "snapshot: " << num_iters << " ";
    }
    proto << extra_proto;
    Caffe::set_random_seed(this->seed_);
    this->InitSolverFromProtoString(proto.str());
    if (from_snapshot != NULL) {
      this->solver_->Restore(from_snapshot);
      vector<Blob<Dtype>*> empty_bottom_vec;
      for (int i = 0; i < this->solver_->iter(); ++i) {
        this->solver_->net()->Forward(empty_bottom_vec);
      }
    }
    if (devices == 1) {
      this->solver_->Solve();
    } else {
      LOG(INFO) << "Multi-GPU test on " << devices << " devices";
      vector<int> gpus;
      // put current device at the beginning
      int device_id = solver_->param().device_id();
      gpus.push_back(device_id);
      for (int i = 0; gpus.size() < devices; ++i) {
        if (i != device_id)
          gpus.push_back(i);
      }
      Caffe::set_solver_count(gpus.size());
      this->sync_.reset(new P2PSync<Dtype>(
          this->solver_, NULL, this->solver_->param()));
      this->sync_->Run(gpus);
      Caffe::set_solver_count(1);
    }
    if (snapshot) {
      ostringstream resume_file, trace_file;
      resume_file << snapshot_prefix_ << "/_iter_" << num_iters
                  << ".solverstate";
      string resume_filename = resume_file.str();
      trace_file << snapshot_prefix_ << "/trace.caffetrace";
      string trace_filename = trace_file.str();
      return std::pair<string, string>(resume_filename, trace_filename);
    }
    return std::pair<string, string>(string(), string());
  }

  void TestSolverWeightTrace(const Dtype learning_rate = 1.0,
      const Dtype weight_decay = 0.0, const Dtype momentum = 0.0,
      const int num_iters = 1) {
    // Run the solver for num_iters * 2 iterations.
    const int total_num_iters = num_iters * 2;
    bool snapshot = false;
    const char* from_snapshot = NULL;
    const char* trace_snapshot = NULL;
    int weight_trace_interval = 1;
    int num_traces = 5;
    const int kIterSize = 1;
    const int kDevices = 1;

    MakeTempDir(&snapshot_prefix_);
    ostringstream extra_proto;
    extra_proto <<
      "solver_trace_param { "
      "  save_interval: 1 "
      "  trace_filename: '" << snapshot_prefix_ << "/trace' "
      "  weight_trace_interval: " << weight_trace_interval << " "
      "  num_weight_traces: " << num_traces << " "
      "  create_train_trace: true "
      "  create_test_trace: false "
      "} ";

    RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
        total_num_iters, kIterSize, kDevices, snapshot, from_snapshot,
        trace_snapshot, extra_proto.str());

    string string_snapshot, string_no_snapshot;
    const TraceDigest& digest = solver_->get_digest();
    TextFormat::PrintToString(digest, &string_no_snapshot);

    // map from a name to the weight
    map<string, float> selected_weights;

    vector<vector<int> > trace_per_blob_iter(0);
    for (int i = 0; i < 2; ++i) {
      trace_per_blob_iter.push_back(vector<int>(total_num_iters + 1));
    }
    int share_mult = share_ ? 2 : 1;
    // We take a trace before we start and after we end and at every step
    // and we save 1 traces for four blobs
    EXPECT_EQ(digest.weight_trace_point_size(),
              share_mult * 2 * (total_num_iters + 1));
    for (int i = 0; i < digest.weight_trace_point_size(); ++i) {
      string layer_name = digest.weight_trace_point(i).layer_name();
      int param_id = digest.weight_trace_point(i).param_id();
      ASSERT_GE(param_id, 0);
      ASSERT_LE(param_id, 1);
      int iter = digest.weight_trace_point(i).iter();
      stringstream ss;
      ss << param_id << "_" << iter;
      float weight = digest.weight_trace_point(i).value(0);
      if (share_ && layer_name == "innerprod") {
        selected_weights[ss.str()] = weight;
      }

      // param_id == 0 means weights, otherwise bias of which there is only one
      if (param_id == 0) {
        ASSERT_EQ(digest.weight_trace_point(i).value_size(), num_traces);
      } else {
        ASSERT_EQ(digest.weight_trace_point(i).value_size(), 1);
      }

      // make sure the layer name is correct
      if (share_) {
        EXPECT_TRUE(layer_name == "innerprod" || layer_name == "innerprod2");
      } else {
        EXPECT_EQ(layer_name, "innerprod");
      }
      EXPECT_LE(weight, 100.);
      EXPECT_GE(weight, -100.);
      ASSERT_LE(iter, total_num_iters);
      ASSERT_GE(iter, 0);
      trace_per_blob_iter[param_id][iter]++;
    }
    // per layer / blob / iteration there should be exactly one weight trace
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < total_num_iters + 1; ++j) {
        EXPECT_EQ(trace_per_blob_iter[i][j], share_mult);
      }
    }
    // make sure the weights of the shared blobs are the same
    if (share_) {
      for (int i = 0; i < digest.weight_trace_point_size(); ++i) {
        string layer_name = digest.weight_trace_point(i).layer_name();
        int param_id = digest.weight_trace_point(i).param_id();
        int iter = digest.weight_trace_point(i).iter();
        stringstream ss;
        ss << param_id << "_" << iter;
        float weight = digest.weight_trace_point(i).value(0);
        if (layer_name == "innerprod2") {
          EXPECT_FLOAT_EQ(selected_weights[ss.str()], weight);
        }
      }
    }
  }

  void TestSolverActivationTrace(const Dtype learning_rate = 1.0,
      const Dtype weight_decay = 0.0, const Dtype momentum = 0.0,
      const int num_iters = 1) {
    // Run the solver for num_iters * 2 iterations.
    const int total_num_iters = num_iters * 2;
    bool snapshot = false;
    const char* from_snapshot = NULL;
    const char* trace_snapshot = NULL;
    int trace_interval = 1;
    int num_traces = 5;
    const int kIterSize = 1;
    const int kDevices = 1;

    MakeTempDir(&snapshot_prefix_);
    ostringstream extra_proto;
    extra_proto <<
      "solver_trace_param { "
      "  save_interval: 1 "
      "  trace_filename: '" << snapshot_prefix_ << "/trace' "
      "  activation_trace_interval: " << trace_interval << " "
      "  num_activation_traces: " << num_traces << " "
      "  create_train_trace: true "
      "  create_test_trace: false "
      "} ";

    RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
        total_num_iters, kIterSize, kDevices, snapshot, from_snapshot,
        trace_snapshot, extra_proto.str());

    string string_snapshot, string_no_snapshot;
    const TraceDigest& digest = solver_->get_digest();
    TextFormat::PrintToString(digest, &string_no_snapshot);

    int num_blobs = solver_->net()->blobs().size();
    ASSERT_EQ(digest.activation_trace_size(), num_blobs);

    for (int i = 0; i < digest.activation_trace_size(); ++i) {
      int trace_size = digest.activation_trace(i).activation_trace_point_size();
      EXPECT_TRUE(digest.activation_trace(i).has_blob_name());
      ASSERT_EQ(trace_size, total_num_iters);
      vector<int> iter_counts(total_num_iters + 1, 0);
      for (int j = 0; j < trace_size; ++j) {
        int iter = digest.activation_trace(i).activation_trace_point(j).iter();
        ASSERT_LE(iter, total_num_iters);
        iter_counts[iter]++;
      }
      for (int j = 1; j < trace_size; ++j) {
        EXPECT_EQ(iter_counts[j], 1);
      }
    }
  }

  void TestSolverTrainTrace(const Dtype learning_rate = 1.0,
      const Dtype weight_decay = 0.0, const Dtype momentum = 0.0,
      const int num_iters = 1) {
    // Run the solver for num_iters * 2 iterations.
    const int total_num_iters = num_iters * 2;
    bool snapshot = false;
    const char* from_snapshot = NULL;
    const char* trace_snapshot = NULL;
    const int kIterSize = 1;
    const int kDevices = 1;

    MakeTempDir(&snapshot_prefix_);
    ostringstream extra_proto;
    extra_proto <<
      "solver_trace_param { "
      "  save_interval: 1 "
      "  trace_filename: '" << snapshot_prefix_ << "/trace' "
      "  trace_interval: 0 "
      "  num_traces: 0 "
      "  create_train_trace: true "
      "  create_test_trace: false "
      "} ";

    RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
        total_num_iters, kIterSize, kDevices, snapshot, from_snapshot,
        trace_snapshot, extra_proto.str());

    const TraceDigest digest(solver_->get_digest());
    EXPECT_EQ(digest.train_trace_point_size(), total_num_iters);

    // Run the solver for num_iters iterations and snapshot.
    snapshot = true;
    pair<string, string> snapshot_name = RunLeastSquaresSolver(learning_rate,
        weight_decay, momentum, num_iters, kIterSize, kDevices,
        snapshot, from_snapshot, trace_snapshot, extra_proto.str());

    // Reinitialize the solver and run for num_iters more iterations.
    snapshot = false;
    RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
        total_num_iters, kIterSize, kDevices, snapshot,
        snapshot_name.first.c_str(), snapshot_name.second.c_str(),
        extra_proto.str());

    const TraceDigest& digest_snapshot = solver_->get_digest();
    // Here we can't do the string compare trick, b/c its impossible to get the
    // smoothed loss right
    ASSERT_EQ(digest.train_trace_point_size(),
              digest_snapshot.train_trace_point_size());
    for (int i = 0; i < digest.train_trace_point_size(); ++i) {
      EXPECT_EQ(digest.train_trace_point(i).iter(),
                digest_snapshot.train_trace_point(i).iter());
      EXPECT_EQ(digest.train_trace_point(i).train_loss(),
                digest_snapshot.train_trace_point(i).train_loss());
    }
  }
};

template <typename TypeParam>
class SGDSolverTraceTest : public SolverTraceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new SGDSolver<Dtype>(param));
  }
};

TYPED_TEST_CASE(SGDSolverTraceTest, TestDtypesAndDevices);

TYPED_TEST(SGDSolverTraceTest, TestSolverWeightTrace) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSolverWeightTrace(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTraceTest, TestSolverWeightTraceShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSolverWeightTrace(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTraceTest, TestSolverActivationTrace) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSolverActivationTrace(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTraceTest, TestSolverActivationTraceShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSolverActivationTrace(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTraceTest, TestSolverTrainTrace) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSolverTrainTrace(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTraceTest, TestSolverTrainTraceShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSolverTrainTrace(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

}  // namespace caffe
