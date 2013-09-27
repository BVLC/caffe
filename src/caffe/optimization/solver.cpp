// Copyright Yangqing Jia 2013

#include <fstream>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"
#include "caffe/optimization/solver.hpp"

using std::stringstream;
using std::ofstream;

namespace caffe {

template <typename Dtype>
void Solver<Dtype>::Solve(Net<Dtype>* net) {
  net_ = net;
  LOG(INFO) << "Solving net " << net_->name();
  iter_ = 0;
  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  vector<Blob<Dtype>*> bottom_vec;
  vector<Blob<Dtype>*> top_vec;
  while (iter_++ < param_.max_iter()) {
    Dtype loss = net_->ForwardBackWard(bottom_vec, &top_vec);
    ComputeUpdateValue();
    net->Update();

    // Check if we need to do snapshot
    if (iter_ % param_.snapshot()) {
      // TODO(Yangqing): snapshot
    }
    LOG(INFO) << "Iteration" << iter_ << ", loss=" << loss;
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::Snapshot(bool is_final) {
  NetParameter net_param;
  net_->ToProto(&net_param);
  stringstream ss;
  ss << param_.snapshot_prefix();
  if (is_final) {
    ss << "_final";
  } else {
    ss << "_iter_" << iter_;
  }
  ofstream output_file;
  output_file.open(ss.str().c_str());
  CHECK(net_param.SerializeToOstream(&output_file));
  output_file.close();
}

template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            this->param_.power());
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  rate = min(max(rate, this->param_.min_pr()), this->param_.max_lr());
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue() {
  // First of all, see if we need to initialize the history
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_.params();
  if (this->iter_ == 1 && this->param_.momentum() > 0) {
    LOG(INFO) << "Using momentum " << this->param_.momentum();
    for (int i = 0; i < net_params.size(); ++i) {
      const Blob<Dtype>* net_param = net_params[i].get();
      history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
          net_param->num(), net_param->channels(), net_param->height(),
          net_param->width())));
    }
  }
  // get the learning rate
  Dtype rate = GetLearningRate();
  if (this->param_.momentum == 0) {
    for (int i = 0; i < net_params.size(); ++i) {
      switch (Caffe::mode()) {
      case Caffe::CPU:
        caffe_scal(net_params[i]->count(), rate,
            net_params[i]->mutable_cpu_data());
        break;
      case Caffe::GPU:
        caffe_gpu_scal(net_params[i]->count(), rate,
            net_params[i]->mutable_gpu_data());
        break;
      default:
        LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
      }
    }
  } else {
    NOT_IMPLEMENTED;
  }
}



INSTANTIATE_CLASS(Solver);

}  // namespace caffe