// Copyright 2014 BVLC and contributors.
#pragma once

#include <string>
#include "caffe/iter_callback.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template< typename Dtype >
struct DefaultSolverActions {
  DefaultSolverActions() {
  }

  explicit DefaultSolverActions(const SolverParameter& solver_param):
      param_(solver_param) {
  }

  void SetResumeFile(const std::string& resume_file) {
      resume_file_ = resume_file;
  }

  // Function operator overload implements the functor needed by
  // solver to handle training iteration notifications and provide
  // indications as to what to do.
  IterActions<Dtype> operator()(const TrainingStats<Dtype>& stats) {
    IterActions<Dtype> actions;

    if (stats.GetIter() < param_.max_iter()) {
      actions.SetShouldContinue();
    } else {
      actions.SetShouldStop();
    }

    if (this->param_.display() &&
         (stats.GetIter() % this->param_.display() == 0)) {
      actions.SetShouldDisplay();
    }

    actions.SetLearnRate(this->GetLearningRate(stats));
    actions.SetMomentum(this->param_.momentum());
    actions.SetWeightDecay(this->param_.weight_decay());

    // We haven't started training yet when GetIter() == 0.
    if (stats.GetIter() == 0) {
      // Resume before starting if a resume file was specified.
      if ( !resume_file_.empty() ) {
        actions.SetResumeFile(resume_file_);
      }

      // Run a test pass before doing any training to avoid waiting a
      // potentially very long time (param_.test_interval() training iterations)
      // to report that there's not enough memory to run the test net and crash,
      // etc.; and to gauge the effect of the first training iterations.
      if (param_.test_interval()) {
        actions.SetShouldTest();
      }
    } else {
      // We have begun training.
      if (this->param_.display() && (stats.GetIter() %
                                       param_.display() == 0) ) {
         actions.SetShouldDisplay();
      }

      if (param_.test_interval() && ( stats.GetIter() %
                                       param_.test_interval() == 0)) {
        actions.SetShouldTest();
      }

      if (param_.snapshot() &&
         (stats.GetIter() % param_.snapshot() == 0) &&
         (stats.GetIter() > stats.GetStartIter())) {
        actions.SetShouldSnapshot();
      }
    }
    return actions;
  }

  // Return the current learning rate. The currently implemented learning rate
  // policies are as follows:
  //    - fixed: always return base_lr.
  //    - step: return base_lr * gamma ^ (floor(iter / step))
  //    - exp: return base_lr * gamma ^ iter
  //    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
  // where base_lr, gamma, step and power are defined in the solver parameter
  // protocol buffer, and iter is the current iteration.
  Dtype GetLearningRate(const TrainingStats<Dtype>& stats) {
    Dtype rate;
    const string& lr_policy = this->param_.lr_policy();
    if (lr_policy == "fixed") {
      rate = this->param_.base_lr();
    } else if (lr_policy == "step") {
      int current_step = stats.GetIter() / this->param_.stepsize();
      rate = this->param_.base_lr() *
          pow(this->param_.gamma(), current_step);
    } else if (lr_policy == "exp") {
      rate = this->param_.base_lr() *
          pow(this->param_.gamma(), stats.GetIter());
    } else if (lr_policy == "inv") {
      rate = this->param_.base_lr() *
          pow(Dtype(1) + this->param_.gamma() * stats.GetIter(),
              - this->param_.power());
    } else {
      LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
    }
    return rate;
  }

 private:
  // The solver parameter object specificed in proto file.
  SolverParameter param_;
  // File to resume from, if any.
  std::string resume_file_;
};

}  // namespace caffe
