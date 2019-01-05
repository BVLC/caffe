#include <cstdio>

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/algorithm/string.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/gan_solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
GANSolver<Dtype>::GANSolver(const SolverParameter& g_param, const SolverParameter& d_param) {
  g_solver.reset(caffe::SolverRegistry<Dtype>::CreateSolver(g_param));
  d_solver.reset(caffe::SolverRegistry<Dtype>::CreateSolver(d_param));
  iter_ = 0;
}

template <typename Dtype>
void GANSolver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  CHECK_EQ(d_solver->net_->num_inputs(), 1);
  LOG(INFO) << "Total iter: " << iter_ << " / " << d_solver->param_.max_iter();
  LOG(INFO) << "Solve\t\tGenerator\t\tDiscriminator";
  LOG(INFO) << "\t\t\t" << g_solver->net_->name() << "\t\t\t" << d_solver->net_->name();
  LOG(INFO) << "LR Policy\t\t" << g_solver->param_.lr_policy() << "\t\t" << d_solver->param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  /*
  auto vecs = d_solver->net_->bottom_need_backward(), vec_names = d_solver->net_->layer_names();
  for (int i = 0; i < vecs.size(); i ++) {
    for (int j = 0; j < vecs[i].size(); j ++) {
      LOG(INFO) << vec_names[i] << " " << i << " " << j << " " << vecs[i][j];
    }
  }
  */
 int ind = d_solver->net_->base_layer_index();
 d_solver->net_->set_bottom_need_backward(ind, true);

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  // use d_solver as standard
  Step(d_solver->param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (d_solver->param_.snapshot_after_train()
      && (!d_solver->param_.snapshot() || iter_ % d_solver->param_.snapshot() != 0)) {
    d_solver->Snapshot();
    g_solver->Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (d_solver->param_.display() && iter_ % d_solver->param_.display() == 0) {
    // int average_loss = d_solver->param_.average_loss();
    // Dtype loss;
    // d_solver->net_->Forward(&loss);

    // UpdateSmoothedLoss(loss, start_iter, average_loss);

    // LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (d_solver->param_.test_interval() && iter_ % d_solver->param_.test_interval() == 0) {
    // d_solver->TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void print_max_diff(Blob<Dtype> *blob) {
  float maxx = -100, minn = 100;
  for (int i = 0; i < 28 * 28; i ++) {
    float elem = blob->cpu_diff()[i];
    if (elem > maxx) maxx = elem;
    if (elem < minn) minn = elem;
  }
  LOG(INFO) << "Gradient " << maxx << " " << minn;
}

template <typename Dtype>
void GANSolver<Dtype>::Step(int iters) {
  int start_iter = iter_;
  int stop_iter = iter_ + iters;
  int base_ind = d_solver->net_->base_layer_index();
  int end_ind = d_solver->net_->layers().size() - 1;
  int g_last_layer = g_solver->net_->layers().size() - 1;
  
  d_solver->losses_.clear();
  g_solver->losses_.clear();

  iteration_timer_.Start();

  // zero-init the params
  d_solver->net_->ClearParamDiffs();
  g_solver->net_->ClearParamDiffs();

  // label placeholder
  Blob<Dtype>* disc_label = d_solver->net_->input_blobs()[0];
  // ones, zeros
  Blob<Dtype> ones, zeros;
  ones.ReshapeLike(*disc_label);
  zeros.ReshapeLike(*disc_label);
  auto ones_data = ones.mutable_cpu_data(), zeros_data = zeros.mutable_cpu_data();
  for(int i = 0; i < disc_label->shape()[0]; i++) {
    ones_data[i] = 1.0;
    zeros_data[i] = 0.0;
  }
  
  Dtype disc_real_loss = 0, disc_fake_loss = 0, gen_loss = 0, _tmp;
  while (++iter_ < stop_iter) {
    if (d_solver->param_.test_interval() && iter_ % d_solver->param_.test_interval() == 0) {
      LOG(INFO) << "Iter=" << iter_ << "\tDisc Real\t" << "Disc Fake\t" << "Gen";
      LOG(INFO) << "\t\t\t" << disc_real_loss / d_solver->param_.test_interval() << "\t" << disc_fake_loss / d_solver->param_.test_interval() << "\t" << gen_loss / d_solver->param_.test_interval();
      disc_real_loss = disc_fake_loss = gen_loss = 0;
      if (Caffe::root_solver())
        TestAll();
      if (requested_early_exit_)
        break;
    }

    /// Train D
    auto x_fake = g_solver->net_->Forward(); // G(z)

    disc_label->CopyFrom(ones); //CHECK_EQ((int)disc_label->cpu_data()[23], 1);
    d_solver->net_->Forward(&_tmp); // D(real)
    disc_real_loss += _tmp;
    d_solver->net_->Backward(); // accumulate gradient for D(real)

    disc_label->CopyFrom(zeros); //CHECK_EQ((int)disc_label->cpu_data()[19], 0);
    disc_fake_loss += d_solver->net_->ForwardFromTo(x_fake, base_ind, end_ind); // D(G(z))

    d_solver->net_->Backward(); // accumulate gradient for D(G(z))
    d_solver->ApplyUpdate();
    d_solver->net_->ClearParamDiffs();
    
    /// Train G
    x_fake = g_solver->net_->Forward(); // G(z)

    disc_label->CopyFrom(ones); //CHECK_EQ((int)disc_label->cpu_data()[49], 1);
    gen_loss += d_solver->net_->ForwardFromTo(x_fake, base_ind, end_ind); // D(G(z))
    d_solver->net_->Backward(); // calculate gradient
    auto d_bottom = d_solver->net_->bottom_vecs()[base_ind][0];
    // LOG_IF(INFO, Caffe::root_solver()) << "d bottom " << d_bottom->shape_string();

    // TODO: do not caculate gradient for weights
    auto g_top = g_solver->net_->mutable_top_vecs()[g_last_layer][0];
    // LOG_IF(INFO, Caffe::root_solver()) << "g top    " << g_top->shape_string();
    g_top->CopyFrom(*d_bottom, true, false);
    //print_max_diff(g_top);

    g_solver->net_->Backward();
    g_solver->ApplyUpdate();

    g_solver->net_->ClearParamDiffs();
    d_solver->net_->ClearParamDiffs();

    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((d_solver->param_.snapshot()
         && iter_ % d_solver->param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      d_solver->Snapshot();
      g_solver->Snapshot();
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      break;
    }
  }
}

template <typename Dtype>
void GANSolver<Dtype>::SetActionFunction(ActionCallback func) {
  g_solver->SetActionFunction(func);
  d_solver->SetActionFunction(func);
}

template<typename Dtype>
SolverAction::Enum GANSolver<Dtype>::GetRequestedAction() {
  if (d_solver->action_request_function_) {
    // If the external request function has been set, call it.
    // Only call on discriminator
    return d_solver->action_request_function_();
  }
  return SolverAction::NONE;
}

INSTANTIATE_CLASS(GANSolver);

}  // namespace caffe
