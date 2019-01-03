#include <cstdio>

#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
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
}

template <typename Dtype>
void GANSolver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  CHECK_EQ(d_solver->net_->num_inputs(), 1);
  LOG(INFO) << "Solve\t\tGenerator\t\tDiscriminator";
  LOG(INFO) << "\t\t\t" << g_solver->net_->name() << "\t\t\t" << d_solver->net_->name();
  LOG(INFO) << "LR Policy\t\t" << g_solver->param_.lr_policy() << "\t\t" << d_solver->param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

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
void GANSolver<Dtype>::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int d_average_loss = d_solver->param_.average_loss();
  int g_average_loss = g_solver->param_.average_loss();
  d_solver->losses_.clear();
  g_solver->losses_.clear();
  d_smoothed_loss_ = 0;
  g_smoothed_loss_ = 0;

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

  while (iter_ < stop_iter) {
    if (d_solver->param_.test_interval() && iter_ % d_solver->param_.test_interval() == 0
        && (iter_ > 0 || d_solver->param_.test_initialization())) {
      if (Caffe::root_solver()) {
        // TestAll();
      }
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

    /*
    for (int i = 0; i < d_solver->callbacks_.size(); ++i)
      d_solver->callbacks_[i]->on_start();
    for (int i = 0; i < g_solver->callbacks_.size(); ++i)
      g_solver->callbacks_[i]->on_start();
    */

    const bool d_display = d_solver->param_.display() && iter_ % d_solver->param_.display() == 0;
    const bool g_display = g_solver->param_.display() && iter_ % g_solver->param_.display() == 0;
    d_solver->net_->set_debug_info(d_display && d_solver->param_.debug_info());
    g_solver->net_->set_debug_info(g_display && g_solver->param_.debug_info());

    disc_label->CopyFrom(ones); CHECK_EQ((int)disc_label->cpu_data()[0], 1);

    // accumulate the loss and gradient
    Dtype d_loss = 0, g_loss = 0;
    for (int i = 0; i < d_solver->param_.iter_size(); ++i) {
      /// Train D
      auto x_fake = g_solver->net_->Forward(); // G(z)

      auto res = d_solver->net_->Forward(&d_loss); // D(real)
      d_solver->net_->Backward(); // accumulate gradient for D(real)

      LOG_IF(INFO, Caffe::root_solver()) << "Disc real:";
      for (i = 0; i < res.size(); i++)
        LOG_IF(INFO, Caffe::root_solver()) << res[i]->shape_string();

      disc_label->CopyFrom(zeros); CHECK_EQ((int)disc_label->cpu_data()[0], 0);
      d_solver->net_->ForwardFromTo(x_fake, d_solver->net_->base_layer_index(), d_solver->net_->layers().size() - 1); // D(G(z))
      d_solver->net_->Backward(); // accumulate gradient for D(G(z))
      d_solver->ApplyUpdate();
      d_solver->net_->ClearParamDiffs();

      /// Train G
      x_fake = g_solver->net_->Forward(); // G(z)

      disc_label->CopyFrom(ones); CHECK_EQ((int)disc_label->cpu_data()[0], 1);
      d_solver->net_->ForwardFromTo(x_fake, d_solver->net_->base_layer_index(), d_solver->net_->layers().size() - 1); // D(G(z))
      d_solver->net_->Backward(); // calculate gradient
      // TODO: do not caculate gradient for weights
      g_solver->net->Backward(d_solver->net_->bottom_vecs()[0], true, g_solver->net_->bottom_vecs()[0]);
      g_solver->ApplyUpdate();

      d_solver->net_->ClearParamDiffs();
      g_solver->net_->ClearParamDiffs();
    }
    
    /*
    loss /= param_.iter_size();
    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    if (display) {
      float lapse = iteration_timer_.Seconds();
      float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() << " iters), loss = " << smoothed_loss_;
      iteration_timer_.Start();
      iterations_last_ = iter_;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    */
   
    // Do not support gradients ready call back
    /*
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
    */

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
      // Break out of training loop.
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
