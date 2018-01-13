/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CAFFE_MLSLSOLVER_HPP_
#define CAFFE_MLSLSOLVER_HPP_

#ifdef USE_MLSL

#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver.hpp"
#include "caffe/solver_factory.hpp"

namespace caffe {

template <typename Dtype>
class MultiSolver {
 public:
  explicit MultiSolver(boost::shared_ptr<Solver<Dtype>> root_solver)
    : root_solver_(root_solver),
      iter_size(root_solver_->param().iter_size()) {
    root_solver_->set_forward_backward(
      boost::bind(&MultiSolver<Dtype>::ForwardBackward, this));
#ifdef FW_OVERLAP_OPT    
    Net<Dtype>& net = *root_solver_->net();
    const std::vector<shared_ptr<Layer<Dtype>>> & layers{ net.layers() };
    layer_finished_flags_.resize(layers.size());
    std::fill(layer_finished_flags_.begin(), layer_finished_flags_.end(), true);
#endif
  }


  // Invoked at specific points during an iteration
  class Callback : public Solver<Dtype>::Callback {
  protected:
    virtual ~Callback() {
    }
    virtual void on_backward_finished(int layer_id) = 0;
    virtual void on_delwt_wait(int layer_id) = 0;
    virtual void apply_updates(int layer_id) = 0;

    template <typename T>
    friend class MultiSolver;
  };

  void add_callback(Callback* value) {
    root_solver_->add_callback(value);
    callbacks_.push_back(value);
  }

  Dtype ForwardBackward();

  void Solve() {
    root_solver_->Solve();
  }

  Net<Dtype>& net() {
    return *root_solver_->net();
  }

  const SolverParameter& param() const {
    return root_solver_->param();
  }

  boost::shared_ptr<Solver<Dtype>> root_solver() {
    return root_solver_;
  }
#ifdef FW_OVERLAP_OPT
  void set_layer_finished_flag(int layer_id, bool flag) {
    layer_finished_flags_[layer_id] = flag;
  }
#endif
 private:
  virtual Dtype ForwardBackwardImpl(bool first, bool last);
  bool IsSkipSyncGradient(int layer_id);
  bool WaitGradient(int layer_id);
  void UpdateGradient(int layer_id);

 protected:
  boost::shared_ptr<Solver<Dtype>> root_solver_;
  int iter_size;
  vector<Callback*> callbacks_;
#ifdef FW_OVERLAP_OPT
  vector<bool> layer_finished_flags_;
#endif
};

}  // namespace caffe

#endif // USE_MLSL

#endif  // CAFFE_MLSLSOLVER_HPP_
