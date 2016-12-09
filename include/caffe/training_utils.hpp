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

#include <string>
#include <vector>

using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::vector;

vector<string> get_stages_from_flags(const std::string& stages_flag) {
  vector<string> stages;
  boost::split(stages, stages_flag, boost::is_any_of(","));
  return stages;
}

void use_flags(caffe::SolverParameter* solver_param,
    const std::string& flag_solver,
    const std::string& flag_engine,
    const int& flag_level,
    const std::string& stages_flag) {
  caffe::UpgradeSolverAsNeeded(flag_solver, solver_param);
  vector<string> stages = get_stages_from_flags(stages_flag);

  // Override engine if provided in cmd line
  if (flag_engine != "") {
    solver_param->set_engine(flag_engine);
  }

  solver_param->mutable_train_state()->set_level(flag_level);
  for (int i = 0; i < stages.size(); i++) {
    solver_param->mutable_train_state()->add_stage(stages[i]);
  }
}

int multiphase_train(caffe::MultiPhaseSolverParameter* multi_solver_params,
    const std::string& flag_solver,
    const std::string& flag_engine,
    const int& flag_level,
    const std::string& stages_flag) {
  LOG(INFO) << "Running multiphase solver.";
  caffe::NetParameter solver_phase_net_param;
  caffe::NetParameter topology_net_param;
  caffe::SolverParameter solver_param;
  CHECK(multi_solver_params->params_pair(0).has_solver_params())
      << "Solver parameters should be provided in at least first params pair";
  CHECK(caffe::ReadProtoFromTextFile(
      multi_solver_params->params_pair(0).solver_params().net(),
      &topology_net_param))
        << "Could not read from net parameter of solver proto file";
  string snapshot_prefix = multi_solver_params->
    params_pair(0).solver_params().snapshot_prefix() + "_phase_";

  for (int j = 0; j < multi_solver_params->params_pair_size(); j++) {
    if (multi_solver_params->params_pair(j).has_solver_params()) {
      solver_param = multi_solver_params->params_pair(j).solver_params();

      if (solver_param.solver_mode() !=
        caffe::SolverParameter_SolverMode_CPU) {
          LOG(ERROR) << "CPU mode supported only";
          return -1;
      }
    }

    if (multi_solver_params->params_pair(j).has_batch_size()) {
      for (int i = 0; i < topology_net_param.layer_size(); i++) {
        if (topology_net_param.layer(i).type() == "Data") {
          topology_net_param.mutable_layer(i)->mutable_data_param()->
              set_batch_size(multi_solver_params->params_pair(j).batch_size());
          break;
        }
      }
    }

    solver_param.set_snapshot_prefix(snapshot_prefix
      + boost::lexical_cast<string>(j));

    solver_param.set_allocated_net_param(&topology_net_param);
    solver_param.clear_net();

    use_flags(
      &solver_param,
      flag_solver,
      flag_engine,
      flag_level,
      stages_flag);

    shared_ptr<caffe::Solver<float> >
        solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

    topology_net_param = *solver_param.release_net_param();

    solver->net()->CopyTrainedLayersFrom(solver_phase_net_param);
    for (int i = 0; i < solver->test_nets().size(); ++i) {
      solver->test_nets()[i]->CopyTrainedLayersFrom(solver_phase_net_param);
    }

    solver->Solve();
    solver->net()->ToProto(
      &solver_phase_net_param,
     solver->param().snapshot_diff());
  }

  LOG(INFO) << "Optimization Done.";
  return 0;
}
