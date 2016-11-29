
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
  CHECK(caffe::ReadProtoFromTextFile(
      multi_solver_params->params_pair(0).solver_params().net(),
      &topology_net_param))
        << "Could not read from net parameter of solver proto file";

  for (int j = 0; j < multi_solver_params->params_pair_size(); j++) {
    solver_param = multi_solver_params->params_pair(j).solver_params();
    solver_param.set_snapshot_prefix(solver_param.snapshot_prefix() +
        "_phase_" + boost::lexical_cast<string>(j));
    if (solver_param.solver_mode() != caffe::SolverParameter_SolverMode_CPU) {
        LOG(ERROR) << "CPU mode supported only";
        return -1;
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
