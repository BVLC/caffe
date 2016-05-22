#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "caffe/common.hpp"
#ifdef USE_LIBDNN
#include "caffe/device.hpp"
#include "caffe/greentea/libdnn_tuner.hpp"


namespace caffe {

void LibDNNTuner::set_setup_routine(std::function<bool()> fun) {
  this->setup_routine_ = fun;
}

void LibDNNTuner::set_benchmark_routine(std::function<double()> fun) {
  this->benchmark_routine_ = fun;
}

void LibDNNTuner::Tune(libdnnTunerMethod_t method) {
  bool setup_success = setup_routine_();
  int_tp current_param = 0;
  double baseline_score = 0;
  double best_score = 0;
  for (int i = 0; i < 5; ++i) {
    baseline_score += benchmark_routine_();
  }
  baseline_score /= 5;
  best_score = baseline_score;

  if (method == LIBDNN_TUNER_METHOD_ALL) {
    while (true) {
      bool setup_success = setup_routine_();
      if (setup_success) {
        double score = benchmark_routine_();
        if (score > best_score) {
          best_score = score;
        }
        std::cout << "Score: "
            << (100.0/baseline_score)*score <<  "% (best: "
            << (100.0/baseline_score)*best_score << "%)"<< std::endl;
      }

      bool overflow = false;
      while (true) {
        overflow = params_[current_param]->advance(1);
        if (overflow) {
          // Parameter is at default value again
          // Switch to the next parameter
          ++current_param;
          if (current_param >= params_.size()) {
            // Through all parameters, stop
            break;
          }
        } else {
          // Current parameter has changed to a new value, stop
          break;
        }
      }
      if (current_param >= params_.size()) {
        // Through all parameters, stop
        break;
      }
      current_param = 0;
    }
  }
  if (method == LIBDNN_TUNER_METHOD_ANNEALING) {
    double temp = 1.0;
    double temp_min = 0.01;
    double alpha = 0.95;
    double old_score = baseline_score;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, params_.size() - 1);
    std::uniform_int_distribution<int> adv(1, 3);
    std::uniform_int_distribution<int> dir(0, 1);
    std::uniform_real_distribution<double> aprn(0.0, 1.0);

    // Initial state snapshot
    Snapshot(baseline_score);

    while (temp > temp_min) {
      for (int i = 0; i < 100; ++i) {
        int next_param = uni(rng);
        libdnnTunerParamStatus_t status;
        while (true) {
          status = params_[next_param]->advance(dir(rng) == 0?-1:1*adv(rng));
          if (status != LIBDNN_TUNER_PARAM_STAT_NO_SOLUTION) {
            break;
          }
        }
        std::cout << "Changing parameter: " << params_[next_param]->get_name()
            << ", new index: "
            << params_[next_param]->get_curr_idx()
            << ", new value: "
            << get_param<double>(params_[next_param]->get_name()) << std::endl;
        bool setup_success = setup_routine_();
        double score = -1.0;
        if (setup_success) {
          score = benchmark_routine_();
          if (score > best_score) {
            best_score = score;
          }
          std::cout << "Score: "
              << (100.0/baseline_score)*score <<  "% (best: "
              << (100.0/baseline_score)*best_score << "%) temp: "
              << temp << ", step: " << i << std::endl;
        } else {
          std::cout << "Setup failure" << std::endl;
          RestoreSnapshot(snapshots_[snapshots_.size()-1]);
        }
        double ap = std::exp(((1.0/old_score)-(1.0/score))/temp);
        if (ap > aprn(rng)) {
          // Accept solution, create a snapshot
          Snapshot(score);
          old_score = score;
        } else {
          // Reject solution, restore the last snapshot
          RestoreSnapshot(snapshots_[snapshots_.size()-1]);
        }
      }
      temp *= alpha;
    }
    // Restore the best solution
    RestoreSnapshot(snapshot_queue_.top());
    setup_routine_();
    std::cout << "Final score: "
        << ((100.0/baseline_score)*benchmark_routine_()) << std::endl;
  }
  // Cleanup
  // TODO
}

void LibDNNTuner::Snapshot(double score) {
  std::shared_ptr<LibDNNTunerSnapshot>
        snapshot(new LibDNNTunerSnapshot(score, &params_));
  snapshots_.push_back(snapshot);
  snapshot_queue_.push(snapshot);
}

void LibDNNTuner::RestoreSnapshot(
    std::shared_ptr<LibDNNTunerSnapshot> snapshot) {
  std::vector<std::shared_ptr<LibDNNTunerParam>>* params =
      snapshot->get_params();
  for (int i = 0; i < params_.size(); ++i) {
    params_[i]->update((*params)[i]);
  }
}

template<class T>
void LibDNNTuner::add_range_param(std::string name,
                                  T def_value, T min, T max, T step) {
  std::vector<T> values;

  T value = static_cast<T>(def_value);

  T vmin = std::min(max, min);
  T vmax = std::max(max, min);

  values.push_back(value);

  while (value >= vmin) {
    value -= step;
    if (value <= vmax && value >= vmin) {
      values.insert(values.begin(), value);
    }
  }

  value = static_cast<T>(def_value);

  while (value <= vmax) {
    value += step;
    if (value >= vmin && value <= vmax) {
      values.push_back(value);  std::vector<T> set_values;
    }
  }

  add_set_param(name, def_value, values);
}
template void LibDNNTuner::add_range_param(std::string name, float def_value,
                                  float min, float max, float step);
template void LibDNNTuner::add_range_param(std::string name, double def_value,
                                  double min, double max, double step);
template void LibDNNTuner::add_range_param(std::string name, int32_t def_value,
                                  int32_t min, int32_t max, int32_t step);
template void LibDNNTuner::add_range_param(std::string name, int64_t def_value,
                                  int64_t min, int64_t max, int64_t step);

template<class T>
void LibDNNTuner::add_range_param(const char* name,
                                  T def_value, T min, T max, T step) {
  std::string str(name);
  add_range_param<T>(str, def_value, min, max, step);
}
template void LibDNNTuner::add_range_param(const char* name, float def_value,
                                  float min, float max, float step);
template void LibDNNTuner::add_range_param(const char* name, double def_value,
                                  double min, double max, double step);
template void LibDNNTuner::add_range_param(const char* name, int32_t def_value,
                                  int32_t min, int32_t max, int32_t step);
template void LibDNNTuner::add_range_param(const char* name, int64_t def_value,
                                  int64_t min, int64_t max, int64_t step);


template<class T>
void LibDNNTuner::add_set_param(std::string name,
                                T def_value, std::vector<T> values) {
  if (is_same<T, float>::value || is_same<T, double>::value) {
    std::vector<double> set_values;
    int_tp def_idx = -1;
    for (int_tp i = 0; i < values.size(); ++i) {
      set_values.push_back(values[i]);
      if (def_value == values[i]) {
        def_idx = i;
      }
    }
    if (def_idx == -1) {
      def_idx = set_values.size();
      set_values.push_back(def_value);
    }
    std::shared_ptr<LibDNNTunerParam> param(
        new LibDNNTunerParamReal(this, name, set_values, def_idx));
    params_.push_back(param);
    param_map_.insert(std::pair<std::string,
                      std::shared_ptr<LibDNNTunerParam>>(name, param));
  }

  if (is_same<T, bool>::value) {
    std::vector<bool> set_values;
    int_tp def_idx = -1;
    for (int_tp i = 0; i < values.size(); ++i) {
      set_values.push_back(values[i]);
      if (def_value == values[i]) {
        def_idx = i;
      }
    }
    if (def_idx == -1) {
      def_idx = set_values.size();
      set_values.push_back(def_value);
    }
    std::shared_ptr<LibDNNTunerParam> param(
        new LibDNNTunerParamBool(this, name, set_values, def_idx));
    params_.push_back(param);
    param_map_.insert(std::pair<std::string,
                      std::shared_ptr<LibDNNTunerParam>>(name, param));
  }

  if (is_same<T, int32_t>::value || is_same<T, int64_t>::value) {
    std::vector<int64_t> set_values;
    int_tp def_idx = -1;
    for (int_tp i = 0; i < values.size(); ++i) {
      set_values.push_back(values[i]);
      if (def_value == values[i]) {
        def_idx = i;
      }
    }
    if (def_idx == -1) {
      def_idx = set_values.size();
      set_values.push_back(def_value);
    }
    std::shared_ptr<LibDNNTunerParam>
          param(new LibDNNTunerParamInt(this, name, set_values, def_idx));
    params_.push_back(param);
    param_map_.insert(std::pair<std::string,
                      std::shared_ptr<LibDNNTunerParam>>(name, param));
  }
}
template void LibDNNTuner::add_set_param(std::string name,
                            float def_value, std::vector<float> values);
template void LibDNNTuner::add_set_param(std::string name,
                            double def_value, std::vector<double> values);
template void LibDNNTuner::add_set_param(std::string name,
                            int32_t def_value, std::vector<int32_t> values);
template void LibDNNTuner::add_set_param(std::string name,
                            int64_t def_value, std::vector<int64_t> values);

template<>
void LibDNNTuner::add_constraint(std::vector<std::string> con_params,
                    std::vector<std::string> con_adapt,
                    std::function<bool(std::vector<bool>)> con_func) {
  std::shared_ptr<LibDNNTunerConstraint> constraint;
  constraint = std::shared_ptr<LibDNNTunerConstraint>(
      new LibDNNTunerConstraintBool(
      this, con_params, con_adapt, con_func));
  constraints_.push_back(constraint);
  for (int_tp i = 0; i < con_params.size(); ++i) {
    std::shared_ptr<LibDNNTunerParam> param = param_map_.at(con_params[i]);
    param->add_constraint(constraint);
  }
}
template<>
void LibDNNTuner::add_constraint(std::vector<std::string> con_params,
                    std::vector<std::string> con_adapt,
                    std::function<bool(std::vector<double>)> con_func) {
  std::shared_ptr<LibDNNTunerConstraint> constraint;
  constraint = std::shared_ptr<LibDNNTunerConstraint>(
      new LibDNNTunerConstraintReal(
      this, con_params, con_adapt, con_func));
  constraints_.push_back(constraint);
  for (int_tp i = 0; i < con_params.size(); ++i) {
    std::shared_ptr<LibDNNTunerParam> param = param_map_.at(con_params[i]);
    param->add_constraint(constraint);
  }
}
template<>
void LibDNNTuner::add_constraint(std::vector<std::string> con_params,
                    std::vector<std::string> con_adapt,
                    std::function<bool(std::vector<int64_t>)> con_func) {
  std::shared_ptr<LibDNNTunerConstraint> constraint;
  constraint = std::shared_ptr<LibDNNTunerConstraint>(
      new LibDNNTunerConstraintInt(
      this, con_params, con_adapt, con_func));
  constraints_.push_back(constraint);
  for (int_tp i = 0; i < con_params.size(); ++i) {
    std::shared_ptr<LibDNNTunerParam> param = param_map_.at(con_params[i]);
    param->add_constraint(constraint);
  }
}

template<class T>
void LibDNNTuner::add_constraint(std::vector<const char*> con_params,
                    std::vector<const char*> con_adapt,
                    std::function<bool(std::vector<T>)> con_func) {
  std::vector<std::string> con_params_str;
  std::vector<std::string> con_adapt_str;

  for (int_tp i = 0; i < con_params.size(); ++i) {
    std::string str(con_params[i]);
    con_params_str.push_back(str);
  }

  for (int_tp i = 0; i < con_adapt.size(); ++i) {
    std::string str(con_adapt[i]);
    con_adapt_str.push_back(str);
  }

  add_constraint(con_params_str, con_adapt_str, con_func);
}
template void LibDNNTuner::add_constraint(std::vector<const char*> con_params,
                           std::vector<const char*> con_adapt,
                           std::function<bool(std::vector<bool>)> con_func);
template void LibDNNTuner::add_constraint(std::vector<const char*> con_params,
                           std::vector<const char*> con_adapt,
                           std::function<bool(std::vector<double>)> con_func);
template void LibDNNTuner::add_constraint(std::vector<const char*> con_params,
                           std::vector<const char*> con_adapt,
                           std::function<bool(std::vector<int64_t>)> con_func);

template<class T>
void LibDNNTuner::add_constraint(std::vector<const char*> con_params,
                    std::vector<std::string> con_adapt,
                    std::function<bool(std::vector<T>)> con_func) {
  std::vector<std::string> con_params_str;
  std::vector<std::string> con_adapt_str;

  for (int_tp i = 0; i < con_params.size(); ++i) {
    std::string str(con_params[i]);
    con_params_str.push_back(str);
  }

  for (int_tp i = 0; i < con_adapt.size(); ++i) {
    std::string str(con_adapt[i]);
    con_adapt_str.push_back(str);
  }
}
template void LibDNNTuner::add_constraint(std::vector<const char*> con_params,
                           std::vector<std::string> con_adapt,
                           std::function<bool(std::vector<bool>)> con_func);
template void LibDNNTuner::add_constraint(std::vector<const char*> con_params,
                           std::vector<std::string> con_adapt,
                           std::function<bool(std::vector<double>)> con_func);
template void LibDNNTuner::add_constraint(std::vector<const char*> con_params,
                           std::vector<std::string> con_adapt,
                           std::function<bool(std::vector<int64_t>)> con_func);

template<class T>
void LibDNNTuner::add_constraint(std::vector<std::string> con_params,
                    std::vector<const char*> con_adapt,
                    std::function<bool(std::vector<T>)> con_func) {
  std::vector<std::string> con_params_str;
  std::vector<std::string> con_adapt_str;

  for (int_tp i = 0; i < con_params.size(); ++i) {
    std::string str(con_params[i]);
    con_params_str.push_back(str);
  }

  for (int_tp i = 0; i < con_adapt.size(); ++i) {
    std::string str(con_adapt[i]);
    con_adapt_str.push_back(str);
  }
}
template void LibDNNTuner::add_constraint(std::vector<std::string> con_params,
                           std::vector<const char*> con_adapt,
                           std::function<bool(std::vector<bool>)> con_func);
template void LibDNNTuner::add_constraint(std::vector<std::string> con_params,
                           std::vector<const char*> con_adapt,
                           std::function<bool(std::vector<double>)> con_func);
template void LibDNNTuner::add_constraint(std::vector<std::string> con_params,
                           std::vector<const char*> con_adapt,
                           std::function<bool(std::vector<int32_t>)> con_func);

template<class T>
void LibDNNTuner::add_set_param(const char* name,
                                T def_value, std::vector<T> values) {
  std::string str(name);
  add_set_param<T>(str, def_value, values);
}
template void LibDNNTuner::add_set_param(const char* name,
                            float def_value, std::vector<float> values);
template void LibDNNTuner::add_set_param(const char* name,
                            double def_value, std::vector<double> values);
template void LibDNNTuner::add_set_param(const char* name,
                            int32_t def_value, std::vector<int32_t> values);
template void LibDNNTuner::add_set_param(const char* name,
                            int64_t def_value, std::vector<int64_t> values);

void LibDNNTuner::add_boolean_param(std::string name, bool def_value) {
  std::vector<bool> set_values;
  set_values.push_back(def_value);
  set_values.push_back(!def_value);
  std::shared_ptr<LibDNNTunerParam> param(
      new LibDNNTunerParamBool(this, name, set_values, 0));
  params_.push_back(param);
  param_map_.insert(std::pair<std::string,
                    std::shared_ptr<LibDNNTunerParam>>(name, param));
}

void LibDNNTuner::add_boolean_param(const char* name, bool def_value) {
  std::string str(name);
  add_boolean_param(str, def_value);
}


template<class T>
T LibDNNTuner::get_param(std::string name) {
  T value;
  std::shared_ptr<LibDNNTunerParam> param = param_map_.at(name);

  std::shared_ptr<LibDNNTunerParamBool> param_bool =
      std::dynamic_pointer_cast<LibDNNTunerParamBool>(param);
  if (param_bool.get() != nullptr) {
    value = static_cast<T>(param_bool->get_value());
    return value;
  }

  std::shared_ptr<LibDNNTunerParamInt> param_int =
      std::dynamic_pointer_cast<LibDNNTunerParamInt>(param);
  if (param_int.get() != nullptr) {
    value = static_cast<T>(param_int->get_value());
    return value;
  }

  std::shared_ptr<LibDNNTunerParamReal> param_real =
      std::dynamic_pointer_cast<LibDNNTunerParamReal>(param);
  if (param_real.get() != nullptr) {
    value = static_cast<T>(param_real->get_value());
    return value;
  }

  return value;
}
template float LibDNNTuner::get_param(std::string name);
template double LibDNNTuner::get_param(std::string name);
template int32_t LibDNNTuner::get_param(std::string name);
template int64_t LibDNNTuner::get_param(std::string name);
template bool LibDNNTuner::get_param(std::string name);

template<class T>
T LibDNNTuner::get_param(const char* name) {
  std::string str(name);
  return get_param<T>(str);
}
template float LibDNNTuner::get_param(const char* name);
template double LibDNNTuner::get_param(const char* name);
template int32_t LibDNNTuner::get_param(const char* name);
template int64_t LibDNNTuner::get_param(const char* name);
template bool LibDNNTuner::get_param(const char* name);

std::string LibDNNTunerParam::get_name() {
  return name_;
}

libdnnTunerParamStatus_t LibDNNTunerParam::advance(int_tp offset) {
  for (int i = 0; i < abs(offset); ++i) {
    if (offset > 0) {
      ++curr_idx_;
    } else {
      --curr_idx_;
    }
    if (curr_idx_ >= count_values()) {
       curr_idx_ = 0;
    }
    if (curr_idx_ < 0) {
      curr_idx_ = count_values() - 1;
    }
  }
  if (curr_idx_ == def_idx_) {
    return LIBDNN_TUNER_PARAM_STAT_OVERFLOW;
  }

  bool constraints_ok = true;
  for (int i = 0; i < constraints_.size(); ++i) {
    constraints_ok &= constraints_[i]->evaluate();
  }

  if (constraints_ok) {
    return LIBDNN_TUNER_PARAM_STAT_OK;
  } else {
    return LIBDNN_TUNER_PARAM_STAT_NO_SOLUTION;
  }
}

int_tp LibDNNTunerParam::get_curr_idx() {
  return curr_idx_;
}

int_tp LibDNNTunerParam::get_def_idx() {
  return def_idx_;
}

void LibDNNTunerParam::set_curr_idx(int_tp curr_idx) {
  curr_idx_ = curr_idx;
}

void LibDNNTunerParam::set_def_idx(int_tp def_idx) {
  def_idx_ = def_idx;
}

void LibDNNTunerParam::add_constraint(
    std::shared_ptr<LibDNNTunerConstraint> constraint) {
  constraints_.push_back(constraint);
}

double LibDNNTunerSnapshot::get_score() {
  return score_;
}

std::vector<std::shared_ptr<LibDNNTunerParam>>*
  LibDNNTunerSnapshot::get_params() {
  return &params_;
}


int_tp LibDNNTunerParamInt::count_values() {
  return values_.size();
}
int_tp LibDNNTunerParamReal::count_values() {
  return values_.size();
}
int_tp LibDNNTunerParamBool::count_values() {
  return values_.size();
}

int64_t LibDNNTunerParamInt::get_value() {
  // std::cout << name_ << ", value: " << values_[curr_idx_] << std::endl;
  return values_[curr_idx_];
}
double LibDNNTunerParamReal::get_value() {
  // std::cout << name_ << ", value: " << values_[curr_idx_] << std::endl;
  return values_[curr_idx_];
}
bool LibDNNTunerParamBool::get_value() {
  // std::cout << name_ << ", value: " << values_[curr_idx_] << std::endl;
  return values_[curr_idx_];
}

const std::vector<int64_t>& LibDNNTunerParamInt::get_values() {
  return values_;
}
const std::vector<double>& LibDNNTunerParamReal::get_values() {
  return values_;
}
const std::vector<bool>& LibDNNTunerParamBool::get_values() {
  return values_;
}


std::shared_ptr<LibDNNTunerParam> LibDNNTunerParamInt::clone() {
  return std::shared_ptr<LibDNNTunerParamInt>
      (new LibDNNTunerParamInt(*this));
}

std::shared_ptr<LibDNNTunerParam> LibDNNTunerParamReal::clone() {
  return std::shared_ptr<LibDNNTunerParamReal>
      (new LibDNNTunerParamReal(*this));
}

std::shared_ptr<LibDNNTunerParam> LibDNNTunerParamBool::clone() {
  return std::shared_ptr<LibDNNTunerParamBool>
      (new LibDNNTunerParamBool(*this));
}


void LibDNNTunerParam::update(std::shared_ptr<LibDNNTunerParam> other) {
  curr_idx_ = other->get_curr_idx();
  def_idx_ = other->get_def_idx();
}

bool LibDNNTunerConstraintBool::evaluate() {
  std::vector<bool> values;

  for (int_tp i = 0; i < con_params_.size(); ++i) {
    values.push_back(tuner_->get_param<bool>(con_params_[i]));
  }

  return func_(values);
}

bool LibDNNTunerConstraintInt::evaluate() {
  std::vector<int64_t> values;

  for (int_tp i = 0; i < con_params_.size(); ++i) {
    values.push_back(tuner_->get_param<int64_t>(con_params_[i]));
  }

  return func_(values);
}

bool LibDNNTunerConstraintReal::evaluate() {
  std::vector<double> values;

  for (int_tp i = 0; i < con_params_.size(); ++i) {
    values.push_back(tuner_->get_param<double>(con_params_[i]));
  }

  return func_(values);
}

}  // namespace caffe

#endif  // USE_LIBDNN
