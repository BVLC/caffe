#ifdef USE_LIBDNN

#include <algorithm>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/libdnn/libdnn_tuner.hpp"


namespace caffe {

void LibDNNTuner::set_setup_routine(std::function<bool()> fun) {
  this->setup_routine_ = fun;
}

void LibDNNTuner::set_benchmark_routine(std::function<double()> fun) {
  this->benchmark_routine_ = fun;
}

void LibDNNTuner::load_params(std::map<string, int64_t> params) {
  std::map<string, int64_t>::iterator it;
  for (it = params.begin(); it != params.end(); ++it) {
    if (param_map_.find(it->first) != param_map_.end()) {
      param_map_[it->first]->set_value(it->second);
    }
  }
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
  shared_ptr<LibDNNTunerSnapshot>
        snapshot(new LibDNNTunerSnapshot(score, &params_));
  snapshots_.push_back(snapshot);
  snapshot_queue_.push(snapshot);
}

void LibDNNTuner::RestoreSnapshot(
    shared_ptr<LibDNNTunerSnapshot> snapshot) {
  vector<shared_ptr<LibDNNTunerParam>>* params =
      snapshot->get_params();
  for (int i = 0; i < params_.size(); ++i) {
    params_[i]->update((*params)[i]);
  }
}

template<class T>
void LibDNNTuner::add_range_param(string name,
                                  T def_value, T min, T max, T step) {
  vector<T> values;

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
      values.push_back(value);  vector<T> set_values;
    }
  }

  add_set_param(name, def_value, values);
}
template void LibDNNTuner::add_range_param(string name, float def_value,
                                  float min, float max, float step);
template void LibDNNTuner::add_range_param(string name, double def_value,
                                  double min, double max, double step);
template void LibDNNTuner::add_range_param(string name, int32_t def_value,
                                  int32_t min, int32_t max, int32_t step);
template void LibDNNTuner::add_range_param(string name, int64_t def_value,
                                  int64_t min, int64_t max, int64_t step);

template<class T>
void LibDNNTuner::add_range_param(const char* name,
                                  T def_value, T min, T max, T step) {
  string str(name);
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
void LibDNNTuner::add_set_param(string name,
                                T def_value, vector<T> values) {
  if (is_same<T, float>::value || is_same<T, double>::value) {
    vector<double> set_values;
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
    shared_ptr<LibDNNTunerParam> param(
        new LibDNNTunerParamReal(this, name, set_values, def_idx));
    params_.push_back(param);
    param_map_.insert(std::pair<string,
                      shared_ptr<LibDNNTunerParam>>(name, param));
  }

  if (is_same<T, bool>::value) {
    vector<bool> set_values;
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
    shared_ptr<LibDNNTunerParam> param(
        new LibDNNTunerParamBool(this, name, set_values, def_idx));
    params_.push_back(param);
    param_map_.insert(std::pair<string,
                      shared_ptr<LibDNNTunerParam>>(name, param));
  }

  if (is_same<T, int32_t>::value || is_same<T, int64_t>::value) {
    vector<int64_t> set_values;
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
    shared_ptr<LibDNNTunerParam>
          param(new LibDNNTunerParamInt(this, name, set_values, def_idx));
    params_.push_back(param);
    param_map_.insert(std::pair<string,
                      shared_ptr<LibDNNTunerParam>>(name, param));
  }
}
template void LibDNNTuner::add_set_param(string name,
                            float def_value, vector<float> values);
template void LibDNNTuner::add_set_param(string name,
                            double def_value, vector<double> values);
template void LibDNNTuner::add_set_param(string name,
                            int32_t def_value, vector<int32_t> values);
template void LibDNNTuner::add_set_param(string name,
                            int64_t def_value, vector<int64_t> values);

template<class T>
void LibDNNTuner::restrict_param(const char* name, T min_value, T max_value) {
  string str(name);
  restrict_param<T>(str, min_value, max_value);
}
template void LibDNNTuner::restrict_param(const char* name,
                            float min_value, float max_value);
template void LibDNNTuner::restrict_param(const char* name,
                            double min_value, double max_value);
template void LibDNNTuner::restrict_param(const char* name,
                            int32_t min_value, int32_t max_value);
template void LibDNNTuner::restrict_param(const char* name,
                            int64_t min_value, int64_t max_value);


template<class T>
void LibDNNTuner::restrict_param(string name,
                                          T min_value, T max_value) {
  shared_ptr<LibDNNTunerParam> param = param_map_.at(name);

  shared_ptr<LibDNNTunerParamBool> param_bool =
      std::dynamic_pointer_cast<LibDNNTunerParamBool>(param);
  if (param_bool.get() != nullptr) {
    param_bool->restrict_values(min_value, max_value);
  }

  shared_ptr<LibDNNTunerParamInt> param_int =
      std::dynamic_pointer_cast<LibDNNTunerParamInt>(param);
  if (param_int.get() != nullptr) {
    param_int->restrict_values(min_value, max_value);
  }

  shared_ptr<LibDNNTunerParamReal> param_real =
      std::dynamic_pointer_cast<LibDNNTunerParamReal>(param);
  if (param_real.get() != nullptr) {
    param_real->restrict_values(min_value, max_value);
  }
}
template void LibDNNTuner::restrict_param(string name,
                            float min_value, float max_value);
template void LibDNNTuner::restrict_param(string name,
                            double min_value, double max_value);
template void LibDNNTuner::restrict_param(string name,
                            int32_t min_value, int32_t max_value);
template void LibDNNTuner::restrict_param(string name,
                            int64_t min_value, int64_t max_value);


template<>
void LibDNNTuner::add_constraint(vector<string> con_params,
                    vector<string> con_adapt,
                    std::function<bool(vector<bool>)> con_func) {
  shared_ptr<LibDNNTunerConstraint> constraint;
  constraint = shared_ptr<LibDNNTunerConstraint>(
      new LibDNNTunerConstraintBool(
      this, con_params, con_adapt, con_func));
  constraints_.push_back(constraint);
  for (int_tp i = 0; i < con_params.size(); ++i) {
    shared_ptr<LibDNNTunerParam> param = param_map_.at(con_params[i]);
    param->add_constraint(constraint);
  }
}
template<>
void LibDNNTuner::add_constraint(vector<string> con_params,
                    vector<string> con_adapt,
                    std::function<bool(vector<double>)> con_func) {
  shared_ptr<LibDNNTunerConstraint> constraint;
  constraint = shared_ptr<LibDNNTunerConstraint>(
      new LibDNNTunerConstraintReal(
      this, con_params, con_adapt, con_func));
  constraints_.push_back(constraint);
  for (int_tp i = 0; i < con_params.size(); ++i) {
    shared_ptr<LibDNNTunerParam> param = param_map_.at(con_params[i]);
    param->add_constraint(constraint);
  }
}
template<>
void LibDNNTuner::add_constraint(vector<string> con_params,
                    vector<string> con_adapt,
                    std::function<bool(vector<int64_t>)> con_func) {
  shared_ptr<LibDNNTunerConstraint> constraint;
  constraint = shared_ptr<LibDNNTunerConstraint>(
      new LibDNNTunerConstraintInt(
      this, con_params, con_adapt, con_func));
  constraints_.push_back(constraint);
  for (int_tp i = 0; i < con_params.size(); ++i) {
    shared_ptr<LibDNNTunerParam> param = param_map_.at(con_params[i]);
    param->add_constraint(constraint);
  }
}

template<class T>
void LibDNNTuner::add_constraint(vector<const char*> con_params,
                    vector<const char*> con_adapt,
                    std::function<bool(vector<T>)> con_func) {
  vector<string> con_params_str;
  vector<string> con_adapt_str;

  for (int_tp i = 0; i < con_params.size(); ++i) {
    string str(con_params[i]);
    con_params_str.push_back(str);
  }

  for (int_tp i = 0; i < con_adapt.size(); ++i) {
    string str(con_adapt[i]);
    con_adapt_str.push_back(str);
  }

  add_constraint(con_params_str, con_adapt_str, con_func);
}
template void LibDNNTuner::add_constraint(vector<const char*> con_params,
                           vector<const char*> con_adapt,
                           std::function<bool(vector<bool>)> con_func);
template void LibDNNTuner::add_constraint(vector<const char*> con_params,
                           vector<const char*> con_adapt,
                           std::function<bool(vector<double>)> con_func);
template void LibDNNTuner::add_constraint(vector<const char*> con_params,
                           vector<const char*> con_adapt,
                           std::function<bool(vector<int64_t>)> con_func);

template<class T>
void LibDNNTuner::add_constraint(vector<const char*> con_params,
                    vector<string> con_adapt,
                    std::function<bool(vector<T>)> con_func) {
  vector<string> con_params_str;
  vector<string> con_adapt_str;

  for (int_tp i = 0; i < con_params.size(); ++i) {
    string str(con_params[i]);
    con_params_str.push_back(str);
  }

  for (int_tp i = 0; i < con_adapt.size(); ++i) {
    string str(con_adapt[i]);
    con_adapt_str.push_back(str);
  }
}
template void LibDNNTuner::add_constraint(vector<const char*> con_params,
                           vector<string> con_adapt,
                           std::function<bool(vector<bool>)> con_func);
template void LibDNNTuner::add_constraint(vector<const char*> con_params,
                           vector<string> con_adapt,
                           std::function<bool(vector<double>)> con_func);
template void LibDNNTuner::add_constraint(vector<const char*> con_params,
                           vector<string> con_adapt,
                           std::function<bool(vector<int64_t>)> con_func);

template<class T>
void LibDNNTuner::add_constraint(vector<string> con_params,
                    vector<const char*> con_adapt,
                    std::function<bool(vector<T>)> con_func) {
  vector<string> con_params_str;
  vector<string> con_adapt_str;

  for (int_tp i = 0; i < con_params.size(); ++i) {
    string str(con_params[i]);
    con_params_str.push_back(str);
  }

  for (int_tp i = 0; i < con_adapt.size(); ++i) {
    string str(con_adapt[i]);
    con_adapt_str.push_back(str);
  }
}
template void LibDNNTuner::add_constraint(vector<string> con_params,
                           vector<const char*> con_adapt,
                           std::function<bool(vector<bool>)> con_func);
template void LibDNNTuner::add_constraint(vector<string> con_params,
                           vector<const char*> con_adapt,
                           std::function<bool(vector<double>)> con_func);
template void LibDNNTuner::add_constraint(vector<string> con_params,
                           vector<const char*> con_adapt,
                           std::function<bool(vector<int32_t>)> con_func);

template<class T>
void LibDNNTuner::add_set_param(const char* name,
                                T def_value, vector<T> values) {
  string str(name);
  add_set_param<T>(str, def_value, values);
}
template void LibDNNTuner::add_set_param(const char* name,
                            float def_value, vector<float> values);
template void LibDNNTuner::add_set_param(const char* name,
                            double def_value, vector<double> values);
template void LibDNNTuner::add_set_param(const char* name,
                            int32_t def_value, vector<int32_t> values);
template void LibDNNTuner::add_set_param(const char* name,
                            int64_t def_value, vector<int64_t> values);

void LibDNNTuner::add_boolean_param(string name,
                                    bool def_value, bool inverse) {
  vector<bool> set_values;
  set_values.push_back(def_value);
  if (inverse) {
    set_values.push_back(!def_value);
  }
  shared_ptr<LibDNNTunerParam> param(
      new LibDNNTunerParamBool(this, name, set_values, 0));
  params_.push_back(param);
  param_map_.insert(std::pair<string,
                    shared_ptr<LibDNNTunerParam>>(name, param));
}

void LibDNNTuner::add_boolean_param(const char* name,
                                    bool def_value, bool inverse) {
  string str(name);
  add_boolean_param(str, def_value, inverse);
}


template<class T>
T LibDNNTuner::get_param(string name) {
  T value;
  shared_ptr<LibDNNTunerParam> param = param_map_.at(name);

  shared_ptr<LibDNNTunerParamBool> param_bool =
      std::dynamic_pointer_cast<LibDNNTunerParamBool>(param);
  if (param_bool.get() != nullptr) {
    value = static_cast<T>(param_bool->get_value());
    return value;
  }

  shared_ptr<LibDNNTunerParamInt> param_int =
      std::dynamic_pointer_cast<LibDNNTunerParamInt>(param);
  if (param_int.get() != nullptr) {
    value = static_cast<T>(param_int->get_value());
    return value;
  }

  shared_ptr<LibDNNTunerParamReal> param_real =
      std::dynamic_pointer_cast<LibDNNTunerParamReal>(param);
  if (param_real.get() != nullptr) {
    value = static_cast<T>(param_real->get_value());
    return value;
  }

  return value;
}
template float LibDNNTuner::get_param(string name);
template double LibDNNTuner::get_param(string name);
template int32_t LibDNNTuner::get_param(string name);
template int64_t LibDNNTuner::get_param(string name);
template bool LibDNNTuner::get_param(string name);

template<class T>
T LibDNNTuner::get_param(const char* name) {
  string str(name);
  return get_param<T>(str);
}
template float LibDNNTuner::get_param(const char* name);
template double LibDNNTuner::get_param(const char* name);
template int32_t LibDNNTuner::get_param(const char* name);
template int64_t LibDNNTuner::get_param(const char* name);
template bool LibDNNTuner::get_param(const char* name);

string LibDNNTunerParam::get_name() {
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
    shared_ptr<LibDNNTunerConstraint> constraint) {
  constraints_.push_back(constraint);
}

double LibDNNTunerSnapshot::get_score() {
  return score_;
}

vector<shared_ptr<LibDNNTunerParam>>*
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

void LibDNNTunerParamInt::set_value(int64_t value) {
  for (int i = 0; i < values_.size(); ++i) {
    if (values_[i] == value) {
      curr_idx_ = i;
    }
  }
}
void LibDNNTunerParamReal::set_value(int64_t value) {
  for (int i = 0; i < values_.size(); ++i) {
    if (values_[i] == value) {
      curr_idx_ = i;
    }
  }
}
void LibDNNTunerParamBool::set_value(int64_t value) {
  for (int i = 0; i < values_.size(); ++i) {
    if (values_[i] == value) {
      curr_idx_ = i;
    }
  }
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

const vector<int64_t>& LibDNNTunerParamInt::get_values() {
  return values_;
}
const vector<double>& LibDNNTunerParamReal::get_values() {
  return values_;
}
const vector<bool>& LibDNNTunerParamBool::get_values() {
  return values_;
}


shared_ptr<LibDNNTunerParam> LibDNNTunerParamInt::clone() {
  return shared_ptr<LibDNNTunerParamInt>
      (new LibDNNTunerParamInt(*this));
}

shared_ptr<LibDNNTunerParam> LibDNNTunerParamReal::clone() {
  return shared_ptr<LibDNNTunerParamReal>
      (new LibDNNTunerParamReal(*this));
}

shared_ptr<LibDNNTunerParam> LibDNNTunerParamBool::clone() {
  return shared_ptr<LibDNNTunerParamBool>
      (new LibDNNTunerParamBool(*this));
}

void LibDNNTunerParamInt::restrict_values(
    int64_t min_value, int64_t max_value) {
  vector<int64_t> new_values;
  int64_t def_value = values_[def_idx_];
  int64_t curr_value = values_[curr_idx_];
  for (int_tp i = 0; i < values_.size(); ++i) {
    int64_t value = values_[i];
    if (value >= min_value && value <= max_value) {
      new_values.push_back(value);
    }
  }
  if (new_values.size() > 0) {
    values_ = new_values;
    int_tp min_idx = 0;
    int64_t min_set_value = values_[0];
    int_tp max_idx = 0;
    int64_t max_set_value = values_[0];
    for (int_tp i = 0; i < values_.size(); ++i) {
      if (values_[i] < min_set_value) {
        min_set_value = values_[i];
        min_idx = i;
      }
      if (values_[i] > max_set_value) {
        max_set_value = values_[i];
        max_idx = i;
      }
      if (def_value == values_[i]) {
        def_idx_ = i;
      }
      if (curr_value == values_[i]) {
        curr_idx_ = i;
      }
    }
    if (def_value < min_set_value) {
      def_idx_ = min_idx;
    }
    if (def_value > max_set_value) {
      def_idx_ = max_idx;
    }
    if (curr_value < min_set_value) {
      curr_idx_ = min_idx;
    }
    if (curr_value > max_set_value) {
      curr_idx_ = max_idx;
    }
  }
}

void LibDNNTunerParamReal::restrict_values(
    double min_value, double max_value) {
  vector<double> new_values;
  double def_value = values_[def_idx_];
  double curr_value = values_[curr_idx_];
  for (int_tp i = 0; i < values_.size(); ++i) {
    double value = values_[i];
    if (value >= min_value && value <= max_value) {
      values_.push_back(value);
    }
  }
  if (new_values.size() > 0) {
    values_ = new_values;
    int_tp min_idx = 0;
    double min_set_value = values_[0];
    int_tp max_idx = 0;
    double max_set_value = values_[0];
    for (int_tp i = 0; i < values_.size(); ++i) {
      if (values_[i] < min_set_value) {
        min_set_value = values_[i];
        min_idx = i;
      }
      if (values_[i] > max_set_value) {
        max_set_value = values_[i];
        max_idx = i;
      }
      if (def_value == values_[i]) {
        def_idx_ = i;
      }
      if (curr_value == values_[i]) {
        curr_idx_ = i;
      }
    }
    if (def_value < min_set_value) {
      def_idx_ = min_idx;
    }
    if (def_value > max_set_value) {
      def_idx_ = max_idx;
    }
    if (curr_value < min_set_value) {
      curr_idx_ = min_idx;
    }
    if (curr_value > max_set_value) {
      curr_idx_ = max_idx;
    }
  }
}

void LibDNNTunerParamBool::restrict_values(
    bool min_value, bool max_value) {
  vector<bool> new_values;
  bool def_value = values_[def_idx_];
  bool curr_value = values_[curr_idx_];
  for (int_tp i = 0; i < values_.size(); ++i) {
    bool value = values_[i];
    if (value >= min_value && value <= max_value) {
      values_.push_back(value);
    }
  }
  if (new_values.size() > 0) {
    values_ = new_values;
    int_tp min_idx = 0;
    bool min_set_value = values_[0];
    int_tp max_idx = 0;
    bool max_set_value = values_[0];
    for (int_tp i = 0; i < values_.size(); ++i) {
      if (values_[i] < min_set_value) {
        min_set_value = values_[i];
        min_idx = i;
      }
      if (values_[i] > max_set_value) {
        max_set_value = values_[i];
        max_idx = i;
      }
      if (def_value == values_[i]) {
        def_idx_ = i;
      }
      if (curr_value == values_[i]) {
        curr_idx_ = i;
      }
    }
    if (def_value < min_set_value) {
      def_idx_ = min_idx;
    }
    if (def_value > max_set_value) {
      def_idx_ = max_idx;
    }
    if (curr_value < min_set_value) {
      curr_idx_ = min_idx;
    }
    if (curr_value > max_set_value) {
      curr_idx_ = max_idx;
    }
  }
}

void LibDNNTunerParam::update(shared_ptr<LibDNNTunerParam> other) {
  curr_idx_ = other->get_curr_idx();
  def_idx_ = other->get_def_idx();
}

bool LibDNNTunerConstraintBool::evaluate() {
  vector<bool> values;

  for (int_tp i = 0; i < con_params_.size(); ++i) {
    values.push_back(tuner_->get_param<bool>(con_params_[i]));
  }

  return func_(values);
}

bool LibDNNTunerConstraintInt::evaluate() {
  vector<int64_t> values;

  for (int_tp i = 0; i < con_params_.size(); ++i) {
    values.push_back(tuner_->get_param<int64_t>(con_params_[i]));
  }

  return func_(values);
}

bool LibDNNTunerConstraintReal::evaluate() {
  vector<double> values;

  for (int_tp i = 0; i < con_params_.size(); ++i) {
    values.push_back(tuner_->get_param<double>(con_params_[i]));
  }

  return func_(values);
}

}  // namespace caffe

#endif  // USE_LIBDNN
