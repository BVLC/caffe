#ifndef CAFFE_LIBDNN_LIBDNN_TUNER_HPP_
#define CAFFE_LIBDNN_LIBDNN_TUNER_HPP_
#include <functional>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <type_traits>
#include <vector>
#include "caffe/common.hpp"

namespace caffe {

typedef enum {
    LIBDNN_TUNER_METHOD_ALL          = 0,
    LIBDNN_TUNER_METHOD_ANNEALING    = 1,
} libdnnTunerMethod_t;

typedef enum {
    LIBDNN_TUNER_PARAM_STAT_OK           = 0,
    LIBDNN_TUNER_PARAM_STAT_OVERFLOW     = 1,
    LIBDNN_TUNER_PARAM_STAT_NO_SOLUTION  = 2,
} libdnnTunerParamStatus_t;

class LibDNNTuner;

class LibDNNTunerConstraint {
 public:
  LibDNNTunerConstraint(LibDNNTuner* tuner, vector<string> con_params,
                        vector<string> con_adapt) :
  tuner_(tuner), con_params_(con_params), con_adapt_(con_adapt) {
  }
  virtual bool evaluate() = 0;
 protected:
  LibDNNTuner* tuner_;
  vector<string> con_params_;
  vector<string> con_adapt_;
};

class LibDNNTunerConstraintBool : public LibDNNTunerConstraint {
 public:
  LibDNNTunerConstraintBool(LibDNNTuner* tuner,
                            vector<string> con_params,
                            vector<string> con_adapt,
                            std::function<bool(vector<bool>)> func) :
                            LibDNNTunerConstraint(tuner, con_params, con_adapt),
                            func_(func) {
  }
  bool evaluate();
 protected:
  std::function<bool(vector<bool>)> func_;
};

class LibDNNTunerConstraintReal : public LibDNNTunerConstraint {
 public:
  LibDNNTunerConstraintReal(LibDNNTuner* tuner,
                            vector<string> con_params,
                            vector<string> con_adapt,
                            std::function<bool(vector<double>)> func) :
                            LibDNNTunerConstraint(tuner, con_params, con_adapt),
                            func_(func) {
  }
  bool evaluate();
 protected:
  std::function<bool(vector<double>)> func_;
};

class LibDNNTunerConstraintInt : public LibDNNTunerConstraint {
 public:
  LibDNNTunerConstraintInt(LibDNNTuner* tuner,
                           vector<string> con_params,
                           vector<string> con_adapt,
                           std::function<bool(vector<int64_t>)> func) :
                           LibDNNTunerConstraint(tuner, con_params, con_adapt),
                           func_(func) {
  }
  bool evaluate();
 protected:
  std::function<bool(vector<int64_t>)> func_;
};

class LibDNNTunerParam {
 public:
  LibDNNTunerParam(LibDNNTuner* tuner, string name, int_tp def_idx) :
    constraints_(), tuner_(tuner), name_(name),
    curr_idx_(def_idx), def_idx_(def_idx)
  {}
  LibDNNTunerParam(LibDNNTuner* tuner, LibDNNTunerParam& other) :  // NOLINT
    constraints_(other.constraints_), tuner_(tuner),
    name_(other.name_), curr_idx_(other.curr_idx_), def_idx_(other.def_idx_)
  {}

  virtual void set_value(int64_t value) = 0;
  virtual int_tp count_values() = 0;
  virtual shared_ptr<LibDNNTunerParam> clone() = 0;

  string get_name();

  libdnnTunerParamStatus_t advance(int_tp offset);

  int_tp get_curr_idx();
  int_tp get_def_idx();
  void set_curr_idx(int_tp curr_idx);
  void set_def_idx(int_tp def_idx);
  void update(shared_ptr<LibDNNTunerParam> other);
  void add_constraint(shared_ptr<LibDNNTunerConstraint> constraint);

 protected:
  vector<shared_ptr<LibDNNTunerConstraint>> constraints_;
  LibDNNTuner* tuner_;
  string name_;
  int_tp curr_idx_;
  int_tp def_idx_;
};

class LibDNNTunerParamInt: public LibDNNTunerParam {
 public:
  LibDNNTunerParamInt(LibDNNTuner* tuner,
                      string name, vector<int64_t> values,
                      int_tp def_idx) :
                      LibDNNTunerParam(tuner, name, def_idx) {
    values_ = values;
  }
  LibDNNTunerParamInt(LibDNNTunerParamInt& other) :  // NOLINT
    LibDNNTunerParam(other), values_(other.values_) {
  }
  void set_value(int64_t value);
  int64_t get_value();
  const vector<int64_t>& get_values();
  int_tp count_values();
  shared_ptr<LibDNNTunerParam> clone();
  void restrict_values(int64_t min_value, int64_t max_value);
 protected:
  vector<int64_t> values_;
};

class LibDNNTunerParamBool: public LibDNNTunerParam {
 public:
  LibDNNTunerParamBool(LibDNNTuner* tuner,
                       string name, vector<bool> values,
                       int_tp def_idx) :
                       LibDNNTunerParam(tuner, name, def_idx) {
    values_ = values;
  }
  LibDNNTunerParamBool(LibDNNTunerParamBool& other) :  // NOLINT
    LibDNNTunerParam(other), values_(other.values_) {
  }
  void set_value(int64_t value);
  bool get_value();
  const vector<bool>& get_values();
  int_tp count_values();
  virtual shared_ptr<LibDNNTunerParam> clone();
  void restrict_values(bool min_value, bool max_value);
 protected:
  vector<bool> values_;
};

class LibDNNTunerParamReal: public LibDNNTunerParam {
 public:
  LibDNNTunerParamReal(LibDNNTuner* tuner,
                       string name, vector<double> values,
                       int_tp def_idx) :
                       LibDNNTunerParam(tuner, name, def_idx) {
    values_ = values;
  }
  LibDNNTunerParamReal(LibDNNTunerParamReal& other) :  // NOLINT
    LibDNNTunerParam(other), values_(other.values_) {
  }
  void set_value(int64_t value);
  double get_value();
  const vector<double>& get_values();
  int_tp count_values();
  virtual shared_ptr<LibDNNTunerParam> clone();
  void restrict_values(double min_value, double max_value);
 protected:
  vector<double> values_;
};



class LibDNNTunerSnapshot {
 public:
  LibDNNTunerSnapshot(double score,
                      vector<shared_ptr<LibDNNTunerParam>>* params) :
    score_(score) {
      for (int i = 0; i < params->size(); ++i) {
        shared_ptr<LibDNNTunerParam> param((*params)[i]->clone());
        params_.push_back(param);
      }
  }
  double get_score();
  vector<shared_ptr<LibDNNTunerParam>>* get_params();
 protected:
  double score_;
  vector<shared_ptr<LibDNNTunerParam>> params_;
};

class LibDNNTunerSnapshotCompare {
 public:
  explicit LibDNNTunerSnapshotCompare(const bool& revparam = false)
    { reverse_ = revparam; }
  bool operator() (shared_ptr<LibDNNTunerSnapshot>& lhs,  // NOLINT
                   shared_ptr<LibDNNTunerSnapshot>& rhs) const {  // NOLINT
    if (reverse_)
      return (lhs->get_score() > rhs->get_score());
    else
      return (lhs->get_score() < rhs->get_score());
  }
 private:
  bool reverse_;
};


class LibDNNTuner {
 public:
  explicit LibDNNTuner() :
  constraints_(), params_() {
  }

  void Tune(libdnnTunerMethod_t method);

  string Serialize();

  void Restore(string json);

  void Snapshot(double score);
  void RestoreSnapshot(shared_ptr<LibDNNTunerSnapshot> snapshot);

  void set_setup_routine(std::function<bool()> fun);

  void set_benchmark_routine(std::function<double()> fun);

  void add_boolean_param(string name, bool def_value, bool inverse);
  void add_boolean_param(const char* name, bool def_value, bool inverse);

  template<class T>
  void add_range_param(string name, T def_value, T min, T max, T step);
  template<class T>
  void add_range_param(const char* name, T def_value, T min, T max, T step);

  template<class T>
  void add_set_param(string name, T def_value, vector<T> values);
  template<class T>
  void add_set_param(const char* name, T def_value, vector<T> values);

  template<class T>
  void restrict_param(string name, T min_value, T max_value);
  template<class T>
  void restrict_param(const char* name, T min_value, T max_value);

  template<class T>
  void add_constraint(vector<string> con_params,
                      vector<string> con_adapt,
                      std::function<bool(vector<T>)> con_func);

  template<class T>
  void add_constraint(vector<const char*> con_params,
                      vector<const char*> con_adapt,
                      std::function<bool(vector<T>)> con_func);

  template<class T>
  void add_constraint(vector<const char*> con_params,
                      vector<string> con_adapt,
                      std::function<bool(vector<T>)> con_func);


  template<class T>
  void add_constraint(vector<string> con_params,
                      vector<const char*> con_adapt,
                      std::function<bool(vector<T>)> con_func);

  template<class T>
  T get_param(string name);
  template<class T>
  T get_param(const char* name);

  void load_params(std::map<string, int64_t> params);

 protected:
  void snapshot();

 private:
  std::function<bool()> setup_routine_;
  std::function<double()> benchmark_routine_;

  std::priority_queue<shared_ptr<LibDNNTunerSnapshot>,
    vector<shared_ptr<LibDNNTunerSnapshot> >,
    LibDNNTunerSnapshotCompare> snapshot_queue_;

  vector<shared_ptr<LibDNNTunerSnapshot> > snapshots_;

  vector<shared_ptr<LibDNNTunerConstraint> > constraints_;
  vector<shared_ptr<LibDNNTunerParam> > params_;
  std::map<string, shared_ptr<LibDNNTunerParam> > param_map_;
};

}  // namespace caffe




#endif /* CAFFE_LIBDNN_LIBDNN_TUNER_HPP_ */
