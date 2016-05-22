#ifndef CAFFE_GREENTEA_LIBDNN_TUNER_HPP_
#define CAFFE_GREENTEA_LIBDNN_TUNER_HPP_
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
  LibDNNTunerConstraint(LibDNNTuner* tuner, std::vector<std::string> con_params,
                        std::vector<std::string> con_adapt) :
  tuner_(tuner), con_params_(con_params), con_adapt_(con_adapt) {
  }
  virtual bool evaluate() = 0;
 protected:
  LibDNNTuner* tuner_;
  std::vector<std::string> con_params_;
  std::vector<std::string> con_adapt_;
};

class LibDNNTunerConstraintBool : public LibDNNTunerConstraint {
 public:
  LibDNNTunerConstraintBool(LibDNNTuner* tuner,
                            std::vector<std::string> con_params,
                            std::vector<std::string> con_adapt,
                            std::function<bool(std::vector<bool>)> func) :
                            LibDNNTunerConstraint(tuner, con_params, con_adapt),
                            func_(func) {
  }
  bool evaluate();
 protected:
  std::function<bool(std::vector<bool>)> func_;
};

class LibDNNTunerConstraintReal : public LibDNNTunerConstraint {
 public:
  LibDNNTunerConstraintReal(LibDNNTuner* tuner,
                            std::vector<std::string> con_params,
                            std::vector<std::string> con_adapt,
                            std::function<bool(std::vector<double>)> func) :
                            LibDNNTunerConstraint(tuner, con_params, con_adapt),
                            func_(func) {
  }
  bool evaluate();
 protected:
  std::function<bool(std::vector<double>)> func_;
};

class LibDNNTunerConstraintInt : public LibDNNTunerConstraint {
 public:
  LibDNNTunerConstraintInt(LibDNNTuner* tuner,
                           std::vector<std::string> con_params,
                           std::vector<std::string> con_adapt,
                           std::function<bool(std::vector<int64_t>)> func) :
                           LibDNNTunerConstraint(tuner, con_params, con_adapt),
                           func_(func) {
  }
  bool evaluate();
 protected:
  std::function<bool(std::vector<int64_t>)> func_;
};

class LibDNNTunerParam {
 public:
  LibDNNTunerParam(LibDNNTuner* tuner, std::string name, int_tp def_idx) :
    constraints_(), tuner_(tuner), name_(name),
    curr_idx_(def_idx), def_idx_(def_idx)
  {}
  LibDNNTunerParam(LibDNNTuner* tuner, LibDNNTunerParam& other) :  // NOLINT
    constraints_(other.constraints_), tuner_(tuner),
    name_(other.name_), curr_idx_(other.curr_idx_), def_idx_(other.def_idx_)
  {}

  virtual int_tp count_values() = 0;
  virtual std::shared_ptr<LibDNNTunerParam> clone() = 0;

  std::string get_name();

  libdnnTunerParamStatus_t advance(int_tp offset);

  int_tp get_curr_idx();
  int_tp get_def_idx();
  void set_curr_idx(int_tp curr_idx);
  void set_def_idx(int_tp def_idx);
  void update(std::shared_ptr<LibDNNTunerParam> other);
  void add_constraint(std::shared_ptr<LibDNNTunerConstraint> constraint);

 protected:
  LibDNNTuner* tuner_;
  std::string name_;
  int_tp curr_idx_;
  int_tp def_idx_;
  std::vector<std::shared_ptr<LibDNNTunerConstraint>> constraints_;
};

class LibDNNTunerParamInt: public LibDNNTunerParam {
 public:
  LibDNNTunerParamInt(LibDNNTuner* tuner,
                      std::string name, std::vector<int64_t> values,
                      int_tp def_idx) :
                      LibDNNTunerParam(tuner, name, def_idx) {
    values_ = values;
  }
  LibDNNTunerParamInt(LibDNNTunerParamInt& other) :  // NOLINT
    LibDNNTunerParam(other), values_(other.values_) {
  }
  int64_t get_value();
  const std::vector<int64_t>& get_values();
  int_tp count_values();
  std::shared_ptr<LibDNNTunerParam> clone();
 protected:
  std::vector<int64_t> values_;
};

class LibDNNTunerParamBool: public LibDNNTunerParam {
 public:
  LibDNNTunerParamBool(LibDNNTuner* tuner,
                       std::string name, std::vector<bool> values,
                       int_tp def_idx) :
                       LibDNNTunerParam(tuner, name, def_idx) {
    values_ = values;
  }
  LibDNNTunerParamBool(LibDNNTunerParamBool& other) :  // NOLINT
    LibDNNTunerParam(other), values_(other.values_) {
  }
  bool get_value();
  const std::vector<bool>& get_values();
  int_tp count_values();
  virtual std::shared_ptr<LibDNNTunerParam> clone();
 protected:
  std::vector<bool> values_;
};

class LibDNNTunerParamReal: public LibDNNTunerParam {
 public:
  LibDNNTunerParamReal(LibDNNTuner* tuner,
                       std::string name, std::vector<double> values,
                       int_tp def_idx) :
                       LibDNNTunerParam(tuner, name, def_idx) {
    values_ = values;
  }
  LibDNNTunerParamReal(LibDNNTunerParamReal& other) :  // NOLINT
    LibDNNTunerParam(other), values_(other.values_) {
  }
  double get_value();
  const std::vector<double>& get_values();
  int_tp count_values();
  virtual std::shared_ptr<LibDNNTunerParam> clone();
 protected:
  std::vector<double> values_;
};



class LibDNNTunerSnapshot {
 public:
  LibDNNTunerSnapshot(double score,
                      std::vector<std::shared_ptr<LibDNNTunerParam>>* params) :
    score_(score) {
      for (int i = 0; i < params->size(); ++i) {
        std::shared_ptr<LibDNNTunerParam> param((*params)[i]->clone());
        params_.push_back(param);
      }
  }
  double get_score();
  std::vector<std::shared_ptr<LibDNNTunerParam>>* get_params();
 protected:
  double score_;
  std::vector<std::shared_ptr<LibDNNTunerParam>> params_;
};

class LibDNNTunerSnapshotCompare {
 public:
  explicit LibDNNTunerSnapshotCompare(const bool& revparam = false)
    { reverse_ = revparam; }
  bool operator() (std::shared_ptr<LibDNNTunerSnapshot>& lhs,  // NOLINT
                   std::shared_ptr<LibDNNTunerSnapshot>& rhs) const {  // NOLINT
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

  std::string Serialize();

  void Restore(std::string json);

  void Snapshot(double score);
  void RestoreSnapshot(std::shared_ptr<LibDNNTunerSnapshot> snapshot);

  void set_setup_routine(std::function<bool()> fun);

  void set_benchmark_routine(std::function<double()> fun);

  void add_boolean_param(std::string name, bool def_value);
  void add_boolean_param(const char* name, bool def_value);

  template<class T>
  void add_range_param(std::string name, T def_value, T min, T max, T step);
  template<class T>
  void add_range_param(const char* name, T def_value, T min, T max, T step);

  template<class T>
  void add_set_param(std::string name, T def_value, std::vector<T> values);
  template<class T>
  void add_set_param(const char* name, T def_value, std::vector<T> values);

  template<class T>
  void add_constraint(std::vector<std::string> con_params,
                      std::vector<std::string> con_adapt,
                      std::function<bool(std::vector<T>)> con_func);

  template<class T>
  void add_constraint(std::vector<const char*> con_params,
                      std::vector<const char*> con_adapt,
                      std::function<bool(std::vector<T>)> con_func);

  template<class T>
  void add_constraint(std::vector<const char*> con_params,
                      std::vector<std::string> con_adapt,
                      std::function<bool(std::vector<T>)> con_func);


  template<class T>
  void add_constraint(std::vector<std::string> con_params,
                      std::vector<const char*> con_adapt,
                      std::function<bool(std::vector<T>)> con_func);

  template<class T>
  T get_param(std::string name);
  template<class T>
  T get_param(const char* name);

 protected:
  void snapshot();

 private:
  std::function<bool()> setup_routine_;
  std::function<double()> benchmark_routine_;

  std::priority_queue<std::shared_ptr<LibDNNTunerSnapshot>,
    std::vector<std::shared_ptr<LibDNNTunerSnapshot>>,
    LibDNNTunerSnapshotCompare> snapshot_queue_;

  std::vector<std::shared_ptr<LibDNNTunerSnapshot>> snapshots_;

  std::vector<std::shared_ptr<LibDNNTunerConstraint> > constraints_;
  std::vector<std::shared_ptr<LibDNNTunerParam> > params_;
  std::map<std::string, std::shared_ptr<LibDNNTunerParam>> param_map_;
};

}  // namespace caffe




#endif /* CAFFE_GREENTEA_LIBDNN_TUNER_HPP_ */
