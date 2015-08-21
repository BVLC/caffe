#ifndef CAFFE_APOLLO_NET_HPP_
#define CAFFE_APOLLO_NET_HPP_

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
class ApolloNet {
 public:
  explicit ApolloNet();
  virtual ~ApolloNet() {}

  void Init() {
    phase_ = TRAIN;
  }

  Dtype ForwardLayer(const string& layer_param_string);

  Dtype ForwardLayer(const LayerParameter& layer_param);

  Dtype f(shared_ptr<Layer<Dtype> > layer);

  Dtype f(const string& layer_prototxt);

  void BackwardLayer(const string& layer_name);

  void Backward();

  void ResetForward() {
    active_layers_vec_.clear();
    active_layers_set_.clear();
    active_params_set_.clear();
    active_blobs_set_.clear();
  }

  void AddLayerParams(shared_ptr<Layer<Dtype> > layer);

  Dtype DiffL2Norm();

  void CopyTrainedLayersFrom(const NetParameter& param);

  void CopyTrainedLayersFrom(const string trained_filename) {
    NetParameter param;
    ReadNetParamsFromBinaryFileOrDie(trained_filename, &param);
    CopyTrainedLayersFrom(param);
  }

  void CopyLayerFrom(const LayerParameter& source_layer);

  void SaveTrainedLayersTo(const string trained_filename) const;

  void Update(Dtype lr, Dtype momentum, Dtype clip_gradients,
    Dtype weight_decay);

  /// @brief returns the phase: TRAIN or TEST
  inline Phase phase() const { return phase_; }

  inline void set_phase_test() {
    phase_ = TEST;
  }
  inline void set_phase_train() {
    phase_ = TRAIN;
  }
  inline map<string, shared_ptr<Blob<Dtype> > >& params() {
    return params_;
  }
  inline map<string, Dtype>& param_decay_mults() {
    return param_decay_mults_;
  }
  inline map<string, Dtype>& param_lr_mults() {
    return param_lr_mults_;
  }
  inline map<string, shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }
  inline map<string, shared_ptr<Layer<Dtype> > >& layers() {
    return layers_map_;
  }
  inline const vector<string>& active_layer_names() const {
    return active_layers_vec_;
  }
  inline const set<string>& active_param_names() const {
    return active_params_set_;
  }

 protected:
  /// @brief The phase: TRAIN or TEST
  Phase phase_;
  /// @brief Individual layers in the net
  map<string, shared_ptr<Layer<Dtype> > > layers_map_;
  /// @brief the blobs storing top results after each layer.
  map<string, shared_ptr<Blob<Dtype> > > blobs_;
  map<string, shared_ptr<Blob<Dtype> > > params_;
  map<string, shared_ptr<Blob<Dtype> > > local_params_;
  map<string, Dtype> param_decay_mults_;
  map<string, Dtype> param_lr_mults_;
  map<string, vector<shared_ptr<Blob<Dtype> > > > bottom_blobs_;
  map<string, vector<string> > bottom_blob_names_;
  map<string, LayerParameter> param_cache_;
  vector<string> active_layers_vec_;
  set<string> active_layers_set_;
  set<string> active_params_set_;
  set<string> active_blobs_set_;

  DISABLE_COPY_AND_ASSIGN(ApolloNet);
};

}  // namespace caffe

#endif  // CAFFE_APOLLO_NET_HPP_
