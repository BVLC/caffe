#include <google/protobuf/text_format.h>

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "caffe/apollonet.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe {

template <typename Dtype>
ApolloNet<Dtype>::ApolloNet() {
  Init();
}

template <typename Dtype>
Dtype ApolloNet<Dtype>::f(const string& layer_prototxt) {
  LayerParameter p;
  bool success =
    google::protobuf::TextFormat::ParseFromString(layer_prototxt, &p);
  ASSERT(success, "Invalid prototxt string");
  return ForwardLayer(p);
}

template <typename Dtype>
Dtype ApolloNet<Dtype>::ForwardLayer(const string& layer_param_string) {
  /* This function will
   * 1) Check if the layer name is in the cache
   * 2) Create the layer if it is new
   * 3) Set up the top blobs
   * 4) Set up the bottom blobs
   * 5) Set up the parameters
   * 6) Call the Forward function */

  LayerParameter active_layer_param;
  ASSERT(active_layer_param.ParseFromString(layer_param_string), "");
  return ForwardLayer(active_layer_param);
}

template <typename Dtype>
Dtype ApolloNet<Dtype>::f(shared_ptr<Layer<Dtype> > layer) {
  /* This function will
   * 1) Add the layer to the cache if it's new
   * 2) Set up the top blobs
   * 3) Set up the bottom blobs
   * 4) Set up the parameters
   * 5) Call the Forward function */

  const LayerParameter& active_layer_param = layer->layer_param();
  string layer_name = active_layer_param.name();

  const bool new_layer = layers_map_.find(layer_name) == layers_map_.end();
  if (new_layer) {
    LOG(INFO) << "Adding Layer " << layer_name;
    LOG(INFO) << active_layer_param.DebugString();
    layers_map_[layer_name] = layer;
    active_layers_set_.insert(layer_name);
  }

  active_layers_vec_.push_back(layer_name);
  vector<Blob<Dtype>*> bottom_vec;
  vector<Blob<Dtype>*> top_vec;

  const vector<string>& bottom_names = bottom_blob_names_[layer_name];
  bool reset_bottoms = (active_layer_param.bottom_size()
      != bottom_names.size());
  for (int i = 0; i < active_layer_param.bottom_size(); ++i) {
    const string& blob_name = active_layer_param.bottom(i);
    ASSERT(blobs_.find(blob_name) != blobs_.end(),
      "Could not find bottom: '" << blob_name <<
      "' for layer: " << layer_name);
    if (bottom_names.size() > i &&
        bottom_names[i] != blob_name) {
      reset_bottoms = true;
    }
  }

  if (new_layer || reset_bottoms) {
    // Reset the bottom blobs
    bottom_blobs_[layer_name].clear();
    bottom_blob_names_[layer_name].clear();
    for (int i = 0; i < active_layer_param.bottom_size(); ++i) {
      const string& blob_name = active_layer_param.bottom(i);
      shared_ptr<Blob<Dtype> > top_blob = blobs_[blob_name];
      bottom_blob_names_[layer_name].push_back(blob_name);
      shared_ptr<Blob<Dtype> > bottom_blob(
        new Blob<Dtype>(top_blob->shape()));
      bottom_blobs_[layer_name].push_back(bottom_blob);
    }
    layer->reset_bottoms(bottom_blob_names_[layer_name]);
  }

  for (int i = 0; i < active_layer_param.bottom_size(); ++i) {
    // Reshape bottom_blobs to match their respective top blobs
    const string& blob_name = active_layer_param.bottom(i);
    shared_ptr<Blob<Dtype> > top_blob = blobs_[blob_name];
    shared_ptr<Blob<Dtype> > bottom_blob = bottom_blobs_[layer_name][i];

    bottom_blob->ReshapeLike(*top_blob);
    bottom_blob->ShareData(*top_blob);
    if (layer->in_place_layer() || !layer->overwrites_bottom_diffs()) {
      // save memory when layer accumulates delta rather than overwriting
      bottom_blob->ShareDiff(*top_blob);
    }
  }

  for (int i = 0; i < active_layer_param.bottom_size(); ++i) {
    bottom_vec.push_back(bottom_blobs_[layer_name][i].get());
  }

  ASSERT(layer->layer_param().top_size()
      == active_layer_param.top_size(), "top vec cannot change");
  for (int top_id = 0; top_id < active_layer_param.top_size(); ++top_id) {
    ASSERT(layer->layer_param().top(top_id)
      == active_layer_param.top(top_id), "top vec cannot change");
  }

  for (int top_id = 0; top_id < active_layer_param.top_size(); ++top_id) {
    const string& blob_name = active_layer_param.top(top_id);
    if (blobs_.find(blob_name) == blobs_.end()) {
      shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
      blobs_[blob_name] = blob_pointer;
    }
    Blob<Dtype>* top_blob = blobs_[blob_name].get();
    if (!layer->in_place_layer()) {
      std::pair<set<string>::iterator, bool> ret;
      ret = active_blobs_set_.insert(blob_name);
      ASSERT(ret.second, "Top with name '"
          << blob_name << "' is already used");
    }
    top_vec.push_back(top_blob);
    if (top_blob->DiffInitialized() && !layer->is_loss()) {
      // Zero out top_diffs, except for loss blobs, which never change
      top_blob->SetDiffValues(0.);
    }
  }

  if (new_layer) {
    layer->SetUp(bottom_vec, top_vec);
    AddLayerParams(layer);
    if (param_cache_.find(layer_name) != param_cache_.end()) {
      CopyLayerFrom(param_cache_[layer_name]);
    }
  }

  for (int param_id = 0; param_id < layer->param_names().size(); ++param_id) {
    const string& param_name = layer->param_names()[param_id];
    active_params_set_.insert(param_name);
  }

  Dtype loss = 0;
  layer->set_phase(phase_);
  loss = layer->Forward(bottom_vec, top_vec);
  return loss;
}

template <typename Dtype>
Dtype ApolloNet<Dtype>::ForwardLayer(const LayerParameter& active_layer_param) {
  /* This function will
   * 1) Check if the layer name is in the cache
   * 2) Create the layer if it is new
   * 3) Set up the top blobs
   * 4) Set up the bottom blobs
   * 5) Set up the parameters
   * 6) Call the Forward function */

  RuntimeParameter runtime_param = active_layer_param.rp();
  ASSERT(active_layer_param.has_name(), "");
  const string& layer_name = active_layer_param.name();
  shared_ptr<Layer<Dtype> > layer;
  const bool new_layer = layers_map_.find(layer_name) == layers_map_.end();
  if (new_layer) {
    layer = LayerRegistry<Dtype>::CreateLayer(active_layer_param);;
    LOG(INFO) << "Creating Layer " << layer_name;
    LOG(INFO) << active_layer_param.DebugString();
    layers_map_[layer_name] = layer;
    active_layers_set_.insert(layer_name);
  } else {
    layer = layers_map_[layer_name];
    std::pair<set<string>::iterator, bool> ret;
    ret = active_layers_set_.insert(layer_name);
    ASSERT(ret.second, "Layer with name '" << layer_name
        << "' is already used");
    ASSERT(layer->layer_param().type() == active_layer_param.type(),
        "WARNING: layer with name '" << active_layer_param.name()
        << "' and different type already exists.");
  }
  layer->set_runtime_param(runtime_param);

  active_layers_vec_.push_back(layer_name);
  vector<Blob<Dtype>*> bottom_vec;
  vector<Blob<Dtype>*> top_vec;

  const vector<string>& bottom_names = bottom_blob_names_[layer_name];
  bool reset_bottoms = (active_layer_param.bottom_size()
      != bottom_names.size());
  for (int i = 0; i < active_layer_param.bottom_size(); ++i) {
    const string& blob_name = active_layer_param.bottom(i);
    ASSERT(blobs_.find(blob_name) != blobs_.end(),
      "Could not find bottom: '" << blob_name <<
      "' for layer: " << layer_name);
    if (bottom_names.size() > i &&
        bottom_names[i] != blob_name) {
      reset_bottoms = true;
    }
  }

  if (new_layer || reset_bottoms) {
    // Reset the bottom blobs
    bottom_blobs_[layer_name].clear();
    bottom_blob_names_[layer_name].clear();
    for (int i = 0; i < active_layer_param.bottom_size(); ++i) {
      const string& blob_name = active_layer_param.bottom(i);
      shared_ptr<Blob<Dtype> > top_blob = blobs_[blob_name];
      bottom_blob_names_[layer_name].push_back(blob_name);
      shared_ptr<Blob<Dtype> > bottom_blob(
        new Blob<Dtype>(top_blob->shape()));
      bottom_blobs_[layer_name].push_back(bottom_blob);
    }
    layer->reset_bottoms(bottom_blob_names_[layer_name]);
  }

  for (int i = 0; i < active_layer_param.bottom_size(); ++i) {
    // Reshape bottom_blobs to match their respective top blobs
    const string& blob_name = active_layer_param.bottom(i);
    shared_ptr<Blob<Dtype> > top_blob = blobs_[blob_name];
    shared_ptr<Blob<Dtype> > bottom_blob = bottom_blobs_[layer_name][i];

    bottom_blob->ReshapeLike(*top_blob);
    bottom_blob->ShareData(*top_blob);
    if (layer->in_place_layer() || !layer->overwrites_bottom_diffs()) {
      // save memory when layer accumulates delta rather than overwriting
      bottom_blob->ShareDiff(*top_blob);
    }
  }

  for (int i = 0; i < active_layer_param.bottom_size(); ++i) {
    bottom_vec.push_back(bottom_blobs_[layer_name][i].get());
  }

  ASSERT(layer->layer_param().top_size()
      == active_layer_param.top_size(), "top vec cannot change");
  for (int top_id = 0; top_id < active_layer_param.top_size(); ++top_id) {
    ASSERT(layer->layer_param().top(top_id)
      == active_layer_param.top(top_id), "top vec cannot change");
  }

  for (int top_id = 0; top_id < active_layer_param.top_size(); ++top_id) {
    const string& blob_name = active_layer_param.top(top_id);
    if (blobs_.find(blob_name) == blobs_.end()) {
      shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
      blobs_[blob_name] = blob_pointer;
    }
    Blob<Dtype>* top_blob = blobs_[blob_name].get();
    if (!layer->in_place_layer()) {
      std::pair<set<string>::iterator, bool> ret;
      ret = active_blobs_set_.insert(blob_name);
      ASSERT(ret.second, "Top with name '"
          << blob_name << "' is already used");
    }
    top_vec.push_back(top_blob);
    if (top_blob->DiffInitialized() && !layer->is_loss()) {
      // Zero out top_diffs, except for loss blobs, which never change
      top_blob->SetDiffValues(0.);
    }
  }

  if (new_layer) {
    layer->SetUp(bottom_vec, top_vec);
    AddLayerParams(layer);
    if (param_cache_.find(layer_name) != param_cache_.end()) {
      CopyLayerFrom(param_cache_[layer_name]);
    }
  }

  for (int param_id = 0; param_id < layer->param_names().size(); ++param_id) {
    const string& param_name = layer->param_names()[param_id];
    active_params_set_.insert(param_name);
  }

  Dtype loss = 0;
  layer->set_phase(phase_);
  loss = layer->Forward(bottom_vec, top_vec);
  return loss;
}

template <typename Dtype>
void ApolloNet<Dtype>::AddLayerParams(shared_ptr<Layer<Dtype> > layer) {
  // hook up param names and lr_mults with Net
  vector<string> param_names;
  vector<Dtype> param_decay_mults;
  vector<Dtype> param_lr_mults;
  const LayerParameter& layer_param = layer->layer_param();
  const int param_size = layer_param.param_size();
  const string& layer_name = layer_param.name();
  if (param_size > 0) {
    // new layer has explitily named it's params
    ASSERT(param_size == layer->blobs().size(), "Layer: '"
        << layer_name << "' declared an incorrect number of params");
    for (int i = 0; i < layer->blobs().size(); ++i) {
      string param_name;
      if (layer_param.param(i).has_name()) {
        param_name = layer_param.param(i).name();
        ASSERT(param_name.find(".p") == string::npos,
            "named param '" << param_name << "' cannot contain .p");
      } else {
        stringstream ss;
        ss << layer_param.name() << ".p" << i;
        param_name = ss.str();
      }
      param_names.push_back(param_name);
      param_decay_mults.push_back(layer_param.param(i).decay_mult());
      param_lr_mults.push_back(layer_param.param(i).lr_mult());
    }
  } else {
    // provide default param names
    for (int i = 0; i < layer->blobs().size(); ++i) {
      stringstream ss;
      ss << layer_param.name() << ".p" << i;
      param_names.push_back(ss.str());
      param_decay_mults.push_back(Dtype(1.));
      param_lr_mults.push_back(Dtype(1.));
    }
  }
  layer->set_param_names(param_names);
  for (int i = 0; i < layer->blobs().size(); ++i) {
    const string& param_name = layer->param_names()[i];
    if (local_params_.find(param_name) == local_params_.end()) {
      local_params_[param_name] = layer->blobs()[i];
      params_[param_name] = shared_ptr<Blob<Dtype> >(
        new Blob<Dtype>(layer->blobs()[i]->shape()));
      params_[param_name]->ShareData(*local_params_[param_name]);
      if (!layer->overwrites_param_diffs()) {
        params_[param_name]->ShareDiff(*local_params_[param_name]);
      }
    } else {
      layer->blobs()[i]->ShareData(*local_params_[param_name]);
      layer->blobs()[i]->ShareDiff(*local_params_[param_name]);
    }
    param_decay_mults_[param_name] = param_decay_mults[i];
    param_lr_mults_[param_name] = param_lr_mults[i];
  }
}

template <typename Dtype>
void ApolloNet<Dtype>::Backward() {
  for (int i = active_layers_vec_.size() - 1; i >= 0; --i) {
    BackwardLayer(active_layers_vec_[i]);
  }
}

template <typename Dtype>
void ApolloNet<Dtype>::Update(Dtype lr, Dtype momentum,
  Dtype clip_gradients, Dtype weight_decay) {
  Dtype diffnorm = DiffL2Norm();
  Dtype clip_scale = 1;
  if (clip_gradients > 0) {
    if (diffnorm > clip_gradients) {
      clip_scale = clip_gradients / diffnorm;
    }
  }
  // iterate over active params
  for (set<string>::iterator it = active_params_set_.begin();
      it != active_params_set_.end(); ++it) {
    const string& param_name = *it;
    shared_ptr<Blob<Dtype> > cur_param = params_[param_name];
    Dtype lr_new = lr * clip_scale * param_lr_mults()[param_name];
    Dtype decay_new = weight_decay * param_decay_mults()[param_name];
    const shared_ptr<Tensor<Dtype> > &cur_data = cur_param->data();
    const shared_ptr<Tensor<Dtype> > &cur_diff = cur_param->diff();
    cur_diff->AddMulFrom(*cur_data, decay_new);
    cur_data->AddMulFrom(*cur_diff, -lr_new);
    cur_param->scale_diff(momentum);
  }
}

template <typename Dtype>
void ApolloNet<Dtype>::BackwardLayer(const string& layer_name) {
  shared_ptr<Layer<Dtype> > layer = layers_map_[layer_name];
  const LayerParameter& layer_param = layer->layer_param();
  vector<Blob<Dtype>*> bottom_vec;
  vector<Blob<Dtype>*> top_vec;
  for (int top_id = 0; top_id < layer_param.top_size(); ++top_id) {
    const string& blob_name = layer_param.top(top_id);
    top_vec.push_back(blobs_[blob_name].get());
  }
  vector<shared_ptr<Blob<Dtype> > > bottom_blobs = bottom_blobs_[layer_name];
  vector<bool> propagate_down;
  for (int bottom_id = 0; bottom_id < bottom_blobs.size(); ++bottom_id) {
    bottom_vec.push_back(bottom_blobs[bottom_id].get());
    propagate_down.push_back(true);
  }
  layer->Backward(top_vec, propagate_down, bottom_vec);

  if (layer->overwrites_bottom_diffs() && !layer->in_place_layer()) {
    // if layer overwrites bottom_diff
    for (int i = 0; i < layer_param.bottom_size(); ++i) {
      const string& bottom_name = layer_param.bottom(i);
      // add layer's bottom diff buffer to previous layer's top diffs
      blobs_[bottom_name]->AddDiffFrom(*bottom_vec[i]);
    }
  }
  if (layer->overwrites_param_diffs()) {
    for (int i = 0; i < layer->param_names().size(); ++i) {
      const string& param_name = layer->param_names()[i];
      // add param diff to master diff
      params_[param_name]->AddDiffFrom(*layer->blobs()[i]);
    }
  }
}

template <typename Dtype>
Dtype ApolloNet<Dtype>::DiffL2Norm() {
  Dtype sumsq_diff = 0.;
  for (set<string>::iterator it = active_params_set_.begin();
      it != active_params_set_.end(); ++it) {
    const string& param_name = *it;
    sumsq_diff += params_[param_name]->sumsq_diff();
  }
  return std::sqrt(sumsq_diff);
}

template <typename Dtype>
void ApolloNet<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();

    if (layers_map_.find(source_layer_name) == layers_map_.end()) {
      param_cache_[source_layer_name] = source_layer;
      LOG(INFO) << "Caching source layer blobs " << source_layer_name;
      continue;
    }
    CopyLayerFrom(source_layer);
  }
}

template <typename Dtype>
void ApolloNet<Dtype>::CopyLayerFrom(const LayerParameter& source_layer) {
  const string& source_layer_name = source_layer.name();
  LOG(INFO) << "Copying source layer blobs " << source_layer_name;
  vector<shared_ptr<Blob<Dtype> > >& target_blobs =
      layers_map_[source_layer_name]->blobs();

  ASSERT(target_blobs.size() == source_layer.blobs_size(),
      "Incompatible number of blobs for layer " << source_layer_name);
  for (int j = 0; j < target_blobs.size(); ++j) {
    const bool kReshape = false;
    target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
  }
}

template <typename Dtype>
void ApolloNet<Dtype>::SaveTrainedLayersTo(const string trained_filename)
  const {
  NetParameter param;
  DLOG(INFO) << "Serializing " << layers_map_.size() << " layers";
  typename map<string, shared_ptr<Layer<Dtype> > >::const_iterator it =
    layers_map_.begin();
  while (it != layers_map_.end()) {
    shared_ptr<Layer<Dtype> > layer = it->second;
    LayerParameter* layer_param = param.add_layer();
    layer->ToProto(layer_param);
    ++it;
  }
  WriteProtoToBinaryFile(param, trained_filename);
}

INSTANTIATE_CLASS(ApolloNet);

}  // namespace caffe
