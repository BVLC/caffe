// Copyright Yangqing Jia 2013

#include <map>
#include <set>
#include <string>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/layer_factory.hpp"
#include "caffe/net.hpp"

using std::pair;
using std::map;
using std::set;

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param,
    const vector<Blob<Dtype>* >& bottom) {
  // Basically, build all the layers and set up its connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  int num_layers = param.layers_size();
  CHECK_EQ(bottom.size(), param.bottom_size())
      << "Incorrect bottom blob size.";
  // set the input blobs
  for (int i = 0; i < param.bottom_size(); ++i) {
    const string& blob_name = param.bottom(i);
    blobs_.push_back(Blob<Dtype>(*bottom[i]));
    blob_names_.push_back(blob_name);
    net_input_blob_indices_.push_back(i);
    blob_name_to_idx[blob_name] = i;
    available_blobs.insert(blob_name);
  }
  // For each layer, set up their input and output
  layers_.resize(param.layers_size());
  bottom_vecs_.resize(param.layers_size());
  top_vecs_.resize(param.layers_size());
  for (int i = 0; i < param.top_size(); ++i) {
    const LayerConnection& layer_connection = param.layers(i);
    const LayerParameter& layer_param = layer_connection.layer();
    layers_[i].reset(GetLayer<Dtype>(layer_param));
    // Figure out this layer's input and output
    for (int j = 0; j < layer_connection.bottom_size(); ++j) {
      const string& blob_name = layer_connection.bottom(j);
      if (available_blobs.find(blob_name) == available_blobs.end()) {
        LOG(FATAL) << "Unknown blob input " << blob_name <<
            " to layer" << j;
      }
      bottom_vecs_[i].push_back(
          &blobs_[blob_name_to_idx[blob_name]]);
      available_blobs.erase(blob_name);
    }
    for (int j = 0; j < layer_connection.top_size(); ++j) {
      const string& blob_name = layer_connection.top(j);
      if (blob_name_to_idx.find(blob_name) != blob_name_to_idx.end()) {
        LOG(FATAL) << "Duplicate blobs produced by multiple sources.";
      }
      blobs_.push_back(Blob<Dtype>());
      blob_names_.push_back(blob_name);
      blob_name_to_idx[blob_name] = blob_names_.size() - 1;
      available_blobs.insert(blob_name);
      top_vecs_[i].push_back(&blobs_[blob_names_.size() - 1]);
    }
  }
  // In the end, check if all remaining available blobs are top blobs.
  for (int i = 0; i < param.top_size(); ++i) {
    const string& blob_name = param.top(i);
    if (blob_name_to_idx.find(blob_name) == blob_name_to_idx.end()) {
      LOG(FATAL) << "Unknown blob input " << blob_name;
    }
    net_output_blob_indices_.push_back(blob_name_to_idx[blob_name]);
    available_blobs.erase(blob_name);
  }
  if (!available_blobs.empty()) {
    LOG(WARNING) << "There are some internal blobs not used:";
    for (set<string>::iterator it = available_blobs.begin();
        it != available_blobs.end(); ++it) {
      LOG(WARNING) << "    " << *it;
    }
  }

  LOG(INFO) << "Setting up the layers.";
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->SetUp(bottom_vecs_[i], &top_vecs_[i]);
  }
}

template <typename Dtype>
void Net<Dtype>::Forward(const vector<Blob<Dtype>*> & bottom,
    vector<Blob<Dtype>*>* top) {
  // Copy bottom to internal bottom
  for (int i = 0; i < bottom.size(); ++i) {
    blobs_[net_input_blob_indices_[i]] = *bottom[i];
  }
  for (int i = 0; i < layers_.size(); ++i) {
    layers_[i]->Forward(bottom_vecs_[i], &top_vecs_[i]);
  }
  // Copy internal top to top
  for (int i = 0; i < (*top).size(); ++i) {
    NOT_IMPLEMENTED;
  }
}

template <typename Dtype>
Dtype Net<Dtype>::Backward() {
  Dtype loss = 0;
  // TODO(Yangqing): figure out those layers that do not need backward.
  for (int i = layers_.size() - 1; i >= 0; --i) {
    loss += layers_[i]->Backward(top_vecs_[i], true, &bottom_vecs_[i]);
  }
  return loss;
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layers_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layers(i).layer();
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    LOG(INFO) << "Loading source layer " << source_layer_name;
    vector<Blob<Dtype> >& target_blobs = layers_[target_layer_id]->params();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      target_blobs[j].FromProto(source_layer.blobs(j));
    }
  }
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  for (int i = 0; i < net_input_blob_indices_.size(); ++i) {
    param->add_bottom(blob_names_[net_input_blob_indices_[i]]);
  }
  for (int i = 0; i < net_input_blob_indices_.size(); ++i) {
    param->add_bottom(blob_names_[net_input_blob_indices_[i]]);
  }
  for (int i = 0; i < layers_.size(); ++i) {
    LayerConnection* layer_connection = param->add_layers();
  }
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
