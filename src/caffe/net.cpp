// Copyright Yangqing Jia 2013

#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"

using std::pair;
using std::map;
using std::set;

namespace caffe {

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file) {
  NetParameter param;
  ReadProtoFromTextFile(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Net<Dtype>::Init(const NetParameter& in_param) {
  // Create a copy of in_param with splits added where necessary.
  NetParameter param;
  AddSplits(in_param, &param);
  // Basically, build all the layers and set up its connections.
  name_ = param.name();
  map<string, int> blob_name_to_idx;
  set<string> available_blobs;
  int num_layers = param.layers_size();
  CHECK_EQ(param.input_size() * 4, param.input_dim_size())
      << "Incorrect bottom blob dimension specifications.";
  // set the input blobs
  for (int i = 0; i < param.input_size(); ++i) {
    const string& blob_name = param.input(i);
    shared_ptr<Blob<Dtype> > blob_pointer(
        new Blob<Dtype>(param.input_dim(i * 4),
                        param.input_dim(i * 4 + 1),
                        param.input_dim(i * 4 + 2),
                        param.input_dim(i * 4 + 3)));
    blobs_.push_back(blob_pointer);
    blob_names_.push_back(blob_name);
    blob_need_backward_.push_back(param.force_backward());
    net_input_blob_indices_.push_back(i);
    net_input_blobs_.push_back(blob_pointer.get());
    blob_name_to_idx[blob_name] = i;
    available_blobs.insert(blob_name);
  }
  // For each layer, set up their input and output
  bottom_vecs_.resize(param.layers_size());
  top_vecs_.resize(param.layers_size());
  bottom_id_vecs_.resize(param.layers_size());
  top_id_vecs_.resize(param.layers_size());
  for (int i = 0; i < param.layers_size(); ++i) {
    const LayerConnection& layer_connection = param.layers(i);
    const LayerParameter& layer_param = layer_connection.layer();
    layers_.push_back(shared_ptr<Layer<Dtype> >(GetLayer<Dtype>(layer_param)));
    layer_names_.push_back(layer_param.name());
    LOG(INFO) << "Creating Layer " << layer_param.name();
    bool need_backward = param.force_backward();
    // Figure out this layer's input and output
    for (int j = 0; j < layer_connection.bottom_size(); ++j) {
      const string& blob_name = layer_connection.bottom(j);
      const int blob_id = blob_name_to_idx[blob_name];
      if (available_blobs.find(blob_name) == available_blobs.end()) {
        LOG(FATAL) << "Unknown blob input " << blob_name <<
            " to layer" << j;
      }
      LOG(INFO) << layer_param.name() << " <- " << blob_name;
      bottom_vecs_[i].push_back(
          blobs_[blob_id].get());
      bottom_id_vecs_[i].push_back(blob_id);
      // If a blob needs backward, this layer should provide it.
      need_backward |= blob_need_backward_[blob_id];
      available_blobs.erase(blob_name);
    }
    for (int j = 0; j < layer_connection.top_size(); ++j) {
      const string& blob_name = layer_connection.top(j);
      // Check if we are doing in-place computation
      if (layer_connection.bottom_size() > j &&
          blob_name == layer_connection.bottom(j)) {
        // In-place computation
        LOG(INFO) << layer_param.name() << " -> " << blob_name << " (in-place)";
        available_blobs.insert(blob_name);
        top_vecs_[i].push_back(
            blobs_[blob_name_to_idx[blob_name]].get());
        top_id_vecs_[i].push_back(blob_name_to_idx[blob_name]);
      } else if (blob_name_to_idx.find(blob_name) != blob_name_to_idx.end()) {
        // If we are not doing in-place computation but has duplicated blobs,
        // raise an error.
        LOG(FATAL) << "Duplicate blobs produced by multiple sources.";
      } else {
        // Normal output.
        LOG(INFO) << layer_param.name() << " -> " << blob_name;
        shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
        blobs_.push_back(blob_pointer);
        blob_names_.push_back(blob_name);
        blob_need_backward_.push_back(param.force_backward());
        blob_name_to_idx[blob_name] = blob_names_.size() - 1;
        available_blobs.insert(blob_name);
        top_vecs_[i].push_back(blobs_[blob_names_.size() - 1].get());
        top_id_vecs_[i].push_back(blob_names_.size() - 1);
      }
    }
    // After this layer is connected, set it up.
    // LOG(INFO) << "Setting up " << layer_names_[i];
    layers_[i]->SetUp(bottom_vecs_[i], &top_vecs_[i]);
    for (int topid = 0; topid < top_vecs_[i].size(); ++topid) {
      LOG(INFO) << "Top shape: " << top_vecs_[i][topid]->channels() << " "
          << top_vecs_[i][topid]->height() << " "
          << top_vecs_[i][topid]->width();
    }
    int blobs_lr_size = layers_[i]->layer_param().blobs_lr_size();
    CHECK(blobs_lr_size == layers_[i]->blobs().size() || blobs_lr_size == 0)
        << "Incorrect blobs lr size: should be either 0 or the same as "
           "the number of the layer's parameter blobs.";
    if (blobs_lr_size) {
      // Check if this layer needs backward operation itself
      for (int j = 0; j < blobs_lr_size; ++j) {
        need_backward |= (layers_[i]->layer_param().blobs_lr(j) > 0);
      }
    } else if (layers_[i]->blobs().size()) {
      // catch: if a layer param does not specify blobs_lr, we should assume the
      // learning rate to be 1. Thus we will need to perform backward.
      need_backward = true;
    }
    // Finally, set the backward flag
    layer_need_backward_.push_back(need_backward);
    if (need_backward) {
      LOG(INFO) << layer_names_[i] << " needs backward computation.";
      for (int j = 0; j < top_id_vecs_[i].size(); ++j) {
        blob_need_backward_[top_id_vecs_[i][j]] = true;
      }
    } else {
      LOG(INFO) << layer_names_[i] << " does not need backward computation.";
    }
  }
  // In the end, all remaining blobs are considered output blobs.
  for (set<string>::iterator it = available_blobs.begin();
      it != available_blobs.end(); ++it) {
    LOG(INFO) << "This network produces output " << *it;
    net_output_blobs_.push_back(blobs_[blob_name_to_idx[*it]].get());
  }
  GetLearningRateAndWeightDecay();
  LOG(INFO) << "Network initialization done.";
}


template <typename Dtype>
void Net<Dtype>::AddSplits(const NetParameter& param,
    NetParameter* param_split) {
  // Initialize by copying from the input NetParameter.
  param_split->CopyFrom(param);
  param_split->clear_layers();
  map<string, int> blob_name_to_bottom_count;
  map<string, int> blob_name_to_bottom_split_idx;
  // Determine for each top blob the number of times it's used as a bottom blob.
  for (int i = 0; i < param.layers_size(); ++i) {
    const LayerConnection& layer_connection = param.layers(i);
    for (int j = 0; j < layer_connection.bottom_size(); ++j) {
      const string& blob_name = layer_connection.bottom(j);
      blob_name_to_bottom_count[blob_name]++;
    }
    for (int j = 0; j < layer_connection.top_size(); ++j) {
      const string& blob_name = layer_connection.top(j);
      blob_name_to_bottom_count[blob_name] = 0;
      blob_name_to_bottom_split_idx[blob_name] = 0;
    }
  }
  for (int i = 0; i < param.layers_size(); ++i) {
    LayerConnection* layer_connection = param_split->add_layers();
    layer_connection->CopyFrom(param.layers(i));
    // Replace any shared bottom blobs with split layer outputs.
    for (int j = 0; j < layer_connection->bottom_size(); ++j) {
      const string& blob_name = layer_connection->bottom(j);
      const int split_count = blob_name_to_bottom_count[blob_name];
      if (split_count > 1) {
        const int suffix_max_length = 16;
        char split_suffix[suffix_max_length];
        const int suffix_length = snprintf(split_suffix, suffix_max_length,
            "_split_%d", blob_name_to_bottom_split_idx[blob_name]++);
        CHECK_LT(suffix_length, suffix_max_length);
        const string& split_blob_name = blob_name + split_suffix;
        layer_connection->set_bottom(j, split_blob_name);
      }
    }
    // Create split blob for any top blobs used by other layers as bottom
    // blobs more than once.
    for (int j = 0; j < layer_connection->top_size(); ++j) {
      const string& blob_name = layer_connection->top(j);
      const int split_count = blob_name_to_bottom_count[blob_name];
      if (split_count > 1) {
        LayerConnection* split_layer_connection = param_split->add_layers();
        split_layer_connection->add_bottom(blob_name);
        LayerParameter* split_layer_param =
            split_layer_connection->mutable_layer();
        split_layer_param->set_name(blob_name + "_split");
        split_layer_param->set_type("split");
        vector<Blob<Dtype>*> split_top_blobs(split_count);
        for (int k = 0; k < split_count; ++k) {
          const int suffix_max_length = 16;
          char split_suffix[suffix_max_length];
          const int suffix_length = snprintf(split_suffix, suffix_max_length,
              "_split_%d", k);
          CHECK_LT(suffix_length, suffix_max_length);
          const string& split_blob_name = blob_name + split_suffix;
          split_layer_connection->add_top(split_blob_name);
        }
      }
    }
  }
}


template <typename Dtype>
void Net<Dtype>::GetLearningRateAndWeightDecay() {
  LOG(INFO) << "Collecting Learning Rate and Weight Decay.";
  for (int i = 0; i < layers_.size(); ++i) {
    vector<shared_ptr<Blob<Dtype> > >& layer_blobs = layers_[i]->blobs();
    for (int j = 0; j < layer_blobs.size(); ++j) {
      params_.push_back(layer_blobs[j]);
    }
    // push the learning rate mutlipliers
    if (layers_[i]->layer_param().blobs_lr_size()) {
      CHECK_EQ(layers_[i]->layer_param().blobs_lr_size(), layer_blobs.size());
      for (int j = 0; j < layer_blobs.size(); ++j) {
        float local_lr = layers_[i]->layer_param().blobs_lr(j);
        CHECK_GE(local_lr, 0.);
        params_lr_.push_back(local_lr);
      }
    } else {
      for (int j = 0; j < layer_blobs.size(); ++j) {
        params_lr_.push_back(1.);
      }
    }
    // push the weight decay multipliers
    if (layers_[i]->layer_param().weight_decay_size()) {
      CHECK_EQ(layers_[i]->layer_param().weight_decay_size(),
          layer_blobs.size());
      for (int j = 0; j < layer_blobs.size(); ++j) {
        float local_decay = layers_[i]->layer_param().weight_decay(j);
        CHECK_GE(local_decay, 0.);
        params_weight_decay_.push_back(local_decay);
      }
    } else {
      for (int j = 0; j < layer_blobs.size(); ++j) {
        params_weight_decay_.push_back(1.);
      }
    }
  }
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::ForwardPrefilled() {
  for (int i = 0; i < layers_.size(); ++i) {
    // LOG(ERROR) << "Forwarding " << layer_names_[i];
    layers_[i]->Forward(bottom_vecs_[i], &top_vecs_[i]);
  }
  return net_output_blobs_;
}

template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(
    const vector<Blob<Dtype>*> & bottom) {
  // Copy bottom to internal bottom
  for (int i = 0; i < bottom.size(); ++i) {
    net_input_blobs_[i]->CopyFrom(*bottom[i]);
  }
  return ForwardPrefilled();
}


template <typename Dtype>
string Net<Dtype>::Forward(const string& input_blob_protos) {
  BlobProtoVector blob_proto_vec;
  if (net_input_blobs_.size()) {
    blob_proto_vec.ParseFromString(input_blob_protos);
    CHECK_EQ(blob_proto_vec.blobs_size(), net_input_blobs_.size())
        << "Incorrect input size.";
    for (int i = 0; i < blob_proto_vec.blobs_size(); ++i) {
      net_input_blobs_[i]->FromProto(blob_proto_vec.blobs(i));
    }
  }
  ForwardPrefilled();
  blob_proto_vec.Clear();
  for (int i = 0; i < net_output_blobs_.size(); ++i) {
    net_output_blobs_[i]->ToProto(blob_proto_vec.add_blobs());
  }
  string output;
  blob_proto_vec.SerializeToString(&output);
  return output;
}


template <typename Dtype>
Dtype Net<Dtype>::Backward() {
  Dtype loss = 0;
  for (int i = layers_.size() - 1; i >= 0; --i) {
    if (layer_need_backward_[i]) {
      Dtype layer_loss = layers_[i]->Backward(
          top_vecs_[i], true, &bottom_vecs_[i]);
      loss += layer_loss;
    }
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
      DLOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      CHECK_EQ(target_blobs[j]->num(), source_layer.blobs(j).num());
      CHECK_EQ(target_blobs[j]->channels(), source_layer.blobs(j).channels());
      CHECK_EQ(target_blobs[j]->height(), source_layer.blobs(j).height());
      CHECK_EQ(target_blobs[j]->width(), source_layer.blobs(j).width());
      target_blobs[j]->FromProto(source_layer.blobs(j));
    }
  }
}

template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const string trained_filename) {
  NetParameter param;
  ReadProtoFromBinaryFile(trained_filename, &param);
  CopyTrainedLayersFrom(param);
}

template <typename Dtype>
void Net<Dtype>::ToProto(NetParameter* param, bool write_diff) {
  param->Clear();
  param->set_name(name_);
  // Add bottom and top
  for (int i = 0; i < net_input_blob_indices_.size(); ++i) {
    param->add_input(blob_names_[net_input_blob_indices_[i]]);
  }
  DLOG(INFO) << "Serializing " << layers_.size() << " layers";
  for (int i = 0; i < layers_.size(); ++i) {
    LayerConnection* layer_connection = param->add_layers();
    for (int j = 0; j < bottom_id_vecs_[i].size(); ++j) {
      layer_connection->add_bottom(blob_names_[bottom_id_vecs_[i][j]]);
    }
    for (int j = 0; j < top_id_vecs_[i].size(); ++j) {
      layer_connection->add_top(blob_names_[top_id_vecs_[i][j]]);
    }
    LayerParameter* layer_parameter = layer_connection->mutable_layer();
    layers_[i]->ToProto(layer_parameter, write_diff);
  }
}

template <typename Dtype>
void Net<Dtype>::Update() {
  for (int i = 0; i < params_.size(); ++i) {
    params_[i]->Update();
  }
}

INSTANTIATE_CLASS(Net);

}  // namespace caffe
