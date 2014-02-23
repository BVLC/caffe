// Copyright 2014 kloudkl@github

#include <stdio.h> // for snprintf
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;


template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  const int num_required_args = 6;
  if (argc < num_required_args) {
    LOG(ERROR)<<
        "This program takes in a trained network and an input data layer, and then"
        "  extract features of the input data produced by the net."
        "Usage: demo_extract_features  pretrained_net_param"
        "  extract_feature_blob_name  data_prototxt  data_layer_name"
        "  save_feature_leveldb_name  [CPU/GPU]  [DEVICE_ID=0]";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  Caffe::set_phase(Caffe::TEST);

  NetParameter pretrained_net_param;

  arg_pos = 0;  // the name of the executable
  // We directly load the net param from trained file
  string pretrained_binary_proto(argv[++arg_pos]);
  ReadProtoFromBinaryFile(pretrained_binary_proto.c_str(),
                          &pretrained_net_param);
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(pretrained_net_param));

  string extract_feature_blob_name(argv[++arg_pos]);
  if (!feature_extraction_net->HasBlob(extract_feature_blob_name)) {
    LOG(ERROR)<< "Unknown feature blob name " << extract_feature_blob_name <<
    " in trained network " << pretrained_binary_proto;
    return 1;
  }

  // Expected prototxt contains at least one data layer to extract features.
  /*
   layers {
   layer {
   name: "data_layer_name"
   type: "data"
   source: "/path/to/your/images/to/extract/feature/images_leveldb"
   meanfile: "/path/to/your/image_mean.binaryproto"
   batchsize: 128
   cropsize: 227
   mirror: false
   }
   top: "data_blob_name"
   top: "label_blob_name"
   }
   */
  string data_prototxt(argv[++arg_pos]);
  string data_layer_name(argv[++arg_pos]);
  NetParameter data_net_param;
  ReadProtoFromTextFile(data_prototxt.c_str(), &data_net_param);
  LayerParameter data_layer_param;
  int num_layer;
  for (num_layer = 0; num_layer < data_net_param.layers_size(); ++num_layer) {
    if (data_layer_name == data_net_param.layers(num_layer).layer().name()) {
      data_layer_param = data_net_param.layers(num_layer).layer();
      break;
    }
  }
  if (num_layer = data_net_param.layers_size()) {
    LOG(ERROR) << "Unknown data layer name " << data_layer_name <<
        " in prototxt " << data_prototxt;
  }

  string save_feature_leveldb_name(argv[++arg_pos]);
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(INFO) << "Opening leveldb " << argv[3];
  leveldb::Status status = leveldb::DB::Open(
      options, save_feature_leveldb_name.c_str(), &db);
  CHECK(status.ok()) << "Failed to open leveldb " << save_feature_leveldb_name;

  LOG(ERROR)<< "Extacting Features";
  DataLayer<Dtype> data_layer(data_layer_param);
  vector<Blob<Dtype>*> bottom_vec_that_data_layer_does_not_need_;
  vector<Blob<Dtype>*> top_vec;
  data_layer.Forward(bottom_vec_that_data_layer_does_not_need_, &top_vec);
  int batch_index = 0;
  int image_index = 0;

  Datum datum;
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  const int max_key_str_length = 100;
  char key_str[max_key_str_length];
  int num_bytes_of_binary_code = sizeof(Dtype);
  // TODO: DataLayer seem to rotate from the last record to the first
  // how to judge that all the data record have been enumerated?
  while (top_vec.size()) { // data_layer still outputs data
    LOG(ERROR)<< "Batch " << batch_index << " feature extraction";
    feature_extraction_net->Forward(top_vec);
    const shared_ptr<Blob<Dtype> > feature_blob =
        feature_extraction_net->GetBlob(extract_feature_blob_name);

    LOG(ERROR) << "Batch " << batch_index << " save extracted features";
    int num_features = feature_blob->num();
    int dim_features = feature_blob->count() / num_features;
    for (int n = 0; n < num_features; ++n) {
       datum.set_height(dim_features);
       datum.set_width(1);
       datum.set_channels(1);
       datum.clear_data();
       datum.clear_float_data();
       string* datum_string = datum.mutable_data();
       const Dtype* feature_blob_data = feature_blob->cpu_data();
       for (int d = 0; d < dim_features; ++d) {
         const char* data_byte = reinterpret_cast<const char*>(feature_blob_data + d);
         for(int i = 0; i < num_bytes_of_binary_code; ++i) {
           datum_string->push_back(data_byte[i]);
         }
       }
       string value;
       datum.SerializeToString(&value);
       snprintf(key_str, max_key_str_length, "%d", image_index);
       batch->Put(string(key_str), value);
       if (++image_index % 1000 == 0) {
         db->Write(leveldb::WriteOptions(), batch);
         LOG(ERROR) << "Extracted features of " << image_index << " query images.";
         delete batch;
         batch = new leveldb::WriteBatch();
       }
    }
    // write the last batch
    if (image_index % 1000 != 0) {
      db->Write(leveldb::WriteOptions(), batch);
      LOG(ERROR) << "Extracted features of " << image_index << " query images.";
      delete batch;
      batch = new leveldb::WriteBatch();
    }

    data_layer.Forward(bottom_vec_that_data_layer_does_not_need_, &top_vec);
    ++batch_index;
  } //  while (top_vec.size()) {

  delete batch;
  delete db;
  LOG(ERROR)<< "Successfully ended!";
  return 0;
}

