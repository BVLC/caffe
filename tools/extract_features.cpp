// Copyright 2014 BVLC and contributors.

#include <stdio.h>  // for snprintf
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

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
    " extract features of the input data produced by the net.\n"
    "Usage: demo_extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name"
    "  save_feature_leveldb_name  num_mini_batches  [CPU/GPU]  [DEVICE_ID=0]";
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

  arg_pos = 0;  // the name of the executable
  string pretrained_binary_proto(argv[++arg_pos]);

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */
  string feature_extraction_proto(argv[++arg_pos]);
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto));
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);

  string extract_feature_blob_name(argv[++arg_pos]);
  CHECK(feature_extraction_net->has_blob(extract_feature_blob_name))
      << "Unknown feature blob name " << extract_feature_blob_name
      << " in the network " << feature_extraction_proto;

  string save_feature_leveldb_name(argv[++arg_pos]);
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(INFO)<< "Opening leveldb " << save_feature_leveldb_name;
  leveldb::Status status = leveldb::DB::Open(options,
                                             save_feature_leveldb_name.c_str(),
                                             &db);
  CHECK(status.ok()) << "Failed to open leveldb " << save_feature_leveldb_name;

  int num_mini_batches = atoi(argv[++arg_pos]);

  LOG(ERROR)<< "Extacting Features";

  Datum datum;
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  const int kMaxKeyStrLength = 100;
  char key_str[kMaxKeyStrLength];
  int num_bytes_of_binary_code = sizeof(Dtype);
  vector<Blob<float>*> input_vec;
  int image_index = 0;
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward(input_vec);
    const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
        ->blob_by_name(extract_feature_blob_name);
    int num_features = feature_blob->num();
    int dim_features = feature_blob->count() / num_features;
    Dtype* feature_blob_data;
    for (int n = 0; n < num_features; ++n) {
      datum.set_height(dim_features);
      datum.set_width(1);
      datum.set_channels(1);
      datum.clear_data();
      datum.clear_float_data();
      feature_blob_data = feature_blob->mutable_cpu_data() +
          feature_blob->offset(n);
      for (int d = 0; d < dim_features; ++d) {
        datum.add_float_data(feature_blob_data[d]);
      }
      string value;
      datum.SerializeToString(&value);
      snprintf(key_str, kMaxKeyStrLength, "%d", image_index);
      batch->Put(string(key_str), value);
      ++image_index;
      if (image_index % 1000 == 0) {
        db->Write(leveldb::WriteOptions(), batch);
        LOG(ERROR)<< "Extracted features of " << image_index <<
            " query images.";
        delete batch;
        batch = new leveldb::WriteBatch();
      }
    }  // for (int n = 0; n < num_features; ++n)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  if (image_index % 1000 != 0) {
    db->Write(leveldb::WriteOptions(), batch);
    LOG(ERROR)<< "Extracted features of " << image_index <<
        " query images.";
  }

  delete batch;
  delete db;
  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}

