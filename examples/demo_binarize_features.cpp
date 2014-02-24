// Copyright 2014 kloudkl@github

#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;

template<typename Dtype>
inline int sign(const Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

template<typename Dtype>
void binarize(const int n, const Dtype* real_valued_feature,
              Dtype* binary_code);

template<typename Dtype>
void binarize(const shared_ptr<Blob<Dtype> > real_valued_features,
              shared_ptr<Blob<Dtype> > binary_codes);

template<typename Dtype>
int features_binarization_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return features_binarization_pipeline<float>(argc, argv);
//  return features_binarization_pipeline<double>(argc, argv);
}

template<typename Dtype>
int features_binarization_pipeline(int argc, char** argv) {
  const int num_required_args = 4;
  if (argc < num_required_args) {
    LOG(ERROR)<<
        "This program compresses real valued features into compact binary codes."
        "Usage: demo_binarize_features  data_prototxt  data_layer_name"
        "  save_binarized_feature_binaryproto_file  [CPU/GPU]  [DEVICE_ID=0]";
    return 1;
  }
  int arg_pos = num_required_args;

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
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

  // Expected prototxt contains at least one data layer as the real valued features.
  /*
   layers {
   layer {
   name: "real_valued_features"
   type: "data"
   source: "/path/to/your/real/valued/features_leveldb"
   batchsize: 256
   }
   top: "real_valued_features"
   top: "label"
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
    LOG(ERROR) << "Unknow data layer name " << data_layer_name <<
        " in prototxt " << data_prototxt;
  }

  string save_binarized_feature_binaryproto_file(argv[++arg_pos]);

  LOG(ERROR)<< "Binarizing features";
  DataLayer<Dtype> data_layer(data_layer_param);
  vector<Blob<Dtype>*> bottom_vec_that_data_layer_does_not_need_;
  vector<Blob<Dtype>*> top_vec;
  data_layer.Forward(bottom_vec_that_data_layer_does_not_need_, &top_vec);
  shared_ptr<Blob<Dtype> > feature_binary_codes;
  BlobProtoVector blob_proto_vector;
  int batch_index = 0;
  // TODO: DataLayer seem to rotate from the last record to the first
  // how to judge that all the data record have been enumerated?
  while (top_vec.size()) { // data_layer still outputs data
    LOG(ERROR)<< "Batch " << batch_index << " feature binarization";
    const shared_ptr<Blob<Dtype> > feature_blob(top_vec[0]);
    binarize<Dtype>(feature_blob, feature_binary_codes);

    LOG(ERROR) << "Batch " << batch_index << " save binarized features";
    feature_binary_codes->ToProto(blob_proto_vector.add_blobs());

    data_layer.Forward(bottom_vec_that_data_layer_does_not_need_, &top_vec);
    ++batch_index;
  } //  while (top_vec.size()) {

  WriteProtoToBinaryFile(blob_proto_vector, save_binarized_feature_binaryproto_file);
  LOG(ERROR)<< "Successfully ended!";
  return 0;
}

template<typename Dtype>
void binarize(const int n, const Dtype* real_valued_feature,
              Dtype* binary_codes) {
  // TODO: more advanced binarization algorithm such as bilinear projection
  // Yunchao Gong, Sanjiv Kumar, Henry A. Rowley, and Svetlana Lazebnik.
  // Learning Binary Codes for High-Dimensional Data Using Bilinear Projections.
  // In IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2013.
  // http://www.unc.edu/~yunchao/bpbc.htm
  int size_of_code = sizeof(Dtype) * 8;
  CHECK_EQ(n % size_of_code, 0);
  int num_binary_codes = n / size_of_code;
  uint64_t code;
  int offset;
  for (int i = 0; i < num_binary_codes; ++i) {
    code = 0;
    offset = i * size_of_code;
    for (int j = 0; j < size_of_code; ++j) {
      code |= sign(real_valued_feature[offset + j]);
      code << 1;
    }
    binary_codes[i] = static_cast<Dtype>(code);
  }
}

template<typename Dtype>
void binarize(const shared_ptr<Blob<Dtype> > real_valued_features,
              shared_ptr<Blob<Dtype> > binary_codes) {
  int num = real_valued_features->num();
  int dim = real_valued_features->count() / num;
  int size_of_code = sizeof(Dtype) * 8;
  CHECK_EQ(dim % size_of_code, 0);
  binary_codes->Reshape(num, dim / size_of_code, 1, 1);
  const Dtype* real_valued_features_data = real_valued_features->cpu_data();
  Dtype* binary_codes_data = binary_codes->mutable_cpu_data();
  for (int n = 0; n < num; ++n) {
    binarize<Dtype>(dim,
                    real_valued_features_data + real_valued_features->offset(n),
                    binary_codes_data + binary_codes->offset(n));
  }
}
