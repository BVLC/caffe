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

// TODO: Replace this with caffe_sign after the PR #159 is merged
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
  const int num_required_args = 5;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program compresses real valued features into compact binary codes.\n"
    "Usage: demo_binarize_features  real_valued_feature_prototxt  feature_blob_name"
    "  save_binarized_feature_binaryproto_file  num_mini_batches  [CPU/GPU]  [DEVICE_ID=0]";
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
  string real_valued_feature_prototxt(argv[++arg_pos]);
  NetParameter real_valued_feature_net_param;
  ReadProtoFromTextFile(real_valued_feature_prototxt,
                        &real_valued_feature_net_param);
  shared_ptr<Net<Dtype> > real_valued_feature_net(
      new Net<Dtype>(real_valued_feature_net_param));

  string feature_blob_name(argv[++arg_pos]);
  CHECK(real_valued_feature_net->HasBlob(feature_blob_name))
      << "Unknown feature blob name " << feature_blob_name << " in the network "
      << real_valued_feature_prototxt;

  string save_binarized_feature_binaryproto_file(argv[++arg_pos]);

  int num_mini_batches = atoi(argv[++arg_pos]);

  LOG(ERROR)<< "Binarizing features";
  vector<Blob<Dtype>*> input_vec;
  shared_ptr<Blob<Dtype> > feature_binary_codes(new Blob<Dtype>());
  BlobProtoVector blob_proto_vector;
  int num_features = 0;
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    real_valued_feature_net->Forward(input_vec);
    const shared_ptr<Blob<Dtype> > feature_blob = real_valued_feature_net
        ->GetBlob(feature_blob_name);
    binarize<Dtype>(feature_blob, feature_binary_codes);
    num_features += feature_binary_codes->num();
    feature_binary_codes->ToProto(blob_proto_vector.add_blobs());
  }  //  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  WriteProtoToBinaryFile(blob_proto_vector,
                         save_binarized_feature_binaryproto_file);
  LOG(ERROR)<< "Successfully binarized " << num_features << " features!";
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
  int num_binary_codes = (n + size_of_code - 1) / size_of_code;
  uint64_t code;
  int offset;
  int count = 0;
  for (int i = 0; i < num_binary_codes; ++i) {
    offset = i * size_of_code;
    int j = 0;
    code = 0;
    for (; j < size_of_code && count++ < n; ++j) {
      code |= sign(real_valued_feature[offset + j]);
      code << 1;
    }
    code << (size_of_code - j);
    binary_codes[i] = static_cast<Dtype>(code);
  }
}

template<typename Dtype>
void binarize(const shared_ptr<Blob<Dtype> > real_valued_features,
              shared_ptr<Blob<Dtype> > binary_codes) {
  int num = real_valued_features->num();
  int dim = real_valued_features->count() / num;
  int size_of_code = sizeof(Dtype) * 8;
  binary_codes->Reshape(num, (dim + size_of_code - 1) / size_of_code, 1, 1);
  const Dtype* real_valued_features_data = real_valued_features->cpu_data();
  Dtype* binary_codes_data = binary_codes->mutable_cpu_data();
  for (int n = 0; n < num; ++n) {
    binarize<Dtype>(dim,
                    real_valued_features_data + real_valued_features->offset(n),
                    binary_codes_data + binary_codes->offset(n));
  }
}
