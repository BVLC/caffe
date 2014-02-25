// Copyright 2014 kloudkl@github

#include <cmath> // for std::signbit
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
void binarize(const vector<shared_ptr<Blob<Dtype> > >& feature_blob_vector,
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
  vector<shared_ptr<Blob<Dtype> > > feature_blob_vector;
  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    real_valued_feature_net->Forward(input_vec);
    const shared_ptr<Blob<Dtype> > feature_blob = real_valued_feature_net
        ->GetBlob(feature_blob_name);
    feature_blob_vector.push_back(feature_blob);
  }
  shared_ptr<Blob<Dtype> > feature_binary_codes(new Blob<Dtype>());
  binarize<Dtype>(feature_blob_vector, feature_binary_codes);
  BlobProto blob_proto;
  feature_binary_codes->ToProto(&blob_proto);
  WriteProtoToBinaryFile(blob_proto, save_binarized_feature_binaryproto_file);
  LOG(ERROR)<< "Successfully binarized " << feature_binary_codes->num() << " features!";
  return 0;
}

// http://scikit-learn.org/stable/modules/preprocessing.html#feature-binarization
template<typename Dtype>
void binarize(const vector<shared_ptr<Blob<Dtype> > >& feature_blob_vector,
              shared_ptr<Blob<Dtype> > binary_codes) {
  CHECK_GT(feature_blob_vector.size(), 0);
  Dtype sum;
  size_t count = 0;
  size_t num_features = 0;
  for (int i = 0; i < feature_blob_vector.size(); ++i) {
    num_features += feature_blob_vector[i]->num();
    const Dtype* data = feature_blob_vector[i]->cpu_data();
    for (int j = 0; j < feature_blob_vector[i]->count(); ++j) {
      sum += data[j];
      ++count;
    }
  }
  Dtype mean = sum / count;
  int dim = feature_blob_vector[0]->count() / feature_blob_vector[0]->num();
  int size_of_code = sizeof(Dtype) * 8;
  binary_codes->Reshape(num_features, (dim + size_of_code - 1) / size_of_code,
                        1, 1);
  Dtype* binary_data = binary_codes->mutable_cpu_data();
  int offset;
  uint64_t code;
  for (int i = 0; i < feature_blob_vector.size(); ++i) {
    const Dtype* data = feature_blob_vector[i]->cpu_data();
    for (int j = 0; j < feature_blob_vector[i]->num(); ++j) {
      offset = j * dim;
      code = 0;
      int k;
      for (k = 0; k < dim;) {
        code |= std::signbit(mean - data[k]);
        ++k;
        if (k % size_of_code == 0) {
          binary_data[(k + size_of_code - 1) / size_of_code] = code;
          code = 0;
        } else {
          code <<= 1;
        }
      }  // for k
      if (k % size_of_code != 0) {
        code <<= (size_of_code - 1 - k % size_of_code);
        binary_data[(k + size_of_code - 1) / size_of_code] = code;
      }
    }  // for j
  }  // for i
}
