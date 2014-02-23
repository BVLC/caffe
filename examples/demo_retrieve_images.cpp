// Copyright 2014 kloudkl@github
//
// This program takes in a trained network and an input blob, and then
// extract features of the input blobs produced by the net to retrieve similar images.
// Usage:
//    retrieve_image pretrained_net_param input_blob output_filename top_k_results [CPU/GPU] [DEVICE_ID=0]

#include <stdio.h> // for snprintf
#include <fstream> // for std::ofstream
#include <queue> // for std::priority_queue
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

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
void similarity_search(const shared_ptr<Blob<Dtype> > sample_images_feature,
                       const shared_ptr<Blob<Dtype> > query_image_feature,
                       const int top_k_results,
                       shared_ptr<Blob<Dtype> > retrieval_results);

template<typename Dtype>
int image_retrieval_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return image_retrieval_pipeline<float>(argc, argv);
//  return image_retrieval_pipeline<double>(argc, argv);
}

template<typename Dtype>
int image_retrieval_pipeline(int argc, char** argv) {
  const int num_required_args = 7;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "retrieve_image pretrained_net_param extract__feature_blob_name"
    " sample_images_feature_blob_binaryproto data_prototxt data_layer_name"
    " save_feature_leveldb_name save_retrieval_result_filename"
    " [top_k_results=1] [CPU/GPU] [DEVICE_ID=0]";
    return 1;
  }
  int arg_pos = num_required_args;

  int top_k_results;
  if (argc <= num_required_args) {
    top_k_results = 1;
  } else {
    top_k_results = atoi(argv[arg_pos]);
    CHECK_GE(top_k_results, 0);
  }

  arg_pos = num_required_args + 1;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos) {
      device_id = atoi(argv[arg_pos]);
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

  string sample_images_feature_blob_binaryproto(argv[++arg_pos]);
  BlobProto sample_images_feature_blob_proto;
  ReadProtoFromBinaryFile(argv[++arg_pos], &sample_images_feature_blob_proto);
  shared_ptr<Blob<Dtype> > sample_images_feature_blob(new Blob<Dtype>());
  sample_images_feature_blob->FromProto(sample_images_feature_blob_proto);

  // Expected prototxt contains at least one data layer as the query images.
  /*
   layers {
   layer {
   name: "query_images"
   type: "data"
   source: "/path/to/your/images/to/extract/feature/and/retrieve/similar/images_leveldb"
   meanfile: "/path/to/your/image_mean.binaryproto"
   batchsize: 128
   cropsize: 115
   mirror: false
   }
   top: "query_images"
   top: "ground_truth_labels" // TODO: Add MultiLabelDataLayer support for image retrieval, annotations etc.
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

  string save_retrieval_result_filename(argv[++arg_pos]);
  std::ofstream retrieval_result_ofs(save_retrieval_result_filename.c_str(),
                                     std::ofstream::out);

  LOG(ERROR)<< "Extacting Features and retrieving images";
  DataLayer<Dtype> data_layer(data_layer_param);
  vector<Blob<Dtype>*> bottom_vec_that_data_layer_does_not_need_;
  vector<Blob<Dtype>*> top_vec;
  data_layer.Forward(bottom_vec_that_data_layer_does_not_need_, &top_vec);
  int batch_index = 0;
  shared_ptr<Blob<Dtype> > feature_binary_codes;
  shared_ptr<Blob<Dtype> > retrieval_results;
  int query_image_index = 0;

  Datum datum;
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  const int max_key_str_length = 100;
  char key_str[max_key_str_length];
  int num_bytes_of_binary_code = sizeof(Dtype);
  int count_query_images = 0;
  while (top_vec.size()) { // data_layer still outputs data
    LOG(ERROR)<< "Batch " << batch_index << " feature extraction";
    feature_extraction_net->Forward(top_vec);
    const shared_ptr<Blob<Dtype> > feature_blob =
    feature_extraction_net->GetBlob(extract_feature_blob_name);
    feature_binary_codes.reset(new Blob<Dtype>());
    binarize<Dtype>(feature_blob, feature_binary_codes);

    LOG(ERROR) << "Batch " << batch_index << " save extracted features";
    const Dtype* retrieval_results_data = retrieval_results->cpu_data();
    int num_features = feature_binary_codes->num();
    int dim_features = feature_binary_codes->count() / num_features;
    for (int n = 0; n < num_features; ++n) {
       datum.set_height(dim_features);
       datum.set_width(1);
       datum.set_channels(1);
       datum.clear_data();
       datum.clear_float_data();
       string* datum_string = datum.mutable_data();
       for (int d = 0; d < dim_features; ++d) {
         const Dtype data = feature_binary_codes->data_at(n, d, 0, 0);
         const char* data_byte = reinterpret_cast<const char*>(&data);
         for(int i = 0; i < num_bytes_of_binary_code; ++i) {
           datum_string->push_back(data_byte[i]);
         }
       }
       string value;
       datum.SerializeToString(&value);
       snprintf(key_str, max_key_str_length, "%d", query_image_index);
       batch->Put(string(key_str), value);
       if (++count_query_images % 1000 == 0) {
         db->Write(leveldb::WriteOptions(), batch);
         LOG(ERROR) << "Extracted features of " << count_query_images << " query images.";
         delete batch;
         batch = new leveldb::WriteBatch();
       }
    }
    // write the last batch
    if (count_query_images % 1000 != 0) {
      db->Write(leveldb::WriteOptions(), batch);
      LOG(ERROR) << "Extracted features of " << count_query_images << " query images.";
      delete batch;
      batch = new leveldb::WriteBatch();
    }

    LOG(ERROR) << "Batch " << batch_index << " image retrieval";
    similarity_search<Dtype>(sample_images_feature_blob, feature_binary_codes,
        top_k_results, retrieval_results);

    LOG(ERROR) << "Batch " << batch_index << " save image retrieval results";
    int num_results = retrieval_results->num();
    int dim_results = retrieval_results->count() / num_results;
    for (int i = 0; i < num_results; ++i) {
      retrieval_result_ofs << query_image_index;
      for (int k = 0; k < dim_results; ++k) {
        retrieval_result_ofs << " " << retrieval_results->data_at(i, k, 0, 0);
      }
      retrieval_result_ofs << "\n";
    }
    ++query_image_index;

    data_layer.Forward(bottom_vec_that_data_layer_does_not_need_, &top_vec);
    ++batch_index;
  } //  while (top_vec.size()) {

  delete batch;
  delete db;
  retrieval_result_ofs.close();
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

class MinHeapComparison {
 public:
  bool operator()(const std::pair<int, int>& lhs,
                  const std::pair<int, int>&rhs) const {
    return (lhs.first > rhs.first);
  }
};

template<typename Dtype>
void similarity_search(const shared_ptr<Blob<Dtype> > sample_images_feature,
                       const shared_ptr<Blob<Dtype> > query_image_feature,
                       const int top_k_results,
                       shared_ptr<Blob<Dtype> > retrieval_results) {
  int num_samples = sample_images_feature->num();
  int num_queries = query_image_feature->num();
  int dim = query_image_feature->count() / num_queries;
  retrieval_results->Reshape(num_queries, std::min(num_samples, top_k_results), 1, 1);
  Dtype* retrieval_results_data = retrieval_results->mutable_cpu_data();
  int hamming_dist;
  for (int i = 0; i < num_queries; ++i) {
    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int> >,
        MinHeapComparison> results;
    for (int j = 0; j < num_samples; ++j) {
      hamming_dist = caffe_hamming_distance(
          dim, query_image_feature->cpu_data() + query_image_feature->offset(i),
          sample_images_feature->cpu_data() + sample_images_feature->offset(j));
      if (results.empty()) {
        results.push(std::make_pair(-hamming_dist, j));
      } else if (-hamming_dist > results.top().first) { // smaller hamming dist
        results.push(std::make_pair(-hamming_dist, j));
        if (results.size() > top_k_results) {
          results.pop();
        }
      }
    }  // for (int j = 0; j < num_samples; ++j) {
    retrieval_results_data += retrieval_results->offset(i);
    for (int k = 0; k < results.size(); ++k) {
      retrieval_results_data[k] = results.top().second;
      results.pop();
    }
  }  // for (int i = 0; i < num_queries; ++i) {
}
