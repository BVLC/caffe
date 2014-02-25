// Copyright 2014 kloudkl@github

#include <fstream> // for std::ofstream
#include <queue> // for std::priority_queue
#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using namespace caffe;

template<typename Dtype>
void similarity_search(
    const vector<shared_ptr<Blob<Dtype> > >& sample_binary_feature_blobs,
    const shared_ptr<Blob<Dtype> > query_binary_feature,
    const int top_k_results, vector<vector<Dtype> >* retrieval_results);

template<typename Dtype>
int image_retrieval_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return image_retrieval_pipeline<float>(argc, argv);
//  return image_retrieval_pipeline<double>(argc, argv);
}

template<typename Dtype>
int image_retrieval_pipeline(int argc, char** argv) {
  const int num_required_args = 4;
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in binarized features of query images and sample images"
    " extracted by Caffe to retrieve similar images.\n"
    "Usage: demo_retrieve_images  sample_binary_features_binaryproto_file"
    "  query_binary_features_binaryproto_file  save_retrieval_result_filename"
    "  [top_k_results=1]  [CPU/GPU]  [DEVICE_ID=0]";
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

  arg_pos = 0;  // the name of the executable

  LOG(ERROR)<< "Loading sample binary features";
  string sample_binary_features_binaryproto_file(argv[++arg_pos]);
  BlobProtoVector sample_binary_features;
  ReadProtoFromBinaryFile(sample_binary_features_binaryproto_file,
                          &sample_binary_features);
  vector<shared_ptr<Blob<Dtype> > > sample_binary_feature_blobs;
  int num_samples;
  for (int i = 0; i < sample_binary_features.blobs_size(); ++i) {
    shared_ptr<Blob<Dtype> > blob(new Blob<Dtype>());
    blob->FromProto(sample_binary_features.blobs(i));
    sample_binary_feature_blobs.push_back(blob);
    num_samples += blob->num();
  }
  if (top_k_results > num_samples) {
    top_k_results = num_samples;
  }

  LOG(ERROR)<< "Loading query binary features";
  string query_images_feature_blob_binaryproto(argv[++arg_pos]);
  BlobProtoVector query_images_features;
  ReadProtoFromBinaryFile(query_images_feature_blob_binaryproto,
                          &query_images_features);
  vector<shared_ptr<Blob<Dtype> > > query_binary_feature_blobs;
  for (int i = 0; i < query_images_features.blobs_size(); ++i) {
    shared_ptr<Blob<Dtype> > blob(new Blob<Dtype>());
    blob->FromProto(query_images_features.blobs(i));
    query_binary_feature_blobs.push_back(blob);
  }

  string save_retrieval_result_filename(argv[++arg_pos]);
  LOG(ERROR)<< "Opening result file " << save_retrieval_result_filename;
  std::ofstream retrieval_result_ofs(save_retrieval_result_filename.c_str(),
                                     std::ofstream::out);

  LOG(ERROR)<< "Retrieving images";
  vector<vector<Dtype> > retrieval_results;
  int query_image_index = 0;

  int num_query_batches = query_binary_feature_blobs.size();
  for (int batch_index = 0; batch_index < num_query_batches; ++batch_index) {
    similarity_search<Dtype>(sample_binary_feature_blobs,
                             query_binary_feature_blobs[batch_index],
                             top_k_results, &retrieval_results);
    int num_results = retrieval_results.size();
    for (int i = 0; i < num_results; ++i) {
      retrieval_result_ofs << query_image_index++;
      for (int j = 0; j < retrieval_results[i].size(); ++j) {
        retrieval_result_ofs << " " << retrieval_results[i][j];
      }
      retrieval_result_ofs << "\n";
    }
  }  //  for (int batch_index = 0; batch_index < num_query_batches; ++batch_index) {

  retrieval_result_ofs.close();
  LOG(ERROR)<< "Successfully retrieved similar images for " << query_image_index << " queries!";
  return 0;
}

class MinHeapComparison {
 public:
  bool operator()(const std::pair<int, int>& lhs,
                  const std::pair<int, int>&rhs) const {
    return (lhs.first > rhs.first);
  }
};

template<typename Dtype>
void similarity_search(
    const vector<shared_ptr<Blob<Dtype> > >& sample_images_feature_blobs,
    const shared_ptr<Blob<Dtype> > query_image_feature, const int top_k_results,
    vector<vector<Dtype> >* retrieval_results) {
  int num_queries = query_image_feature->num();
  int dim = query_image_feature->count() / num_queries;
  int hamming_dist;
  retrieval_results->resize(num_queries);
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int> >,
      MinHeapComparison> results;
  for (int i = 0; i < num_queries; ++i) {
    while (!results.empty()) {
      results.pop();
    }
    for (int j = 0; j < sample_images_feature_blobs.size(); ++j) {
      int num_samples = sample_images_feature_blobs[j]->num();
      for (int k = 0; k < num_samples; ++k) {
        hamming_dist = caffe_hamming_distance(
            dim,
            query_image_feature->cpu_data() + query_image_feature->offset(i),
            sample_images_feature_blobs[j]->cpu_data()
                + sample_images_feature_blobs[j]->offset(k));
        if (results.size() < top_k_results) {
          results.push(std::make_pair(-hamming_dist, k));
        } else if (-hamming_dist > results.top().first) {  // smaller hamming dist
          results.pop();
          results.push(std::make_pair(-hamming_dist, k));
        }
      }  // for (int k = 0; k < num_samples; ++k) {
    }  // for (int j = 0; j < sample_images_feature_blobs.size(); ++j)
    retrieval_results->at(i).resize(results.size());
    for (int k = results.size() - 1; k >= 0; --k) {
      retrieval_results->at(i)[k] = results.top().second;
      results.pop();
    }
  }  // for (int i = 0; i < num_queries; ++i) {
}
