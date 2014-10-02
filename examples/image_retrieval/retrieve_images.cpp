// Copyright 2014 kloudkl@github

#include <cuda_runtime.h>
#include <google/protobuf/text_format.h>
#include <stdio.h>
#include <queue>  // for std::priority_queue
#include <string>
#include <utility>  // for pair
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

template<typename Dtype>
void similarity_search(
    const shared_ptr<Blob<Dtype> > sample_binary_feature_blobs,
    const shared_ptr<Blob<Dtype> > query_binary_feature,
    const int top_k_results,
    vector<vector<std::pair<int, int> > >* retrieval_results);

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
  BlobProto sample_binary_features;
  ReadProtoFromBinaryFile(sample_binary_features_binaryproto_file,
                          &sample_binary_features);
  shared_ptr<Blob<Dtype> > sample_binary_feature_blob(new Blob<Dtype>());
  sample_binary_feature_blob->FromProto(sample_binary_features);
  int num_samples = sample_binary_feature_blob->num();
  if (top_k_results > num_samples) {
    top_k_results = num_samples;
  }

  LOG(ERROR)<< "Loading query binary features";
  string query_images_feature_blob_binaryproto(argv[++arg_pos]);
  BlobProto query_images_features;
  ReadProtoFromBinaryFile(query_images_feature_blob_binaryproto,
                          &query_images_features);
  shared_ptr<Blob<Dtype> > query_binary_feature_blob(new Blob<Dtype>());
  query_binary_feature_blob->FromProto(query_images_features);

  string save_retrieval_result_filename(argv[++arg_pos]);
  LOG(ERROR)<< "Opening result file " << save_retrieval_result_filename;
  FILE * result_fileid = fopen(save_retrieval_result_filename.c_str(),
                                         "w");

  LOG(ERROR)<< "Retrieving images";
  vector<vector<std::pair<int, int> > > retrieval_results;
  int query_image_index = 0;

  similarity_search<Dtype>(sample_binary_feature_blob,
                           query_binary_feature_blob, top_k_results,
                           &retrieval_results);
  int num_results = retrieval_results.size();
  for (int i = 0; i < num_results; ++i) {
    fprintf(result_fileid, "%d", query_image_index++);
    for (int j = 0; j < retrieval_results[i].size(); ++j) {
      fprintf(result_fileid, " %d:%d", retrieval_results[i][j].first,
              retrieval_results[i][j].second);
    }
    fprintf(result_fileid, "\n");
  }
  if (result_fileid != NULL) {
    fclose(result_fileid);
  }
  LOG(ERROR) << "Successfully retrieved similar images for " << num_results
      << " queries!";
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
    const shared_ptr<Blob<Dtype> > sample_images_feature_blob,
    const shared_ptr<Blob<Dtype> > query_binary_feature_blob,
    const int top_k_results,
    vector<vector<std::pair<int, int> > >* retrieval_results) {
  int num_samples = sample_images_feature_blob->num();
  int num_queries = query_binary_feature_blob->num();
  int dim = query_binary_feature_blob->count() / num_queries;
  LOG(ERROR)<< "num_samples " << num_samples << ", num_queries " <<
  num_queries << ", dim " << dim;
  int hamming_dist;
  int neighbor_index;
  retrieval_results->resize(num_queries);
  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int> >,
      MinHeapComparison> results;
  for (int i = 0; i < num_queries; ++i) {
    while (!results.empty()) {
      results.pop();
    }
    const Dtype* query_data = query_binary_feature_blob->cpu_data()
        + query_binary_feature_blob->offset(i);
    for (int k = 0; k < num_samples; ++k) {
      const Dtype* sample_data = sample_images_feature_blob->cpu_data()
          + sample_images_feature_blob->offset(k);
      hamming_dist = caffe_hamming_distance(dim, query_data, sample_data);
      if (results.size() < top_k_results) {
        results.push(std::make_pair(-hamming_dist, k));
      } else if (-hamming_dist > results.top().first) {
        // smaller hamming dist, nearer neighbor
        results.pop();
        results.push(std::make_pair(-hamming_dist, k));
      }
    }  // for (int k = 0; k < num_samples; ++k) {
    retrieval_results->at(i).resize(results.size());
    for (int k = results.size() - 1; k >= 0; --k) {
      hamming_dist = -results.top().first;
      neighbor_index = results.top().second;
      retrieval_results->at(i)[k] = std::make_pair(neighbor_index,
                                                   hamming_dist);
      results.pop();
    }
  }  // for (int i = 0; i < num_queries; ++i) {
}
