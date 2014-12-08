#include <cuda_runtime.h>

#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>

#include "caffe/caffe.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
  if (argc < 3) {
    LOG(ERROR) << std::endl
      << "extract-layer-features-from-images" << std::endl
      << "    data-mean.proto" << std::endl
      << "    output-mean-file";
    return 0;
  }

  // Read mean file
  LOG(INFO) << "Loading mean file from" << argv[3];
  BlobProto blob_proto;
  ReadProtoFromBinaryFile(argv[1], &blob_proto);
  Blob<float> data_mean_;
  data_mean_.FromProto(blob_proto);
  const float* mean = data_mean_.cpu_data();
  LOG(INFO) << "Load mean done!";

  // Read layers
  string output_file_name = argv[2];
  FILE *output_file;
  output_file = fopen(output_file_name.c_str(), "w");

  int n_channel = 3;
  int image_height = 256;
  int image_width = 256;

  LOG(INFO) << "Write to output file " << output_file_name.c_str();
  fwrite(&n_channel, sizeof(int), 1, output_file);
  fwrite(&image_height, sizeof(int), 1, output_file);
  fwrite(&image_width, sizeof(int), 1, output_file);
  fwrite(mean, sizeof(float), n_channel * image_height * image_width, output_file);

  fclose(output_file);

  return 0;
}
