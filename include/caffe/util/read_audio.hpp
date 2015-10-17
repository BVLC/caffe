#pragma once

#include <string>

namespace caffe {

int ReadAudioFile(const std::string& filePath, float* data, int capacity,
  int offset = 0);
int ReadAudioFile(const std::string& filePath, double* data, int capacity,
  int offset = 0);

}  // namespace caffe
