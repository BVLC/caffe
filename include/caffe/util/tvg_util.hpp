#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PrintBlob(const Blob<Dtype>* blob, bool print_diff = false, const char* info = 0) {

  const Dtype* data = print_diff ? blob->cpu_diff() : blob->cpu_data();

  if (info != 0) {
    printf("%s: \n", info);
  }

  for (int n = 0; n < blob->num(); n++) {
    for (int c = 0; c < blob->channels(); c++) {
      for (int h = 0; h < blob->height(); h++) {
        for (int w = 0; w < blob->width(); w++) {
          int offset = ((n * blob->channels() + c) * blob->height() + h) * blob->width() + w;
          printf("%11.6f ", *(data + offset));
        }
        printf("\n");
      }
      printf("\n");
    }
   // printf("=================\n");
  }

  printf("-- End of Blob --\n\n");
}

template void PrintBlob(const Blob<float>* blob, bool print_diff = false, const char* info = 0);

template void PrintBlob(const Blob<double>* blob, bool print_diff = false, const char* info = 0);


template <typename Dtype>
void FillWithMax(Blob<Dtype>* blob, float max_value = 1) {

  srand(2000);

  for (int i = 0; i < blob->count(); ++i) {
    blob->mutable_cpu_data()[i] = ((double) rand() / RAND_MAX) * max_value;
  }
}
template void FillWithMax(Blob<float>* const blob, float max_value = 1);
template void FillWithMax(Blob<double>* const blob, float max_value = 1);


template <typename Dtype>
void FillAsRGB(Blob<Dtype>* blob) {

  srand(2000);

  for (int i = 0; i < blob->count(); ++i) {
    blob->mutable_cpu_data()[i] = rand() % 256;
  }
}
template void FillAsRGB(Blob<float>* const blob);
template void FillAsRGB(Blob<double>* const blob);

template<typename Dtype>
void FillAsProb(Blob<Dtype>* blob) {

  srand(1000);//time(NULL));

  for (int i = 0; i < blob->count(); ++i) {
    double num = (double) rand() / (double) RAND_MAX;
    blob->mutable_cpu_data()[i] = static_cast<Dtype>((num != 0) ? num : 0.0002);
  }

  for (int n = 0; n < blob->num(); ++n) {
    for (int h = 0; h < blob->height(); ++h) {
      for (int w = 0; w < blob->width(); ++w) {

        Dtype total = 0;

        for (int c = 0; c < blob->channels(); ++c) {
          total += blob->data_at(n, c, h, w);
        }

        for (int c = 0; c < blob->channels(); ++c) {
          blob->mutable_cpu_data()[blob->offset(n, c, h, w)] = blob->data_at(n, c, h, w) / total;
        }
      }
    }
  }
}
template void FillAsProb(Blob<float>* const blob);
template void FillAsProb(Blob<double>* const blob);


template<typename Dtype>
void FillAsLogProb(Blob<Dtype>* blob) {
  FillAsProb(blob);

  for (int i = 0; i < blob->count(); ++i) {
    blob->mutable_cpu_data()[i] = log(blob->cpu_data()[i]);
  }
}
template void FillAsLogProb(Blob<float>* const blob);
template void FillAsLogProb(Blob<double>* const blob);

}
