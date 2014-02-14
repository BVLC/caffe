#include <cuda_runtime.h>
#include <fcntl.h>

#include <cstring>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/vision_layers.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  
  if (argc < 9) {
    LOG(ERROR) << "padded_im2col_test [CPU/GPU] input_size channel pad filter_size filter_number stride batch_size";
    return 0;
  }

  int input_size = atoi(argv[2]);
  int channel = atoi(argv[3]);
  int pad = atoi(argv[4]);
  int filter_size = atoi(argv[5]);
  int filter_number = atoi(argv[6]);
  int stride = atoi(argv[7]);
  int batch_size = atoi(argv[8]);
  int output_size = (input_size + 2 * pad - filter_size) / stride + 1;
  int middle_size = input_size + 2 * pad;

  // Setup convolution layer that supports padding

  vector<Blob<float>*> bottom1;
  vector<Blob<float>*> top1;
  
  bottom1.push_back(new Blob<float>(batch_size, channel, input_size, input_size));
  top1.push_back(new Blob<float>(batch_size, filter_number, output_size, output_size));

  LayerParameter layer_param;
  layer_param.set_kernelsize(filter_size);
  layer_param.set_pad(pad);
  layer_param.set_stride(stride);
  layer_param.set_num_output(filter_number);

  ConvolutionLayer<float> conv_layer(layer_param);
  conv_layer.SetUp(bottom1, &top1);

  // Setup convolution layer that does not support padding

  vector<Blob<float>*> bottom2;
  vector<Blob<float>*> top2;
  vector<Blob<float>*> middle;
  bottom2.push_back(new Blob<float>(batch_size, channel, input_size, input_size));
  middle.push_back(new Blob<float>(batch_size, channel, middle_size, middle_size));
  top2.push_back(new Blob<float>(batch_size, filter_number, output_size, output_size));
  
  PaddingLayer<float> padding_layer(layer_param);
  padding_layer.SetUp(bottom2, &middle);

  layer_param.set_pad(0);
  ConvolutionLayer<float> conv_layer_nopad(layer_param);
  conv_layer_nopad.SetUp(middle, &top2);

  // Setup euclidean_loss

  vector<Blob<float>*> loss_input;
  loss_input.push_back(top1[0]);
  loss_input.push_back(top2[0]);
  vector<Blob<float>*> diff_loss_input;
  diff_loss_input.push_back(bottom1[0]);
  diff_loss_input.push_back(bottom2[0]);

  // Fill bottom data
  caffe_vRngGaussian<float>(bottom1[0]->count(), bottom1[0]->mutable_cpu_data(), float(0), float(0.01));
  bottom2[0]->CopyFrom(*bottom1[0]);
  
  // Fill top diff
  caffe_vRngGaussian<float>(top1[0]->count(), top1[0]->mutable_cpu_diff(), float(0), float(0.01));
  top2[0]->CopyFrom(*top1[0], true);

  // Fill Conv Filter
  caffe_vRngGaussian<float>(conv_layer.blobs()[0].get()->count(), conv_layer.blobs()[0].get()->mutable_cpu_data(), float(0), float(0.01));
  caffe_vRngGaussian<float>(conv_layer.blobs()[1].get()->count(), conv_layer.blobs()[1].get()->mutable_cpu_data(), float(0), float(0.01));
  
  // Fill conv nopad with the same weights
  conv_layer_nopad.blobs()[0].get()->CopyFrom(*conv_layer.blobs()[0].get());
  conv_layer_nopad.blobs()[1].get()->CopyFrom(*conv_layer.blobs()[1].get());

  if (strcmp(argv[1], "GPU") == 0) {
    LOG(ERROR) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }
  
  // Forward
  conv_layer.Forward(bottom1, &top1);
  padding_layer.Forward(bottom2, &middle);
  conv_layer_nopad.Forward(middle, &top2);
  Blob<float> difference_(batch_size, filter_number, output_size, output_size);
  int count = loss_input[0]->count();
  int num = loss_input[0]->num();
  caffe_sub(count, loss_input[0]->cpu_data(), loss_input[1]->cpu_data(), difference_.mutable_cpu_data());
  float loss_input0 = caffe_cpu_dot(count, loss_input[0]->cpu_data(), loss_input[0]->cpu_data()) / num / float(2);
  float loss_input1 = caffe_cpu_dot(count, loss_input[1]->cpu_data(), loss_input[1]->cpu_data()) / num / float(2);
  float loss = caffe_cpu_dot(count, difference_.cpu_data(), difference_.cpu_data()) / num / float(2);

  // Backward
  conv_layer.Backward(top1, true, &bottom1);
  conv_layer_nopad.Backward(top2, true, &middle);
  padding_layer.Backward(middle, true, &bottom2);
  Blob<float> diff_difference_(batch_size, channel, input_size, input_size);
  count = diff_loss_input[0]->count();
  num = diff_loss_input[0]->num();
  caffe_sub(count, diff_loss_input[0]->cpu_diff(), diff_loss_input[1]->cpu_diff(), diff_difference_.mutable_cpu_data());
  float diff_loss_input0 = caffe_cpu_dot(count, diff_loss_input[0]->cpu_diff(), diff_loss_input[0]->cpu_diff()) / num / float(2);
  float diff_loss_input1 = caffe_cpu_dot(count, diff_loss_input[1]->cpu_diff(), diff_loss_input[1]->cpu_diff()) / num / float(2);
  float diff_loss = caffe_cpu_dot(count, diff_difference_.cpu_data(), diff_difference_.cpu_data()) / num / float(2);

  LOG(ERROR) << "loss0: " << loss_input0 << " loss1: " << loss_input1 << " euclidean distance: " << loss;
  LOG(ERROR) << "diff_loss0: " << diff_loss_input0 << " diff_loss1: " << diff_loss_input1 << " diff euclidean distance: " << diff_loss;
  return 0;
}
