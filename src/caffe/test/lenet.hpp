#ifndef CAFFE_TEST_LENET_HPP_
#define CAFFE_TEST_LENET_HPP_

#include <string>

namespace caffe {

const char* kLENET = "name: \"LeNet\"\n\
bottom: \"data\"\n\
bottom: \"label\"\n\
layers {\n\
  layer {\n\
    name: \"conv1\"\n\
    type: \"conv\"\n\
    num_output: 20\n\
    kernelsize: 5\n\
    stride: 1\n\
    weight_filler {\n\
      type: \"xavier\"\n\
    }\n\
    bias_filler {\n\
      type: \"constant\"\n\
    }\n\
  }\n\
  bottom: \"data\"\n\
  top: \"conv1\"\n\
}\n\
layers {\n\
  layer {\n\
    name: \"pool1\"\n\
    type: \"pool\"\n\
    kernelsize: 2\n\
    stride: 2\n\
    pool: MAX\n\
  }\n\
  bottom: \"conv1\"\n\
  top: \"pool1\"\n\
}\n\
layers {\n\
  layer {\n\
    name: \"conv2\"\n\
    type: \"conv\"\n\
    num_output: 50\n\
    kernelsize: 5\n\
    stride: 1\n\
    weight_filler {\n\
      type: \"xavier\"\n\
    }\n\
    bias_filler {\n\
      type: \"constant\"\n\
    }\n\
  }\n\
  bottom: \"pool1\"\n\
  top: \"conv2\"\n\
}\n\
layers {\n\
  layer {\n\
    name: \"pool2\"\n\
    type: \"pool\"\n\
    kernelsize: 2\n\
    stride: 2\n\
    pool: MAX\n\
  }\n\
  bottom: \"conv2\"\n\
  top: \"pool2\"\n\
}\n\
layers {\n\
  layer {\n\
    name: \"ip1\"\n\
    type: \"innerproduct\"\n\
    num_output: 500\n\
    weight_filler {\n\
      type: \"xavier\"\n\
    }\n\
    bias_filler {\n\
      type: \"constant\"\n\
    }\n\
  }\n\
  bottom: \"pool2\"\n\
  top: \"ip1\"\n\
}\n\
layers {\n\
  layer {\n\
    name: \"relu1\"\n\
    type: \"relu\"\n\
  }\n\
  bottom: \"ip1\"\n\
  top: \"relu1\"\n\
}\n\
layers {\n\
  layer {\n\
    name: \"ip2\"\n\
    type: \"innerproduct\"\n\
    num_output: 10\n\
    weight_filler {\n\
      type: \"xavier\"\n\
    }\n\
    bias_filler {\n\
      type: \"constant\"\n\
    }\n\
  }\n\
  bottom: \"relu1\"\n\
  top: \"ip2\"\n\
}\n\
layers {\n\
  layer {\n\
    name: \"prob\"\n\
    type: \"softmax\"\n\
  }\n\
  bottom: \"ip2\"\n\
  top: \"prob\"\n\
}\n\
layers {\n\
  layer {\n\
    name: \"loss\"\n\
    type: \"multinomial_logistic_loss\"\n\
  }\n\
  bottom: \"prob\"\n\
  bottom: \"label\"\n\
}";

}  // namespace caffe

#endif
