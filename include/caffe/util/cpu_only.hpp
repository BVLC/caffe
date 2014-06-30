// Copyright 2014 BVLC and contributors.

#ifndef CAFFE_UTIL_CPU_ONLY_H_
#define CAFFE_UTIL_CPU_ONLY_H_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

// For CPU-only Caffe, stub out GPU calls as unavailable.
#ifdef CPU_ONLY

#define NO_GPU LOG(FATAL) << "CPU-only Mode"

#define STUB_GPU(classname) \
template <typename Dtype> \
Dtype classname<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, \
    vector<Blob<Dtype>*>* top) { NO_GPU; } \
template <typename Dtype> \
void classname<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    vector<Blob<Dtype>*>* bottom) { NO_GPU; } \

#define STUB_GPU_FORWARD(classname, funcname) \
template <typename Dtype> \
Dtype classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& bottom, \
    vector<Blob<Dtype>*>* top) { NO_GPU; } \

#define STUB_GPU_BACKWARD(classname, funcname) \
template <typename Dtype> \
void classname<Dtype>::funcname##_##gpu(const vector<Blob<Dtype>*>& top, \
    const vector<bool>& propagate_down, \
    vector<Blob<Dtype>*>* bottom) { NO_GPU; } \

#endif  // CPU_ONLY
#endif  // CAFFE_UTIL_CPU_ONLY_H_
