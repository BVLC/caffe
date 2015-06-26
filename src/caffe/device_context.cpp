/*
 * device_context.cpp
 *
 *  Created on: Jun 26, 2015
 *      Author: Fabian Tschopp
 */

#include <vector>
#include "caffe/device_context.hpp"
#include "caffe/greentea/greentea.hpp"
#include "caffe/util/device_alternate.hpp"


namespace caffe {

DeviceContext::DeviceContext()
    : workgroup_sizes_(3, 0), id_(0), backend_(Backend::BACKEND_CUDA) {
  this->Init();
}

DeviceContext::DeviceContext(int id, Backend backend)
    : workgroup_sizes_(3, 0), id_(id), backend_(backend) {
  this->Init();
}

void DeviceContext::Init() {
if(backend_ == BACKEND_CUDA) {
#ifdef USE_CUDA
    workgroup_sizes_[0] = CAFFE_CUDA_NUM_THREADS;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    std::vector<size_t> temp(3);
    clGetDeviceInfo(viennacl::ocl::get_context(id_).devices()[0].id(),
           CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t), &temp[0], NULL);
    workgroup_sizes_[0] = temp[0];
    workgroup_sizes_[1] = temp[1];
    workgroup_sizes_[2] = temp[2];
#endif  // USE_GREENTEA
  }
}

Backend DeviceContext::backend() const {
  return backend_;
}

int DeviceContext::id() const {
  return id_;
}

int DeviceContext::WorkgroupSize(int id) {
  return workgroup_sizes_[id];
  return 0;
}

template<>
Blob<float> *DeviceContext::Buffer(int id) {
  if (buff_f_.size() <= id) {
    buff_f_.push_back(Blob<float>(this));
  }
  return &(buff_f_[id]);
}

template<>
Blob<double> *DeviceContext::Buffer(int id) {
  if (buff_d_.size() <= id) {
    buff_d_.push_back(Blob<double>(this));
  }
  return &(buff_d_[id]);
}

}  // namespace caffe
