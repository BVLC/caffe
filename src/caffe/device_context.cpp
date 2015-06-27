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
    : current_queue_id_(0), workgroup_sizes_(3, 0), id_(0),
      backend_(Backend::BACKEND_CUDA) {
}

DeviceContext::DeviceContext(int id, Backend backend)
    : current_queue_id_(0), workgroup_sizes_(3, 0), id_(id),
      backend_(backend) {
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

int DeviceContext::num_queues() {
  if (backend_ == BACKEND_CUDA) {
#ifdef USE_CUDA
    return 1;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    return GREENTEA_QUEUE_COUNT;
#endif  // USE_GREENTEA
  }
  return 1;
}

template<>
shared_ptr< Blob<float> > DeviceContext::Buffer(int id) {
  if (buff_f_.size() <= id) {
    shared_ptr<Blob<float> > blob_pointer(new Blob<float>(this));
    buff_f_.push_back(blob_pointer);
  }
  return buff_f_[id];
}

template<>
shared_ptr< Blob<double> > DeviceContext::Buffer(int id) {
  if (buff_d_.size() <= id) {
    shared_ptr<Blob<double> > blob_pointer(new Blob<double>(this));
    buff_d_.push_back(blob_pointer);
  }
  return buff_d_[id];
}

int DeviceContext::current_queue_id() {
  return current_queue_id_;
}

void DeviceContext::SwitchQueue(int id) {
  if (backend_ == BACKEND_CUDA) {
#ifdef USE_CUDA
    (void) id;
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
    viennacl::ocl::context &ctx =
        viennacl::ocl::get_context(id_);
    ctx.switch_queue(id % num_queues());
    current_queue_id_ = id % num_queues();
#endif  // USE_GREENTEA
  }
}

void DeviceContext::FinishQueues() {
  if (backend_ == BACKEND_CUDA) {
#ifdef USE_CUDA
#endif  // USE_CUDA
  } else {
  #ifdef USE_GREENTEA
    viennacl::ocl::context &ctx =
        viennacl::ocl::get_context(id_);
    for (int i = 0; i < num_queues(); ++i) {
      ctx.switch_queue(i);
      ctx.get_queue().finish();
    }
    ctx.switch_queue(0);
    current_queue_id_ = 0;
  #endif  // USE_GREENTEA
  }
}

}  // namespace caffe
