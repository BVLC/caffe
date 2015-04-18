/*
 * greentea.cpp
 *
 *  Created on: Apr 6, 2015
 *      Author: Fabian Tschopp
 */

#include "caffe/greentea/greentea.hpp"

namespace caffe {

#ifdef USE_GREENTEA
template<typename Dtype>
cl_mem Subregion(cl_mem in, size_t off, size_t size) {
  cl_buffer_region* region = new cl_buffer_region();
  region->origin = sizeof(Dtype) * off;
  region->size = sizeof(Dtype) * size;
  cl_int status;
  const cl_mem out = clCreateSubBuffer(in, CL_MEM_READ_WRITE,
  CL_BUFFER_CREATE_TYPE_REGION,
                                       region, &status);
  std::cout << "Subregion: " << status << std::endl;
  return out;
}

template cl_mem Subregion<float>(cl_mem in, size_t off, size_t size);
template cl_mem Subregion<double>(cl_mem in, size_t off, size_t size);

template<typename Dtype>
viennacl::vector<Dtype> WrapVector(cl_mem in) {
  if (in == NULL) {
    size_t size;
    clGetMemObjectInfo(in, CL_MEM_SIZE, sizeof(size_t), &size, NULL);
    viennacl::vector<Dtype> out(in, viennacl::OPENCL_MEMORY,
                                size / sizeof(Dtype));
    return out;
  } else {
    std::cout << "HERE!" << std::endl;
    void* ptr = NULL;
    viennacl::vector<Dtype> out((cl_mem)&ptr, viennacl::OPENCL_MEMORY, 0);
    return out;
  }
}

template viennacl::vector<float> WrapVector<float>(cl_mem in);
template viennacl::vector<double> WrapVector<double>(cl_mem in);
template viennacl::vector<int> WrapVector<int>(cl_mem in);
template viennacl::vector<long> WrapVector<long>(cl_mem in);

#endif

DeviceContext::DeviceContext()
    : id_(0),
      backend_(Backend::BACKEND_CUDA) {

}

DeviceContext::DeviceContext(int id, Backend backend)
    : id_(id),
      backend_(backend) {

}

Backend DeviceContext::backend() const {
  return backend_;
}

int DeviceContext::id() const {
  return id_;
}

}
