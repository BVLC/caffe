#ifndef CAFFE_UTIL_FLUENT_H // CAFFE_UTIL_FLUENT_H
#define CAFFE_UTIL_FLUENT_H

#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

namespace cpu {

template <typename Dtype>
struct BlobAccessor
{
  Blob<Dtype>* blob;
  bool diff;
  BlobAccessor(shared_ptr<Blob<Dtype> >& blob, bool diff = false)
  :blob(blob.get()), diff(diff)
  {}

  Dtype* write() { return diff ? blob->mutable_cpu_diff() : blob->mutable_cpu_data(); }
  const Dtype* read() const { return diff ? blob->cpu_diff() : blob->cpu_data(); }
};

template <typename Dtype>
struct Fluent
{
  typedef BlobAccessor<Dtype> Accessor;

  int N;
  Fluent(int N)
  : N(N) 
  {}

  inline Fluent& axpy(Dtype a, const Accessor& x, Accessor& y)
  {
    caffe_axpy(N,a,x.read(),y.write());
    return *this;
  }

  inline Fluent& axpby(Dtype a, const Accessor& x, Dtype b, Accessor& y)
  {
    caffe_cpu_axpby(N,a,x.read(),b,y.write());
    return *this;
  }

  inline Fluent& sign(const Accessor& x, Accessor& y)
  {
    caffe_cpu_sign(N,x.read(),y.write());
    return *this;
  }

  inline Fluent& powx(const Accessor& x, Dtype pow, Accessor& y)
  {
    caffe_powx(N,x.read(),pow,y.write());
    return *this;
  }

  inline Fluent& mul(const Accessor& a, const Accessor& b, Accessor& y)
  {
    caffe_mul(N,a.read(),b.read(),y.write());
    return *this;
  }

  inline Fluent& div(const Accessor& a, const Accessor& b, Accessor& y)
  {
    caffe_div(N,a.read(),b.read(),y.write());
    return *this;
  }

  inline Fluent& clamp(Dtype a, Dtype b, Accessor& y)
  {
    caffe_clamp(N,a,b,y.write());
    return *this;
  }

  inline Fluent& add(const Accessor& a, const Accessor& b, Accessor& y)
  {
    caffe_add(N,a.read(),b.read(),y.write());
    return *this;
  }

  inline Fluent& add_scalar(Dtype a, Accessor& y)
  {
    caffe_add_scalar(N,a,y.write());
    return *this;
  }

  inline Fluent& copy(const Accessor& a, Accessor& y)
  {
    caffe_copy(N,a.read(),y.write());
    return *this;
  }

  inline Fluent& sqrt(const Accessor& a, Accessor& y)
  {
    caffe_sqrt(N,a.read(),y.write());
    return *this;
  }

  inline Fluent& sqr(const Accessor& a, Accessor& y)
  {
    caffe_sqr(N,a.read(),y.write());
    return *this;
  }

  inline Fluent& set(Dtype a, Accessor& y)
  {
    caffe_set(N, a, y.write()); 
    return *this;
  }

  inline Fluent& scal(Dtype a, Accessor& y)
  {
    caffe_scal(N, a, y.write()); 
    return *this;
  }  
};

}  // namespace cpu

namespace gpu {

template <typename Dtype>
struct BlobAccessor
{
  Blob<Dtype>* blob;
  bool diff;
  BlobAccessor(shared_ptr<Blob<Dtype> >& blob, bool diff = false)
  :blob(blob.get()), diff(diff)
  {}

  Dtype* write() { return diff ? blob->mutable_gpu_diff() : blob->mutable_gpu_data(); }
  const Dtype* read() const { return diff ? blob->gpu_diff() : blob->gpu_data(); }
};

template <typename Dtype>
struct Fluent
{
  typedef BlobAccessor<Dtype> Accessor;

  int N;
  Fluent(int N)
  : N(N) 
  {}

  inline Fluent& axpy(Dtype a, const Accessor& x, Accessor& y)
  {
    caffe_gpu_axpy(N,a,x.read(),y.write());
    return *this;
  }

  inline Fluent& axpby(Dtype a, const Accessor& x, Dtype b, Accessor& y)
  {
    caffe_gpu_axpby(N,a,x.read(),b,y.write());
    return *this;
  }

  inline Fluent& sign(const Accessor& x, Accessor& y)
  {
    caffe_gpu_sign(N,x.read(),y.write());
    return *this;
  }

  inline Fluent& powx(const Accessor& x, Dtype pow, Accessor& y)
  {
    caffe_gpu_powx(N,x.read(),pow,y.write());
    return *this;
  }

  inline Fluent& mul(const Accessor& a, const Accessor& b, Accessor& y)
  {
    caffe_gpu_mul(N,a.read(),b.read(),y.write());
    return *this;
  }

  inline Fluent& div(const Accessor& a, const Accessor& b, Accessor& y)
  {
    caffe_gpu_div(N,a.read(),b.read(),y.write());
    return *this;
  }

  inline Fluent& clamp(Dtype a, Dtype b, Accessor& y)
  {
    caffe_gpu_clamp(N,a,b,y.write());
    return *this;
  }

  inline Fluent& add(const Accessor& a, const Accessor& b, Accessor& y)
  {
    caffe_gpu_add(N,a.read(),b.read(),y.write());
    return *this;
  }

  inline Fluent& add_scalar(Dtype a, Accessor& y)
  {
    caffe_gpu_add_scalar(N,a,y.write());
    return *this;
  }

  inline Fluent& copy(const Accessor& a, Accessor& y)
  {
    caffe_copy(N,a.read(),y.write());
    return *this;
  }

  inline Fluent& sqrt(const Accessor& a, Accessor& y)
  {
    caffe_gpu_sqrt(N,a.read(),y.write());
    return *this;
  }

  inline Fluent& sqr(const Accessor& a, Accessor& y)
  {
    caffe_gpu_mul(N,a.read(),a.read(),y.write());
    return *this;
  }

  inline Fluent& set(Dtype a, Accessor& y)
  {
    caffe_gpu_set(N, a, y.write()); 
    return *this;
  }

  inline Fluent& scal(Dtype a, Accessor& y)
  {
    caffe_gpu_scal(N, a, y.write()); 
    return *this;
  }  
};

}  // namespace gpu

}  // namespace caffe

#endif  // CAFFE_UTIL_FLUENT_H