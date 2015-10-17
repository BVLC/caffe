#ifdef USE_AUDIO
#ifndef CAFFE_UTIL_FFT_HPP
#define CAFFE_UTIL_FFT_HPP

#include <boost/scoped_ptr.hpp>

#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
struct FastFourierTransformPImpl;

template <typename Dtype>
class FastFourierTransform_cpu {
 public:
  explicit FastFourierTransform_cpu(int packetSize);
  ~FastFourierTransform_cpu();

  int process(Dtype* input_data, Dtype* output_data, int size);

 private:
  const int _log2Size;
  const int _packetSize;

  boost::scoped_ptr<FastFourierTransformPImpl<Dtype> > _pimpl;
};

template <typename Dtype>
class FastFourierTransform_gpu {
 public:
  explicit FastFourierTransform_gpu(int packetSize);
  ~FastFourierTransform_gpu();

  int process(Dtype* input_data, Dtype* output_data, int size);

 private:
  const int _log2Size;
  const int _packetSize;

  boost::scoped_ptr<FastFourierTransformPImpl<Dtype> > _pimpl;
};

}  // namespace caffe

#endif  // CAFFE_UTIL_FFT_HPP
#endif  // USE_AUDIO
