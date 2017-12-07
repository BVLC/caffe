#ifndef CAFFE_LIBDNN_LIBDNN_BLAS_HPP_
#define CAFFE_LIBDNN_LIBDNN_BLAS_HPP_

#include "caffe/libdnn/libdnn.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
class LibDNNBlas : public LibDNN<Dtype, MItype, MOtype> {
  explicit LibDNNBlas(Device* dev_ptr);

  // BLAS 1
  void axpy(const uint_tp n, const Dtype alpha, vptr<const MItype> x,
            vptr<MOtype> y);
  void scal(const uint_tp n, const Dtype alpha, vptr<MItype> x);
  void dot(const uint_tp n, vptr<const MItype> x, vptr<const MItype> y,
           MOtype* out);
  void asum(const uint_tp n, vptr<MItype> x, MOtype* y);
  void scale(const uint_tp n, const Dtype alpha, vptr<const MItype> x,
             vptr<MOtype> y);

  // BLAS 2
  void gemv(const CBLAS_TRANSPOSE trans_a, const uint_tp m, const uint_tp n,
            const Dtype alpha, vptr<const MItype> a, vptr<const MItype> x,
            const Dtype beta, vptr<MOtype> y);

  // BLAS 3
  void gemm(const CBLAS_TRANSPOSE trans_a, const CBLAS_TRANSPOSE trans_b,
            const uint_tp m, const uint_tp n, const uint_tp k,
            const Dtype alpha, vptr<const MItype> a, vptr<const MItype> b,
            const Dtype beta, vptr<MOtype> c);

 private:
  shared_ptr<DeviceProgram> GetProgram(string identifier);

  vector<shared_ptr<DeviceProgram> > programs_;
  std::map<string, size_t> program_map_;
};


}  // namespace caffe

#endif  // CAFFE_LIBDNN_LIBDNN_BLAS_HPP_
