#ifndef CAFFE_LIBDNN_LIBDNN_BLAS_HPP_
#define CAFFE_LIBDNN_LIBDNN_BLAS_HPP_

#ifdef USE_LIBDNN

#include <unordered_map>
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
  void gemv(const CBLAS_TRANSPOSE trans_A, const uint_tp M, const uint_tp N,
            const Dtype alpha, vptr<const MItype> A, vptr<const MItype> X,
            const Dtype beta, vptr<MOtype> Y);

  // BLAS 3
  void gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
            const uint_tp M, const uint_tp N, const uint_tp K,
            const Dtype alpha, vptr<const MItype> A, vptr<const MItype> B,
            const Dtype beta, vptr<MOtype> C);

 private:
  size_t get_id(string identifier);

  string generate_axpy_kernel();
  string axpy_string_identifier();

  string generate_scal_kernel();
  string scal_string_identifier();

  string generate_dot_kernel();
  string dot_string_identifier();

  string generate_asum_kernel();
  string asum_string_identifier();

  string generate_scale_kernel();
  string scale_string_identifier();

  string generate_gemv_kernel();
  string gemv_string_identifier();

  string generate_gemm_kernel();
  string gemm_string_identifier();

  std::mutex program_mutex_;
  vector<bool> program_ready_;
  vector<shared_ptr<LibDNNTuner> > program_tuners_;
  vector<shared_ptr<DeviceProgram> > programs_;
  std::unordered_map<string, size_t> program_map_;
};


}  // namespace caffe


#endif  // USE_LIBDNN

#endif  // CAFFE_LIBDNN_LIBDNN_BLAS_HPP_
