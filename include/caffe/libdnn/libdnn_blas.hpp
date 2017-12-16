#ifndef CAFFE_LIBDNN_LIBDNN_BLAS_HPP_
#define CAFFE_LIBDNN_LIBDNN_BLAS_HPP_

#ifdef USE_LIBDNN

#include <unordered_map>
#include "caffe/common.hpp"
#include "caffe/definitions.hpp"
#include "caffe/quantizer.hpp"
#include "caffe/libdnn/libdnn.hpp"

namespace caffe {

template<typename MItype, typename MOtype>
class LibDNNBlas : public LibDNN<MItype, MOtype> {
  explicit LibDNNBlas(Device* dev_ptr);

  // BLAS 1
  void axpy(const uint_tp n, const MItype alpha, vptr<const MItype> x,
            vptr<MOtype> y);
  void scal(const uint_tp n, const MItype alpha, vptr<MItype> x);
  void dot(const uint_tp n, vptr<const MItype> x, vptr<const MItype> y,
           MOtype* out);
  void asum(const uint_tp n, vptr<MItype> x, MOtype* y);
  void scale(const uint_tp n, const MItype alpha, vptr<const MItype> x,
             vptr<MOtype> y);

  // BLAS 2
  void gemv(const CBLAS_TRANSPOSE trans_A, const uint_tp M, const uint_tp N,
            const MItype alpha, vptr<const MItype> A, vptr<const MItype> X,
            const MItype beta, vptr<MOtype> Y);

  // BLAS 3
  void gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
            const uint_tp M, const uint_tp N, const uint_tp K,
            const MItype alpha, vptr<const MItype> A, vptr<const MItype> B,
            const MItype beta, vptr<MOtype> C, libdnnAccumulatePrecision_t prec,
            shared_ptr<Quantizer<MItype, MItype> > in_quantizer,
            shared_ptr<Quantizer<MItype, MOtype> > out_quantizer);

 private:
  size_t get_id(string identifier);

  string generate_axpy_source();
  string axpy_string_identifier();

  string generate_scal_source();
  string scal_string_identifier();

  string generate_dot_source();
  string dot_string_identifier();

  string generate_asum_source();
  string asum_string_identifier();

  string generate_scale_source();
  string scale_string_identifier();

  string generate_gemv_source();
  string gemv_string_identifier();

  string generate_gemm_source(shared_ptr<DeviceProgram> program,
         shared_ptr<LibDNNTuner> tuner, bool trans_A, bool trans_B,
         const uint_tp M, const uint_tp N, const uint_tp K,
         bool alpha_term, bool beta_term, libdnnAccumulatePrecision_t prec,
         shared_ptr<Quantizer<MItype, MItype> > in_quantizer,
         shared_ptr<Quantizer<MItype, MOtype> > out_quantizer);
  string gemm_string_identifier(const CBLAS_TRANSPOSE trans_A,
         const CBLAS_TRANSPOSE trans_B, const uint_tp M, const uint_tp N,
         const uint_tp K, bool alpha_term, bool beta_term,
         libdnnAccumulatePrecision_t prec,
         shared_ptr<Quantizer<MItype, MItype> > in_quantizer,
         shared_ptr<Quantizer<MItype, MOtype> > out_quantizer);
  void initialize_gemm_tuner(shared_ptr<DeviceProgram> program,
         shared_ptr<LibDNNTuner> tuner);

  std::mutex program_mutex_;
  vector<bool> program_ready_;
  vector<shared_ptr<LibDNNTuner> > program_tuners_;
  vector<shared_ptr<DeviceProgram> > programs_;
  std::unordered_map<string, size_t> program_map_;
};


}  // namespace caffe


#endif  // USE_LIBDNN

#endif  // CAFFE_LIBDNN_LIBDNN_BLAS_HPP_
