#ifndef CAFFE_LIBDNN_LIBDNN_BLAS_HPP_
#define CAFFE_LIBDNN_LIBDNN_BLAS_HPP_

#ifdef USE_LIBDNN

#include <boost/thread.hpp>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <unordered_map>

#include "caffe/common.hpp"
#include "caffe/definitions.hpp"
#include "caffe/quantizer.hpp"
#include "caffe/libdnn/libdnn.hpp"
namespace caffe {

template<typename MItype, typename MOtype>
class LibDNNBlas : public LibDNN<MItype, MOtype> {
 public:
  explicit LibDNNBlas(Device* dev_ptr);

  // BLAS 1
  void axpby(const uint_tp n, const MItype alpha, vptr<const MItype> x,
             const MOtype beta, vptr<MOtype> y,
             shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  void scal(const uint_tp n, const MItype alpha, vptr<MItype> x,
            shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  void dot(const uint_tp n, vptr<const MItype> x, vptr<const MItype> y,
           MOtype* out,
           libdnnAccumulatePrecision_t prec,
           shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  void asum(const uint_tp n, vptr<MItype> x, MOtype* y,
            libdnnAccumulatePrecision_t prec,
            shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  void scale(const uint_tp n, const MItype alpha, vptr<const MItype> x,
             vptr<MOtype> y, shared_ptr<Quantizer<MItype, MOtype> > quantizer);

  // BLAS 2
  void gemv(const CBLAS_TRANSPOSE trans_A, const uint_tp M, const uint_tp N,
            const MItype alpha, vptr<const MItype> A, vptr<const MItype> x,
            const MItype beta, vptr<MOtype> y,
            libdnnAccumulatePrecision_t prec,
            shared_ptr<Quantizer<MItype, MOtype> > quantizer);

  // BLAS 3
  void gemm(const CBLAS_TRANSPOSE trans_A, const CBLAS_TRANSPOSE trans_B,
            const uint_tp M, const uint_tp N, const uint_tp K,
            const MItype alpha, vptr<const MItype> A, vptr<const MItype> B,
            const MItype beta, vptr<MOtype> C, libdnnAccumulatePrecision_t prec,
            shared_ptr<Quantizer<MItype, MOtype> > quantizer);

 private:
  int_tp get_id(string identifier);
  int_tp get_id_or_new(string identifier);

  string generate_scale_source(shared_ptr<DeviceProgram> program,
                              shared_ptr<LibDNNTuner> tuner,
                              shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  string scale_string_identifier(
                              shared_ptr<Quantizer<MItype, MOtype> > quantizer);

  string generate_axpby_source(shared_ptr<DeviceProgram> program,
                              shared_ptr<LibDNNTuner> tuner,
                              shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  string axpby_string_identifier(
                              shared_ptr<Quantizer<MItype, MOtype> > quantizer);

  string generate_dot_source(shared_ptr<DeviceProgram> program,
                             shared_ptr<LibDNNTuner> tuner,
                             libdnnAccumulatePrecision_t prec,
                             shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  string dot_string_identifier(libdnnAccumulatePrecision_t prec,
                              shared_ptr<Quantizer<MItype, MOtype> > quantizer);

  string generate_asum_source(shared_ptr<DeviceProgram> program,
                              shared_ptr<LibDNNTuner> tuner,
                              libdnnAccumulatePrecision_t prec,
                              shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  string asum_string_identifier(libdnnAccumulatePrecision_t prec,
                              shared_ptr<Quantizer<MItype, MOtype> > quantizer);



  string generate_gemv_source(
         shared_ptr<DeviceProgram> program, shared_ptr<LibDNNTuner> tuner,
         bool trans_A, const uint_tp M, const uint_tp N, bool alpha_term,
         bool beta_term, libdnnAccumulatePrecision_t prec,
         shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  string gemv_string_identifier(const CBLAS_TRANSPOSE trans_A,
         const uint_tp M, const uint_tp N, bool alpha_term, bool beta_term,
         libdnnAccumulatePrecision_t prec,
         shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  void initialize_gemv_tuner(shared_ptr<DeviceProgram> program,
         shared_ptr<LibDNNTuner> tuner);

  string generate_gemm_source(shared_ptr<DeviceProgram> program,
         shared_ptr<LibDNNTuner> tuner, bool trans_A, bool trans_B,
         const uint_tp M, const uint_tp N, const uint_tp K,
         bool alpha_term, bool beta_term, libdnnAccumulatePrecision_t prec,
         shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  string gemm_string_identifier(const CBLAS_TRANSPOSE trans_A,
         const CBLAS_TRANSPOSE trans_B, const uint_tp M, const uint_tp N,
         const uint_tp K, bool alpha_term, bool beta_term,
         libdnnAccumulatePrecision_t prec,
         shared_ptr<Quantizer<MItype, MOtype> > quantizer);
  void initialize_gemm_tuner(shared_ptr<DeviceProgram> program,
         shared_ptr<LibDNNTuner> tuner);

  boost::shared_mutex program_mutex_;
  vector<bool> program_ready_;
  vector<shared_ptr<LibDNNTuner> > program_tuners_;
  vector<shared_ptr<DeviceProgram> > programs_;
  std::unordered_map<string, size_t> program_map_;
};


}  // namespace caffe


#endif  // USE_LIBDNN

#endif  // CAFFE_LIBDNN_LIBDNN_BLAS_HPP_
