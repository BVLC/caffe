#ifndef CAFFE_LIBDNN_LIBDNN_HPP_
#define CAFFE_LIBDNN_LIBDNN_HPP_

#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/backend/device.hpp"
#include "caffe/libdnn/libdnn_tuner.hpp"

namespace caffe {

class LibDNNBase {
 protected:
  LibDNNBase(Device* dev_ptr);

  Device* dev_ptr_;
  shared_ptr<DeviceProgram> program_;
  bool fast_unsafe_math_;
};

template<typename MItype, typename MOtype>
class LibDNN : public LibDNNBase {
 protected:
  LibDNN(Device* dev_ptr);
  string generate_accreg_init(shared_ptr<LibDNNTuner> tuner, bool dterm,
                              bool load, bool beta_term, bool beta_exactly_one);
  string generate_gemm_core(shared_ptr<LibDNNTuner> tuner, bool dterm,
                            bool alpha_term, bool alpha_exactly_one);

  std::map<string, int64_t> gemm_like_default_parameters();

};

}  // namespace caffe

#endif /* CAFFE_GREENTEA_LIBDNN_HPP_ */
