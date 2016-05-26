/*
 * device_context.hpp
 *
 *  Created on: Jun 26, 2015
 *      Author: Fabian Tschopp
 */

#ifndef CAFFE_device_HPP_
#define CAFFE_device_HPP_

#ifdef CMAKE_BUILD
#include "caffe_config.h"
#endif

#include <boost/shared_ptr.hpp>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/greentea/greentea.hpp"

using std::vector;

namespace caffe {

class device {
 public:
  explicit device();
  explicit device(int id, int list_id, Backend backend);
  Backend backend() const;
  int id() const;
  int list_id() const;
  int current_queue_id();
  int workgroup_size(int id);

#ifdef USE_GREENTEA
  viennacl::ocl::program &program();
  void SetProgram();
  bool is_host_unified();
#endif  // USE_GREENTEA

  template<typename Dtype>
  shared_ptr<Blob<Dtype> > Buffer(int id);

  int num_queues();
  void SwitchQueue(int id);
  void FinishQueues();

  void Init();

  uint_tp memory_usage();
  uint_tp peak_memory_usage();
  void IncreaseMemoryUsage(uint_tp bytes);
  void DecreaseMemoryUsage(uint_tp bytes);
  void ResetPeakMemoryUsage();
  bool CheckCapability(std::string cap);
  bool CheckVendor(std::string vendor);

 private:
  int current_queue_id_;
  std::vector<int> workgroup_sizes_;
  int id_;
  int list_id_;
  Backend backend_;
  uint_tp memory_usage_;
  uint_tp peak_memory_usage_;
  std::vector<shared_ptr<Blob<float> > > buff_f_;
  std::vector<shared_ptr<Blob<double> > > buff_d_;
  bool host_unified_;
#ifdef USE_GREENTEA
  viennacl::ocl::program ocl_program_;
#endif  // USE_GREENTEA
};
}  // namespace caffe

#endif /* CAFFE_device_HPP_ */
