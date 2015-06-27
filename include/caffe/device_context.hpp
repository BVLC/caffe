/*
 * device_context.hpp
 *
 *  Created on: Jun 26, 2015
 *      Author: Fabian Tschopp
 */

#ifndef CAFFE_DEVICE_CONTEXT_HPP_
#define CAFFE_DEVICE_CONTEXT_HPP_

#include <boost/shared_ptr.hpp>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/greentea/greentea.hpp"


using std::vector;

namespace caffe {

class DeviceContext {
 public:
  explicit DeviceContext();
  explicit DeviceContext(int id, Backend backend);
  Backend backend() const;
  int id() const;
  int current_queue_id();
  int WorkgroupSize(int id);

  template<typename Dtype>
  shared_ptr< Blob<Dtype> > Buffer(int id);

  int num_queues();
  void SwitchQueue(int id);
  void FinishQueues();

  void Init();
 private:
  int current_queue_id_;
  std::vector<int> workgroup_sizes_;
  int id_;
  Backend backend_;
  std::vector< shared_ptr< Blob<float> > > buff_f_;
  std::vector< shared_ptr< Blob<double> > > buff_d_;
};
}  // namespace caffe

#endif /* CAFFE_DEVICE_CONTEXT_HPP_ */
