
#ifndef MULTI_NODE_PS_NODE_H_
#define MULTI_NODE_PS_NODE_H_

#include <string>

#include "caffe/multi_node/msg_hub.hpp"
#include "caffe/multi_node/ps_thread.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
class ParamServer : public MsgHub<Dtype> {
 public:
  explicit ParamServer(int nthreads);
  virtual ~ParamServer() { }

 public:
  virtual int Init();

  virtual int RouteMsg();

DISABLE_COPY_AND_ASSIGN(ParamServer);
};

}  // end namespace caffe

#endif


