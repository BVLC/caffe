
#ifndef MULTI_NODE_CPU_DISPATCHER_HPP_
#define MULTI_NODE_CPU_DISPATCHER_HPP_

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/multi_node/node_env.hpp"

namespace caffe {

/*
 * Statically generate affinity maps to different CPU and MKL threads
 */

class CPUDispatcher {
 public:
  CPUDispatcher() { }

  // statically assign threads to cpu cores
  void Dispatch(vector<vector<int> > *pthread_arr);

 protected:
  // parse CPU info from /proc/cpuinfo
  void ParseCPUInfo();

 protected:
  // number of physical sockets
  int num_sockets_;

  // number of online cores
  int num_cores_;

  // cores per socket
  int cores_per_socket_;

DISABLE_COPY_AND_ASSIGN(CPUDispatcher);
};

}  // end namespace caffe

#endif


