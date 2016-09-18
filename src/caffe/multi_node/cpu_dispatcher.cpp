
#include <string>
#include <vector>
#include "caffe/multi_node/cpu_dispatcher.hpp"


namespace caffe {

void CPUDispatcher::Dispatch(vector<vector<int> > *pthread_arr) {
  ParseCPUInfo();

#if (defined USE_MKL) && (defined _OPENMP)
  int num_threads = pthread_arr->size();
  int omp_threads = omp_get_max_threads();

  CHECK_LE(omp_threads * num_threads, num_cores_)
                << "too many threads to be scheduled";

  if (pthread_arr->size() <= 0) {
    LOG(ERROR) << "thread number cannot be 0";
  } else if (pthread_arr->size() == 1) {
    // use all the cores if only 1 thread
    for (int i = 0; i < omp_threads; i++) {
      pthread_arr->at(0).push_back(i);
    }
  } else {
    // evenly distribute the threads to each socket
    int thread_per_socket = num_threads / num_sockets_;
    CHECK_GE(cores_per_socket_, thread_per_socket * omp_threads)
          << "too many threads to be scheduled";

    vector<int> thrd_num_arr;
    thrd_num_arr.resize(num_sockets_);
    for (int i = 0; i < thrd_num_arr.size(); i++) {
      thrd_num_arr[i] = thread_per_socket;
    }
    int remain_threads = num_threads - thread_per_socket * num_sockets_;

    for (int i = 0; i < remain_threads; i++) {
      thrd_num_arr[i]++;
    }

    int thrd_idx = 0;
    for (int i = 0; i < num_sockets_; i++) {
      for (int j = 0; j < thrd_num_arr[i]; j++) {
        int core_idx = i * cores_per_socket_ + j * omp_threads;

        for (int k = 0; k < omp_threads; k++) {
          pthread_arr->at(thrd_idx).push_back(core_idx + k);
        }
        thrd_idx++;
      }
    }  // end for
  }

#else
  LOG(ERROR) << "cpu dispatcher only works when MKL and OMP are enabled";
#endif
}

void CPUDispatcher::ParseCPUInfo() {
  num_cores_ = NodeEnv::Instance()->GetOnlineCores();
  num_sockets_ = NodeEnv::Instance()->GetSockets();

  // TODO: parse from cpu info
  cores_per_socket_ = num_cores_ / num_sockets_;
}

}  // end namespace caffe


