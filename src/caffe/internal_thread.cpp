/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/cpu_info.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

InternalThread::~InternalThread() {
  StopInternalThread();
}

bool InternalThread::is_started() const {
  return thread_ && thread_->joinable();
}

bool InternalThread::must_stop() {
  return thread_ && thread_->interruption_requested();
}

void InternalThread::StartInternalThread() {
  CHECK(!is_started()) << "Threads should persist and not be restarted.";

  int device = 0;
#ifndef CPU_ONLY
  CUDA_CHECK(cudaGetDevice(&device));
#endif
  Caffe::Brew mode = Caffe::mode();
  int rand_seed = caffe_rng_rand();
  int solver_count = Caffe::solver_count();
  bool root_solver = Caffe::root_solver();

  try {
    thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
          rand_seed, solver_count, root_solver));
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
}

void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, bool root_solver) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device));
#endif
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  Caffe::set_root_solver(root_solver);

#ifdef _OPENMP
  caffe::cpu::OpenMpManager::bindCurrentThreadToNonPrimaryCoreIfPossible();
#endif

  SetThreadAffinity();
  
  InternalThreadEntry();
}

void InternalThread::SetThreadAffinity() {
#define MAX_CORES 64

  static int count = 0;
  static int ncores = 0;
  static int affinity_cores[MAX_CORES];
  static boost::mutex internal_thread_mutex;

  boost::mutex::scoped_lock lock(internal_thread_mutex);
  if (count == 0) {
    char * pin_cores = getenv("INTERNAL_THREADS_PIN");
    if (pin_cores != NULL) {
      char * token = strtok(pin_cores, ",");
      while (token != NULL) {
        affinity_cores[ncores] = atoi(token);
        token = strtok(NULL, ",");
        ncores++;
        if (ncores >= MAX_CORES) {
          LOG(INFO) << "Too many cores used for internal threads. Just take first " << ncores << " cores.";
          break;
        }
      }
    }
  }

  if (ncores > 0) {
    int pin_core_id = count % ncores;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(affinity_cores[pin_core_id], &set);
    pthread_t thread = pthread_self();
    int s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &set);
    if (s != 0) {
      LOG(WARNING) << "Cannot set affinity for internal thread!";
    }
    for (int j=0; j<CPU_SETSIZE; j++) {
      if (CPU_ISSET(j, &set)) {
        LOG(INFO) << "Internal thread is affinitized to core " << j;
      }
    }
  }
  count++;
}


void InternalThread::StopInternalThread() {
  if (is_started()) {
    thread_->interrupt();
    try {
      thread_->join();
    } catch (boost::thread_interrupted&) {
    } catch (std::exception& e) {
      LOG(FATAL) << "Thread exception: " << e.what();
    }
  }
}

}  // namespace caffe
