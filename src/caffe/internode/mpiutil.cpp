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

#include <glog/logging.h>
#include <stdlib.h>
#include <cassert>
#include <stdexcept>
#include <string>
#include "caffe/internode/mpiutil.hpp"
#ifdef USE_MPI
#include <mpi.h>
#endif

namespace caffe {
namespace internode {

int init_count = 0;

/**
 * Returns current process rank.
 * Default is MPI_COMM_WORLD communicator.
 */
int mpi_get_current_proc_rank() {
#ifdef USE_MPI

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  return rank;
#else
  throw std::runtime_error("can't use mpi");
  return 0;
#endif
}

/**
 * Returns current process rank as string, ex: [1].
 * Default is MPI_COMM_WORLD communicator.
 */
std::string mpi_get_current_proc_rank_as_string() {
#ifdef USE_MPI

  std::ostringstream sstream;
  sstream << "[" << mpi_get_current_proc_rank() << "]";

  return sstream.str();
#else
  throw std::runtime_error("can't use mpi");
  return "";
#endif
}

/**
 * Returns number of processes in the group of communicator.
 * Default is MPI_COMM_WORLD communicator.
 */
int mpi_get_comm_size() {
#ifdef USE_MPI

  int size = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &size);

  return size;
#else
  throw std::runtime_error("can't use mpi");
  return 0;
#endif
}

/**
 * Returns processor name.
 */
std::string mpi_get_current_proc_name() {
#ifdef USE_MPI

  char name[MPI_MAX_PROCESSOR_NAME];
  int size = 0;

  name[0] = 0;
  if (MPI_Get_processor_name(name, &size) != MPI_SUCCESS) return "";

  return std::string(name, size);
#else
  throw std::runtime_error("can't use mpi");
  return 0;
#endif
}

/**
 * Return a string for a given error code.
 */
std::string mpi_get_error_string(int errorcode) {
#ifdef USE_MPI
  char errbuf[MPI_MAX_ERROR_STRING];
  int errbuflen = 0;

  MPI_Error_string(errorcode, errbuf, &errbuflen);

  return std::string(errbuf, errbuflen);
#else
  throw std::runtime_error("can't use mpi");
  return 0;
#endif
}

void mpi_init(int argc, char** argv) {
#ifdef USE_MPI
  int rank = 0, size = 0, namelen = 0;
  char name[MPI_MAX_PROCESSOR_NAME];

  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  assert(provided == MPI_THREAD_FUNNELED);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Get_processor_name(name, &namelen);

  LOG(INFO) << "Process rank " << rank << " from number of " << size
            << " processes running on " << name;
#endif
}

void mpi_finalize() {
#ifdef USE_MPI
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  LOG(INFO) << "Process rank " << rank << " exitted";
  MPI_Finalize();
#endif
}

}  // namespace internode
}  // namespace caffe

