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

