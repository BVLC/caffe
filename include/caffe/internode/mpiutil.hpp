#ifndef CAFFE_UTIL_MPIUTIL_H_
#define CAFFE_UTIL_MPIUTIL_H_

#include <map>
#include <string>

namespace caffe {
namespace internode {

int mpi_get_current_proc_rank();
std::string mpi_get_current_proc_rank_as_string();
int mpi_get_comm_size();
std::string mpi_get_current_proc_name();
std::string mpi_get_error_string(int errorcode);

void mpi_init(int argc, char** argv);
void mpi_finalize();

}  // namespace internode
}  // namespace caffe

#endif   // CAFFE_UTIL_MPIUTIL_H_

