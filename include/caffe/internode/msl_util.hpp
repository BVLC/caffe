#ifdef CAFFE_MSL

#ifndef CAFFE_UTIL_MSL_UTIL_H_
#define CAFFE_UTIL_MSL_UTIL_H_

#include "msl.h"

namespace caffe {
namespace internode {

extern MSL::Distribution *data_parallelism;
extern MSL::Distribution *model_parallelism;
//extern MSL::Distribution *hybrid_parallelism;

void msl_init(int argc, char** argv);
void msl_finalize();
void msl_init_distributions();

}  // namespace internode
}  // namespace caffe

#endif   // CAFFE_UTIL_MSL_UTIL_H_

#endif // CAFFE_MSL
