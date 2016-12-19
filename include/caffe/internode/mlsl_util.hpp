#ifdef USE_MLSL

#ifndef CAFFE_UTIL_MLSL_UTIL_H_
#define CAFFE_UTIL_MLSL_UTIL_H_

#include "mlsl.h"

namespace caffe {
namespace internode {

extern MLSL::Distribution *data_parallelism;
extern MLSL::Distribution *model_parallelism;
//extern MLSL::Distribution *hybrid_parallelism;

void mlsl_init(int argc, char** argv);
void mlsl_finalize();
void mlsl_init_distributions();

}  // namespace internode
}  // namespace caffe

#endif   // CAFFE_UTIL_MLSL_UTIL_H_

#endif // USE_MLSL
