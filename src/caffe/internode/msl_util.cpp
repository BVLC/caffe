#ifdef CAFFE_MSL

#include <glog/logging.h>
#include <stdlib.h>
#include <cassert>
#include <stdexcept>
#include <string>
#include "caffe/internode/msl_util.hpp"

namespace caffe {
namespace internode {

MSL::Distribution *data_parallelism;
MSL::Distribution *model_parallelism;
//MSL::Distribution *hybrid_parallelism;
bool isDistributionsInited = false;

void msl_init(int argc, char** argv) {
  LOG(INFO) << "MSL init";
  MSL::Init(&argc, &argv);
}

void msl_finalize() {
  LOG(INFO) << "MSL finalize";

  if (isDistributionsInited) {
      isDistributionsInited = false;
      delete data_parallelism;
      delete model_parallelism;
      //delete hybrid_parallelism;
  }

  MSL::Finalize();
}

void msl_init_distributions() {

    if (!isDistributionsInited) {
        isDistributionsInited = true;
        const int num_groups = 2;
        data_parallelism = new MSL::Distribution(MSL::GetNumNodes(), 1);
        model_parallelism = new MSL::Distribution(1, MSL::GetNumNodes());
        //hybrid_parallelism = new MSL::Distribution(MSL::GetNumNodes()/num_groups, num_groups);
    }
}

}  // namespace internode
}  // namespace caffe

#endif /* CAFFE_MSL */

