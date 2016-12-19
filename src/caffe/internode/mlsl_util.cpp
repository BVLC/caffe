#ifdef USE_MLSL

#include <glog/logging.h>
#include <stdlib.h>
#include <cassert>
#include <stdexcept>
#include <string>
#include "caffe/internode/mlsl_util.hpp"

namespace caffe {
namespace internode {

MLSL::Distribution *data_parallelism;
MLSL::Distribution *model_parallelism;
//MLSL::Distribution *hybrid_parallelism;
bool isDistributionsInited = false;

void mlsl_init(int argc, char** argv) {
  LOG(INFO) << "MLSL init";
  MLSL::Init(&argc, &argv);
}

void mlsl_finalize() {
  LOG(INFO) << "MLSL finalize";

  if (isDistributionsInited) {
      isDistributionsInited = false;
      delete data_parallelism;
      delete model_parallelism;
      //delete hybrid_parallelism;
  }

  MLSL::Finalize();
}

void mlsl_init_distributions() {

    if (!isDistributionsInited) {
        isDistributionsInited = true;
        data_parallelism = new MLSL::Distribution(MLSL::GetNumNodes(), 1);
        model_parallelism = new MLSL::Distribution(1, MLSL::GetNumNodes());
    }
}

}  // namespace internode
}  // namespace caffe

#endif /* USE_MLSL */

