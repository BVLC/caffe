#include "caffe/solver_factory.hpp"

namespace caffe {

// Hide the implementation of SolverRegistry::CreatorRegistry singleton
template <typename Dtype>
typename SolverRegistry<Dtype>::CreatorRegistry&
SolverRegistry<Dtype>::Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

INSTANTIATE_CLASS(SolverRegistry);

}  // namespace caffe
