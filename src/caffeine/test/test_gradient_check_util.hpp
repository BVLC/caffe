#ifndef CAFFEINE_TEST_GRADIENT_CHECK_UTIL_H_
#define CAFFEINE_TEST_GRADIENT_CHECK_UTIL_H_

#include "caffeine/layer.hpp"

namespace caffeine {

// The gradient checker adds a L2 normalization loss function on top of the
// top blobs, and checks the gradient.
template <typename Dtype>
class GradientChecker {
 public:
  GradientChecker(const Dtype stepsize, const Dtype threshold,
      const unsigned int seed = 1701)
      : stepsize_(stepsize), threshold_(threshold), seed_(seed) {};
  // Checks the gradient of a layer, with provided bottom layers and top
  // layers. The gradient checker will check the gradient with respect to
  // the parameters of the layer, as well as the input blobs if check_through
  // is set True.
  // Note that after the gradient check, we do not guarantee that the data
  // stored in the layer parameters and the blobs.
  void CheckGradient(Layer<Dtype>& layer, vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>& top, int check_bottom = -1);
 protected:
  Dtype GetObjAndGradient(vector<Blob<Dtype>*>& top);
  Dtype stepsize_;
  Dtype threshold_;
  unsigned int seed_;
};

}  // namespace caffeine

#endif  // CAFFEINE_TEST_GRADIENT_CHECK_UTIL_H_