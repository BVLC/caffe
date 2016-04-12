#ifndef CAFFE_MeanfieldIteration_LAYER_HPP_
#define CAFFE_MeanfieldIteration_LAYER_HPP_
#include <string>
#include <utility>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/split_layer.hpp"

#include "caffe/util/modified_permutohedral.hpp"
#include <boost/shared_array.hpp>

namespace caffe {
template <typename Dtype>
class MeanfieldIteration {

 public:
  /**
   * Must be invoked only once after the construction of the layer.
   */
  void OneTimeSetUp(
      Blob<Dtype>* const unary_terms,
      Blob<Dtype>* const softmax_input,
      Blob<Dtype>* const output_blob,
      const shared_ptr<ModifiedPermutohedral> spatial_lattice,
      const Blob<Dtype>* const spatial_norm);

  /**
   * Must be invoked before invoking {@link Forward_cpu()}
   */
  virtual void PrePass(
      const vector<shared_ptr<Blob<Dtype> > >&  parameters_to_copy_from,
      const vector<shared_ptr<ModifiedPermutohedral> >* const bilateral_lattices,
      const Blob<Dtype>* const bilateral_norms);

  /**
   * Forward pass - to be called during inference.
   */
  virtual void Forward_cpu();

  /**
   * Backward pass - to be called during training.
   */
  virtual void Backward_cpu();

  // A quick hack. This should be properly encapsulated.
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

 protected:
  vector<shared_ptr<Blob<Dtype> > > blobs_;

  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int num_pixels_;

  Blob<Dtype> spatial_out_blob_;
  Blob<Dtype> bilateral_out_blob_;
  Blob<Dtype> pairwise_;
  Blob<Dtype> softmax_input_;
  Blob<Dtype> prob_;
  Blob<Dtype> message_passing_;

  vector<Blob<Dtype>*> softmax_top_vec_;
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> sum_top_vec_;
  vector<Blob<Dtype>*> sum_bottom_vec_;

  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  shared_ptr<EltwiseLayer<Dtype> > sum_layer_;

  shared_ptr<ModifiedPermutohedral> spatial_lattice_;
  const vector<shared_ptr<ModifiedPermutohedral> >* bilateral_lattices_;

  const Blob<Dtype>* spatial_norm_;
  const Blob<Dtype>* bilateral_norms_;

};

}

#endif
