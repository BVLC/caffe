#ifndef CAFFE_HEATMAPS_FROM_VEC_LAYER_HPP_
#define CAFFE_HEATMAPS_FROM_VEC_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Projects 3D positions onto the image plane of a virtual camera and creates heatmaps.
 */
template <typename Dtype>
class HeatmapsFromVecLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides options:
   *  - num_iter. The number of IK iterations to best fit the 3D vector to the input heatmaps.
   */
	 explicit HeatmapsFromVecLayer(const LayerParameter& param) 
		 : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "HeatmapsFromVec"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  float fx_; // focal length x
  float fy_; // focal length y
  float ux_; // principal point x
  float uy_; // principal point y
  int num_vecs_; // number of 3D vectors to be transformed to heatmaps
  int heatmap_size_; // resolution of heatmaps (always square)
  int kernel_size_; // size of Gaussian kernel to put is 2*kernel_size_+1 
  float range_; // range -val to val corresponds to heatmap width and height
  std::vector<float> gaussian_; // Gaussian kernel to be put around every projected 2D location (linearized)
  std::vector<int> proj_vecs_; // save projection of 3D vecs for gradient computation
  float gradient_fact_; // constant factor in the Jacobi matrix of projection (for orthographic camera)
};

}  // namespace caffe

#endif  // CAFFE_HEATMAPS_FROM_VEC_LAYER_HPP_
