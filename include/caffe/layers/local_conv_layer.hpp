#pragma once

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class LocalConvolutionLayer : public BaseConvolutionLayer<Dtype> {
public:
  /**
  local convolution[1]
  [1]http://www.cv-foundation.org/openaccess/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html
  Only support 2D convolution till now
  */
  explicit LocalConvolutionLayer(const LayerParameter &param)
      : BaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char *type() const { return "LocalConvolution"; }

protected:
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  void Reshape(const vector<Blob<Dtype> *> &bottom,
               const vector<Blob<Dtype> *> &top);
  /*
  void Reshape_const(const vector<Blob<Dtype> *> &bottom,
               const vector<Blob<Dtype> *> &top) const override;
               */
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  /*
  void Forward_const_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top) const override;
                           */
  virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual inline bool reverse_dimensions() { return false; }
  virtual vector<int> compute_output_shape() const;
  void init_local_offset(int bottom_width, int bottom_height) const;
  void crop_loc_patch_cpu(const Dtype *src, int src_w, int src_h, int src_c,
                          int crop_width, int crop_height, int w_off, int h_off,
                          Dtype *local_patch_data);
  void crop_loc_patch_gpu(const Dtype *src, int src_w, int src_h, int src_c,
                          int crop_width, int crop_height, int w_off, int h_off,
                          Dtype *local_patch_data);
  void realign_loc_conv_result_cpu(const Dtype *local_conv_data,
                                   Dtype *dst_data);
  void realign_loc_conv_result_gpu(const Dtype *local_conv_data,
                                   Dtype *dst_data);

  float local_region_ratio_w_, local_region_ratio_h_;
  int local_region_num_w_, local_region_num_h_;
  int local_region_step_w_, local_region_step_h_;
  int L_;
  mutable ::boost::thread_specific_ptr<Blob<int>> loc_idx_to_offset_ptr_; // Blob saving the map from local region index
                              // to local region offset
private:
  mutable ::boost::thread_specific_ptr<Blob<Dtype>> loc_bottom_buffer_ptr_;
  mutable ::boost::thread_specific_ptr<Blob<Dtype>> loc_top_buffer_ptr_;
};

} // namespace caffe
