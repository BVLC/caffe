#ifndef CAFFE_XXX_LAYER_HPP_
#define CAFFE_XXX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Spatial and chromatic data augmentation
 */

template <typename Dtype>
class DataAugmentationLayer : public Layer<Dtype> {
 public:
  explicit DataAugmentationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual ~DataAugmentationLayer() {};
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)  { LOG(FATAL) << "DataAugmentationLayer does not do backward"; }

      
  virtual void generate_spatial_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff = 1);
  virtual void generate_chromatic_coeffs(const AugmentationParameter& aug, AugmentationCoeff& coeff, Dtype discount_coeff = 1);
  virtual void clear_spatial_coeffs(AugmentationCoeff& coeff);
  virtual void clear_defaults(AugmentationCoeff& coeff);
  virtual void coeff_to_array(const AugmentationCoeff& coeff, Dtype* out);
  virtual void array_to_coeff(Dtype* in, AugmentationCoeff& coeff);

  int cropped_height_;
  int cropped_width_;
  bool output_params_;
  bool input_params_;
  int num_params_;
  Blob<Dtype> data_mean_;
  int num_iter_;
  AugmentationParameter aug_;
  CoeffScheduleParameter discount_coeff_schedule_;
};

}  // namespace caffe

#endif  // CAFFE_XXX_LAYER_HPP_


