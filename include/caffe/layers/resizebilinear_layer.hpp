#ifndef CAFFE_RESIZEBILINEAR_LAYER_HPP_
#define CAFFE_RESIZEBILINEAR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
template <typename Dtype>
class ResizeBilinearLayer : public Layer<Dtype> {
  public:
	explicit ResizeBilinearLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
  	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      	const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "ResizeBilinear"; }
	virtual inline int ExactNumBottomBlobs() const { return 1; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int input_height_, input_width_;	
	int output_height_,output_width_;
	int num_, channels_;
	int factor_;
	
};

} // namespace caffe

#endif // CAFFE_BILINEAR_LAYER_HPP
