#ifndef CAFFE_LOCAL_CONV_LAYER_HPP_
#define CAFFE_LOCAL_CONV_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/base_conv_layer.hpp"

namespace caffe{

template <typename Dtype>
class LocalConvolutionLayer: public BaseConvolutionLayer<Dtype> {
public:
	/**
	local convolution[1]
	[1]http://www.cv-foundation.org/openaccess/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html
	Only support 2D convolution till now
	*/
	explicit LocalConvolutionLayer(const LayerParameter& param)
		: BaseConvolutionLayer<Dtype>(param) {}

	virtual inline const char* type() const { return "LocalConvolution"; }

protected:
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual inline bool reverse_dimensions() { return false; }
	virtual void compute_output_shape();
	void init_local_offset();
	void crop_loc_patch_cpu(const Dtype *src, int src_w, int src_h, int src_c, int crop_width, int crop_height, int w_off, int h_off, Dtype *local_patch_data);
	void crop_loc_patch_gpu(const Dtype *src, int src_w, int src_h, int src_c, int crop_width, int crop_height, int w_off, int h_off, Dtype *local_patch_data);
	void realign_loc_conv_result_cpu(const Dtype *local_conv_data, Dtype *dst_data);
	void realign_loc_conv_result_gpu(const Dtype *local_conv_data, Dtype *dst_data);
	void realign_bottom_diff_cpu(const Dtype *loc_bottom_diff_buffer, Dtype *bottom_diff);
	void realign_bottom_diff_gpu(const Dtype *loc_bottom_diff_buffer, Dtype *bottom_diff);

	float local_region_ratio_w_, local_region_ratio_h_;
	int local_region_num_w_, local_region_num_h_;
	int local_region_step_w_, local_region_step_h_;
	int bottom_width_, bottom_height_;
	int top_height_, top_width_;
	int L_;
	Blob<int> loc_idx_to_offset_; //Blob saving the map from local region index to local region offset
private:

	Blob<Dtype> loc_bottom_buffer_;
	Blob<Dtype> loc_top_buffer_;
};

}//namespace caffe
#endif