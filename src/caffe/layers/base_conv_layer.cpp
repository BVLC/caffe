#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

#ifdef FOR_SCNN_PAPER
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::WeightAlign(){
	//move nonzero weights to continuous memory space
	int slice_sum = 0;
	CHECK_EQ(forward_channel_group_sizes_.size(),group_);
	CHECK_EQ(forward_output_group_sizes_.size(),group_);
	for(int ii=0;ii<group_;ii++){
		slice_sum += forward_channel_group_sizes_[ii]*forward_output_group_sizes_[ii];
	}
	//weight_buffer_.Reshape(1,(this->blobs_[0]->shape(0)/group_) * forward_channels_.size(),this->blobs_[0]->shape(2),this->blobs_[0]->shape(3));
	weight_buffer_.Reshape(1,slice_sum,this->blobs_[0]->shape(2),this->blobs_[0]->shape(3));
	int slice_size = weight_buffer_.shape(2)*weight_buffer_.shape(3);
	int group_output_size = num_output_/group_;
	int group_channel_size = channels_/group_;
	int slice_index = 0;
	for(int n=0;n<this->blobs_[0]->shape(0);n++){
		for(int c=0;c<this->blobs_[0]->shape(1);c++){
			int forward_c = c + (n/group_output_size)*group_channel_size;
			CHECK_LT(forward_c,channels_)
					 << "Input channel index is out of boundary";
			if(forwarding_channel_mask_[forward_c] && forwarding_output_mask_[n]){
				int src_offset = this->blobs_[0]->offset(n,c,0,0);
				CHECK_LT(slice_index,slice_sum)
					<<"Weight slice is out of boundary";
				int dst_offset = weight_buffer_.offset(0,slice_index,0,0);
				caffe_copy(slice_size,this->blobs_[0]->cpu_data()+src_offset,weight_buffer_.mutable_cpu_data()+dst_offset);
				slice_index++;
			}
		}
	}
	CHECK_EQ(slice_index,slice_sum)
		<< "Not enough weight slices are stored";
	//this->blobs_[0]->Snapshot(Layer<Dtype>::layer_param().name()+".blob");
}
#endif

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  CHECK(!conv_param.has_kernel_size() !=
      !(conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(conv_param.has_kernel_size() ||
      (conv_param.has_kernel_h() && conv_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!conv_param.has_pad() && conv_param.has_pad_h()
      && conv_param.has_pad_w())
      || (!conv_param.has_pad_h() && !conv_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!conv_param.has_stride() && conv_param.has_stride_h()
      && conv_param.has_stride_w())
      || (!conv_param.has_stride_h() && !conv_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (conv_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = conv_param.kernel_size();
  } else {
    kernel_h_ = conv_param.kernel_h();
    kernel_w_ = conv_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!conv_param.has_pad_h()) {
    pad_h_ = pad_w_ = conv_param.pad();
  } else {
    pad_h_ = conv_param.pad_h();
    pad_w_ = conv_param.pad_w();
  }
  if (!conv_param.has_stride_h()) {
    stride_h_ = stride_w_ = conv_param.stride();
  } else {
    stride_h_ = conv_param.stride_h();
    stride_w_ = conv_param.stride_w();
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = kernel_w_ == 1 && kernel_h_ == 1
      && stride_h_ == 1 && stride_w_ == 1 && pad_h_ == 0 && pad_w_ == 0;
  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(
        conv_out_channels_, conv_in_channels_ / group_, kernel_h_, kernel_w_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
#ifdef FOR_SCNN_PAPER
  //For partial forwarding
  CHECK(!reverse_dimensions()
		  || ((conv_param.forward_channels_size()==channels_ || conv_param.forward_channels_size()==0)
		  && (conv_param.forward_outputs_size()==0 || conv_param.forward_outputs_size()==num_output_)))
  			  << "Partial forwarding is unsupported for deconv";
  if(!conv_param.forward_channels_size()){
	  for(int ii=0;ii<channels_;ii++){
		  forward_channels_.push_back(ii);
	  }
  }else{
	  for(int ii=0;ii<conv_param.forward_channels_size();ii++){
	  	  	forward_channels_.push_back(conv_param.forward_channels(ii));
	  }
  }
  forwarding_channel_mask_ = vector<int> (channels_,0);
  for(int ii=0;ii<forward_channels_.size();ii++){
	  forwarding_channel_mask_[forward_channels_[ii]]=1;
  }


  if(!conv_param.forward_outputs_size()){
  	  for(int ii=0;ii<num_output_;ii++){
  	  	  forward_outputs_.push_back(ii);
  	  }
  }else{
	  for(int ii=0;ii<conv_param.forward_outputs_size();ii++){
	  	  forward_outputs_.push_back(conv_param.forward_outputs(ii));
	  }
  }
  forwarding_output_mask_ = vector<int> (num_output_,0);
    for(int ii=0;ii<forward_outputs_.size();ii++){
    	forwarding_output_mask_[forward_outputs_[ii]]=1;
  }

  is_skip_channels_ = forward_channels_.size()<channels_;

  forward_channel_group_sizes_ = vector<int> (group_,0);
  int group_size = channels_/group_;
  for(int forward_c=0;forward_c<forward_channels_.size();forward_c++){
	  int c = forward_channels_[forward_c];
	  CHECK_LT(c,channels_)
  		  << "Input channel index is out of boundary";
	  forward_channel_group_sizes_[c/group_size]++;
  }
  group_size = num_output_/group_;//change it
  forward_output_group_sizes_ = vector<int> (group_,0);
  for(int forward_n=0;forward_n<forward_outputs_.size();forward_n++){
  	  int n = forward_outputs_[forward_n];
  	  CHECK_LT(n,num_output_)
    		  << "output map index is out of boundary";
  	forward_output_group_sizes_[n/group_size]++;
  }

  is_scnn_ = is_skip_channels_ || forward_outputs_.size()<num_output_;
#endif
  is_sparse_feature_maps_ = false;
  dense_feature_map_mask_.Reshape(1,1,1,channels_);
  squeezed_weight_buffer_.Reshape(this->blobs_[0]->shape(0)/group_,this->blobs_[0]->shape(1),this->blobs_[0]->shape(2),this->blobs_[0]->shape(3));
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  compute_output_shape();
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_out_, width_out_);
  }
  if (reverse_dimensions()) {
    conv_in_height_ = height_out_;
    conv_in_width_ = width_out_;
    conv_out_spatial_dim_ = height_ * width_;
  } else {
    conv_in_height_ = height_;
    conv_in_width_ = width_;
    conv_out_spatial_dim_ = height_out_ * width_out_;
  }
  kernel_dim_ = conv_in_channels_ * kernel_h_ * kernel_w_;
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  if (reverse_dimensions()) {
    col_buffer_.Reshape(1, kernel_dim_, height_, width_);
  } else {
#ifndef FOR_SCNN_PAPER
    col_buffer_.Reshape(1, kernel_dim_, height_out_, width_out_);
#else
	int left_kernel_dim = forward_channels_.size() * kernel_h_ * kernel_w_;
	CHECK_LE(left_kernel_dim,kernel_dim_)
		<< "Convoluted channels must not be more than all channels";
	col_buffer_.Reshape(1, left_kernel_dim, height_out_, width_out_);
#endif
	col_buf_mask_.Reshape(1,1,1,kernel_dim_);

#ifdef	GPU_USE_CUSPARSE
	nonzero_elements_buffer_.Reshape(1, 1, 1, col_buffer_.count());//WARNING: real sparse matrix needs many less memory
	nonzero_indices_buffer_.Reshape(1,1,1,nonzero_elements_buffer_.count());
	index_pointers_buffer_.Reshape(1,1,1,col_buffer_.shape(1)+1);
	nonzero_per_rowcol_buffer_.Reshape(1,1,1,col_buffer_.shape(1));
#endif
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, height_out_ * width_out_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  // WARNING WARNING WARNING, DOESN'T WORK FOR SCNN PAPER ANYMORE
  // WARNING WARNING WARNING, DOESN'T WORK FOR SCNN PAPER ANYMORE
  // WARNING WARNING WARNING, DOESN'T WORK FOR SCNN PAPER ANYMORE
  if (!is_1x1_ ||  is_sparse_feature_maps_) {
    if (!skip_im2col) {
      //conv_im2col_cpu(input, col_buffer_.mutable_cpu_data(),col_buf_mask_.mutable_cpu_data(), dense_feature_map_mask_.mutable_cpu_data());
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
  int masked_map_num = 0;
  for(int map_idx=0; map_idx<conv_in_channels_;++map_idx){
	  if(dense_feature_map_mask_.cpu_data()[map_idx]){
		  masked_map_num++;
	  }
  }
  //Dtype sparsity = (Dtype)1.0 - (Dtype)masked_map_num/(Dtype)conv_in_channels_;
  //LOG(INFO)<<Layer<Dtype>::layer_param().name()<<" sparsity: "<<sparsity;
  /*
  col_buffer_.Snapshot(Layer<Dtype>::layer_param().name()+".blob");
  LOG(INFO)<<Layer<Dtype>::layer_param().name();
  for(int tmp_i=0;tmp_i<dense_feature_map_mask_.count();tmp_i++){
  	  LOG(INFO) << tmp_i<<"\t"<<dense_feature_map_mask_.cpu_data()[tmp_i];
  }
  for(int tmp_i=0;tmp_i<col_buf_mask_.count();tmp_i++){
    	  LOG(INFO) << tmp_i<<"\t"<<col_buf_mask_.cpu_data()[tmp_i];
  }*/

#ifdef FOR_SCNN_PAPER
  int cur_weight_offset = 0;
  int cur_col_offset = 0;
  int cur_output_offset = 0;
#endif
  int offset_sum = 0;
  for (int g = 0; g < group_; ++g) {
#ifdef FOR_SCNN_PAPER
	  if(!is_scnn_){
#endif
		  //if(sparsity>-1){
			  int left_cols = 0;
			  caffe_cpu_del_zero_cols(conv_out_channels_ /group_,
					  kernel_dim_ / group_,
					  weights + weight_offset_ * g,
					  squeezed_weight_buffer_.mutable_cpu_data(),
					  &left_cols,
					  col_buf_mask_.cpu_data() + kernel_dim_ / group_ * g );
			  //assert(left_cols<=kernel_dim_ / group_);
			  caffe_cpu_cblas_gemm(conv_out_channels_ /
					  group_, conv_out_spatial_dim_, left_cols,
					  (Dtype)1., squeezed_weight_buffer_.cpu_data(),
					  kernel_dim_ / group_, col_buff + offset_sum,
					conv_out_spatial_dim_, (Dtype)0., output + output_offset_ * g, conv_out_spatial_dim_);

			  offset_sum += left_cols * conv_out_spatial_dim_;
		  //}else{
		//		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
		//				  group_, conv_out_spatial_dim_, kernel_dim_ / group_,
		//				  (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
		//				  (Dtype)0., output + output_offset_ * g);
		//  }
#ifdef FOR_SCNN_PAPER
	  }else{
		  //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
		  //    group_, conv_out_spatial_dim_, forward_channel_group_sizes_[g] * kernel_h_ * kernel_w_,
		  //    (Dtype)1., weight_buffer_.cpu_data() + cur_weight_offset, col_buff + cur_col_offset,
		  //    (Dtype)0., output + output_offset_ * g);
		  //cur_weight_offset += (num_output_ / group_) * forward_channel_group_sizes_[g] * kernel_h_ * kernel_w_;
		  //cur_col_offset += conv_out_spatial_dim_ * forward_channel_group_sizes_[g] * kernel_h_ * kernel_w_;
		  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, forward_output_group_sizes_[g],
			  conv_out_spatial_dim_, forward_channel_group_sizes_[g] * kernel_h_ * kernel_w_,
	  	      (Dtype)1., weight_buffer_.cpu_data() + cur_weight_offset, col_buff + cur_col_offset,
	  	      (Dtype)0., output + cur_output_offset);
		  cur_weight_offset += forward_output_group_sizes_[g] * forward_channel_group_sizes_[g] * kernel_h_ * kernel_w_;
		  cur_col_offset += conv_out_spatial_dim_ * forward_channel_group_sizes_[g] * kernel_h_ * kernel_w_;
		  cur_output_offset += forward_output_group_sizes_[g] * conv_out_spatial_dim_;
	  }
#endif
  }

}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_cpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_cpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.cpu_data(), 1., bias);
}

#ifndef CPU_ONLY

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }

  for (int g = 0; g < group_; ++g) {
#ifdef	GPU_USE_CUSPARSE
	  int total_nonzero = 0;
	  caffe_gpu_sparse_dense2csr(kernel_dim_ / group_, conv_out_spatial_dim_,
						  col_buff + col_offset_ * g,
						  nonzero_per_rowcol_buffer_.mutable_gpu_data(),
						  nonzero_elements_buffer_.mutable_gpu_data(),
						  index_pointers_buffer_.mutable_gpu_data(),
						  nonzero_indices_buffer_.mutable_gpu_data(), &total_nonzero);
	  Dtype sparsity = (Dtype)1.0 - (Dtype)total_nonzero/(Dtype)(kernel_dim_*height_out_*width_out_);
	  //LOG(INFO)<<"Sparsity of "<< Layer<Dtype>::layer_param().name() << ": "<< sparsity;
	  if(sparsity<(Dtype)0.9){
#endif
		 caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
			 group_, conv_out_spatial_dim_, kernel_dim_ / group_,
			 (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
			 (Dtype)0., output + output_offset_ * g);
#ifdef	GPU_USE_CUSPARSE
	  }else{
		 //dense weight matrix multi. sparse feature map matrix
		 //WARNING WARNING WARNING: When A*B, B in format CSR is slow
		 caffe_gpu_sparse_mmcsr(conv_out_channels_ /group_, conv_out_spatial_dim_, kernel_dim_ / group_,
				  (Dtype)1., weights + weight_offset_ * g,
				  total_nonzero,
				  nonzero_elements_buffer_.gpu_data(),
				  index_pointers_buffer_.gpu_data(),
				  nonzero_indices_buffer_.gpu_data(),
				  (Dtype)0., output + output_offset_ * g);
	  }
#endif
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  if (is_1x1_) {
    col_buff = input;
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output + output_offset_ * g,
        (Dtype)0., col_buff + col_offset_ * g);
  }
  if (!is_1x1_) {
    conv_col2im_gpu(col_buff, input);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_,
        (Dtype)1., output + output_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(BaseConvolutionLayer);

}  // namespace caffe
