/*!
 *  \brief     The Caffe layer that implements the CRF-RNN described in the paper:
 *             Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *  \authors   Sadeep Jayasumana, Bernardino Romera-Paredes, Shuai Zheng, Zhizhong Su.
 *  \version   1.0
 *  \date      2015
 *  \copyright Torr Vision Group, University of Oxford.
 *  \details   If you use this code, please consider citing the paper:
 *             Shuai Zheng, Sadeep Jayasumana, Bernardino Romera-Paredes, Vibhav Vineet, Zhizhong Su, Dalong Du,
 *             Chang Huang, Philip H. S. Torr. Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             For more information about CRF-RNN, please visit the project website http://crfasrnn.torr.vision.
 */
#include <vector>
#include <boost/lexical_cast.hpp>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/layers/meanfield_layers.hpp"
#include "caffe/util/tvg_common_utils.hpp"

namespace caffe {

/*
 * bottom[0] = Unary
 * bottom[1] = Unary
 * bottom[2] = RGB Image
 * top[0]    = Output. Ie, the final Q distribution
 */
template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  init_cpu_ = false;
  init_gpu_ = false;

  const caffe::MultiStageMeanfieldParameter meanfield_param = this->layer_param_.multi_stage_meanfield_param();

  num_iterations_ = meanfield_param.num_iterations();
  CHECK_GT(num_iterations_, 1) << "Number of iterations must be greater than 1.";

  theta_gamma_ = meanfield_param.theta_gamma();
  theta_alpha_ = meanfield_param.theta_alpha();
  theta_beta_ = meanfield_param.theta_beta();


  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  num_pixels_ = height_ * width_;

  LOG(INFO) << "This implementation has not been tested batch size > 1.";
  //CHECK_EQ(num_, 1) << "This implementation currently only supports batch size 1";

  top[0]->Reshape(num_, channels_, height_, width_);

  // Initialize the parameters that will updated by backpropagation.
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Multimeanfield layer skipping parameter initialization.";
  } else {
    init_param_blobs(meanfield_param);
  }

  init_spatial_lattice();

  // Allocate space for bilateral kernels. This is a temporary buffer used to compute bilateral lattices later.
  // Also allocate space for holding bilateral filter normalization values.
  /*bilateral_kernel_buffer_.reset(new float[5 * num_pixels_]);
  bilateral_norms_.Reshape(num_, 1, height_, width_);*/
  init_bilateral_buffers();

  // Configure the split layer that is used to make copies of the unary term. One copy for each iteration.
  // It may be possible to optimize this calculation later.
  split_layer_bottom_vec_.clear();
  split_layer_bottom_vec_.push_back(bottom[0]);

  split_layer_top_vec_.clear();

  split_layer_out_blobs_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; i++) {
    split_layer_out_blobs_[i].reset(new Blob<Dtype>());
    split_layer_top_vec_.push_back(split_layer_out_blobs_[i].get());
  }

  LayerParameter split_layer_param;
  split_layer_.reset(new SplitLayer<Dtype>(split_layer_param));
  split_layer_->SetUp(split_layer_bottom_vec_, split_layer_top_vec_);



  unary_prob_.Reshape(num_, channels_, height_, width_);


  // Make blobs store outputs of each meanfield iteration. Output of the last iteration is stored in top[0].
  // So we need only (num_iterations_ - 1) blobs.
  iteration_output_blobs_.resize(num_iterations_ - 1);
  for (int i = 0; i < num_iterations_ - 1; ++i) {
    iteration_output_blobs_[i].reset(new Blob<Dtype>(num_, channels_, height_, width_));
  }

  // Make instances of MeanfieldIteration and initialize them.
  meanfield_iterations_.resize(num_iterations_);
  for (int i = 0; i < num_iterations_; ++i) {
    meanfield_iterations_[i].reset(new MeanfieldIteration<Dtype>(this));
    meanfield_iterations_[i]->OneTimeSetUp(
        split_layer_out_blobs_[i].get(), // unary terms
        (i == 0) ? bottom[1] : iteration_output_blobs_[i - 1].get(), // softmax input
        (i == num_iterations_ - 1) ? top[0] : iteration_output_blobs_[i].get(), // output blob
        spatial_lattice_, // spatial lattice
        &spatial_norm_); // spatial normalization factors.
  }
  meanfield_iterations_[0]->is_first_iteration_ = true; // TODO: a nasty hack. Fix later.

  this->param_propagate_down_.resize(this->blobs_.size(), true);

  LOG(INFO) << ("MultiStageMeanfieldLayer initialized.");
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// Do nothing.
}

/**
 * Performs filter-based mean field inference given the image and unaries.
 *
 * bottom[0] - Unary terms
 * bottom[1] - Softmax input (a copy of the unary terms)
 * bottom[2] - RGB images
 *
 * top[0] - Output of the mean field inference (not normalized).
 */
template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {


  split_layer_bottom_vec_[0] = bottom[0];
  split_layer_->Forward(split_layer_bottom_vec_, split_layer_top_vec_);

  // Initialize the bilateral lattice.
  bilateral_lattices_.resize(num_);
  for (int n = 0; n < num_; ++n) {
  compute_bilateral_kernel(bottom[2], n, bilateral_kernel_buffer_); // only batch_size = 1 is supported
  bilateral_lattices_[n].reset(new ModifiedPermutohedral());
  bilateral_lattices_[n]->init(bilateral_kernel_buffer_, 5, width_, height_);

  // Calculate bilateral filter normalization factors.
  Dtype *norm_output_data = bilateral_norms_.mutable_cpu_data() + bilateral_norms_.offset(n);
  bilateral_lattices_[n]->compute_cpu(norm_output_data, norm_feed_, 1);
  for (int i = 0; i < num_pixels_; ++i) {
    norm_output_data[i] = 1.f / (norm_output_data[i] + 1e-20f);
  }
 }
  for (int i = 0; i < num_iterations_; ++i) {

    meanfield_iterations_[i]->PrePass(this->blobs_, &bilateral_lattices_, &bilateral_norms_);

    meanfield_iterations_[i]->Forward_cpu();
  }

}

/**
 * Backprop through filter-based mean field inference.
 */
template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {


  for (int i = (num_iterations_ - 1); i >= 0; --i) {
    meanfield_iterations_[i]->Backward_cpu();
  }

  const vector<bool> split_layer_propagate_down(1, true);
  split_layer_->Backward(split_layer_top_vec_, split_layer_propagate_down, split_layer_bottom_vec_);

  // Accumulate diffs from mean field iterations.
  for (int blob_id = 0; blob_id < this->blobs_.size(); ++blob_id) {

    Blob<Dtype>* cur_blob = this->blobs_[blob_id].get();

    if (this->param_propagate_down_[blob_id]) {

      caffe_set(cur_blob->count(), Dtype(0), cur_blob->mutable_cpu_diff());

      for (int i = 0; i < num_iterations_; ++i) {
        const Dtype* diffs_to_add = meanfield_iterations_[i]->blobs()[blob_id]->cpu_diff();
        caffe_axpy(cur_blob->count(), Dtype(1.), diffs_to_add, cur_blob->mutable_cpu_diff());
      }
    }
  }
}

// TODO: A GPU version of this kernel computation
template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n,
                                                               float* const output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[5 * p] = static_cast<float>(p % width_) / theta_alpha_;
    output_kernel[5 * p + 1] = static_cast<float>(p / width_) / theta_alpha_;

    const Dtype * const rgb_data_start = rgb_blob->cpu_data() + rgb_blob->offset(n);
    output_kernel[5 * p + 2] = static_cast<float>(rgb_data_start[p] / theta_beta_);
    output_kernel[5 * p + 3] = static_cast<float>((rgb_data_start + num_pixels_)[p] / theta_beta_);
    output_kernel[5 * p + 4] = static_cast<float>((rgb_data_start + num_pixels_ * 2)[p] / theta_beta_);
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::compute_spatial_kernel(float* const output_kernel) {

  for (int p = 0; p < num_pixels_; ++p) {
    output_kernel[2*p] = static_cast<float>(p % width_) / theta_gamma_;
    output_kernel[2*p + 1] = static_cast<float>(p / width_) / theta_gamma_;
  }
}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::init_param_blobs(const MultiStageMeanfieldParameter &meanfield_param) {

  // blobs_[0] - spatial kernel weights
  // blobs_[1] - bilateral kernel weights
  // blobs_[2] - compatability matrix
  this->blobs_.resize(3);

  // Allocate space for kernel weights.
  this->blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_));
  this->blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_));

  // Initialize the kernels weights.
  tvg::CommonUtils::read_into_the_diagonal(meanfield_param.spatial_filter_weights_str(), *(this->blobs_[0]));
  tvg::CommonUtils::read_into_the_diagonal(meanfield_param.bilateral_filter_weights_str(), *(this->blobs_[1]));

  // Initialize the compatibility matrix.
  this->blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_));
  caffe_set(channels_ * channels_, Dtype(0.), this->blobs_[2]->mutable_cpu_data());

  // Initialize it to have the Potts model.
  for (int c = 0; c < channels_; ++c) {
    (this->blobs_[2]->mutable_cpu_data())[c * channels_ + c] = Dtype(-1.);
  }

}

template <typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::init_spatial_lattice(void) {

  // This should be done on GPU if the GPU is available.
  // Right now, the spatial kernel is computed on CPU, then transferred over to the GPU
  float * spatial_kernel = new float[2 * num_pixels_];
  compute_spatial_kernel(spatial_kernel);
  bool force_cpu = this->layer_param().multi_stage_meanfield_param().force_cpu();
  spatial_lattice_.reset(new ModifiedPermutohedral());
  spatial_norm_.Reshape(1, 1, height_, width_);

  if ( Caffe::mode() == Caffe::CPU || force_cpu) {
    spatial_lattice_->init_cpu(spatial_kernel, 2, num_pixels_);

    // Compute normalisation factors
    norm_feed_ = new Dtype[num_pixels_];
    caffe_set(num_pixels_, Dtype(1.0), norm_feed_);
    Dtype* norm_data = spatial_norm_.mutable_cpu_data();
    spatial_lattice_->compute_cpu(norm_data, norm_feed_, 1);

    delete[] spatial_kernel;
    init_cpu_ = true;

  }else if( Caffe::mode() == Caffe::GPU){
    float* spatial_kernel_gpu;
    Dtype* norm_data_gpu;

    CUDA_CHECK( cudaMalloc( (void**)&spatial_kernel_gpu, 2*num_pixels_ * sizeof(float) ) );
    CUDA_CHECK( cudaMemcpy(spatial_kernel_gpu, spatial_kernel, 2*num_pixels_ * sizeof(float), cudaMemcpyHostToDevice ) );
    spatial_lattice_->init_gpu(spatial_kernel_gpu, 2, width_, height_);
    CUDA_CHECK( cudaFree(spatial_kernel_gpu) );

    CUDA_CHECK( cudaMalloc( (void**)&norm_feed_, num_pixels_ * sizeof(float)) );
    caffe_gpu_set(num_pixels_, Dtype(1.0), norm_feed_);
    norm_data_gpu = spatial_norm_.mutable_gpu_data();
    spatial_lattice_->compute_gpu(norm_data_gpu, norm_feed_, 1);

    init_gpu_ = true;
  }
  else{
    LOG(FATAL) << "Unknown Caffe mode. Neither CPU nor GPU";
  }

  Dtype* norm_data = spatial_norm_.mutable_cpu_data(); // This value has been computed either on the GPU or CPU. May be more efficient to just do everything on CPU.
  for (int i = 0; i < num_pixels_; ++i) {
    norm_data[i] = 1.0f / (norm_data[i] + 1e-20f);
  }
}

template<typename Dtype>
void MultiStageMeanfieldLayer<Dtype>::init_bilateral_buffers(void) {

  if (init_cpu_) {
    bilateral_kernel_buffer_ = new float[5 * num_pixels_];
  }
  else if (init_gpu_){
    CUDA_CHECK( cudaMalloc( (void**)&bilateral_kernel_buffer_, 5 * num_pixels_ * sizeof(float) ) );
  }
  else{
    LOG(FATAL) << "Should not have been able to get here";
  }
  bilateral_norms_.Reshape(num_, 1, height_, width_);

}

template<typename Dtype>
MultiStageMeanfieldLayer<Dtype>::~MultiStageMeanfieldLayer(){
  if(init_cpu_){
    delete[] bilateral_kernel_buffer_;
    delete[] norm_feed_;
  }
#ifndef CPU_ONLY
  if(init_gpu_){
    CUDA_CHECK(cudaFree(bilateral_kernel_buffer_));
    CUDA_CHECK(cudaFree(norm_feed_));
  }
#endif
}


INSTANTIATE_CLASS(MultiStageMeanfieldLayer);
REGISTER_LAYER_CLASS(MultiStageMeanfield);
}  // namespace caffe
