/*!
 *  \brief     A helper class for {@link MultiStageMeanfieldLayer} class, which is the Caffe layer that implements the
 *             CRF-RNN described in the paper: Conditional Random Fields as Recurrent Neural Networks. IEEE ICCV 2015.
 *
 *             This class itself is not a proper Caffe layer although it behaves like one to some degree.
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

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/meanfield_iteration.hpp"

namespace caffe {

/**
 * To be invoked once only immediately after construction.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::OneTimeSetUp(
    Blob<Dtype>* const unary_terms,
    Blob<Dtype>* const softmax_input,
    Blob<Dtype>* const output_blob,
    const shared_ptr<ModifiedPermutohedral> spatial_lattice,
    const Blob<Dtype>* const spatial_norm) {

  spatial_lattice_ = spatial_lattice;
  spatial_norm_ = spatial_norm;

  count_ = unary_terms->count();
  num_ = unary_terms->num();
  channels_ = unary_terms->channels();
  height_ = unary_terms->height();
  width_ = unary_terms->width();
  num_pixels_ = height_ * width_;

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Meanfield iteration skipping parameter initialization.";
  } else {
    blobs_.resize(3);
    blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // spatial kernel weight
    blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // bilateral kernel weight
    blobs_[2].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // compatibility transform matrix
  }

  pairwise_.Reshape(num_, channels_, height_, width_);
  spatial_out_blob_.Reshape(num_, channels_, height_, width_);
  bilateral_out_blob_.Reshape(num_, channels_, height_, width_);
  message_passing_.Reshape(num_, channels_, height_, width_);

  // Softmax layer configuration
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(softmax_input);

  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);

  LayerParameter softmax_param;
  softmax_layer_.reset(new SoftmaxLayer<Dtype>(softmax_param));
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  // Sum layer configuration
  sum_bottom_vec_.clear();
  sum_bottom_vec_.push_back(unary_terms);
  sum_bottom_vec_.push_back(&pairwise_);

  sum_top_vec_.clear();
  sum_top_vec_.push_back(output_blob);

  LayerParameter sum_param;
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.));
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(-1.));
  sum_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  sum_layer_.reset(new EltwiseLayer<Dtype>(sum_param));
  sum_layer_->SetUp(sum_bottom_vec_, sum_top_vec_);
}

/**
 * To be invoked before every call to the Forward_cpu() method.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::PrePass(
    const vector<shared_ptr<Blob<Dtype> > >& parameters_to_copy_from,
    const vector<shared_ptr<ModifiedPermutohedral> >* const bilateral_lattices,
    const Blob<Dtype>* const bilateral_norms) {

  bilateral_lattices_ = bilateral_lattices;
  bilateral_norms_ = bilateral_norms;

  // Get copies of the up-to-date parameters.
  for (int i = 0; i < parameters_to_copy_from.size(); ++i) {
    blobs_[i]->CopyFrom(*(parameters_to_copy_from[i].get()));
  }
}

/**
 * Forward pass during the inference.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::Forward_cpu() {


  //------------------------------- Softmax normalization--------------------
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  //-----------------------------------Message passing-----------------------
  for (int n = 0; n < num_; ++n) {

    Dtype* spatial_out_data = spatial_out_blob_.mutable_cpu_data() + spatial_out_blob_.offset(n);
    const Dtype* prob_input_data = prob_.cpu_data() + prob_.offset(n);

    spatial_lattice_->compute(spatial_out_data, prob_input_data, channels_, false);

    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, spatial_norm_->cpu_data(),
          spatial_out_data + channel_id * num_pixels_,
          spatial_out_data + channel_id * num_pixels_);
    }

    Dtype* bilateral_out_data = bilateral_out_blob_.mutable_cpu_data() + bilateral_out_blob_.offset(n);

    (*bilateral_lattices_)[n]->compute(bilateral_out_data, prob_input_data, channels_, false);
    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, bilateral_norms_->cpu_data() + bilateral_norms_->offset(n),
          bilateral_out_data + channel_id * num_pixels_,
          bilateral_out_data + channel_id * num_pixels_);
    }
  }

  caffe_set(count_, Dtype(0.), message_passing_.mutable_cpu_data());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        this->blobs_[0]->cpu_data(), spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n), (Dtype) 0.,
        message_passing_.mutable_cpu_data() + message_passing_.offset(n));
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
        this->blobs_[1]->cpu_data(), bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n), (Dtype) 1.,
        message_passing_.mutable_cpu_data() + message_passing_.offset(n));
  }

  //--------------------------- Compatibility multiplication ----------------
  //Result from message passing needs to be multiplied with compatibility values.
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
        channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
        message_passing_.cpu_data() + message_passing_.offset(n), (Dtype) 0.,
        pairwise_.mutable_cpu_data() + pairwise_.offset(n));
  }

  //------------------------- Adding unaries, normalization is left to the next iteration --------------
  // Add unary
  sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
}


template<typename Dtype>
void MeanfieldIteration<Dtype>::Backward_cpu() {


  //---------------------------- Add unary gradient --------------------------
  vector<bool> eltwise_propagate_down(2, true);
  sum_layer_->Backward(sum_top_vec_, eltwise_propagate_down, sum_bottom_vec_);

  //---------------------------- Update compatibility diffs ------------------
  caffe_set(this->blobs_[2]->count(), Dtype(0.), this->blobs_[2]->mutable_cpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., pairwise_.cpu_diff() + pairwise_.offset(n),
                          message_passing_.cpu_data() + message_passing_.offset(n), (Dtype) 1.,
                          this->blobs_[2]->mutable_cpu_diff());
  }

  //-------------------------- Gradient after compatibility transform--- -----
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
                          channels_, (Dtype) 1., this->blobs_[2]->cpu_data(),
                          pairwise_.cpu_diff() + pairwise_.offset(n), (Dtype) 0.,
                          message_passing_.mutable_cpu_diff() + message_passing_.offset(n));
  }

  // ------------------------- Gradient w.r.t. kernels weights ------------
  caffe_set(this->blobs_[0]->count(), Dtype(0.), this->blobs_[0]->mutable_cpu_diff());
  caffe_set(this->blobs_[1]->count(), Dtype(0.), this->blobs_[1]->mutable_cpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., message_passing_.cpu_diff() + message_passing_.offset(n),
                          spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n), (Dtype) 1.,
                          this->blobs_[0]->mutable_cpu_diff());
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, (Dtype) 1., message_passing_.cpu_diff() + message_passing_.offset(n),
                          bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n), (Dtype) 1.,
                          this->blobs_[1]->mutable_cpu_diff());
  }

  /*Dtype* tmp = new Dtype[count_];
  caffe_mul<Dtype>(count_, message_passing_.cpu_diff(), spatial_out_blob_.cpu_data(), tmp);

  for (int c = 0; c < count_; ++c) {
    (this->blobs_[0]->mutable_cpu_diff())[0] += tmp[c];
  }

  caffe_mul<Dtype>(count_, message_passing_.cpu_diff(), bilateral_out_blob_.cpu_data(), tmp);
  for (int c = 0; c < count_; ++c) {
    (this->blobs_[1]->mutable_cpu_diff())[0] += tmp[c];
  }

  delete[] tmp;*/

  // TODO: Check whether there's a way to improve the accuracy of this calculation.
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                          this->blobs_[0]->cpu_data(), message_passing_.cpu_diff() + message_passing_.offset(n),
                          (Dtype) 0.,
                          spatial_out_blob_.mutable_cpu_diff() + spatial_out_blob_.offset(n));
  }
  //caffe_cpu_scale<Dtype>(count_, (this->blobs_[0]->cpu_data())[0],
  //    message_passing_.cpu_diff(), spatial_out_blob_.mutable_cpu_diff());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                          this->blobs_[1]->cpu_data(), message_passing_.cpu_diff() + message_passing_.offset(n),
                          (Dtype) 0.,
                          bilateral_out_blob_.mutable_cpu_diff() + bilateral_out_blob_.offset(n));
  }
  //caffe_cpu_scale<Dtype>(count_, (this->blobs_[1]->cpu_data())[0],
  //    message_passing_.cpu_diff(), bilateral_out_blob_.mutable_cpu_diff());


  //---------------------------- BP thru normalization --------------------------
  for (int n = 0; n < num_; ++n) {

    Dtype *spatial_out_diff = spatial_out_blob_.mutable_cpu_diff() + spatial_out_blob_.offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, spatial_norm_->cpu_data(),
                spatial_out_diff + channel_id * num_pixels_,
                spatial_out_diff + channel_id * num_pixels_);
    }

    Dtype *bilateral_out_diff = bilateral_out_blob_.mutable_cpu_diff() + bilateral_out_blob_.offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, bilateral_norms_->cpu_data() + bilateral_norms_->offset(n),
                bilateral_out_diff + channel_id * num_pixels_,
                bilateral_out_diff + channel_id * num_pixels_);
    }
  }

  //--------------------------- Gradient for message passing ---------------
  for (int n = 0; n < num_; ++n) {

    spatial_lattice_->compute(prob_.mutable_cpu_diff() + prob_.offset(n),
                              spatial_out_blob_.cpu_diff() + spatial_out_blob_.offset(n), channels_,
                              true, false);

    (*bilateral_lattices_)[n]->compute(prob_.mutable_cpu_diff() + prob_.offset(n),
                                       bilateral_out_blob_.cpu_diff() + bilateral_out_blob_.offset(n),
                                       channels_, true, true);
  }

  //--------------------------------------------------------------------------------
  vector<bool> propagate_down(2, true);
  softmax_layer_->Backward(softmax_top_vec_, propagate_down, softmax_bottom_vec_);
}

INSTANTIATE_CLASS(MeanfieldIteration);
}  // namespace caffe
