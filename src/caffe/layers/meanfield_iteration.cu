#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/meanfield_layers.hpp"

namespace caffe {

/**
 * Forward pass during the inference.
 */
    template <typename Dtype>
    void MeanfieldIteration<Dtype>::Forward_gpu() {


        //------------------------------- Softmax normalization--------------------
        softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

        //-----------------------------------Message passing-----------------------
        for (int n = 0; n < num_; ++n) {
        Dtype* spatial_out_data = spatial_out_blob_.mutable_gpu_data();
        const Dtype* prob_input_data = prob_.gpu_data();

        spatial_lattice_->compute_gpu(spatial_out_data, prob_input_data, channels_, false);

        // Pixel-wise normalization.
        for (int channel_id = 0; channel_id < channels_; ++channel_id) {
            caffe_gpu_mul(num_pixels_, spatial_norm_->gpu_data(),
                          spatial_out_data + channel_id * num_pixels_,
                          spatial_out_data + channel_id * num_pixels_);
        }

        Dtype* bilateral_out_data = bilateral_out_blob_.mutable_gpu_data();

        (*bilateral_lattices_)[n]->compute_gpu(bilateral_out_data, prob_input_data, channels_, false);
        // Pixel-wise normalization.
        for (int channel_id = 0; channel_id < channels_; ++channel_id) {
            caffe_gpu_mul(num_pixels_, bilateral_norms_->gpu_data(),
                          bilateral_out_data + channel_id * num_pixels_,
                          bilateral_out_data + channel_id * num_pixels_);
           }
        }

        caffe_gpu_set(count_, Dtype(0.), message_passing_.mutable_gpu_data());

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                              this->blobs_[0]->gpu_data(), spatial_out_blob_.gpu_data(), (Dtype) 0.,
                              message_passing_.mutable_gpu_data() );

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                              this->blobs_[1]->gpu_data(), bilateral_out_blob_.gpu_data(), (Dtype) 1.,
                              message_passing_.mutable_gpu_data() );

        //--------------------------- Compatibility multiplication ----------------
        //Result from message passing needs to be multiplied with compatibility values.
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_,
                              channels_, (Dtype) 1., this->blobs_[2]->gpu_data(),
                              message_passing_.gpu_data(), (Dtype) 0.,
                              pairwise_.mutable_gpu_data() );

        //------------------------- Adding unaries, normalization is left to the next iteration --------------
        // Add unary
        sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);
    }


    template<typename Dtype>
    void MeanfieldIteration<Dtype>::Backward_gpu() {


        //---------------------------- Add unary gradient --------------------------
        vector<bool> eltwise_propagate_down(2, true);
        sum_layer_->Backward(sum_top_vec_, eltwise_propagate_down, sum_bottom_vec_);

        //---------------------------- Update compatibility diffs ------------------
        caffe_gpu_set(this->blobs_[2]->count(), Dtype(0.), this->blobs_[2]->mutable_gpu_diff());

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                              num_pixels_, (Dtype) 1., pairwise_.gpu_diff(),
                              message_passing_.gpu_data(), (Dtype) 1.,
                              this->blobs_[2]->mutable_gpu_diff());


        //-------------------------- Gradient after compatibility transform--- -----
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_,
                              channels_, (Dtype) 1., this->blobs_[2]->gpu_data(),
                              pairwise_.gpu_diff(), (Dtype) 0.,
                              message_passing_.mutable_gpu_diff() );

        // ------------------------- Gradient w.r.t. kernels weights ------------
        caffe_gpu_set(this->blobs_[0]->count(), Dtype(0.), this->blobs_[0]->mutable_gpu_diff());
        caffe_gpu_set(this->blobs_[1]->count(), Dtype(0.), this->blobs_[1]->mutable_gpu_diff());

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                              num_pixels_, (Dtype) 1., message_passing_.gpu_diff(),
                              spatial_out_blob_.gpu_data(), (Dtype) 1.,
                              this->blobs_[0]->mutable_gpu_diff());

        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                              num_pixels_, (Dtype) 1., message_passing_.gpu_diff(),
                              bilateral_out_blob_.gpu_data(), (Dtype) 1.,
                              this->blobs_[1]->mutable_gpu_diff());

        // TODO: Check whether there's a way to improve the accuracy of this calculation.
        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                              this->blobs_[0]->gpu_data(), message_passing_.gpu_diff(),
                              (Dtype) 0.,
                              spatial_out_blob_.mutable_gpu_diff() );

        caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, (Dtype) 1.,
                              this->blobs_[1]->gpu_data(), message_passing_.gpu_diff(),
                              (Dtype) 0.,
                              bilateral_out_blob_.mutable_gpu_diff() );

        //---------------------------- BP thru normalization --------------------------
        Dtype *spatial_out_diff = spatial_out_blob_.mutable_gpu_diff();
        for (int channel_id = 0; channel_id < channels_; ++channel_id) {
            caffe_gpu_mul(num_pixels_, spatial_norm_->gpu_data(),
                          spatial_out_diff + channel_id * num_pixels_,
                          spatial_out_diff + channel_id * num_pixels_);
        }

        Dtype *bilateral_out_diff = bilateral_out_blob_.mutable_gpu_diff();
        for (int channel_id = 0; channel_id < channels_; ++channel_id) {
            caffe_gpu_mul(num_pixels_, bilateral_norms_->gpu_data(),
                          bilateral_out_diff + channel_id * num_pixels_,
                          bilateral_out_diff + channel_id * num_pixels_);
        }

        //--------------------------- Gradient for message passing ---------------
        for (int n = 0; n < num_; ++n) {
           spatial_lattice_->compute_gpu(prob_.mutable_gpu_diff(),
                                  spatial_out_blob_.gpu_diff(), channels_,
                                  true, false);

           (*bilateral_lattices_)[n]->compute_gpu(prob_.mutable_gpu_diff(),
                                           bilateral_out_blob_.gpu_diff(),
                                           channels_, true, true);
        }
        //--------------------------------------------------------------------------------
        vector<bool> propagate_down(2, true);
        softmax_layer_->Backward(softmax_top_vec_, propagate_down, softmax_bottom_vec_);
    }
// Instantiate class
    template void MeanfieldIteration<float>::Forward_gpu();
    template void MeanfieldIteration<double>::Forward_gpu();
    template void MeanfieldIteration<float>::Backward_gpu();
    template void MeanfieldIteration<double>::Backward_gpu();
}  // namespace caffe
