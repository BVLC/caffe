#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"
namespace caffe {

/**
 * @brief Compute triplet loss with multiple negative samples.
 */
template <typename Dtype>
  class TripletLossLayer : public LossLayer<Dtype> {
   public:
    explicit TripletLossLayer(const LayerParameter& param)
    : LossLayer<Dtype>(param), diff_() {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline const char* type() const { return "TripletLoss"; }
    virtual inline bool AllowForceBackward(const int bottom_index) const {
      return bottom_index != 1;
    }

   protected:
    /// @copydoc TripletLossLayer
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    /**
     * There are 3 types of triplet loss which are:
     * L_0(x_a, x_p, x_n) = max(0, m - || x_a - x_n ||_2^2 + || x_a - x_p ||_2^2)
     * used in FaceNet: A Unified Embedding for Face Recognition and Clustering
     * L_1(x_a, x_p, x_n) = max(0, 1 - || x_a - x_n ||_2^2 / (|| x_a - x_p ||_2^2 + m))
     * used in Learning Descriptors for Object Recognition and 3D Pose Estimation
     * L_2(x_a, x_p, x_n) = max(0, 1 - exp(|| x_a - x_n ||_2^2) / (exp(|| x_a - x_p ||_2^2) + m))
     * used in Learning Descriptors for Object Recognition and 3D Pose Estimation
     */

    /** One of the partial deriviation is:
     * \begin{eqnarray}
     * \mathcal{L}_{tri}(s_i,s_j,s_k) = max(0,1-\frac{||f(x_i)-f(x_k)||_2^2}{||f(x_i)-f(x_j)||_2^2+m})
     * \end{eqnarray}®
     * where $ f(x) $ is the input of the loss layer for sample $ x $ and m is the margin for triplet.
     * Denote that $D_{ij}=||f(x_i)-f(x_j)||_2^2$ and $D_{ik}=||f(x_i)-f(x_k)||_2^2$,
     * so the partial differential equations for the input of triplet loss layer are:
     * \begin{eqnarray}
     * \dfrac{\partial \mathcal{L}_{tri}}{\partial f(x_i)}=
     * &\frac{D_{ik}(f(x_i)-f(x_j))-(D_{ij}+m)(f(x_i)-f(x_k))}{(D_{ij}+m)^2} \nonumber \\
     * \dfrac{\partial \mathcal{L}_{tri}}{\partial f(x_j)}=
     * &\frac{D_{ik}(f(x_j)-f(x_i))}{(D_{ij}+m)^2} \nonumber \\
     * \dfrac{\partial \mathcal{L}_{tri}}{\partial f(x_k)}=
     * &\frac{f(x_i)-f(x_k)}{D_{ij}+m}
     * \end{eqnarray}®
     */
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom);
    Blob<Dtype> diff_;  // cached for backward pass
    Blob<Dtype> diff_pos;
    Blob<Dtype> diff_neg;
    Blob<Dtype> dist_sq_;  // cached for backward pass
    Blob<Dtype> dist_sq_pos;
    Blob<Dtype> dist_sq_neg;
    Blob<Dtype> diff_sq_;  // tmp storage for gpu forward pass
    Blob<Dtype> diff_sq_pos;
    Blob<Dtype> diff_sq_neg;
    Blob<Dtype> summer_vec_;  // tmp storage for gpu forward pass
  };

}  // namespace caffe

#endif  // CAFFE_TRIPLET_LOSS_LAYER_HPP_
