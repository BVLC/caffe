#include <algorithm>
#include <vector>

#include "caffe/layers/lifted_struct_similarity_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LiftedStructSimilaritySoftmaxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  // List of member variables defined in /include/caffe/loss_layers.hpp;
  //   diff_, dist_sq_, summer_vec_, loss_aug_inference_;
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  dot_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
  ones_.Reshape(bottom[0]->num(), 1, 1, 1);  // n by 1 vector of ones.
  for (int i=0; i < bottom[0]->num(); ++i){
    ones_.mutable_cpu_data()[i] = Dtype(1);
  }
  blob_pos_diff_.Reshape(bottom[0]->channels(), 1, 1, 1);
  blob_neg_diff_.Reshape(bottom[0]->channels(), 1, 1, 1);
} 

template <typename Dtype>
void LiftedStructSimilaritySoftmaxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  const int channels = bottom[0]->channels();
  for (int i = 0; i < bottom[0]->num(); i++){
    dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels, bottom[0]->cpu_data() + (i*channels), bottom[0]->cpu_data() + (i*channels));
  }

  int M_ = bottom[0]->num();
  int N_ = bottom[0]->num();
  int K_ = bottom[0]->channels();

  const Dtype* bottom_data1 = bottom[0]->cpu_data();
  const Dtype* bottom_data2 = bottom[0]->cpu_data();

  Dtype dot_scaler(-2.0);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, dot_scaler, bottom_data1, bottom_data2, (Dtype)0., dot_.mutable_cpu_data());

  // add ||x_i||^2 to all elements in row i
  for (int i=0; i<N_; i++){
    caffe_axpy(N_, dist_sq_.cpu_data()[i], ones_.cpu_data(), dot_.mutable_cpu_data() + i*N_);
  }

  // add the norm vector to row i
  for (int i=0; i<N_; i++){
    caffe_axpy(N_, Dtype(1.0), dist_sq_.cpu_data(), dot_.mutable_cpu_data() + i*N_);
  }

  // construct pairwise label matrix
  vector<vector<bool> > label_mat(N_, vector<bool>(N_, false));
  for (int i=0; i<N_; i++){
    for (int j=0; j<N_; j++){
      label_mat[i][j] = (bottom[1]->cpu_data()[i] == bottom[1]->cpu_data()[j]);
    }
  }

  Dtype margin = this->layer_param_.lifted_struct_sim_softmax_loss_param().margin();
  Dtype loss(0.0);
  num_constraints = Dtype(0.0); 
  const Dtype* bin = bottom[0]->cpu_data();
  Dtype* bout = bottom[0]->mutable_cpu_diff();

  // zero initialize bottom[0]->mutable_cpu_diff();
  for (int i=0; i<N_; i++){
    caffe_set(K_, Dtype(0.0), bout + i*K_);
  }

  // loop upper triangular matrix and look for positive anchors
  for (int i=0; i<N_; i++){
    for (int j=i+1; j<N_; j++){

      // found a positive pair @ anchor (i, j)
      if (label_mat[i][j]){
        Dtype dist_pos = sqrt(dot_.cpu_data()[i*N_ + j]);

        caffe_sub(K_, bin + i*K_, bin + j*K_, blob_pos_diff_.mutable_cpu_data());

        // 1.count the number of negatives for this positive
        int num_negatives = 0;
        for (int k=0; k<N_; k++){
          if (!label_mat[i][k]){
            num_negatives += 1;
          }
        }

        for (int k=0; k<N_; k++){
          if (!label_mat[j][k]){
            num_negatives += 1;
          }
        }

        loss_aug_inference_.Reshape(num_negatives, 1, 1, 1);

        // vector of ones used to sum along channels
        summer_vec_.Reshape(num_negatives, 1, 1, 1);
        for (int ss = 0; ss < num_negatives; ++ss){
          summer_vec_.mutable_cpu_data()[ss] = Dtype(1);
        }

        // 2. compute loss augmented inference
        int neg_idx = 0;
        // mine negative (anchor i, neg k)
        for (int k=0; k<N_; k++){
          if (!label_mat[i][k]){
            loss_aug_inference_.mutable_cpu_data()[neg_idx] = margin - sqrt(dot_.cpu_data()[i*N_ + k]);
            neg_idx++;
          }
        }

        // mine negative (anchor j, neg k)
        for (int k=0; k<N_; k++){
          if (!label_mat[j][k]){
            loss_aug_inference_.mutable_cpu_data()[neg_idx] = margin - sqrt(dot_.cpu_data()[j*N_ + k]);
            neg_idx++;
          }
        }

        // compute softmax of loss aug inference vector;
        Dtype max_elem = *std::max_element(loss_aug_inference_.cpu_data(), loss_aug_inference_.cpu_data() + num_negatives);

        caffe_add_scalar(loss_aug_inference_.count(), Dtype(-1.0)*max_elem, loss_aug_inference_.mutable_cpu_data());
        caffe_exp(loss_aug_inference_.count(), loss_aug_inference_.mutable_cpu_data(), loss_aug_inference_.mutable_cpu_data());
        Dtype soft_maximum = log(caffe_cpu_dot(num_negatives, summer_vec_.cpu_data(), loss_aug_inference_.mutable_cpu_data())) + max_elem;

        // hinge the soft_maximum - S_ij (positive pair similarity)
        Dtype this_loss = std::max(soft_maximum + dist_pos, Dtype(0.0));

        // squared hinge
        loss += this_loss * this_loss; 
        num_constraints += Dtype(1.0);

        // 3. compute gradients
        Dtype sum_exp = caffe_cpu_dot(num_negatives, summer_vec_.cpu_data(), loss_aug_inference_.mutable_cpu_data());

        // update from positive distance dJ_dD_{ij}; update x_i, x_j
        Dtype scaler(0.0);

        scaler = Dtype(2.0)*this_loss / dist_pos;
        // update x_i
        caffe_axpy(K_, scaler * Dtype(1.0), blob_pos_diff_.cpu_data(), bout + i*K_);
        // update x_j
        caffe_axpy(K_, scaler * Dtype(-1.0), blob_pos_diff_.cpu_data(), bout + j*K_);

        // update from negative distance dJ_dD_{ik}; update x_i, x_k
        neg_idx = 0;
        Dtype dJ_dDik(0.0);
        for (int k=0; k<N_; k++){
          if (!label_mat[i][k]){
            caffe_sub(K_, bin + i*K_, bin + k*K_, blob_neg_diff_.mutable_cpu_data());

            dJ_dDik = Dtype(2.0)*this_loss * Dtype(-1.0)* loss_aug_inference_.cpu_data()[neg_idx] / sum_exp;
            neg_idx++;

            scaler = dJ_dDik / sqrt(dot_.cpu_data()[i*N_ + k]);

            // update x_i
            caffe_axpy(K_, scaler * Dtype(1.0), blob_neg_diff_.cpu_data(), bout + i*K_);
            // update x_k
            caffe_axpy(K_, scaler * Dtype(-1.0), blob_neg_diff_.cpu_data(), bout + k*K_);
          }
        }

        // update from negative distance dJ_dD_{jk}; update x_j, x_k
        Dtype dJ_dDjk(0.0);
        for (int k=0; k<N_; k++){
          if (!label_mat[j][k]){
            caffe_sub(K_, bin + j*K_, bin + k*K_, blob_neg_diff_.mutable_cpu_data());

            dJ_dDjk = Dtype(2.0)*this_loss * Dtype(-1.0)*loss_aug_inference_.cpu_data()[neg_idx] / sum_exp;
            neg_idx++;

            scaler = dJ_dDjk / sqrt(dot_.cpu_data()[j*N_ + k]);

            // update x_j
            caffe_axpy(K_, scaler * Dtype(1.0), blob_neg_diff_.cpu_data(), bout + j*K_);
            // update x_k
            caffe_axpy(K_, scaler * Dtype(-1.0), blob_neg_diff_.cpu_data(), bout + k*K_);
          }
        }
      } // close this postive pair
    }
  }
  loss = loss / num_constraints / Dtype(2.0);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void LiftedStructSimilaritySoftmaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype alpha = top[0]->cpu_diff()[0] / num_constraints / Dtype(2.0);

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  for (int i = 0; i < num; i++){
    Dtype* bout = bottom[0]->mutable_cpu_diff();
    caffe_scal(channels, alpha, bout + (i*channels));
  }
}

#ifdef CPU_ONLY
STUB_GPU(LiftedStructSimilaritySoftmaxLossLayer);
#endif

INSTANTIATE_CLASS(LiftedStructSimilaritySoftmaxLossLayer);
REGISTER_LAYER_CLASS(LiftedStructSimilaritySoftmaxLoss);

}  // namespace caffe



