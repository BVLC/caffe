#include <vector>

#include "caffe/layers/dice_coef_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  const int batchsize = bottom[0]->num();
  const int dim = bottom[0]->count(1);
  nclasses_ = bottom[1]->channels();
  const int imgsize = bottom[1]->count(2);


  vector<int> multiplier_shape(1, dim);
  vector<int> result_shape(1, batchsize);
  result_.Reshape(result_shape);
  result_tmp_.Reshape(result_shape);
  multiplier_.Reshape(multiplier_shape);
  tmp_.ReshapeLike(*bottom[0]);
  caffe_set(dim, Dtype(1.), multiplier_.mutable_cpu_data());

  smooth_ = Dtype(1.0/(double)batchsize);


  switch (this->layer_param_.dice_coef_loss_param().generalization()) {
  case DiceCoefLossParameter_GeneralizationMode_NONE:
    CHECK_EQ(nclasses_, 1) << "channel != 1 for single dice";
    do_weight_ = false;
    ignore_label_ = -1;
    break;
  case DiceCoefLossParameter_GeneralizationMode_MULTICLASS:
    CHECK_NE(nclasses_, 1) << "channel == 1 for multiclass";
    do_weight_ = false;
    if (this->layer_param_.dice_coef_loss_param().has_ignore_label())
      ignore_label_ = this->layer_param_.dice_coef_loss_param().ignore_label();
    else
      ignore_label_ = -1;
    break;
  case DiceCoefLossParameter_GeneralizationMode_MULTICLASS_WEIGHTED_ALL:
    norm_all_ = true;
  case DiceCoefLossParameter_GeneralizationMode_MULTICLASS_WEIGHTED_BATCH:
    norm_batch_ = true;
  case DiceCoefLossParameter_GeneralizationMode_MULTICLASS_WEIGHTED:
    CHECK_NE(nclasses_, 1) << "channel == 1 for multiclass";
    do_weight_ = true;
    if (this->layer_param_.dice_coef_loss_param().has_ignore_label())
      ignore_label_ = this->layer_param_.dice_coef_loss_param().ignore_label();
    else
      ignore_label_ = -1;
    if (this->layer_param_.dice_coef_loss_param().has_weight_mode())
      {
        switch (this->layer_param_.dice_coef_loss_param().weight_mode())
          {
          case DiceCoefLossParameter_WeightMode_INVERSE_VOLUME: weight_pow_ = -2; break;
          case DiceCoefLossParameter_WeightMode_EXTRA_SMALL_VOLUMES: weight_pow_ = -3; break;
          case DiceCoefLossParameter_WeightMode_EQUALIZE_CLASSES: weight_pow_ = -1; break;
          }
        ignore_label_ = this->layer_param_.dice_coef_loss_param().ignore_label();
      }
    smooth_ = Dtype(0.0);
    break;
  }
  if (do_weight_)
    {
      vector<int> weight_shape = {batchsize, nclasses_};
      weights_.Reshape(weight_shape);
      if (norm_batch_)
        {
          vector<int> batch_size_multiplier_shape = {batchsize};
          batchsize_multiplier_.Reshape(batch_size_multiplier_shape);
          caffe_set(batchsize, Dtype(1.0), batchsize_multiplier_.mutable_cpu_data());
          if (norm_all_)
            {
              vector<int> weight_perclass_shape = {nclasses_};
              weights_perclass_mem_.Reshape(weight_perclass_shape);
            }
        }
      // set minimal weight to 1 for 1/w^2 not to nan
      caffe_set(batchsize*nclasses_, Dtype(1.), weights_.mutable_cpu_data());
      vector<int> mask_shape = {nclasses_, dim};
      mask_.Reshape(mask_shape);
      // populate mask, can be transposed
      caffe_set(dim*nclasses_, Dtype(0.), mask_.mutable_cpu_data());
      for (unsigned int i = 0; i< nclasses_; ++i)
        caffe_set(imgsize, Dtype(1.), mask_.mutable_cpu_data()+(dim+imgsize)*i);
      if (ignore_label_ != -1)
          caffe_set(imgsize, Dtype(0.), mask_.mutable_cpu_data()+(dim+imgsize)*ignore_label_);
      weight_multiplier_.ReshapeLike(*bottom[0]);
    }
  else if (ignore_label_ != -1 && nclasses_ > 1)
    caffe_set(imgsize, Dtype(0.), multiplier_.mutable_cpu_data()+imgsize*ignore_label_);

  caffe_set(batchsize, smooth_, result_tmp_.mutable_cpu_data());
  caffe_set(batchsize, smooth_, result_.mutable_cpu_data());
}


template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const int batchsize = bottom[0]->num();
  caffe_set(batchsize, smooth_, result_tmp_.mutable_cpu_data());
  caffe_set(batchsize, smooth_, result_.mutable_cpu_data());

  for (unsigned int i=0; i< bottom[1]->count(); ++i)
    {
      if (bottom[1]->cpu_data()[i] <0.5)
        bottom[1]->mutable_cpu_data()[i] = 0.0;
      else
        bottom[1]->mutable_cpu_data()[i] = 1.0;
    }

  if (do_weight_)
    {
      Dtype unit_weight = Dtype(1.0);
      caffe_set(batchsize*nclasses_, unit_weight, weights_.mutable_cpu_data());
      // compute weights per label per image
      caffe_cpu_gemm(CblasNoTrans, CblasTrans, bottom[1]->num(), bottom[1]->channels(),
                     bottom[1]->count(1), unit_weight, bottom[1]->cpu_data(), mask_.cpu_data(),
                     Dtype(1.), weights_.mutable_cpu_data());

      // below normalize over batch
      if (norm_batch_)
        {
          Blob<Dtype> weights_perclass;
          vector<int> weight_perclass_shape = {nclasses_};
          weights_perclass.Reshape(weight_perclass_shape);

          caffe_cpu_gemv(CblasTrans,batchsize,nclasses_,Dtype(1.0),weights_.cpu_data(),
                         batchsize_multiplier_.cpu_data(),Dtype(0.0),weights_perclass.mutable_cpu_data());
          caffe_scal(nclasses_, Dtype(1.0)/Dtype(batchsize), weights_perclass.mutable_cpu_data());
          if (norm_all_)
            {
              if (numit_ != 0)
                {
                  caffe_cpu_axpby(nclasses_,Dtype(numit_),weights_perclass_mem_.cpu_data(),Dtype(1.0), weights_perclass.mutable_cpu_data());
                  caffe_scal(nclasses_, Dtype(1.0)/Dtype(++numit_), weights_perclass.mutable_cpu_data());
                }
              else
                numit_ = 1;
              caffe_copy(nclasses_,weights_perclass.cpu_data(),weights_perclass_mem_.mutable_cpu_data());
            }
          else
            numit_++;
          caffe_cpu_gemm(CblasNoTrans,CblasNoTrans,batchsize,nclasses_,1,Dtype(1.0),
                         batchsize_multiplier_.cpu_data(),weights_perclass.cpu_data(),
                         Dtype(0.0), weights_.mutable_cpu_data());
        }

      // do 1/w^2
      caffe_powx(bottom[1]->num() * bottom[1]->channels(),weights_.cpu_data(), Dtype(weight_pow_),
                     weights_.mutable_cpu_data());

      //  put them into our multiplexed multiplier
      caffe_cpu_gemm(CblasTrans, CblasNoTrans,
                     bottom[1]->num(), bottom[1]->count(1), bottom[1]->channels(),
                     Dtype(1.), weights_.cpu_data(), mask_.cpu_data(), Dtype(0.),
                     weight_multiplier_.mutable_cpu_data());
    }


  caffe_mul(bottom[0]->count(), bottom[0]->cpu_data(), bottom[0]->cpu_data(),
            tmp_.mutable_cpu_data());
  if (do_weight_)
		caffe_mul(bottom[0]->count(), weight_multiplier_.cpu_data(), tmp_.cpu_data(),
							tmp_.mutable_cpu_data());
  // result_tmp_ <- 1.*tmp_ * multiplier + 1*results_tmp_
  caffe_cpu_gemv(CblasNoTrans, bottom[0]->num(), bottom[0]->count(1), Dtype(1.), tmp_.cpu_data(),
                 multiplier_.cpu_data(), Dtype(1.), result_tmp_.mutable_cpu_data());

  // tmp_ <- b1 * b1
	caffe_mul(bottom[0]->count(), bottom[1]->cpu_data(), bottom[1]->cpu_data(),
						tmp_.mutable_cpu_data());
  if (do_weight_)
		caffe_mul(bottom[0]->count(), weight_multiplier_.cpu_data(), tmp_.cpu_data(),
							tmp_.mutable_cpu_data());
  // result_tmp_ <- 1.*tmp_ * multiplier + 1*results_tmp_
  caffe_cpu_gemv(CblasNoTrans, bottom[0]->num(), bottom[0]->count(1), Dtype(1.), tmp_.cpu_data(),
                 multiplier_.cpu_data(), Dtype(1.), result_tmp_.mutable_cpu_data());

  caffe_mul(bottom[0]->count(), bottom[0]->cpu_data(), bottom[1]->cpu_data(),
            tmp_.mutable_cpu_data());

  if (do_weight_)
		caffe_mul(bottom[0]->count(), weight_multiplier_.cpu_data(), tmp_.cpu_data(),
							tmp_.mutable_cpu_data());
  caffe_cpu_gemv(CblasNoTrans, bottom[1]->num(), bottom[1]->count(1), Dtype(2.), tmp_.cpu_data(),
                 multiplier_.cpu_data(), Dtype(1.), result_.mutable_cpu_data());
  caffe_div(bottom[0]->num(), result_.cpu_data(), result_tmp_.cpu_data(),
            result_.mutable_cpu_data());


  Dtype loss = Dtype(1) - caffe_cpu_asum(bottom[0]->num(), result_.cpu_data()) / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = Dtype(1.0);
      const int index = (i == 0) ? 1 : 0;

      caffe_copy(bottom[i]->count(), bottom[i]->cpu_data(), bottom[i]->mutable_cpu_diff());

      for (int j = 0; j < bottom[i]->num(); j++) {
        const Dtype alpha = sign * top[0]->cpu_diff()[0] / result_tmp_.cpu_data()[j];
        caffe_cpu_axpby(
                        bottom[i]->count(1),              // count
                        alpha*Dtype(-2),                  // alpha
                        bottom[index]->cpu_data()+j*bottom[i]->count(1),        // a
                        alpha*result_.cpu_data()[j]*Dtype(2),                      // beta
                        bottom[i]->mutable_cpu_diff()+j*bottom[i]->count(1)
                        );  // b
      }
      if (do_weight_)
        caffe_mul(bottom[i]->count(), weight_multiplier_.cpu_data(), bottom[i]->cpu_diff(),
                  bottom[i]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(DiceCoefLossLayer);
#endif

INSTANTIATE_CLASS(DiceCoefLossLayer);
REGISTER_LAYER_CLASS(DiceCoefLoss);

}  // namespace caffe
