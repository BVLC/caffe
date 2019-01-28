#include <vector>

#include "caffe/layers/dice_coef_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  int weightType = -1;
  if (this->layer_param_.dice_coef_loss_param().has_contour_shape())
    {
      switch (this->layer_param_.dice_coef_loss_param().contour_shape())
        {
        case DiceCoefLossParameter_ContourShape_NO: break;
        case DiceCoefLossParameter_ContourShape_SIMPLE: weightType = 0; break;
        case DiceCoefLossParameter_ContourShape_SHARP: weightType = 1; break;
        }
    }

  if (this->layer_param_.dice_coef_loss_param().has_contour_amplitude())
    contour_amplitude_ = this->layer_param_.dice_coef_loss_param().contour_amplitude();


  if (weightType != -1)
    {
      do_contour_weights_ = true;
      int csize = 3;
      if (this->layer_param_.dice_coef_loss_param().has_contour_size())
        csize = this->layer_param_.dice_coef_loss_param().contour_size();
      if (csize < 3)
        csize = 3;
      csize = (csize % 2 == 0? csize+1:csize);
      vector<int> cwShape{1,1,csize,csize};
      contour_weights_kernel_.Reshape(cwShape);




      switch (weightType)
        {
        case 1:
          {
            double sum = 0.0;
            for (int i=0; i<csize; ++i)
              for (int j=0; j<csize; ++j)
                {
                  int ti = i-csize/2;
                  int tj = j-csize/2;
                  if (ti==0 && tj ==0)
                    continue;
                  double val = -1.0/((double)(abs(ti)+abs(tj) * 4));
                  sum += val;
                  caffe_set(1, Dtype(val),
                            contour_weights_kernel_.mutable_cpu_data()+contour_weights_kernel_.offset(0,0,i,j));
                }
            caffe_set(1, Dtype(-sum),
                      contour_weights_kernel_.mutable_cpu_data()+contour_weights_kernel_.offset(0,0,csize/2+1,csize/2+1));
            break;
          }
        default:
          caffe_set(contour_weights_kernel_.count(), Dtype(-1.0/((double)csize*csize-1)),
                    contour_weights_kernel_.mutable_cpu_data());
          caffe_set(1, Dtype(1.0),
                    contour_weights_kernel_.mutable_cpu_data()+contour_weights_kernel_.offset(0,0,csize/2,csize/2));
        }

    }
}

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
  height_ = bottom[1]->height();
  width_ = bottom[1]->width();

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
      }
    smooth_ = Dtype(0.0);
    break;
  }
  if (do_contour_weights_)
    {
      vector<int> sum_contour_shape = {batchsize, nclasses_};
      sum_contour_.Reshape(sum_contour_shape);
      vector<int> mi_shape = {height_,width_};
      mask_inverter_.Reshape(mi_shape);
      caffe_set(height_*width_, Dtype(1.0), mask_inverter_.mutable_cpu_data());
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
    }
  else if (ignore_label_ != -1 && nclasses_ > 1)
    caffe_set(imgsize, Dtype(0.), multiplier_.mutable_cpu_data()+imgsize*ignore_label_);


  caffe_set(batchsize, smooth_, result_tmp_.mutable_cpu_data());
  caffe_set(batchsize, smooth_, result_.mutable_cpu_data());


  vector<Dtype> weights;
  if (this->layer_param_.dice_coef_loss_param().has_weights())
    {
      BlobProto blob_proto;
      ReadProtoFromBinaryFile(
                              this->layer_param_.dice_coef_loss_param().weights(), &blob_proto);
      Blob<Dtype> external_weights_proto;
      external_weights_proto.FromProto(blob_proto);
      CHECK_EQ(external_weights_proto.count(), nclasses_*nclasses_);
      Dtype sum = 0;
      for (int ci = 0; ci < nclasses_; ++ci)
        {
          Dtype cw = external_weights_proto.cpu_data()[external_weights_proto.offset(0,0,ci,ci)];
          if (cw < Dtype(1E-9))
            cw = Dtype(1E-9);
          sum += cw;
          weights.push_back(cw);
        }
      for (int ci = 0; ci < nclasses_; ++ci)
        weights[ci] /= sum;
      has_external_weights_ = true;
    }


  if (do_weight_ || has_external_weights_ || do_contour_weights_)
    {
      weight_multiplier_.ReshapeLike(*bottom[0]);
      caffe_set(weight_multiplier_.count(), Dtype(1.0), weight_multiplier_.mutable_cpu_data());
    }

  if (has_external_weights_)
    {
      external_weights_.ReshapeLike(weight_multiplier_);
      // compute external_weights_ in good shape (same as weight_multiplier_)
      caffe_set(external_weights_.count(), Dtype(1.0), external_weights_.mutable_cpu_data());
      for (int bi = 0; bi < batchsize; ++bi)
        for (int ci = 0; ci < nclasses_; ++ci)
          caffe_set(imgsize, weights[ci], external_weights_.mutable_cpu_data()+external_weights_.offset(bi,ci,0,0));
    }

  if (do_contour_weights_)
    {
      contour_weights_.ReshapeLike(*bottom[0]);
      std::vector<int> col_buffer_shape;
      col_buffer_shape.push_back(contour_weights_kernel_.count(1));
      for (int i = 0; i < 2; ++i) {
        col_buffer_shape.push_back(contour_weights_.shape(i));
      }
      col_buffer_.Reshape(col_buffer_shape);
    }
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
      if (ignore_label_ != -1)
        for (int i=0; i<batchsize; ++i)
          caffe_set(1, Dtype(1E6)*Dtype(bottom[1]->count(2)),
                  weights_.mutable_cpu_data()+i*nclasses_+ignore_label_);
      //      compute weights per label per image
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
      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans,
                     bottom[1]->num(), bottom[1]->count(1), bottom[1]->channels(),
                     Dtype(1.), weights_.cpu_data(), mask_.cpu_data(), Dtype(0.),
                     weight_multiplier_.mutable_cpu_data());
    }

  if (has_external_weights_)
    {
      caffe_mul(weight_multiplier_.count(), external_weights_.cpu_data(), weight_multiplier_.cpu_data(), weight_multiplier_.mutable_cpu_data());
    }
  if (do_contour_weights_)
    {
      for (int n =0; n<batchsize; ++n)
        {
          // first reshape bottom[1] in order to be B C 1 H W (convolutions over 1 not, over channels)
          std::vector<int> old_shape(bottom[1]->shape());
          std::vector<int> new_shape(bottom[1]->shape());
          new_shape.insert(new_shape.begin()+2, 1);
          bottom[1]->Reshape(new_shape);
          contour_weights_.Reshape(new_shape);
          caffe_cpu_axpby(height_*width_, Dtype(1.0), mask_inverter_.cpu_data(), Dtype(-1.0),
                            bottom[1]->mutable_cpu_data()+bottom[1]->offset({n,0,0,0,0}));
          compute_contour_weights_cpu(bottom[1]->cpu_data()+bottom[1]->offset({n,0,0,0,0}),
                                      contour_weights_kernel_.cpu_data(),
                                      contour_weights_.mutable_cpu_data()
                                      +contour_weights_.offset({n,0,0,0,0}));
          caffe_cpu_axpby(height_*width_, Dtype(1.0), mask_inverter_.cpu_data(), Dtype(-1.0),
                            bottom[1]->mutable_cpu_data()+bottom[1]->offset({n,0,0,0,0}));
          for (int c =1; c<nclasses_; ++c)
            compute_contour_weights_cpu(bottom[1]->cpu_data()+bottom[1]->offset({n,c,0,0,0}),
                                        contour_weights_kernel_.cpu_data(),
                                        contour_weights_.mutable_cpu_data()
                                        +contour_weights_.offset({n,c,0,0,0}));
          bottom[1]->Reshape(old_shape);
          contour_weights_.Reshape(old_shape);
          caffe_cpu_gemv(CblasNoTrans,nclasses_,height_*width_,Dtype(1.0),
                         contour_weights_.cpu_data()+contour_weights_.offset(n,0,0,0),
                         multiplier_.cpu_data(), Dtype(0.0),
                         sum_contour_.mutable_cpu_data()+sum_contour_.offset(n));
          for (int c =0; c<nclasses_; ++c)
            {
              Dtype sum = Dtype(sum_contour_.cpu_data()[sum_contour_.offset({n,c})]);
              caffe_scal(height_*width_,
                         Dtype(height_*width_)/sum,
                         contour_weights_.mutable_cpu_data()+contour_weights_.offset(n,c));
            }
        }
      caffe_mul(weight_multiplier_.count(), contour_weights_.cpu_data(),
                weight_multiplier_.cpu_data(), weight_multiplier_.mutable_cpu_data());
    }


  caffe_mul(bottom[0]->count(), bottom[0]->cpu_data(), bottom[0]->cpu_data(),
            tmp_.mutable_cpu_data());
  if (do_weight_ || has_external_weights_)
		caffe_mul(bottom[0]->count(), weight_multiplier_.cpu_data(), tmp_.cpu_data(),
							tmp_.mutable_cpu_data());
  // result_tmp_ <- 1.*tmp_ * multiplier + 1*results_tmp_
  caffe_cpu_gemv(CblasNoTrans, bottom[0]->num(), bottom[0]->count(1), Dtype(1.), tmp_.cpu_data(),
                 multiplier_.cpu_data(), Dtype(1.), result_tmp_.mutable_cpu_data());

  // tmp_ <- b1 * b1
	caffe_mul(bottom[0]->count(), bottom[1]->cpu_data(), bottom[1]->cpu_data(),
						tmp_.mutable_cpu_data());
  if (do_weight_ || has_external_weights_ || do_contour_weights_)
		caffe_mul(bottom[0]->count(), weight_multiplier_.cpu_data(), tmp_.cpu_data(),
							tmp_.mutable_cpu_data());
  // result_tmp_ <- 1.*tmp_ * multiplier + 1*results_tmp_
  caffe_cpu_gemv(CblasNoTrans, bottom[0]->num(), bottom[0]->count(1), Dtype(1.), tmp_.cpu_data(),
                 multiplier_.cpu_data(), Dtype(1.), result_tmp_.mutable_cpu_data());

  caffe_mul(bottom[0]->count(), bottom[0]->cpu_data(), bottom[1]->cpu_data(),
            tmp_.mutable_cpu_data());

  if (do_weight_ || has_external_weights_ || do_contour_weights_)
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
      if (do_weight_ || has_external_weights_ || do_contour_weights_)
        caffe_mul(bottom[i]->count(), weight_multiplier_.cpu_data(), bottom[i]->cpu_diff(),
                  bottom[i]->mutable_cpu_diff());
    }
  }
}

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::conv_im2col_cpu(const Dtype* data, Dtype* col_buff)
{
  int kh = contour_weights_kernel_.height();
  im2col_cpu(data, 1,
             height_,
             width_,
             kh,
             kh,
             (kh-1)/2, (kh-1)/2,
             1, 1,
             1, 1,
             col_buff);
}



template <typename Dtype>
void DiceCoefLossLayer<Dtype>::compute_contour_weights_cpu(const Dtype*input, const Dtype*weights,
                                                           Dtype *output)
{
  conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
  const Dtype *col_buff = col_buffer_.cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                        1,
                        contour_weights_.count(3),
                        contour_weights_kernel_.count(2),
                        (Dtype)1., weights, col_buff,
                        (Dtype)0., output) ;
  caffe_abs(width_*height_, output, output);
  //output is between 0 and 1
  caffe_scal(width_*height_, Dtype(contour_amplitude_ - 1.0), output);
  // now between 0 and amplitude_ -1
  caffe_add_scalar(width_*height_, Dtype(1.0), output);
  // now between 1 and amplitude
}

#ifdef CPU_ONLY
STUB_GPU(DiceCoefLossLayer);
#endif

INSTANTIATE_CLASS(DiceCoefLossLayer);
REGISTER_LAYER_CLASS(DiceCoefLoss);

}  // namespace caffe
