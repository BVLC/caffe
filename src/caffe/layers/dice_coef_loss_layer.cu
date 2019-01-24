#include <vector>

#include "caffe/layers/dice_coef_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {


template <typename Dtype>
  __global__ void StickLabels(const int n, const Dtype* b1in,
                              Dtype* b1out) {
  CUDA_KERNEL_LOOP(index, n)
      {
        if (b1in[index]  < Dtype(0.5))
          b1out[index] = Dtype(0.0);
        else
          b1out[index] = Dtype(1.0);
      }
  }


template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  const int batchsize = bottom[0]->num();
  caffe_gpu_set(batchsize, smooth_, result_tmp_.mutable_gpu_data());
  caffe_gpu_set(batchsize, smooth_, result_.mutable_gpu_data());

  StickLabels<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom[1]->gpu_data(), bottom[1]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;

  if (do_weight_)
    {
      Dtype unit_weight = Dtype(1.);
      caffe_gpu_set(batchsize*nclasses_, unit_weight, weights_.mutable_gpu_data());
      if (ignore_label_ != -1)
        for (int i=0; i<batchsize; ++i)
          caffe_gpu_set(1, Dtype(1E6)*Dtype(bottom[1]->count(2)),
                        weights_.mutable_gpu_data()+i*nclasses_+ignore_label_);

      caffe_gpu_gemm(CblasNoTrans, CblasTrans, bottom[1]->num(), bottom[1]->channels(),
                     bottom[1]->count(1), unit_weight, bottom[1]->gpu_data(), mask_.gpu_data(),
                     Dtype(1.), weights_.mutable_gpu_data());

       //below normalize over batch
      if (norm_batch_)
        {
          Blob<Dtype> weights_perclass;
          vector<int> weight_perclass_shape = {nclasses_};
          weights_perclass.Reshape(weight_perclass_shape);
          caffe_gpu_gemv(CblasTrans,batchsize,nclasses_,Dtype(1.0),weights_.gpu_data(),
                         batchsize_multiplier_.gpu_data(),Dtype(0.0),weights_perclass.mutable_gpu_data());
          caffe_gpu_scal(nclasses_, Dtype(1.0)/Dtype(batchsize), weights_perclass.mutable_gpu_data());

          if (norm_all_)
            {
              if (numit_ != 0)
                {
                  caffe_gpu_axpby(nclasses_,Dtype(numit_),weights_perclass_mem_.gpu_data(),Dtype(1.0), weights_perclass.mutable_gpu_data());
                  caffe_gpu_scal(nclasses_, Dtype(1.0)/Dtype(++numit_), weights_perclass.mutable_gpu_data());
                }
              else
                numit_ = 1;
              caffe_copy(nclasses_,weights_perclass.gpu_data(),weights_perclass_mem_.mutable_gpu_data());
            }

          caffe_gpu_gemm(CblasNoTrans,CblasNoTrans,batchsize,nclasses_,1,Dtype(1.0),
                     batchsize_multiplier_.gpu_data(),weights_perclass.gpu_data(),
                         Dtype(0.0), weights_.mutable_gpu_data());
          // end of normlization over batch
        }

      caffe_gpu_powx(bottom[1]->num() * bottom[1]->channels(),weights_.gpu_data(), Dtype(weight_pow_),
                 weights_.mutable_gpu_data());
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
                     bottom[1]->num(), bottom[1]->count(1), bottom[1]->channels(),
                     Dtype(1.), weights_.gpu_data(), mask_.gpu_data(), Dtype(0.),
                     weight_multiplier_.mutable_gpu_data());
    }

  if (has_external_weights_)
    {
      caffe_gpu_mul(weight_multiplier_.count(), external_weights_.gpu_data(),
                    weight_multiplier_.gpu_data(), weight_multiplier_.mutable_gpu_data());
    }
  if (do_contour_weights_)
    {
      for (int n =0; n<batchsize; ++n)
        {
          std::vector<int> old_shape(bottom[1]->shape());
          std::vector<int> new_shape(bottom[1]->shape());
          new_shape.insert(new_shape.begin()+2, 1);
          bottom[1]->Reshape(new_shape);
          contour_weights_.Reshape(new_shape);
          // for background , ie class 0, one should revert labels in order for zero padding not
          // to give much weight to image border
          caffe_gpu_axpby(height_*width_, Dtype(1.0), mask_inverter_.gpu_data(), Dtype(-1.0),
                          bottom[1]->mutable_gpu_data()+bottom[1]->offset({n,0,0,0,0}));
          compute_contour_weights_gpu(bottom[1]->gpu_data()+bottom[1]->offset({n,0,0,0,0}),
                                      contour_weights_kernel_.gpu_data(),
                                      contour_weights_.mutable_gpu_data()
                                      +contour_weights_.offset({n,0,0,0,0}));
          caffe_gpu_axpby(height_*width_, Dtype(1.0), mask_inverter_.gpu_data(), Dtype(-1.0),
                          bottom[1]->mutable_gpu_data()+bottom[1]->offset({n,0,0,0,0}));

          for (int c =1; c<nclasses_; ++c)
              compute_contour_weights_gpu(bottom[1]->gpu_data()+bottom[1]->offset({n,c,0,0,0}),
                                          contour_weights_kernel_.gpu_data(),
                                          contour_weights_.mutable_gpu_data()
                                          +contour_weights_.offset({n,c,0,0,0}));
          bottom[1]->Reshape(old_shape);
          contour_weights_.Reshape(old_shape);
          caffe_gpu_gemv(CblasNoTrans,nclasses_,height_*width_,Dtype(1.0),
                         contour_weights_.gpu_data()+contour_weights_.offset(n,0,0,0),
                         multiplier_.gpu_data(), Dtype(0.0),
                         sum_contour_.mutable_gpu_data()+sum_contour_.offset(n));
          for (int c =0; c<nclasses_; ++c)
            {
              Dtype sum = Dtype(sum_contour_.cpu_data()[sum_contour_.offset({n,c})]);
              caffe_gpu_scal(height_*width_,
                             Dtype(height_*width_)/sum,
                             contour_weights_.mutable_gpu_data()+contour_weights_.offset({n,c,0,0}));
            }
        }
      caffe_gpu_mul(weight_multiplier_.count(), contour_weights_.gpu_data(),
                    weight_multiplier_.gpu_data(), weight_multiplier_.mutable_gpu_data());
    }


  caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[0]->gpu_data(),
                tmp_.mutable_gpu_data());
  if (do_weight_ || has_external_weights_ || do_contour_weights_)
    caffe_gpu_mul(bottom[0]->count(), weight_multiplier_.gpu_data(), tmp_.gpu_data(),
                  tmp_.mutable_gpu_data());


  caffe_gpu_gemv(CblasNoTrans, bottom[0]->num(), bottom[0]->count(1), Dtype(1.), tmp_.gpu_data(),
                 multiplier_.gpu_data(), Dtype(1.), result_tmp_.mutable_gpu_data());

  caffe_gpu_mul(count, bottom[1]->gpu_data(), bottom[1]->gpu_data(),
                tmp_.mutable_gpu_data());
  if (do_weight_ || has_external_weights_ || do_contour_weights_)
    caffe_gpu_mul(bottom[0]->count(), weight_multiplier_.gpu_data(), tmp_.gpu_data(),
									tmp_.mutable_gpu_data());
  caffe_gpu_gemv(CblasNoTrans, bottom[1]->num(), bottom[1]->count(1), Dtype(1.), tmp_.gpu_data(),
                 multiplier_.gpu_data(), Dtype(1.), result_tmp_.mutable_gpu_data());

	caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                tmp_.mutable_gpu_data());
  if (do_weight_ || has_external_weights_ || do_contour_weights_)
    caffe_gpu_mul(bottom[0]->count(), weight_multiplier_.gpu_data(), tmp_.gpu_data(),
              tmp_.mutable_gpu_data());
  caffe_gpu_gemv(CblasNoTrans, bottom[1]->num(), bottom[1]->count(1), Dtype(2.), tmp_.gpu_data(),
                 multiplier_.gpu_data(), Dtype(1.), result_.mutable_gpu_data());
  caffe_gpu_div(bottom[0]->num(), result_.gpu_data(), result_tmp_.gpu_data(),
                result_.mutable_gpu_data());

  Dtype loss;
  caffe_gpu_asum(bottom[0]->num(), result_.gpu_data(), &loss);
  loss /= bottom[0]->num();
  loss = Dtype(1.) - loss;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void DiceCoefLossBackward(const int n, const Dtype* x,
                                     const Dtype* y, Dtype* out_diff, const Dtype sign,
                                     const Dtype* loss, const Dtype* denominator, const int dim,
                                     const int nclasses) {
  CUDA_KERNEL_LOOP(index, n)
    {
      const int i = index / dim;
			const Dtype alpha = Dtype(-2.) / denominator[i] * sign;
			Dtype beta = Dtype(2.) * loss[i] / denominator[i] * sign;
			out_diff[index] = alpha * x[index] + beta * y[index];
   }
}

template <typename Dtype>
void DiceCoefLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  for (int i = 0; i < 2; ++i) {
        if (propagate_down[i])
          {
          int count = bottom[i]->count();
          const Dtype sign = Dtype(1.0)*top[0]->cpu_diff()[0];
          const int index = (i == 0) ? 1 : 0;
          DiceCoefLossBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom[index]->gpu_data(), bottom[i]->gpu_data(), bottom[i]->mutable_gpu_diff(),
					sign, result_.gpu_data(), result_tmp_.gpu_data(), bottom[i]->count(1),
					bottom[i]->channels());
          CUDA_POST_KERNEL_CHECK;
          }
        if (do_weight_ || has_external_weights_ || do_contour_weights_)
          caffe_gpu_mul(bottom[i]->count(), weight_multiplier_.gpu_data(), bottom[i]->gpu_diff(),
                        bottom[i]->mutable_gpu_diff());
  }
}



template <typename Dtype>
void DiceCoefLossLayer<Dtype>::conv_im2col_gpu(const Dtype* data, Dtype* col_buff)
  {
    int kh = contour_weights_kernel_.height();
    im2col_gpu(data, 1,
               height_, width_,
               kh,kh,
               (kh-1)/2, (kh-1)/2,
               1, 1,
               1, 1,
               col_buff);
 }



template <typename Dtype>
void DiceCoefLossLayer<Dtype>::compute_contour_weights_gpu(const Dtype*input, const Dtype*weights,
                                                             Dtype *output)
  {
    conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    const Dtype *col_buff = col_buffer_.gpu_data();
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
                          1, contour_weights_.count(3),
                          contour_weights_kernel_.count(2),
                          (Dtype)1., weights, col_buff,
                          (Dtype)0., output) ;
    caffe_gpu_abs(width_*height_, output, output);
    caffe_gpu_scal(width_*height_, Dtype(contour_amplitude_ - 1.0), output);
    caffe_gpu_add_scalar(width_*height_, Dtype(1.0), output);
  }


INSTANTIATE_LAYER_GPU_FUNCS(DiceCoefLossLayer);

}  // namespace caffe
