#include <vector>

#include "caffe/layers/mvn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void MVNLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  mean_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  variance_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
  temp_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  if ( this->layer_param_.mvn_param().across_channels() ) {
    sum_multiplier_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),
                            bottom[0]->width());
  } else {
    sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  }
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  eps_ = this->layer_param_.mvn_param().eps();
}

template<typename Dtype, typename MItype, typename MOtype>
void MVNLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int_tp num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int_tp dim = bottom[0]->count() / num;

  // subtract mean
  caffe_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, bottom_data,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());  // EX
  caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
  caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);  // X-EX

  if (this->layer_param_.mvn_param().normalize_variance()) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_powx(bottom[0]->count(), top_data, Dtype(2),
        temp_.mutable_cpu_data());  // (X-EX)^2
    caffe_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, temp_.cpu_data(),
        sum_multiplier_.cpu_data(), 0.,
        variance_.mutable_cpu_data());  // E((X-EX)^2)

    // normalize variance
    caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
          variance_.mutable_cpu_data());

    caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());

    caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          temp_.mutable_cpu_data());

    caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void MVNLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  int_tp num;
  if (this->layer_param_.mvn_param().across_channels())
    num = bottom[0]->num();
  else
    num = bottom[0]->num() * bottom[0]->channels();

  int_tp dim = bottom[0]->count() / num;

  if (this->layer_param_.mvn_param().normalize_variance()) {
    caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
    caffe_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
          sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
    caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
          mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
          bottom_diff);
    caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

    caffe_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_diff,
            sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
    caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
            mean_.cpu_data(), sum_multiplier_.cpu_data(), 1.,
            bottom_diff);

    caffe_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1. / dim),
        bottom_diff);

    // put the squares of bottom into temp_
    caffe_powx(temp_.count(), bottom_data, Dtype(2),
        temp_.mutable_cpu_data());
    caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
        variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
        temp_.mutable_cpu_data());

    caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
  } else {
    caffe_gemv<Dtype>(CblasNoTrans, num, dim, 1. / dim, top_diff,
      sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
    caffe_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      mean_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
      temp_.mutable_cpu_data());
    caffe_add(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(MVNLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(MVNLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(MVNLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(MVNLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(MVN);
REGISTER_LAYER_CLASS_INST(MVN, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(MVN, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(MVN, (double), (double), (double));

}  // namespace caffe
