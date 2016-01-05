#include <vector>
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
inline Dtype sigmoid_diff(Dtype x) {
  return x * (1. - x);
}

template <typename Dtype>
inline Dtype tanh(Dtype x) {
  Dtype exp2x = exp(2 * x);
  return fabs(x) < Dtype(5) ? ((exp2x - Dtype(1)) / (exp2x + Dtype(1)))
    : (x > 0 ? Dtype(1) : Dtype(-1));
}

template <typename Dtype>
inline Dtype tanh_diff(Dtype x) {
  return (1. - x * x);
}

template <typename Dtype>
void LstmUnitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LstmUnitParameter lstm_unit_param = this->layer_param_.lstm_unit_param();
  CHECK((lstm_unit_param.has_num_cells()))
      << "lstm_unit_param.has_num_cells()";
  CHECK((lstm_unit_param.has_input_weight_filler()))
      << "lstm_unit_param.has_input_weight_filler()";
  CHECK((lstm_unit_param.has_input_gate_weight_filler()))
      << "lstm_unit_param.has_input_gate_weight_filler()";
  CHECK((lstm_unit_param.has_forget_gate_weight_filler()))
      << "lstm_unit_param.has_forget_gate_weight_filler()";
  CHECK((lstm_unit_param.has_output_gate_weight_filler()))
      << "lstm_unit_param.has_output_gate_weight_filler()";

  channels_ = lstm_unit_param.num_cells();
  CHECK_EQ(channels_, bottom[1]->shape(1)) <<
    "Number of input memory channels must match the number of lstm mem_cells";
  input_data_size_ = bottom[0]->shape(1);
  N_ = channels_;
  K_ = input_data_size_;

  this->blobs_.resize(4);
  for (int i = 0; i < 4; ++i) {
      this->blobs_[i].reset(new Blob<Dtype>(
          1, channels_, 1, input_data_size_));
  }

  shared_ptr<Filler<Dtype> > input_weight_filler(GetFiller<Dtype>(
      lstm_unit_param.input_weight_filler()));
  input_weight_filler->Fill(this->blobs_[0].get());

  shared_ptr<Filler<Dtype> > input_gate_weight_filler(GetFiller<Dtype>(
      lstm_unit_param.input_gate_weight_filler()));
  input_gate_weight_filler->Fill(this->blobs_[1].get());

  shared_ptr<Filler<Dtype> > forget_gate_weight_filler(GetFiller<Dtype>(
      lstm_unit_param.forget_gate_weight_filler()));
  forget_gate_weight_filler->Fill(this->blobs_[2].get());

  shared_ptr<Filler<Dtype> > output_gate_weight_filler(GetFiller<Dtype>(
      lstm_unit_param.output_gate_weight_filler()));
  output_gate_weight_filler->Fill(this->blobs_[3].get());

  input_gates_data_buffer_.reset(new Blob<Dtype>());
  this->buffers_.push_back(input_gates_data_buffer_);
  forget_gates_data_buffer_.reset(new Blob<Dtype>());
  this->buffers_.push_back(forget_gates_data_buffer_);
  output_gates_data_buffer_.reset(new Blob<Dtype>());
  this->buffers_.push_back(output_gates_data_buffer_);
  input_values_data_buffer_.reset(new Blob<Dtype>());
  this->buffers_.push_back(input_values_data_buffer_);
  gates_diff_buffer_.reset(new Blob<Dtype>());
  next_state_tot_diff_buffer_.reset(new Blob<Dtype>());
  tanh_mem_buffer_.reset(new Blob<Dtype>());
  dldg_buffer_.reset(new Blob<Dtype>());

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void LstmUnitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK((this->layer_param_.bottom_size() == 2
      || this->layer_param_.bottom_size() == 0))
      << "LstmUnit must have a data and cell bottom";
  CHECK((this->layer_param_.top_size() == 2
      || this->layer_param_.top_size() == 0))
      << "LstmUnit must have a data and cell top";
  num_ = bottom[0]->shape(0);
  M_ = num_;
  input_gates_data_buffer_->Reshape(num_, channels_, 1, 1);
  forget_gates_data_buffer_->Reshape(num_, channels_, 1, 1);
  output_gates_data_buffer_->Reshape(num_, channels_, 1, 1);
  input_values_data_buffer_->Reshape(num_, channels_, 1, 1);
  gates_diff_buffer_->Reshape(num_, 4 * channels_, 1, 1);
  next_state_tot_diff_buffer_->Reshape(num_, channels_, 1, 1);
  tanh_mem_buffer_->Reshape(num_, channels_, 1, 1);
  dldg_buffer_->Reshape(num_, channels_, 1, 1);
  vector<int> shape;
  shape.push_back(num_);
  shape.push_back(channels_);
  top[0]->Reshape(shape);
  top[1]->Reshape(shape);
}

template <typename Dtype>
void LstmUnitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* prev_state_data = bottom[1]->cpu_data();

  const Dtype* input_weight = this->blobs_[0]->cpu_data();
  const Dtype* input_gate_weight = this->blobs_[1]->cpu_data();
  const Dtype* forget_gate_weight = this->blobs_[2]->cpu_data();
  const Dtype* output_gate_weight = this->blobs_[3]->cpu_data();

  Dtype* next_hidden_state = top[0]->mutable_cpu_data();
  Dtype* next_memory_state = top[1]->mutable_cpu_data();

  Dtype* input_gates = input_gates_data_buffer_->mutable_cpu_data();
  Dtype* forget_gates = forget_gates_data_buffer_->mutable_cpu_data();
  Dtype* output_gates = output_gates_data_buffer_->mutable_cpu_data();
  Dtype* input_values = input_values_data_buffer_->mutable_cpu_data();
  Dtype* tanh_next_memory_state = tanh_mem_buffer_->mutable_cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., input_data, input_weight,
    (Dtype)0., input_values);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., input_data, input_gate_weight,
    (Dtype)0., input_gates);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., input_data, forget_gate_weight,
    (Dtype)0., forget_gates);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_,
    (Dtype)1., input_data, output_gate_weight,
    (Dtype)0., output_gates);

  if (this->layer_param_.lstm_unit_param().tie_output_forget()) {
    caffe_set(channels_ * num_, Dtype(0), forget_gates);
    caffe_sub(channels_ * num_, forget_gates, output_gates, forget_gates);
  }

  for (int n = 0; n < num_; ++n) {
    for (int i = 0; i < channels_; ++i) {
      const int idx = i + n * channels_;
      input_gates[idx] = sigmoid(input_gates[idx]);
      forget_gates[idx] = sigmoid(forget_gates[idx]);
      output_gates[idx] = sigmoid(output_gates[idx]);
      input_values[idx] = tanh(input_values[idx]);

      next_memory_state[idx] = prev_state_data[idx] * forget_gates[idx] +
          input_gates[idx] * input_values[idx];
      if (this->layer_param_.lstm_unit_param().tanh_hidden()) {
        tanh_next_memory_state[idx] = tanh(next_memory_state[idx]);
      } else {
        tanh_next_memory_state[idx] = next_memory_state[idx];
      }
      next_hidden_state[idx] = tanh_next_memory_state[idx] * output_gates[idx];
    }
  }
}

template <typename Dtype>
void LstmUnitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* prev_state_data = bottom[1]->cpu_data();

  const Dtype* input_weight = this->blobs_[0]->cpu_data();
  const Dtype* input_gate_weight = this->blobs_[1]->cpu_data();
  const Dtype* forget_gate_weight = this->blobs_[2]->cpu_data();
  const Dtype* output_gate_weight = this->blobs_[3]->cpu_data();

  const Dtype* input_gates = input_gates_data_buffer_->cpu_data();
  const Dtype* forget_gates = forget_gates_data_buffer_->cpu_data();
  const Dtype* output_gates = output_gates_data_buffer_->cpu_data();
  const Dtype* input_values = input_values_data_buffer_->cpu_data();
  const Dtype* tanh_next_memory_state = tanh_mem_buffer_->cpu_data();

  Dtype* gates_diff = gates_diff_buffer_->mutable_cpu_data();

  Dtype* input_gates_diff = gates_diff + channels_ * num_ * 0;
  Dtype* forget_gates_diff = gates_diff + channels_ * num_ * 1;
  Dtype* output_gates_diff = gates_diff + channels_ * num_ * 2;
  Dtype* input_values_diff = gates_diff + channels_ * num_ * 3;
  Dtype* tanh_next_memory_diff = tanh_mem_buffer_->mutable_cpu_diff();

  for (int n = 0; n < num_; ++n) {
    for (int i = 0; i < channels_; ++i) {
      const int idx = i + n * channels_;
      input_gates_diff[idx] = sigmoid_diff(input_gates[idx]);
      forget_gates_diff[idx] = sigmoid_diff(forget_gates[idx]);
      output_gates_diff[idx] = sigmoid_diff(output_gates[idx]);
      input_values_diff[idx] = tanh_diff(input_values[idx]);
      if (this->layer_param_.lstm_unit_param().tanh_hidden()) {
        tanh_next_memory_diff[idx] = tanh_diff(tanh_next_memory_state[idx]);
      } else {
        tanh_next_memory_diff[idx] = Dtype(1.);
      }
    }
  }

  Dtype* input_weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* input_gate_weight_diff = this->blobs_[1]->mutable_cpu_diff();
  Dtype* forget_gate_weight_diff = this->blobs_[2]->mutable_cpu_diff();
  Dtype* output_gate_weight_diff = this->blobs_[3]->mutable_cpu_diff();

  Dtype* input_diff = bottom[0]->mutable_cpu_diff();
  Dtype* prev_state_diff = bottom[1]->mutable_cpu_diff();

  const Dtype* next_hidden_state_diff = top[0]->cpu_diff();
  const Dtype* next_memory_state_diff = top[1]->cpu_diff();

  Dtype* next_state_tot_diff = next_state_tot_diff_buffer_->mutable_cpu_data();
  caffe_mul(num_ * channels_, output_gates,
    next_hidden_state_diff, next_state_tot_diff);
  caffe_mul(num_ * channels_, tanh_next_memory_diff,
    next_state_tot_diff, next_state_tot_diff);
  caffe_add(num_ * channels_, next_memory_state_diff,
    next_state_tot_diff, next_state_tot_diff);

  caffe_mul(num_ * channels_, next_state_tot_diff,
    forget_gates, prev_state_diff);

  Dtype* dldg_data = dldg_buffer_->mutable_cpu_data();

  caffe_mul(num_ * channels_, input_gates, input_values_diff, dldg_data);
  caffe_mul(num_ * channels_, next_state_tot_diff, dldg_data, dldg_data);
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    channels_, input_data_size_, num_,
    (Dtype)1., dldg_data, input_data,
    (Dtype)1., input_weight_diff);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    num_, input_data_size_, channels_,
    (Dtype)1., dldg_data, input_weight,
    (Dtype)1., input_diff);

  caffe_mul(num_ * channels_, input_gates_diff, input_values, dldg_data);
  caffe_mul(num_ * channels_, next_state_tot_diff, dldg_data, dldg_data);
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    channels_, input_data_size_, num_,
    (Dtype)1., dldg_data, input_data,
    (Dtype)1., input_gate_weight_diff);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    num_, input_data_size_, channels_,
    (Dtype)1., dldg_data, input_gate_weight,
    (Dtype)1., input_diff);

  if (this->layer_param_.lstm_unit_param().tie_output_forget()) {
    caffe_mul(num_ * channels_, forget_gates_diff, prev_state_data, dldg_data);
    caffe_mul(num_ * channels_, next_state_tot_diff, dldg_data, dldg_data);
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
      channels_, input_data_size_, num_,
      (Dtype)-1., dldg_data, input_data,
      (Dtype)1., output_gate_weight_diff);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      num_, input_data_size_, channels_,
      (Dtype)-1., dldg_data, output_gate_weight,
      (Dtype)1., input_diff);
  } else {
    caffe_mul(num_ * channels_, forget_gates_diff, prev_state_data, dldg_data);
    caffe_mul(num_ * channels_, next_state_tot_diff, dldg_data, dldg_data);
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
      channels_, input_data_size_, num_,
      (Dtype)1., dldg_data, input_data,
      (Dtype)1., forget_gate_weight_diff);
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
      num_, input_data_size_, channels_,
      (Dtype)1., dldg_data, forget_gate_weight,
      (Dtype)1., input_diff);
  }

  caffe_mul(num_ * channels_, output_gates_diff, tanh_next_memory_state,
      dldg_data);
  caffe_mul(num_ * channels_, next_hidden_state_diff, dldg_data, dldg_data);
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
    channels_, input_data_size_, num_,
    (Dtype)1., dldg_data, input_data,
    (Dtype)1., output_gate_weight_diff);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
    num_, input_data_size_, channels_,
    (Dtype)1., dldg_data, output_gate_weight,
    (Dtype)1., input_diff);
}

#ifdef CPU_ONLY
STUB_GPU(LstmUnitLayer);
#endif

INSTANTIATE_CLASS(LstmUnitLayer);
REGISTER_LAYER_CLASS(LstmUnit);

}  // namespace caffe
