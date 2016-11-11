#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/rnn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RNNLayer<Dtype>::RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h_0";
}

template <typename Dtype>
void RNNLayer<Dtype>::RecurrentOutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "h_" + format_int(this->T_);
}

template <typename Dtype>
void RNNLayer<Dtype>::RecurrentInputShapes(vector<BlobShape>* shapes) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  shapes->resize(1);
  (*shapes)[0].Clear();
  (*shapes)[0].add_dim(1);  // a single timestep
  (*shapes)[0].add_dim(this->N_);
  (*shapes)[0].add_dim(num_output);
}

template <typename Dtype>
void RNNLayer<Dtype>::OutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "o";
}

template <typename Dtype>
void RNNLayer<Dtype>::FillUnrolledNet(NetParameter* net_param) const {
  const int num_output = this->layer_param_.recurrent_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";
  const FillerParameter& weight_filler =
      this->layer_param_.recurrent_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.recurrent_param().bias_filler();

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.
  LayerParameter hidden_param;
  hidden_param.set_type("InnerProduct");
  hidden_param.mutable_inner_product_param()->set_num_output(num_output);
  hidden_param.mutable_inner_product_param()->set_bias_term(false);
  hidden_param.mutable_inner_product_param()->set_axis(2);
  hidden_param.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  LayerParameter biased_hidden_param(hidden_param);
  biased_hidden_param.mutable_inner_product_param()->set_bias_term(true);
  biased_hidden_param.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter sum_param;
  sum_param.set_type("Eltwise");
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);

  LayerParameter tanh_param;
  tanh_param.set_type("TanH");

  LayerParameter scale_param;
  scale_param.set_type("Scale");
  scale_param.mutable_scale_param()->set_axis(0);

  LayerParameter slice_param;
  slice_param.set_type("Slice");
  slice_param.mutable_slice_param()->set_axis(0);

  vector<BlobShape> input_shapes;
  RecurrentInputShapes(&input_shapes);
  CHECK_EQ(1, input_shapes.size());

  LayerParameter* input_layer_param = net_param->add_layer();
  input_layer_param->set_type("Input");
  InputParameter* input_param = input_layer_param->mutable_input_param();
  input_layer_param->add_top("h_0");
  input_param->add_shape()->CopyFrom(input_shapes[0]);

  LayerParameter* cont_slice_param = net_param->add_layer();
  cont_slice_param->CopyFrom(slice_param);
  cont_slice_param->set_name("cont_slice");
  cont_slice_param->add_bottom("cont");
  cont_slice_param->mutable_slice_param()->set_axis(0);

  // Add layer to transform all timesteps of x to the hidden state dimension.
  //     W_xh_x = W_xh * x + b_h
  {
    LayerParameter* x_transform_param = net_param->add_layer();
    x_transform_param->CopyFrom(biased_hidden_param);
    x_transform_param->set_name("x_transform");
    x_transform_param->add_param()->set_name("W_xh");
    x_transform_param->add_param()->set_name("b_h");
    x_transform_param->add_bottom("x");
    x_transform_param->add_top("W_xh_x");
    x_transform_param->add_propagate_down(true);
  }

  if (this->static_input_) {
    // Add layer to transform x_static to the hidden state dimension.
    //     W_xh_x_static = W_xh_static * x_static
    LayerParameter* x_static_transform_param = net_param->add_layer();
    x_static_transform_param->CopyFrom(hidden_param);
    x_static_transform_param->mutable_inner_product_param()->set_axis(1);
    x_static_transform_param->set_name("W_xh_x_static");
    x_static_transform_param->add_param()->set_name("W_xh_static");
    x_static_transform_param->add_bottom("x_static");
    x_static_transform_param->add_top("W_xh_x_static_preshape");
    x_static_transform_param->add_propagate_down(true);

    LayerParameter* reshape_param = net_param->add_layer();
    reshape_param->set_type("Reshape");
    BlobShape* new_shape =
         reshape_param->mutable_reshape_param()->mutable_shape();
    new_shape->add_dim(1);  // One timestep.
    // Should infer this->N as the dimension so we can reshape on batch size.
    new_shape->add_dim(-1);
    new_shape->add_dim(
        x_static_transform_param->inner_product_param().num_output());
    reshape_param->set_name("W_xh_x_static_reshape");
    reshape_param->add_bottom("W_xh_x_static_preshape");
    reshape_param->add_top("W_xh_x_static");
  }

  LayerParameter* x_slice_param = net_param->add_layer();
  x_slice_param->CopyFrom(slice_param);
  x_slice_param->set_name("W_xh_x_slice");
  x_slice_param->add_bottom("W_xh_x");

  LayerParameter output_concat_layer;
  output_concat_layer.set_name("o_concat");
  output_concat_layer.set_type("Concat");
  output_concat_layer.add_top("o");
  output_concat_layer.mutable_concat_param()->set_axis(0);

  for (int t = 1; t <= this->T_; ++t) {
    string tm1s = format_int(t - 1);
    string ts = format_int(t);

    cont_slice_param->add_top("cont_" + ts);
    x_slice_param->add_top("W_xh_x_" + ts);

    // Add layer to flush the hidden state when beginning a new sequence,
    // as indicated by cont_t.
    //     h_conted_{t-1} := cont_t * h_{t-1}
    //
    // Normally, cont_t is binary (i.e., 0 or 1), so:
    //     h_conted_{t-1} := h_{t-1} if cont_t == 1
    //                       0   otherwise
    {
      LayerParameter* cont_h_param = net_param->add_layer();
      cont_h_param->CopyFrom(scale_param);
      cont_h_param->set_name("h_conted_" + tm1s);
      cont_h_param->add_bottom("h_" + tm1s);
      cont_h_param->add_bottom("cont_" + ts);
      cont_h_param->add_top("h_conted_" + tm1s);
    }

    // Add layer to compute
    //     W_hh_h_{t-1} := W_hh * h_conted_{t-1}
    {
      LayerParameter* w_param = net_param->add_layer();
      w_param->CopyFrom(hidden_param);
      w_param->set_name("W_hh_h_" + tm1s);
      w_param->add_param()->set_name("W_hh");
      w_param->add_bottom("h_conted_" + tm1s);
      w_param->add_top("W_hh_h_" + tm1s);
      w_param->mutable_inner_product_param()->set_axis(2);
    }

    // Add layers to compute
    //     h_t := \tanh( W_hh * h_conted_{t-1} + W_xh * x_t + b_h )
    //          = \tanh( W_hh_h_{t-1} + W_xh_t )
    {
      LayerParameter* h_input_sum_param = net_param->add_layer();
      h_input_sum_param->CopyFrom(sum_param);
      h_input_sum_param->set_name("h_input_sum_" + ts);
      h_input_sum_param->add_bottom("W_hh_h_" + tm1s);
      h_input_sum_param->add_bottom("W_xh_x_" + ts);
      if (this->static_input_) {
        h_input_sum_param->add_bottom("W_xh_x_static");
      }
      h_input_sum_param->add_top("h_neuron_input_" + ts);
    }
    {
      LayerParameter* h_neuron_param = net_param->add_layer();
      h_neuron_param->CopyFrom(tanh_param);
      h_neuron_param->set_name("h_neuron_" + ts);
      h_neuron_param->add_bottom("h_neuron_input_" + ts);
      h_neuron_param->add_top("h_" + ts);
    }

    // Add layer to compute
    //     W_ho_h_t := W_ho * h_t + b_o
    {
      LayerParameter* w_param = net_param->add_layer();
      w_param->CopyFrom(biased_hidden_param);
      w_param->set_name("W_ho_h_" + ts);
      w_param->add_param()->set_name("W_ho");
      w_param->add_param()->set_name("b_o");
      w_param->add_bottom("h_" + ts);
      w_param->add_top("W_ho_h_" + ts);
      w_param->mutable_inner_product_param()->set_axis(2);
    }

    // Add layers to compute
    //     o_t := \tanh( W_ho h_t + b_o)
    //          = \tanh( W_ho_h_t )
    {
      LayerParameter* o_neuron_param = net_param->add_layer();
      o_neuron_param->CopyFrom(tanh_param);
      o_neuron_param->set_name("o_neuron_" + ts);
      o_neuron_param->add_bottom("W_ho_h_" + ts);
      o_neuron_param->add_top("o_" + ts);
    }
    output_concat_layer.add_bottom("o_" + ts);
  }  // for (int t = 1; t <= this->T_; ++t)

  net_param->add_layer()->CopyFrom(output_concat_layer);
}

INSTANTIATE_CLASS(RNNLayer);
REGISTER_LAYER_CLASS(RNN);

}  // namespace caffe
