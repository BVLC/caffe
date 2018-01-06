#include <vector>

#include "caffe/layers/tile_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void TileLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const TileParameter& tile_param = this->layer_param_.tile_param();
  axis_ = bottom[0]->CanonicalAxisIndex(tile_param.axis());
  CHECK(tile_param.has_tiles()) << "Number of tiles must be specified";
  tiles_ = tile_param.tiles();
  CHECK_GT(tiles_, 0) << "Number of tiles must be positive.";
  vector<int_tp> top_shape = bottom[0]->shape();
  top_shape[axis_] = bottom[0]->shape(axis_) * tiles_;
  top[0]->Reshape(top_shape);
  outer_dim_ = bottom[0]->count(0, axis_);
  inner_dim_ = bottom[0]->count(axis_);

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void TileLayer<Dtype, MItype, MOtype>::Forward_cpu(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int_tp i = 0; i < outer_dim_; ++i) {
    for (int_tp t = 0; t < tiles_; ++t) {
      caffe_copy(inner_dim_, bottom_data, top_data);
      top_data += inner_dim_;
    }
    bottom_data += inner_dim_;
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void TileLayer<Dtype, MItype, MOtype>::Backward_cpu(
    const vector<Blob<MOtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<MItype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int_tp i = 0; i < outer_dim_; ++i) {
    caffe_copy(inner_dim_, top_diff, bottom_diff);
    top_diff += inner_dim_;
    for (int_tp t = 1; t < tiles_; ++t) {
      caffe_axpy(inner_dim_, Dtype(1), top_diff, bottom_diff);
      top_diff += inner_dim_;
    }
    bottom_diff += inner_dim_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(TileLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(TileLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(TileLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(TileLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Tile);
REGISTER_LAYER_CLASS_INST(Tile, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Tile, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Tile, (double), (double), (double));

}  // namespace caffe
