#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/crop_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CropLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the data
  // bottom[1] supplies the size
  const CropParameter& param = this->layer_param_.crop_param();
  CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
  int_tp input_dim = bottom[0]->num_axes();
  const int_tp start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_LT(start_axis, input_dim) << "crop axis bigger than input dim";
  if (param.offset_size() > 1) {
    // the number of crop values specified must be equal to the number
    // of dimensions following axis
    CHECK_EQ(start_axis + param.offset_size(), input_dim)
      << "number of offset values specified must be equal to the number of "
      << "dimensions following axis.";
  }
  this->InitializeQuantizers(bottom, top);
}

template<typename Dtype, typename MItype, typename MOtype>
void CropLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  const CropParameter& param = this->layer_param_.crop_param();
  int_tp input_dim = bottom[0]->num_axes();
  const int_tp start_axis = bottom[0]->CanonicalAxisIndex(param.axis());

  // Initialize offsets to 0 and the new shape to the current shape of the data.
  vector<int_tp> new_shape(bottom[0]->shape());
  vector<int_tp> offsets_shape(1, input_dim);
  offsets.Reshape(offsets_shape);
  int_tp* offset_data = offsets.mutable_cpu_data();
  // Determine crop offsets and the new shape post-crop.
  for (int_tp i = 0; i < input_dim; ++i) {
    int_tp crop_offset = 0;
    int_tp new_size = bottom[0]->shape(i);
    if (i >= start_axis) {
      new_size = bottom[1]->shape(i);
      if (param.offset_size() == 1) {
        // If only one offset is given, all crops have the same offset.
        crop_offset = param.offset(0);
      } else if (param.offset_size() > 1) {
        // For several offsets, the number of offsets must be equal to the
        // number of dimensions to crop, that is dimensions after the axis.
        crop_offset = param.offset(i - start_axis);
      }
      // Check that the crop and offset are within the dimension's bounds.
      CHECK_GE(bottom[0]->shape(i) - crop_offset, bottom[1]->shape(i))
          << "the crop for dimension " << i << " is out-of-bounds with "
          << "size " << bottom[1]->shape(i) << " and offset " << crop_offset;
    }
    new_shape[i] = new_size;
    offset_data[i] = crop_offset;
  }
  top[0]->Reshape(new_shape);
  // Compute strides
  src_strides_.Reshape(offsets_shape);
  dst_strides_.Reshape(offsets_shape);
  for (int_tp i = 0; i < input_dim; ++i) {
    src_strides_.mutable_cpu_data()[i] = bottom[0]->count(i + 1, input_dim);
    dst_strides_.mutable_cpu_data()[i] = top[0]->count(i + 1, input_dim);
  }

  if (Caffe::mode() == Caffe::GPU && this->device_program_.get() == nullptr) {
    this->GenerateProgram();
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void CropLayer<Dtype, MItype, MOtype>::crop_copy(const vector<Blob<MItype>*>& bottom,
             const vector<Blob<MOtype>*>& top,
             const int_tp* offsets,
             vector<int_tp> indices,
             int_tp cur_dim,
             const Dtype* src_data,
             Dtype* dest_data,
             bool is_forward) {
  if (cur_dim + 1 < top[0]->num_axes()) {
    // We are not yet at the final dimension, call copy recursively
    for (int_tp i = 0; i < top[0]->shape(cur_dim); ++i) {
      indices[cur_dim] = i;
      crop_copy(bottom, top, offsets, indices, cur_dim+1,
                src_data, dest_data, is_forward);
    }
  } else {
    // We are at the last dimensions, which is stored continuously in memory
    // prepare index vector reduced(red) and with offsets(off)
    vector<int_tp> ind_red(cur_dim, 0);
    vector<int_tp> ind_off(cur_dim+1, 0);
    for (int_tp j = 0; j < cur_dim; ++j) {
      ind_red[j] = indices[j];
      ind_off[j] = indices[j] + offsets[j];
    }
    ind_off[cur_dim] = offsets[cur_dim];
    // do the copy
    if (is_forward) {
      caffe_copy(top[0]->shape(cur_dim),
          src_data + bottom[0]->offset(ind_off),
          dest_data + top[0]->offset(ind_red));
    } else {
      // in the backwards pass the src_data is top_diff
      // and the dest_data is bottom_diff
      caffe_copy(top[0]->shape(cur_dim),
          src_data + top[0]->offset(ind_red),
          dest_data + bottom[0]->offset(ind_off));
    }
  }
}

template<typename Dtype, typename MItype, typename MOtype>
void CropLayer<Dtype, MItype, MOtype>::Forward_cpu(const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  vector<int_tp> indices(top[0]->num_axes(), 0);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  crop_copy(bottom, top, offsets.cpu_data(), indices, 0, bottom_data, top_data,
      true);
}

template<typename Dtype, typename MItype, typename MOtype>
void CropLayer<Dtype, MItype, MOtype>::Backward_cpu(const vector<Blob<MOtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<MItype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    vector<int_tp> indices(top[0]->num_axes(), 0);
    crop_copy(bottom, top, offsets.cpu_data(), indices, 0, top_diff,
        bottom_diff, false);
  }
}

#ifdef CPU_ONLY
STUB_GPU(CropLayer);
#endif

INSTANTIATE_CLASS_3T_GUARDED(CropLayer, (half_fp), (half_fp), (half_fp));
INSTANTIATE_CLASS_3T_GUARDED(CropLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(CropLayer, (double), (double), (double));

REGISTER_LAYER_CLASS(Crop);
REGISTER_LAYER_CLASS_INST(Crop, (half_fp), (half_fp), (half_fp));
REGISTER_LAYER_CLASS_INST(Crop, (float), (float), (float));
REGISTER_LAYER_CLASS_INST(Crop, (double), (double), (double));

}  // namespace caffe
