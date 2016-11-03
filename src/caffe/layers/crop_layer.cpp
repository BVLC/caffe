/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/crop_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

template <typename Dtype>
void CropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
  // bottom[0] supplies the data
  // bottom[1] supplies the size
  const CropParameter& param = this->layer_param_.crop_param();
  CHECK_EQ(bottom.size(), 2) << "Wrong number of bottom blobs.";
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_LT(start_axis, input_dim) << "crop axis bigger than input dim";
  if (param.offset_size() > 1) {
    // the number of crop values specified must be equal to the number
    // of dimensions following axis
    CHECK_EQ(start_axis + param.offset_size(), input_dim)
      << "number of offset values specified must be equal to the number of "
      << "dimensions following axis.";
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const CropParameter& param = this->layer_param_.crop_param();
  int input_dim = bottom[0]->num_axes();
  const int start_axis = bottom[0]->CanonicalAxisIndex(param.axis());

  // Initialize offsets to 0 and the new shape to the current shape of the data.
  offsets = vector<int>(input_dim, 0);
  vector<int> new_shape(bottom[0]->shape());

  // Determine crop offsets and the new shape post-crop.
  for (int i = 0; i < input_dim; ++i) {
    int crop_offset = 0;
    int new_size = bottom[0]->shape(i);
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
    offsets[i] = crop_offset;
  }
  top[0]->Reshape(new_shape);
}

template <typename Dtype>
template <bool is_forward>
void CropLayer<Dtype>::crop_copy(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top,
             const vector<int>& offsets,
             const Dtype* src_data,
             Dtype* dest_data) {
  int last_dim = top[0]->num_axes() - 1;
  int copy_count = top[0]->count() / top[0]->shape(last_dim);

#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int i = 0; i < copy_count; ++i) {
    // prepare index vector reduced(red) and with offsets(off)
    std::vector<int> ind_red(last_dim, 0);
    std::vector<int> ind_off(last_dim+1, 0);
    int cur_iteration = i;
    for (int j = last_dim - 1; j >=0; --j) {
      int index = cur_iteration % top[0]->shape(j);
      cur_iteration /= top[0]->shape(j);
      ind_red[j] = index;
      ind_off[j] = index + offsets[j];
    }
    ind_off[last_dim] = offsets[last_dim];
    // Last dimensions stored continously in memory
    // do the copy
    if (is_forward) {
      caffe_copy(top[0]->shape(last_dim),
          src_data + bottom[0]->offset(ind_off),
          dest_data + top[0]->offset(ind_red));
    } else {
      // in the backwards pass the src_data is top_diff
      // and the dest_data is bottom_diff
      caffe_copy(top[0]->shape(last_dim),
          src_data + top[0]->offset(ind_red),
          dest_data + bottom[0]->offset(ind_off));
    }
  }
}

template <typename Dtype>
void CropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  crop_copy<true>(bottom, top, offsets, bottom_data, top_data);
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  if (propagate_down[0]) {
    caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
    crop_copy<false>(bottom, top, offsets, top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(CropLayer);
#endif

INSTANTIATE_CLASS(CropLayer);
REGISTER_LAYER_CLASS(Crop);

}  // namespace caffe
