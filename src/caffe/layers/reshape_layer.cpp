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

#include <vector>

#include "caffe/layers/reshape_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  const int ps = this->layer_param_.reshape_param().pixelshuffler();
  if(ps == 1){
    inferred_axis_ = -1;
    copy_axes_.clear();
    const BlobShape& top_blob_shape = this->layer_param_.reshape_param().shape();
    const int top_num_axes = top_blob_shape.dim_size();
    constant_count_ = 1;
    for (int i = 0; i < top_num_axes; ++i) {
      const int top_dim = top_blob_shape.dim(i);
      if (top_dim == 0) {
        copy_axes_.push_back(i);
      } else if (top_dim == -1) {
        CHECK_EQ(inferred_axis_, -1) << "new shape contains multiple "
            << "-1 dims; at most a single (1) value of -1 may be specified";
        inferred_axis_ = i;
      } else {
        constant_count_ *= top_dim;
      }
   }
  }
  
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int ps = this->layer_param_.reshape_param().pixelshuffler();   
  if (ps ==1 ){
        const int input_start_axis = this->layer_param_.reshape_param().axis();
        const int start_axis = (input_start_axis >= 0) ? input_start_axis :
            bottom[0]->num_axes() + input_start_axis + 1;
        CHECK_GE(start_axis, 0) << "axis " << input_start_axis << " out of range";
        CHECK_LE(start_axis, bottom[0]->num_axes()) << "axis " << input_start_axis
            << " out of range for " << bottom[0]->num_axes() << "-D input blob";
        const int num_axes = this->layer_param_.reshape_param().num_axes();
        CHECK_GE(num_axes, -1) << "num_axes must be >= 0, or -1 for all";
        const int end_axis =
            (num_axes == -1) ? bottom[0]->num_axes() : (start_axis + num_axes);
        CHECK_LE(end_axis, bottom[0]->num_axes())
            << "end_axis = axis + num_axes is out of range";
        const int num_axes_replaced = end_axis - start_axis;
        const int num_axes_retained = bottom[0]->num_axes() - num_axes_replaced;
        const BlobShape& top_blob_shape = this->layer_param_.reshape_param().shape();
        const int num_new_axes = top_blob_shape.dim_size();
        vector<int> top_shape(num_axes_retained + num_new_axes);
        int top_shape_index = 0;
        for (int i = 0; i < start_axis; ++i) {
          top_shape[top_shape_index++] = bottom[0]->shape(i);
        }
        for (int i = 0; i < num_new_axes; ++i) {
          top_shape[top_shape_index++] = top_blob_shape.dim(i);
        }
        for (int i = end_axis; i < bottom[0]->num_axes(); ++i) {
          top_shape[top_shape_index++] = bottom[0]->shape(i);
        }
        CHECK_EQ(top_shape_index, top_shape.size());
        for (int i = 0; i < copy_axes_.size(); ++i) {
          const int copy_axis_index = copy_axes_[i];
          CHECK_GT(bottom[0]->num_axes(), start_axis + copy_axis_index)
              << "new shape contains a 0, but there was no corresponding bottom axis "
              << "to copy";
          top_shape[start_axis + copy_axis_index] =
              bottom[0]->shape(start_axis + copy_axis_index);
        }
        if (inferred_axis_ >= 0) {
          // A -1 dim was specified; infer the correct dimension by computing the
          // product of the other dimensions.
          int explicit_count = constant_count_;
          explicit_count *= bottom[0]->count(0, start_axis);
          explicit_count *= bottom[0]->count(end_axis);
          for (int i = 0; i < copy_axes_.size(); ++i) {
            const int copy_axis_index = copy_axes_[i];
            explicit_count *= top_shape[start_axis + copy_axis_index];
          }
          CHECK_EQ(0, bottom[0]->count() % explicit_count) << "bottom count ("
              << bottom[0]->count() << ") must be divisible by the product of "
              << "the specified dimensions (" << explicit_count << ")";
          const int inferred_dim = bottom[0]->count() / explicit_count;
          top_shape[start_axis + inferred_axis_] = inferred_dim;
        }
        top[0]->Reshape(top_shape);
        CHECK_EQ(top[0]->count(), bottom[0]->count())
            << "output count must match input count";
        top[0]->ShareData(*bottom[0]);
        top[0]->ShareDiff(*bottom[0]);
  }
  else {

    // Input: (N, C * r^2, H, W)
    // Output: (N, C, H * r, W * r)
    // Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
    // https://arxiv.org/abs/1609.05158
    vector<int> bottom_shape = bottom[0]->shape();
    const int bn = bottom_shape[0];
    const int bc = bottom_shape[1];
    const int bh = bottom_shape[2];
    const int bw = bottom_shape[3];
    vector<int> top_shape(4);
    top_shape[0] = bn;
    top_shape[1] = bc/(ps * ps);
    top_shape[2] = bh * ps;
    top_shape[3] = bw * ps;
    top[0]->Reshape(top_shape);
    CHECK_EQ(top[0]->count(), bottom[0]->count())
            << "output count must match input count";
  }

}

template <typename Dtype>
void ReshapeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  const int ps = this->layer_param_.reshape_param().pixelshuffler();
  if( ps == 1){}
  else{
    vector<int> bottom_shape = bottom[0]->shape();
    const int bn = bottom_shape[0];
    const int bc = bottom_shape[1];
    const int bh = bottom_shape[2];
    const int bw = bottom_shape[3];
    //const int tn = top_shape[0];
    vector<int> top_shape = top[0]->shape();
    const int tc = top_shape[1];
    const int th = top_shape[2];
    const int tw = top_shape[3];

    int test_r1 = bc/tc;
    const int r = th / bh;
    int test_r2 = r * r;
    CHECK_EQ( test_r1, test_r2) << "Pixelshuffler output is illegal";
    
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    #ifdef _OPENMP
        #if defined(_MSC_EXTENSIONS)
            #pragma omp parallel for
        #else
            #pragma omp parallel for collapse(2)
        #endif
    #endif
    for(int n = 0; n < bn; n++) {
      for(int c = 0; c < bc; c++) {
        for(int h = 0; h < bh; h++) {
          for(int w = 0; w < bw; w++) {
              int new_c = static_cast<int>(floor(c/(r*r)));
              int new_h = h*r + (static_cast<int>(floor(c/r)))%r;
              int new_w = w*r+ (c%(r*r))%r;
              int top_index = n*(tc*th*tw)+ new_c*(th*tw)+ new_h*tw+ new_w;
              int bottom_index = n*bc*bh*bw + c*(bh*bw) + h*bw + w;
              top_data[top_index] = bottom_data[bottom_index];
          }
        }
      }
    }
  }
}

template <typename Dtype>
void ReshapeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  const int ps = this->layer_param_.reshape_param().pixelshuffler();
  if( ps == 1){}
  else{
    vector<int> top_shape = top[0]->shape();
    const int tn = top_shape[0];    
    const int tc = top_shape[1];
    const int th = top_shape[2];
    const int tw = top_shape[3];
    vector<int> bottom_shape = bottom[0]->shape();
    const int bc = bottom_shape[1];
    const int bh = bottom_shape[2];
    const int bw = bottom_shape[3];

    //int test_r1 = bc/tc;
    const int r = th / bh;
    //int test_r2 = r * r;
    //CHECK_EQ( test_r1, test_r2) << "Pixelshuffler output is illegal";
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();

    #ifdef _OPENMP
        #if defined(_MSC_EXTENSIONS)
            #pragma omp parallel for
        #else
            #pragma omp parallel for collapse(2)
        #endif
    #endif
    for(int n = 0; n < tn; n++) {
      for(int c = 0; c < tc; c++) {
        for(int h = 0; h < th; h++) {
          for(int w = 0; w < tw; w++) {
              int old_c = c*r*r + (h%r)*r + w%r;
              int old_h = static_cast<int>(floor(h/r));
              int old_w = static_cast<int>(floor(w/r));
              int bottom_index = n*(bc*bh*bw)+ old_c*(bh*bw)+ old_h*bw+ old_w;
              int top_index = n*(tc*th*tw)+ c*(th*tw)+ h*tw+ w;
              bottom_diff[bottom_index] = top_diff[top_index];
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(ReshapeLayer);
REGISTER_LAYER_CLASS(Reshape);

}  // namespace caffe
