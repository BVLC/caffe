#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  //find channel axis and compute spatial axes constants
  channel_axis_ = bottom[0]->CanonicalAxisIndex(pool_param.axis());
  channels_ = bottom[0]->shape(channel_axis_);
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 1);

  // Setup input dimensions (input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  input_shape_.Reshape(bottom_dim_blob_shape);
  int* input_shape_data = input_shape_.mutable_cpu_data();
  // printf("input_shape_data channel_axis_:%d ::",channel_axis_);
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
     // printf("%d ",input_shape_data[i]);
  }
   // printf("\n");
  vector<int> spatial_dim_blob_shape(1, num_spatial_axes_);

  global_pooling_ = pool_param.global_pooling();
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (global_pooling_) {
    //if global pooling height and width are set the entire blob, 
    //and the layer cannot have a kernel set
    CHECK_GE(0, pool_param.kernel_size_size())
        << "With Global_pooling: true Filter size cannot specified.";
    CHECK(!pool_param.has_kernel_h() || !pool_param.has_kernel_w())
        << "With Global_pooling: true Filter size cannot specified.";
    for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
      kernel_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  } else {
     //if kernel_h or kernel_w are set we cannot set the kernel another way
     //And there must be 2 spatial dims
    if (pool_param.has_kernel_h() || pool_param.has_kernel_w()) {
        CHECK_EQ(num_spatial_axes_, 2)
            << "kernel_h & kernel_w can only be used for 2D pooling.";
        CHECK_EQ(0, pool_param.kernel_size_size())
            << "Either kernel_size or kernel_h/w should be specified; not both.";
        kernel_shape_data[0] = pool_param.kernel_h();
        kernel_shape_data[1] = pool_param.kernel_w();
      } else {
        //using repeated kernel param
        const int num_kernel_dims = pool_param.kernel_size_size();
        CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
            << "kernel_size must be specified once, or once per spatial dimension "
            << "(kernel_size specified " << num_kernel_dims << " times; "
            << num_spatial_axes_ << " spatial dims);";
        for (int i = 0; i < num_spatial_axes_; ++i) {
            kernel_shape_data[i] =
                pool_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
          }
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
      CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }

  // setup padding dimensions (pad_)
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  int pad_sum = 0;
  if (pool_param.has_pad_h() || pool_param.has_pad_w()) {
      //if pad_h or pad_w are set we cannot set the pad another way
      //And there must be 2 spatial dims
      CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
      CHECK_EQ(0, pool_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
      pad_data[0] = pool_param.pad_h();
      pad_data[1] = pool_param.pad_w();
  } else {
    //using repeated pad param
    const int num_pad_dims = pool_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          pool_param.pad((num_pad_dims == 1) ? 0 : i);
      if (global_pooling_) {
          CHECK(pad_data[i] == 0)
            << "With Global_pooling: true; pool = 0";
        }
      CHECK_LT(pad_data[i], kernel_shape_data[i]);
      pad_sum += pad_data[i];
    }
  }
  if (pad_sum != 0 ) {
     CHECK(this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
      }

// Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (pool_param.has_stride_h() || pool_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, pool_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = pool_param.stride_h();
    stride_data[1] = pool_param.stride_w();
  } else {
    //using repeated stride param
    const int num_stride_dims = pool_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          pool_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
      if (global_pooling_) {
        CHECK(stride_data[i] == 1)
          << "With Global_pooling: true; stride = 1";
      }
    }
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(pool_param.axis());
  num_ = bottom[0]->count(0, channel_axis_);
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 1);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  const int* input_shape_data = this->input_shape_.cpu_data();
  if (global_pooling_) {
    for (int i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] = input_shape_data[i+1];
    }
  }
  // printf("num:%d\n",num_spatial_axes_);
  //compute output shape
  const int* pad_data = this->pad_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  
  vector<int> spatial_dim_blob_shape(1, num_spatial_axes_);
  output_shape_.Reshape(spatial_dim_blob_shape);
  int* output_shape_data = output_shape_.mutable_cpu_data();
  int pad_sum = 0;
  for (int i = 0; i < num_spatial_axes_; ++i) {    
    int oc = static_cast<int>(ceil(static_cast<float>(
          input_shape_data[i+1] + 2 * pad_data[i] - kernel_shape_data[i]) / stride_data[i])) + 1;
    pad_sum += pad_data[i];
    output_shape_data[i] = oc;
    // printf("output_shape_ %d\n\n",output_shape_[i]);
  }
  if (pad_sum){
    for(int i = 0; i < num_spatial_axes_; ++i){
        if ( (output_shape_data[i] - 1) * stride_data[i] >= input_shape_data[i+1] + pad_data[i] )
            --output_shape_data[i];
        CHECK_LT((output_shape_data[i] - 1) * stride_data[i],input_shape_data[i+1] + pad_data[i]);
    }
  } 

  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(first_spatial_axis);  // Discard input spatial axes.
  for (int i = 0; i < num_spatial_axes_; ++i) {
      top_shape.push_back(output_shape_data[i]);
  }
  // for (int i = 0; i < top_shape.size(); ++i) {
  //     printf("%d ",top_shape[i]);
  // }
  // printf("\n");
  top[0]->Reshape(top_shape);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }

  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(top_shape);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(top_shape);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  vector<int> offset(2,0);
  offset[1] = 1;

  const int* kernel_shape = kernel_shape_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* input_shape_data = this->input_shape_.cpu_data();
  const int* output_shape_data = this->output_shape_.cpu_data();

  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop

    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
        if (num_spatial_axes_ == 2) {
          for (int ph = 0; ph < output_shape_data[0]; ++ph) {
            for (int pw = 0; pw < output_shape_data[1]; ++pw) {
              int hstart = ph * stride_data[0] - pad_data[0];
              int wstart = pw * stride_data[1] - pad_data[1];
              int hend = min(hstart + kernel_shape[0], input_shape_data[1]);
              int wend = min(wstart + kernel_shape[1], input_shape_data[2]);
              hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              const int pool_index = ph * output_shape_data[1] + pw;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * input_shape_data[2] + w;
                  if (bottom_data[index] > top_data[pool_index]) {
                    top_data[pool_index] = bottom_data[index];
                    if (use_top_mask) {
                      top_mask[pool_index] = static_cast<Dtype>(index);
                    } else {
                      mask[pool_index] = index;
                    }
                  }
                }
              }
            }
          }
        } else if (num_spatial_axes_ == 3) {
          for (int ph = 0; ph < output_shape_data[0]; ++ph) {
            for (int pw = 0; pw < output_shape_data[1]; ++pw) {
              for (int pz = 0; pz< output_shape_data[2]; ++pz) {
                int hstart = ph * stride_data[0] - pad_data[0];
                int wstart = pw * stride_data[1] - pad_data[1];
                int zstart = pz * stride_data[2] - pad_data[2];
                int hend = min(hstart + kernel_shape[0], input_shape_data[1]);
                int wend = min(wstart + kernel_shape[1], input_shape_data[2]);
                int zend = min(zstart + kernel_shape[2], input_shape_data[3]);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                zstart = max(zstart, 0);
                const int pool_index = (ph * output_shape_data[1] + pw)*output_shape_data[2] +pz;
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    for (int z = zstart; z < zend; ++z) {
                      const int index = (h * input_shape_data[2] + w)*input_shape_data[3]+z;
                      if (bottom_data[index] > top_data[pool_index]) {
                        top_data[pool_index] = bottom_data[index];
                        if (use_top_mask) {
                          top_mask[pool_index] = static_cast<Dtype>(index);
                        } else {
                          mask[pool_index] = index;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        } else {
          NOT_IMPLEMENTED;
        }
        // compute offset
        if (num_ > 1 || channels_ > 1) {
          bottom_data += bottom[0]->offset(offset);
          top_data += top[0]->offset(offset);
          if (use_top_mask) {
            top_mask += top[0]->offset(offset);
          } else {
            mask += top[0]->offset(offset);
          }
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    // printf ("Num axes: %d\n",num_spatial_axes_);
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
          if (num_spatial_axes_ == 2) {
            for (int ph = 0; ph < output_shape_data[0]; ++ph) {
              for (int pw = 0; pw < output_shape_data[1]; ++pw) {
                int hstart = ph * stride_data[0] - pad_data[0];
                int wstart = pw * stride_data[1] - pad_data[1];
                int hend = min(hstart + kernel_shape[0], input_shape_data[1] + pad_data[0]);
                int wend = min(wstart + kernel_shape[1], input_shape_data[2] + pad_data[1]);
                int pool_size = (hend - hstart) * (wend - wstart);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                hend = min(hend,input_shape_data[1]);
                wend = min(wend,input_shape_data[2]);
                

                const int pool_index = ph * output_shape_data[1] + pw;
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    const int index = h * input_shape_data[2] + w;
                    top_data[pool_index] += bottom_data[index];
                  }
                }
                top_data[pool_index] /= pool_size;
              }
            }
        } else if (num_spatial_axes_ == 3) {
            for (int ph = 0; ph < output_shape_data[0]; ++ph) {
              for (int pw = 0; pw < output_shape_data[1]; ++pw) {
                for (int pz = 0; pz< output_shape_data[2]; ++pz) {
                  int hstart = ph * stride_data[0] - pad_data[0];
                  int wstart = pw * stride_data[1] - pad_data[1];
                  int zstart = pz * stride_data[2] - pad_data[2];
                  int hend = min(hstart + kernel_shape[0], input_shape_data[1]+ pad_data[0]);
                  int wend = min(wstart + kernel_shape[1], input_shape_data[2]+ pad_data[1]);
                  int zend = min(zstart + kernel_shape[2], input_shape_data[3]+ pad_data[2]);
                  int pool_size = (hend - hstart) * (wend - wstart) * (zend - zstart);
                  hstart = max(hstart, 0);
                  wstart = max(wstart, 0);
                  zstart = max(zstart, 0);
                  hend = min(hend,input_shape_data[1]);
                  wend = min(wend,input_shape_data[2]);
                  zend = min(zend,input_shape_data[3]);

                  const int pool_index = (ph * output_shape_data[1] + pw)*output_shape_data[2] +pz;
                  //printf("n:%d c:%d ph:%d pw:%d pz:%d\n",n,c,ph,pw,pz);
                  
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      for (int z = zstart; z < zend; ++z) {
                        const int index = (h * input_shape_data[2] + w)*input_shape_data[3]+z;
                        top_data[pool_index] += bottom_data[index];
                      }
                    }
                  }
                  top_data[pool_index] /= pool_size;
                }
              }
            }
        } else {
          NOT_IMPLEMENTED;
        }
        // compute offset
        if (num_ > 1 || channels_ > 1) {
          bottom_data += bottom[0]->offset(offset);
          top_data += top[0]->offset(offset);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;

  const int* kernel_shape = this->kernel_shape_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* input_shape_data = this->input_shape_.cpu_data();
  const int* output_shape_data = this->output_shape_.cpu_data();
  int top_num = top[0]->count(0, channel_axis_);
  vector<int> offset(2,0);
  offset[1] = 1;
  //printf("check %d==%d\n",top_num,top[0]->num());
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top_num; ++n) {
      for (int c = 0; c < channels_; ++c) {
        if (num_spatial_axes_ == 2) {
          for (int ph = 0; ph < output_shape_data[0]; ++ph) {
            for (int pw = 0; pw < output_shape_data[1]; ++pw) {
              const int index = ph * output_shape_data[1] + pw;
              const int bottom_index =
                  use_top_mask ? top_mask[index] : mask[index];
              bottom_diff[bottom_index] += top_diff[index];
            }
          }
        } else if (num_spatial_axes_ == 3) {
            for (int ph = 0; ph < output_shape_data[0]; ++ph) {
              for (int pw = 0; pw < output_shape_data[1]; ++pw) {
                for (int pz = 0; pz < output_shape_data[2]; ++pz) {
                const int index = (ph * output_shape_data[1] + pw)*output_shape_data[2] +pz;
                const int bottom_index =
                    use_top_mask ? top_mask[index] : mask[index];
                bottom_diff[bottom_index] += top_diff[index];
              }
            }
          }
        } else {
          NOT_IMPLEMENTED;
        }
        if (num_ > 1 || channels_ > 1) {
          bottom_diff += bottom[0]->offset(offset);
          top_diff += top[0]->offset(offset);
          if (use_top_mask) {
            top_mask += top[0]->offset(offset);
          } else {
            mask += top[0]->offset(offset);
          }
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top_num; ++n) {
      for (int c = 0; c < channels_; ++c) {
        if (num_spatial_axes_ == 2) {
          for (int ph = 0; ph < output_shape_data[0]; ++ph) {
            for (int pw = 0; pw < output_shape_data[1]; ++pw) {
              int hstart = ph * stride_data[0] - pad_data[0];
              int wstart = pw * stride_data[1] - pad_data[1];
              int hend = min(hstart + kernel_shape[0], input_shape_data[1]+pad_data[0]);
              int wend = min(wstart + kernel_shape[1], input_shape_data[2]+pad_data[1]);

              int pool_size = (hend - hstart) * (wend - wstart);
              hstart = max(hstart, 0);
              wstart = max(wstart, 0);
              hend = min(hend, input_shape_data[1]);
              wend = min(wend, input_shape_data[2]);


              const int pool_index = ph * output_shape_data[1] + pw;
              for (int h = hstart; h < hend; ++h) {
                for (int w = wstart; w < wend; ++w) {
                  const int index = h * input_shape_data[2] + w;
                  
                  bottom_diff[index] +=
                    top_diff[pool_index] / pool_size;
                }
              }
            }
          }
        } else if (num_spatial_axes_ == 3) {
            for (int ph = 0; ph < output_shape_data[0]; ++ph) {
              for (int pw = 0; pw < output_shape_data[1]; ++pw) {
                for (int pz = 0; pz< output_shape_data[2]; ++pz) {
                  int hstart = ph * stride_data[0] - pad_data[0];
                  int wstart = pw * stride_data[1] - pad_data[1];
                  int zstart = pz * stride_data[2] - pad_data[2];
                  int hend = min(hstart + kernel_shape[0], input_shape_data[1]);
                  int wend = min(wstart + kernel_shape[1], input_shape_data[2]);
                  int zend = min(zstart + kernel_shape[2], input_shape_data[3]);
                  hstart = max(hstart, 0);
                  wstart = max(wstart, 0);
                  zstart = max(zstart, 0);
                  const int pool_index = (ph * output_shape_data[1] + pw)*output_shape_data[2] +pz;
                  int pool_size = (hend - hstart) * (wend - wstart) * (zend - zstart);

                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      for (int z = zstart; z < zend; ++z) {
                        const int index = (h * input_shape_data[2] + w)*input_shape_data[3]+z;
                        bottom_diff[index] +=
                                      top_diff[pool_index] / pool_size;
                      }
                    }
                  }
                }
              }
            }
        } else {
          NOT_IMPLEMENTED;
        }

        // offset
        if (num_ > 1 || channels_ > 1) {
          bottom_diff += bottom[0]->offset(offset);
          top_diff += top[0]->offset(offset);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
