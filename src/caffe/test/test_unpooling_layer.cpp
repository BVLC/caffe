#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/unpooling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class UnpoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  UnpoolingLayerTest()
      : pool_blob_bottom_(new Blob<Dtype>()),
        blob_bottom_(new Blob<Dtype>()),
        blob_mask_(new Blob<Dtype>()),
        blob_argmax_count_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    pool_blob_bottom_->Reshape(2, 3, 7, 5);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->pool_blob_bottom_);
    pool_blob_bottom_vec_.push_back(pool_blob_bottom_);
    pool_blob_top_vec_.push_back(blob_bottom_);
    pool_blob_top_vec_.push_back(blob_mask_);
    pool_blob_top_vec_.push_back(blob_argmax_count_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_mask_);
    blob_bottom_vec_.push_back(blob_argmax_count_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~UnpoolingLayerTest() {
    delete pool_blob_bottom_;
    delete blob_bottom_;
    delete blob_mask_;
    delete blob_argmax_count_;
    delete blob_top_;
  }
  Blob<Dtype>* const pool_blob_bottom_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_mask_;
  Blob<Dtype>* const blob_argmax_count_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> pool_blob_bottom_vec_;
  vector<Blob<Dtype>*> pool_blob_top_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;

  class PoolUnpoolLayer : public Layer<Dtype> {
   public:
    explicit PoolUnpoolLayer(const LayerParameter& param)
        : Layer<Dtype>(param),
          pool_layer_(new PoolingLayer<Dtype>(param)),
          unpool_layer_(new UnpoolingLayer<Dtype>(param)) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
      pool_top_vec_.push_back(&pooled_data_);
      unpool_bottom_vec_.push_back(&pooled_data_);
      if (this->layer_param_.pooling_param().pool() ==
          PoolingParameter_PoolMethod_MAX) {
        pool_top_vec_.push_back(&mask_);
        const bool overlapping = this->layer_param_.pooling_param().stride() <
            this->layer_param_.pooling_param().kernel_size();
        if (overlapping) {
          pool_top_vec_.push_back(&argmax_count_);
        }
        unpool_bottom_vec_.push_back(&mask_);
        if (overlapping) {
          unpool_bottom_vec_.push_back(&argmax_count_);
        } else {
          // Push back the pool input blob for shape only.
          unpool_bottom_vec_.push_back(bottom[0]);
        }
      } else {  // pool == PoolingParameter_PoolMethod_AVE
        // Push back the pool input blob for shape only.
        unpool_bottom_vec_.push_back(bottom[0]);
      }
      pool_layer_->SetUp(bottom, pool_top_vec_);
      unpool_layer_->SetUp(unpool_bottom_vec_, top);
    }

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
      pool_layer_->Reshape(bottom, pool_top_vec_);
      unpool_layer_->Reshape(unpool_bottom_vec_, top);
    }

   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
      pool_layer_->Forward(bottom, pool_top_vec_);
      unpool_layer_->Forward(unpool_bottom_vec_, top);
    }

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom) {
      unpool_layer_->Backward(top, propagate_down, unpool_bottom_vec_);
      pool_layer_->Backward(pool_top_vec_, propagate_down, bottom);
    }

    shared_ptr<PoolingLayer<Dtype> > pool_layer_;
    shared_ptr<UnpoolingLayer<Dtype> > unpool_layer_;
    Blob<Dtype> pooled_data_;
    Blob<Dtype> mask_;
    Blob<Dtype> argmax_count_;
    vector<Blob<Dtype>*> pool_top_vec_;
    vector<Blob<Dtype>*> unpool_bottom_vec_;
  };

  void PoolUnpoolGradientCheck(const bool overlapping = true,
      const PoolingParameter_PoolMethod pool =
          PoolingParameter_PoolMethod_MAX) {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(2);
    if (overlapping) {
      pooling_param->set_stride(1);
    } else {
      pooling_param->set_stride(2);
    }
    pooling_param->set_pool(pool);
    FillerParameter filler_param;
    filler_param.set_min(1);
    filler_param.set_max(5);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(pool_blob_bottom_);
    PoolUnpoolLayer layer(layer_param);
    GradientChecker<Dtype> checker(1e-4, 3e-2);
    // Check the gradients with respect to only bottom blob 0; we don't
    // compute gradients with respect to the mask (bottom blob 1).
    checker.CheckGradientExhaustive(&layer, this->pool_blob_bottom_vec_,
        this->blob_top_vec_);
  }

  void TestForward(const bool overlapping = true,
      const PoolingParameter_PoolMethod pool =
      PoolingParameter_PoolMethod_MAX) {
    LayerParameter layer_param;
    PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
    pooling_param->set_kernel_size(2);
    if (overlapping) {
      pooling_param->set_stride(1);
    } else {
      pooling_param->set_stride(2);
    }
    pooling_param->set_pool(pool);
    const int num = 2;
    const int channels = 2;
    const bool max_pooling = (pool == PoolingParameter_PoolMethod_MAX);
    pool_blob_bottom_->Reshape(num, channels, 3, 5);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0, c = 1; i < 15 * num * channels; i += 15, ++c) {
      const Dtype sign = (c % 2 == 0) ? 1 : -1;
      pool_blob_bottom_->mutable_cpu_data()[i +  0] = 1 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i +  1] = 2 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i +  2] = 5 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i +  3] = 2 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i +  4] = 3 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i +  5] = 9 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i +  6] = 4 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i +  7] = 1 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i +  8] = 4 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i +  9] = 8 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i + 10] = 1 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i + 11] = 2 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i + 12] = 5 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i + 13] = 2 * c * sign;
      pool_blob_bottom_->mutable_cpu_data()[i + 14] = 3 * c * sign;
    }
    PoolingLayer<Dtype> pool_layer(layer_param);
    int out_height, out_width;
    if (max_pooling && !overlapping) {
      pool_blob_top_vec_.resize(2);
      blob_bottom_vec_[2] = pool_blob_bottom_;
    } else if (!max_pooling) {
      pool_blob_top_vec_.resize(1);
    }
    if (overlapping) {
      out_height = 2;
      out_width = 4;
    } else {
      out_height = 2;
      out_width = 3;
    }
    pool_layer.SetUp(pool_blob_bottom_vec_, pool_blob_top_vec_);
    EXPECT_EQ(blob_bottom_->num(), num);
    EXPECT_EQ(blob_bottom_->channels(), channels);
    EXPECT_EQ(blob_bottom_->height(), out_height);
    EXPECT_EQ(blob_bottom_->width(), out_width);
    if (max_pooling) {
      EXPECT_EQ(blob_mask_->num(), num);
      EXPECT_EQ(blob_mask_->channels(), channels);
      EXPECT_EQ(blob_mask_->height(), out_height);
      EXPECT_EQ(blob_mask_->width(), out_width);
      if (overlapping) {
        EXPECT_EQ(blob_argmax_count_->num(), num);
        EXPECT_EQ(blob_argmax_count_->channels(), channels);
        EXPECT_EQ(blob_argmax_count_->height(), 3);
        EXPECT_EQ(blob_argmax_count_->width(), 5);
      }
    }
    pool_layer.Forward(pool_blob_bottom_vec_, pool_blob_top_vec_);
    if (overlapping) {
      for (int i = 0, c = 1; i < 8 * num * channels; i += 8, ++c) {
        const Dtype sign = (c % 2 == 0) ? 1 : -1;
        if (max_pooling) {
          // Expected maxpool output (& unpool input): 2x 2 channels of:
          //   sign == 1:
          //     [9 5 5 8]
          //     [9 5 5 8]
          //   sign == -1:
          //     [-1 -1 -1 -2]
          //     [-1 -1 -1 -2]
          if (sign == 1) {
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 0], 9 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 1], 5 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 2], 5 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 3], 8 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 4], 9 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 5], 5 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 6], 5 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 7], 8 * c);
          } else {
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 0], -1 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 1], -1 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 2], -1 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 3], -2 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 4], -1 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 5], -1 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 6], -1 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 7], -2 * c);
          }
        } else {
          // Expected avepool output (& unpool input): 2x 2 channels of:
          //     [4 3 3 17/4]
          //     [4 3 3 17/4]
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 0], 4 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 1], 3 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 2], 3 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 3], Dtype(17) / 4 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 4], 4 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 5], 3 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 6], 3 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 7], Dtype(17) / 4 * c * sign);
        }
      }
    } else {
      for (int i = 0, c = 1; i < 6 * num * channels; i += 6, ++c) {
        const Dtype sign = (c % 2 == 0) ? 1 : -1;
        if (pool == PoolingParameter_PoolMethod_MAX) {
          // Expected maxpool output (& unpool input): 2x 2 channels of:
          //   sign == 1:
          //     [9 5 8]
          //     [2 5 3]
          //   sign == -1:
          //     [-1 -1 -3]
          //     [-1 -2 -3]
          if (sign == 1) {
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 0], 9 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 1], 5 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 2], 8 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 3], 2 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 4], 5 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 5], 3 * c);
          } else {
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 0], -1 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 1], -1 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 2], -3 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 3], -1 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 4], -2 * c);
            EXPECT_EQ(blob_bottom_->cpu_data()[i + 5], -3 * c);
          }
        } else {
          // Expected avepool output (& unpool input): 2x 2 channels of:
          //     [4.0 3.0 5.5]
          //     [1.5 3.5 3.0]
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 0], 4.0 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 1], 3.0 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 2], 5.5 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 3], 1.5 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 4], 3.5 * c * sign);
          EXPECT_EQ(blob_bottom_->cpu_data()[i + 5], 3.0 * c * sign);
        }
      }
    }
    if (max_pooling && overlapping) {
      // Expected mask output: 2x 2 channels of:
      //   sign == 1:
      //     [5  2  2 9]
      //     [5 12 12 9]
      //   sign == -1:
      //     [ 0 7 7  3]
      //     [10 7 7 13]
      for (int i = 0, c = 1; i < 8 * num * channels; i += 8, ++c) {
        const Dtype sign = (c % 2 == 0) ? 1 : -1;
        if (sign == 1) {
          EXPECT_EQ(blob_mask_->cpu_data()[i + 0],  5);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 1],  2);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 2],  2);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 3],  9);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 4],  5);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 5], 12);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 6], 12);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 7],  9);
        } else {
          EXPECT_EQ(blob_mask_->cpu_data()[i + 0],  0);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 1],  7);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 2],  7);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 3],  3);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 4], 10);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 5],  7);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 6],  7);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 7], 13);
        }
      }
    } else if (max_pooling) {
      // Expected mask output: 2x 2 channels of:
      //   sign == 1:
      //     [ 5  2  9]
      //     [11 12 14]
      //   sign == -1:
      //     [ 0  7  4]
      //     [10 13 14]
      for (int i = 0, c = 1; i < 6 * num * channels; i += 6, ++c) {
        const Dtype sign = (c % 2 == 0) ? 1 : -1;
        if (sign == 1) {
          EXPECT_EQ(blob_mask_->cpu_data()[i + 0],  5);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 1],  2);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 2],  9);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 3], 11);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 4], 12);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 5], 14);
        } else {
          EXPECT_EQ(blob_mask_->cpu_data()[i + 0],  0);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 1],  7);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 2],  4);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 3], 10);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 4], 13);
          EXPECT_EQ(blob_mask_->cpu_data()[i + 5], 14);
        }
      }
    }
    if (max_pooling && overlapping) {
      // Expected argmax count output: 2x 2 channels of:
      //   sign == 1:
      //     [0 0 2 0 0]
      //     [2 0 0 0 2]
      //     [0 0 2 0 0]
      //   sign == -1:
      //     [1 0 0 1 0]
      //     [0 0 4 0 0]
      //     [1 0 0 1 0]
      for (int i = 0, c = 1; i < 15 * num * channels; i += 15, ++c) {
        const Dtype sign = (c % 2 == 0) ? 1 : -1;
        if (sign == 1) {
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  0], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  1], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  2], 2);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  3], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  4], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  5], 2);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  6], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  7], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  8], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  9], 2);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i + 10], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i + 11], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i + 12], 2);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i + 13], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i + 14], 0);
        } else {
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  0], 1);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  1], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  2], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  3], 1);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  4], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  5], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  6], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  7], 4);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  8], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i +  9], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i + 10], 1);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i + 11], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i + 12], 0);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i + 13], 1);
          EXPECT_EQ(blob_argmax_count_->cpu_data()[i + 14], 0);
        }
      }
    }
    UnpoolingLayer<Dtype> layer(layer_param);
    if (!max_pooling) {
      blob_bottom_vec_.resize(2);
      blob_bottom_vec_[1] = pool_blob_bottom_vec_[0];
    }
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), pool_blob_bottom_->height());
    EXPECT_EQ(blob_top_->width(), pool_blob_bottom_->width());
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    for (int i = 0, c = 1; i < 15 * num * channels; i += 15, ++c) {
      const Dtype sign = (c % 2 == 0) ? 1 : -1;
      if (max_pooling && overlapping) {
        // Expected overlapping max unpool output: 2x 2 channels of:
        //   sign == 1:
        //     [0 0 5 0 0]
        //     [9 0 0 0 8]
        //     [0 0 5 0 0]
        //   sign == -1:
        //     [-1 0  0 -2 0]
        //     [ 0 0 -1  0 0]
        //     [-1 0  0 -2 0]
        if (sign == 1) {
          EXPECT_EQ(blob_top_->cpu_data()[i +  0], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  1], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  2], 5 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  3], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  4], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  5], 9 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  6], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  7], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  8], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  9], 8 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 12], 5 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0 * c);
        } else {
          EXPECT_EQ(blob_top_->cpu_data()[i +  0], -1 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  1],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  2],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  3], -2 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  4],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  5],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  6],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  7], -1 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  8],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  9],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 10], -1 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 11],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 12],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 13], -2 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 14],  0 * c);
        }
      } else if (max_pooling && !overlapping) {
        // Expected non-overlapping max unpool output: 2x 2 channels of:
        //   sign == 1:
        //     [0 0 5 0 0]
        //     [9 0 0 0 8]
        //     [0 2 5 0 3]
        //   sign == -1:
        //     [-1 0  0  0 -3]
        //     [ 0 0 -1  0  0]
        //     [-1 0  0 -2 -3]
        if (sign == 1) {
          EXPECT_EQ(blob_top_->cpu_data()[i +  0], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  1], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  2], 5 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  3], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  4], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  5], 9 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  6], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  7], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  8], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  9], 8 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 11], 2 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 12], 5 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 14], 3 * c);
        } else {
          EXPECT_EQ(blob_top_->cpu_data()[i +  0], -1 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  1],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  2],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  3],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  4], -3 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  5],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  6],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  7], -1 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  8],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i +  9],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 10], -1 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 11],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 12],  0 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 13], -2 * c);
          EXPECT_EQ(blob_top_->cpu_data()[i + 14], -3 * c);
        }
      } else if (!max_pooling && overlapping) {
        // Expected overlapping ave unpool output: 2x 2 channels of:
        //     [4 7/2 3 29/8 17/4]
        //     [4 7/2 3 29/8 17/4]
        //     [4 7/2 3 29/8 17/4]
        EXPECT_EQ(blob_top_->cpu_data()[i +  0], 4 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  1], Dtype(7) / 2 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  2], 3 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  3], Dtype(29) / 8 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  4], Dtype(17) / 4 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  5], 4 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  6], Dtype(7) / 2 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  7], 3 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  8], Dtype(29) / 8 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  9], Dtype(17) / 4 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i + 10], 4 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i + 11], Dtype(7) / 2 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i + 12], 3 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i + 13], Dtype(29) / 8 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i + 14], Dtype(17) / 4 * c * sign);
      } else {  // !max_pooling && !overlapping
        // Expected non-overlapping ave unpool output: 2x 2 channels of:
        //     [4   4   3   3   5.5]
        //     [4   4   3   3   5.5]
        //     [1.5 1.5 3.5 3.5 3.0]
        EXPECT_EQ(blob_top_->cpu_data()[i +  0], 4 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  1], 4 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  2], 3 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  3], 3 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  4], 5.5 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  5], 4 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  6], 4 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  7], 3 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  8], 3 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i +  9], 5.5 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i + 10], 1.5 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i + 11], 1.5 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i + 12], 3.5 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i + 13], 3.5 * c * sign);
        EXPECT_EQ(blob_top_->cpu_data()[i + 14], 3 * c * sign);
      }
    }
  }
};

TYPED_TEST_CASE(UnpoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(UnpoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
  pooling_param->set_kernel_size(3);
  pooling_param->set_stride(2);

  PoolingLayer<Dtype> pool_layer(layer_param);
  pool_layer.SetUp(this->pool_blob_bottom_vec_, this->blob_bottom_vec_);
  EXPECT_EQ(this->blob_bottom_->num(), this->pool_blob_bottom_->num());
  EXPECT_EQ(this->blob_bottom_->channels(),
            this->pool_blob_bottom_->channels());
  EXPECT_EQ(this->blob_bottom_->height(), 3);
  EXPECT_EQ(this->blob_bottom_->width(), 2);
  EXPECT_EQ(this->blob_mask_->num(), this->pool_blob_bottom_->num());
  EXPECT_EQ(this->blob_mask_->channels(), this->pool_blob_bottom_->channels());
  EXPECT_EQ(this->blob_mask_->height(), 3);
  EXPECT_EQ(this->blob_mask_->width(), 2);

  UnpoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->pool_blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->pool_blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->pool_blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->pool_blob_bottom_->width());
}

TYPED_TEST(UnpoolingLayerTest, TestForwardMax) {
  this->TestForward();
}

TYPED_TEST(UnpoolingLayerTest, TestForwardMaxNonOverlapping) {
  const bool kOverlapping = false;
  this->TestForward(kOverlapping);
}

TYPED_TEST(UnpoolingLayerTest, TestForwardAve) {
  const bool kOverlapping = true;
  const PoolingParameter_PoolMethod kPool = PoolingParameter_PoolMethod_AVE;
  this->TestForward(kOverlapping, kPool);
}

TYPED_TEST(UnpoolingLayerTest, TestForwardAveNonOverlapping) {
  const bool kOverlapping = false;
  const PoolingParameter_PoolMethod kPool = PoolingParameter_PoolMethod_AVE;
  this->TestForward(kOverlapping, kPool);
}

TYPED_TEST(UnpoolingLayerTest, TestGradientMax) {
  this->PoolUnpoolGradientCheck();
}

TYPED_TEST(UnpoolingLayerTest, TestGradientMaxNonOverlapping) {
  const bool kOverlapping = false;
  this->PoolUnpoolGradientCheck(kOverlapping);
}

TYPED_TEST(UnpoolingLayerTest, TestGradientAve) {
  const bool kOverlapping = true;
  const PoolingParameter_PoolMethod kPool = PoolingParameter_PoolMethod_AVE;
  this->PoolUnpoolGradientCheck(kOverlapping, kPool);
}

TYPED_TEST(UnpoolingLayerTest, TestGradientAveNonOverlapping) {
  const bool kOverlapping = false;
  const PoolingParameter_PoolMethod kPool = PoolingParameter_PoolMethod_AVE;
  this->PoolUnpoolGradientCheck(kOverlapping, kPool);
}

}  // namespace caffe
