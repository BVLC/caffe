/*************************************************************************
> File Name: test_unpooling_layer.cpp
> Author: 
> Mail: 
> Created Time: 2015年11月19日 星期四 13时43分54秒
************************************************************************/

#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
    template <typename Dtype>
    class UnpoolingLayerTest : public CPUDeviceTest<Dtype> {
        protected:
        UnpoolingLayerTest()
        : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_bottom_mask_(new Blob<Dtype>()) {}

        // Automatically done when calling TYPED_TEST
        virtual void SetUp() {
            Caffe::set_random_seed(1701);
            blob_bottom_->Reshape(2, 2, 2, 4);
            blob_bottom_mask_->Reshape(2, 2, 2, 4);
            // fill the values
            FillerParameter filler_param;
            GaussianFiller<Dtype> filler(filler_param);
            filler.Fill(this->blob_bottom_);
            filler.Fill(this->blob_bottom_mask_);
            blob_bottom_vec_.push_back(blob_bottom_);
            blob_bottom_vec_.push_back(blob_bottom_mask_);
            blob_top_vec_.push_back(blob_top_);

        }

        ~UnpoolingLayerTest() {
            delete blob_bottom_;
            delete blob_top_;
            delete blob_bottom_mask_;
        }
        Blob<Dtype>* const blob_bottom_;
        Blob<Dtype>* const blob_top_;
        Blob<Dtype>* const blob_bottom_mask_;
        vector<Blob<Dtype>*> blob_bottom_vec_;
        vector<Blob<Dtype>*> blob_top_vec_;

        // Test for 2x2 square unpooling layer
        void TestForwardSquare() {
            LayerParameter layer_param;
            PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
            pooling_param->set_kernel_size(2);
            pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
            const int num = 2;
            const int channels = 2;
            blob_bottom_->Reshape(num, channels, 2, 4);
            blob_bottom_mask_->Reshape(num, channels, 2, 4);
            // Input: 2x2 channels of:
            //     [9 5 5 8]
            //     [9 5 5 8]
            // mask: 2x2 channels of:
            //     [5  2  2 9]
            //     [5 12 12 9]
            for (int i = 0; i < 8 * num * channels; i += 8) {
                // Initialize the bottom
                blob_bottom_->mutable_cpu_data()[i + 0] = 9;
                blob_bottom_->mutable_cpu_data()[i + 1] = 5;
                blob_bottom_->mutable_cpu_data()[i + 2] = 5;
                blob_bottom_->mutable_cpu_data()[i + 3] = 8;
                blob_bottom_->mutable_cpu_data()[i + 4] = 9;
                blob_bottom_->mutable_cpu_data()[i + 5] = 5;
                blob_bottom_->mutable_cpu_data()[i + 6] = 5;
                blob_bottom_->mutable_cpu_data()[i + 7] = 8;
                // And the mask
                blob_bottom_mask_->mutable_cpu_data()[i + 0] = 5;
                blob_bottom_mask_->mutable_cpu_data()[i + 1] = 2;
                blob_bottom_mask_->mutable_cpu_data()[i + 2] = 2;
                blob_bottom_mask_->mutable_cpu_data()[i + 3] = 9;
                blob_bottom_mask_->mutable_cpu_data()[i + 4] = 5;
                blob_bottom_mask_->mutable_cpu_data()[i + 5] = 12;
                blob_bottom_mask_->mutable_cpu_data()[i + 6] = 12;
                blob_bottom_mask_->mutable_cpu_data()[i + 7] = 9;
            }

            UnpoolingLayer<Dtype> layer(layer_param);
            layer.SetUp(blob_bottom_vec_, blob_top_vec_);
            EXPECT_EQ(blob_top_->num(), num);
            EXPECT_EQ(blob_top_->channels(), channels);
            EXPECT_EQ(blob_top_->height(), 3);
            EXPECT_EQ(blob_top_->width(), 5);

            // Forward Test
            layer.Forward(blob_bottom_vec_, blob_top_vec_);
            // Expect output: 2x2 channels of
            //      [0 0 5 0 0]
            //      [9 0 0 0 8]
            //      [0 0 5 0 0]
            for (int i = 0; i < 15 * num * channels; i += 15) {
                EXPECT_EQ(blob_top_->cpu_data()[i +  0], 0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  1], 0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  2], 5);
                EXPECT_EQ(blob_top_->cpu_data()[i +  3], 0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  4], 0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  5], 9);
                EXPECT_EQ(blob_top_->cpu_data()[i +  6], 0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  7], 0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  8], 0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  9], 8);
                EXPECT_EQ(blob_top_->cpu_data()[i + 10], 0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 11], 0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 12], 5);
                EXPECT_EQ(blob_top_->cpu_data()[i + 13], 0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 14], 0);
            }
        }

        // Test for 3x2 rectangular unpooling layer with kernel_h > kernel_w
        void TestForwardRectHigh() {
            LayerParameter layer_param;
            PoolingParameter* pooling_param = layer_param.mutable_pooling_param();
            pooling_param->set_kernel_h(3);
            pooling_param->set_kernel_w(2);
            pooling_param->set_pool(PoolingParameter_PoolMethod_MAX);
            const int num = 2;
            const int channels = 2;
            blob_bottom_->Reshape(num, channels, 4, 5);
            blob_bottom_mask_->Reshape(num, channels, 4, 5);
            // Input: 2x 2 channels of:
            // [35    32    26    27    27]
            // [32    33    33    27    27]
            // [31    34    34    27    27]
            // [36    36    34    18    18]
            //
            // mask: 2x2 channels of:
            // [ 1    8     4     17    17]
            // [ 8   21    21     17    17]
            // [13   27    27     17    17]
            // [32   32    27     35    35]

            for (int i = 0; i < 20 * num * channels; i += 20) {
                blob_bottom_->mutable_cpu_data()[i +  0] = 35;
                blob_bottom_->mutable_cpu_data()[i +  1] = 32;
                blob_bottom_->mutable_cpu_data()[i +  2] = 26;
                blob_bottom_->mutable_cpu_data()[i +  3] = 27;
                blob_bottom_->mutable_cpu_data()[i +  4] = 27;
                blob_bottom_->mutable_cpu_data()[i +  5] = 32;
                blob_bottom_->mutable_cpu_data()[i +  6] = 33;
                blob_bottom_->mutable_cpu_data()[i +  7] = 33;
                blob_bottom_->mutable_cpu_data()[i +  8] = 27;
                blob_bottom_->mutable_cpu_data()[i +  9] = 27;
                blob_bottom_->mutable_cpu_data()[i + 10] = 31;
                blob_bottom_->mutable_cpu_data()[i + 11] = 34;
                blob_bottom_->mutable_cpu_data()[i + 12] = 34;
                blob_bottom_->mutable_cpu_data()[i + 13] = 27;
                blob_bottom_->mutable_cpu_data()[i + 14] = 27;
                blob_bottom_->mutable_cpu_data()[i + 15] = 36;
                blob_bottom_->mutable_cpu_data()[i + 16] = 36;
                blob_bottom_->mutable_cpu_data()[i + 17] = 34;
                blob_bottom_->mutable_cpu_data()[i + 18] = 18;
                blob_bottom_->mutable_cpu_data()[i + 19] = 18;

                // For the mask

                blob_bottom_mask_->mutable_cpu_data()[i +  0] =  0;
                blob_bottom_mask_->mutable_cpu_data()[i +  1] =  7;
                blob_bottom_mask_->mutable_cpu_data()[i +  2] =  3;
                blob_bottom_mask_->mutable_cpu_data()[i +  3] = 16;
                blob_bottom_mask_->mutable_cpu_data()[i +  4] = 16;
                blob_bottom_mask_->mutable_cpu_data()[i +  5] =  7;
                blob_bottom_mask_->mutable_cpu_data()[i +  6] = 20;
                blob_bottom_mask_->mutable_cpu_data()[i +  7] = 20;
                blob_bottom_mask_->mutable_cpu_data()[i +  8] = 16;
                blob_bottom_mask_->mutable_cpu_data()[i +  9] = 16;
                blob_bottom_mask_->mutable_cpu_data()[i + 10] = 12;
                blob_bottom_mask_->mutable_cpu_data()[i + 11] = 26;
                blob_bottom_mask_->mutable_cpu_data()[i + 12] = 26;
                blob_bottom_mask_->mutable_cpu_data()[i + 13] = 16;
                blob_bottom_mask_->mutable_cpu_data()[i + 14] = 16;
                blob_bottom_mask_->mutable_cpu_data()[i + 15] = 31;
                blob_bottom_mask_->mutable_cpu_data()[i + 16] = 31;
                blob_bottom_mask_->mutable_cpu_data()[i + 17] = 26;
                blob_bottom_mask_->mutable_cpu_data()[i + 18] = 34;
                blob_bottom_mask_->mutable_cpu_data()[i + 19] = 34;
            }  

            UnpoolingLayer<Dtype> layer(layer_param);
            layer.SetUp(blob_bottom_vec_, blob_top_vec_);
            EXPECT_EQ(blob_top_->num(), num);
            EXPECT_EQ(blob_top_->channels(), channels);
            EXPECT_EQ(blob_top_->height(), 6);
            EXPECT_EQ(blob_top_->width(), 6);
            //EXPECT_EQ(blob_bottom_->height(), 4);
            //EXPECT_EQ(blob_bottom_->width(), 5);


            layer.Forward(blob_bottom_vec_, blob_top_vec_);
            // Expected output: 2x 2 channels of:
                // [35     0     0    26    0    0]
                // [ 0    32     0    0     0    0]
                // [31     0     0    0     27   0]
                // [ 0     0    33    0     0    0]
                // [ 0     0    34    0     0    0]
                // [ 0    36     0    0     18   0]
                // (this is generated by magic(6) in MATLAB)

            for (int i = 0; i < 36 * num * channels; i += 36) {
                EXPECT_EQ(blob_top_->cpu_data()[i +  0] , 35);
                EXPECT_EQ(blob_top_->cpu_data()[i +  1] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  2] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  3] , 26);
                EXPECT_EQ(blob_top_->cpu_data()[i +  4] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  5] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  6] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  7] , 32);
                EXPECT_EQ(blob_top_->cpu_data()[i +  8] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i +  9] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 10] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 11] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 12] , 31);
                EXPECT_EQ(blob_top_->cpu_data()[i + 13] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 14] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 15] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 16] , 27);
                EXPECT_EQ(blob_top_->cpu_data()[i + 17] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 18] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 19] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 20] , 33);
                EXPECT_EQ(blob_top_->cpu_data()[i + 21] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 22] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 23] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 24] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 25] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 26] , 34);
                EXPECT_EQ(blob_top_->cpu_data()[i + 27] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 28] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 29] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 30] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 31] , 36);
                EXPECT_EQ(blob_top_->cpu_data()[i + 32] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 33] ,  0);
                EXPECT_EQ(blob_top_->cpu_data()[i + 34] , 18);
                EXPECT_EQ(blob_top_->cpu_data()[i + 35] ,  0);
            }
        }       
    };

    TYPED_TEST_CASE(UnpoolingLayerTest, TestDtypes);
    
    TYPED_TEST(UnpoolingLayerTest, TestForward) {
        //this->TestForwardSquare();
        this->TestForwardRectHigh();
    }

} // namespace caffe
