#include <vector>
#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layers/reorg_layer.hpp"
#include "caffe/test/test_caffe_main.hpp"
namespace caffe {
    template <typename TypeParam>
    class ReorgLayerTest : public MultiDeviceTest<TypeParam> {
        typedef typename TypeParam::Dtype Dtype;
        protected:
        Blob<Dtype>* const blob_bottom_;
        Blob<Dtype>* const blob_top_;
        vector<Blob<Dtype>*> blob_bottom_vec_;
        vector<Blob<Dtype>*> blob_top_vec_;
        ReorgLayerTest() : blob_bottom_(new Blob<Dtype>(2, 8, 2, 2)),
                           blob_top_(new Blob<Dtype>()) {
            Dtype* bottom_data = blob_bottom_->mutable_cpu_data();
            int_tp count = blob_bottom_->count();
            // n = 0: 0, 1, ..., 31
            // n = 1: 32, 33, ..., 63
            for(int_tp i = 0; i < count; ++i) {
                bottom_data[i] = static_cast<Dtype>(i);
            }
            blob_bottom_vec_.push_back(blob_bottom_);
            blob_top_vec_.push_back(blob_top_);
        }
        virtual ~ReorgLayerTest() {
            delete blob_bottom_;
            delete blob_top_;
        }
    };
    TYPED_TEST_CASE(ReorgLayerTest, TestDtypesAndDevices);
    TYPED_TEST(ReorgLayerTest, TestForward) {
        typedef typename TypeParam::Dtype Dtype;
        LayerParameter layer_param;
        ReorgParameter* reorg_param = layer_param.mutable_reorg_param();
        reorg_param->set_stride(2);
        reorg_param->set_reverse(true);

        ReorgLayer<Dtype> layer(layer_param);
        layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
        layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
        // Test
        // shape check
        EXPECT_EQ(this->blob_bottom_->count(), this->blob_top_->count());
        EXPECT_EQ(this->blob_bottom_->num(), this->blob_top_->num());
        EXPECT_EQ(this->blob_bottom_->channels(), this->blob_top_->channels() * 4);
        EXPECT_EQ(this->blob_bottom_->height(), this->blob_top_->height() / 2);
        EXPECT_EQ(this->blob_bottom_->width(), this->blob_top_->width() / 2);
        // check number
        int_tp val = 0;
        for(int_tp n = 0; n < this->blob_top_->num(); ++n) {
            for(int_tp c = 0; c < this->blob_top_->channels(); ++c) {
                for(int_tp h = 0; h < this->blob_top_->height(); ++h) {
                    for(int_tp w = 0; w < this->blob_top_->width(); ++w) {
                        //int_tp offset = this->blob_top_->offset(n, c, h, w);
                        int_tp tw = w / 2;
                        int_tp th = h / 2;
                        int_tp offset = (w - tw*2) + (h - th*2)*2;
                        int_tp tc = c + offset * this->blob_top_->channels();
                        EXPECT_EQ(this->blob_bottom_->data_at(n, tc, th, tw),
                                  this->blob_top_->data_at(n, c, h, w));
                    }
                }
            }
        }
    }
}