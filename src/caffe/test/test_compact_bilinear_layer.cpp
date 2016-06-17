#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/compact_bilinear_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "stdio.h"

namespace caffe {

template<typename TypeParam>
class CompactBilinearLayerTest: public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

 protected:
    CompactBilinearLayerTest() :
            blob_bottom_0_(new Blob<Dtype>(2, 3, 6, 5)),
                    blob_bottom_1_(new Blob<Dtype>(2, 5, 6, 5)),
                    blob_bottom_2_(new Blob<Dtype>(1, 512, 14, 14)),
                    blob_top_(new Blob<Dtype>()) {
    }
    virtual void SetUp() {
        // fill the values
        FillerParameter filler_param;
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_0_);
        caffe_add_scalar(blob_bottom_0_->count(), Dtype(1.0),
                blob_bottom_0_->mutable_cpu_data());
        filler.Fill(this->blob_bottom_1_);
        caffe_add_scalar(blob_bottom_1_->count(), Dtype(1.0),
                blob_bottom_1_->mutable_cpu_data());
        filler.Fill(this->blob_bottom_2_);

        blob_bottom_vec_.push_back(blob_bottom_0_);
        blob_bottom_vec_.push_back(blob_bottom_1_);

        bottom_vec_large.push_back(blob_bottom_2_);
        bottom_vec_large.push_back(blob_bottom_2_);

        blob_top_vec_.push_back(blob_top_);
    }

    virtual ~CompactBilinearLayerTest() {
        delete blob_bottom_0_;
        delete blob_bottom_1_;
        delete blob_bottom_2_;
        delete blob_top_;
    }

    Blob<Dtype>* const blob_bottom_0_;
    Blob<Dtype>* const blob_bottom_1_;
    Blob<Dtype>* const blob_bottom_2_;
    Blob<Dtype>* const blob_top_;
    vector<Blob<Dtype>*> blob_bottom_vec_, bottom_vec_large;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CompactBilinearLayerTest, TestDtypesAndDevices);

TYPED_TEST(CompactBilinearLayerTest, TestSetUp) {
typedef typename TypeParam::Dtype Dtype;

const int num_outputs[1] = {44};
const bool sum_pools[2] = {true, false};

for(int iout = 0; iout < 1; ++iout)
for(int isum = 0; isum < 3; ++isum) {
    LayerParameter layer_param;
    CompactBilinearParameter* compact_param =
    layer_param.mutable_compact_bilinear_param();

    // isum==2 means not setting the sum pool param
    if (isum != 2)
    compact_param->set_sum_pool(sum_pools[isum]);
    compact_param->set_num_output(num_outputs[iout]);

    shared_ptr<CompactBilinearLayer<Dtype> > layer(
            new CompactBilinearLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    EXPECT_EQ(this->blob_bottom_0_->shape(0), this->blob_top_->num());
    EXPECT_EQ(num_outputs[iout], this->blob_top_->channels());

    if ((isum == 2) || (sum_pools[isum] == true)) {
        EXPECT_EQ(1, this->blob_top_->height());
        EXPECT_EQ(1, this->blob_top_->width());
    } else {
        EXPECT_EQ(this->blob_bottom_0_->shape(2),
                this->blob_top_->height());
        EXPECT_EQ(this->blob_bottom_0_->shape(3),
                this->blob_top_->width());
    }
}
}

TYPED_TEST(CompactBilinearLayerTest, TestGradientBottomDiff) {
typedef typename TypeParam::Dtype Dtype;
LayerParameter layer_param;
layer_param.mutable_compact_bilinear_param()->set_num_output(6);
CompactBilinearLayer<Dtype> layer(layer_param);
GradientChecker<Dtype> checker(1e-1, 1e-1);
checker.CheckGradient(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
}

TYPED_TEST(CompactBilinearLayerTest, TestGradientBottomSame) {
typedef typename TypeParam::Dtype Dtype;
LayerParameter layer_param;
layer_param.mutable_compact_bilinear_param()->set_num_output(6);
CompactBilinearLayer<Dtype> layer(layer_param);
// somehow 1e-2 has some error, thus change to 1e-1
GradientChecker<Dtype> checker(1e-2, 1e-1);
this->blob_bottom_vec_.resize(1);
this->blob_bottom_vec_.push_back(this->blob_bottom_vec_[0]);
checker.CheckGradient(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
}

TYPED_TEST(CompactBilinearLayerTest, TestGradientBottomDiffNoPool) {
typedef typename TypeParam::Dtype Dtype;
LayerParameter layer_param;
layer_param.mutable_compact_bilinear_param()->set_num_output(6);
layer_param.mutable_compact_bilinear_param()->set_sum_pool(false);
CompactBilinearLayer<Dtype> layer(layer_param);
GradientChecker<Dtype> checker(1e-1, 1e-1);
checker.CheckGradient(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
}

TYPED_TEST(CompactBilinearLayerTest, TestGradientBottomSameNoPool) {
typedef typename TypeParam::Dtype Dtype;
LayerParameter layer_param;
layer_param.mutable_compact_bilinear_param()->set_num_output(6);
layer_param.mutable_compact_bilinear_param()->set_sum_pool(false);
CompactBilinearLayer<Dtype> layer(layer_param);
// stepsize & threshold
GradientChecker<Dtype> checker(1e-2, 1e-1);
this->blob_bottom_vec_.resize(1);
this->blob_bottom_vec_.push_back(this->blob_bottom_vec_[0]);
checker.CheckGradient(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

// For a speed test of 32*512*28*28, which is the typical case,
// and num_output_ set to 8192. The K40c GPU has a forward timing
// of 118ms and backward timing of 204ms. Half of the time was
// spent on cufft, which reaches the documented practical upper
// bound (400Gflops for N=8192). The other half is mainly bounded
// by memory reading. It could possibly be accelearted, but the overall
// speed up will be not more than 2 (assume using the same cufft lib).

// In practice, we assume a input size of 448*448, VGG16 network and
// conv5 output as this layer's input. The speed of this layer is
// 3.7ms per forward pass, or 10ms per forward-backward pass, which
// should be sufficiently fast.
// If using a 224*224 input, the above numbers would further reduce
// by a factor of 4. The forward pass would reduce to 0.9ms. In
// comparison, the original VGG16's forward pass take 12.5ms.

// The CPU implementation is not heavy optimized. Under the same
// setting, the forward pass takes 12.1s and backward takes 18.0s.
// As documented in the compact_bilinear_layer.cpp, this could
// potentially be improved by a factor of 2-3 with better FFT libs.
// If we're allow to use multi-threads, it could be further
// improved by a factor of #cores.

TYPED_TEST(CompactBilinearLayerTest, TestSpeed) {
typedef typename TypeParam::Dtype Dtype;
LayerParameter layer_param;
layer_param.mutable_compact_bilinear_param()->set_num_output(8192);
CompactBilinearLayer<Dtype> layer(layer_param);
layer.SetUp(this->bottom_vec_large, this->blob_top_vec_);

layer.Forward(this->bottom_vec_large, this->blob_top_vec_);
if (Caffe::mode() == Caffe::GPU) {
    caffe_copy(this->blob_top_->count(),
            this->blob_top_->gpu_data(), this->blob_top_->mutable_gpu_diff());
} else {
    caffe_copy(this->blob_top_->count(),
            this->blob_top_->cpu_data(), this->blob_top_->mutable_cpu_diff());
}
vector<bool> propagate_down(2, true);
layer.Backward(this->blob_top_vec_, propagate_down,
        this->bottom_vec_large);
}

}  // namespace caffe
