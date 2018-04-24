#include <algorithm>
#include <random>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#ifdef USE_LIBDNN
#include "caffe/layers/libdnn_conv_layer.hpp"
#endif

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

// Comparative check shape size limit
#define ELEMENT_LIMIT 1000000

namespace caffe {

template<typename TypeParam>
class QuantComparativeConvTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  QuantComparativeConvTest() :
        percentile_eps_(0.05),
        blob_bottom_(new Blob<float>()),
        blob_bottom_quant_(new Blob<Dtype>()),
        blob_top_(new Blob<float>()),
        blob_top_quant_(new Blob<Dtype>()),
        blob_top_unquant_(new Blob<float>()),
        rng_(rd_()) {
  }

  virtual void SetUp() {
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_quant_.push_back(blob_bottom_quant_);
    blob_top_vec_.push_back(blob_top_);
    blob_top_vec_quant_.push_back(blob_top_quant_);

    QuantizerParameter quant_param;
    quant_param.set_mode(CAFFE_QUANT_OBSERVE);
    bottom_quant_ = std::make_shared<Quantizer<float, Dtype> >(quant_param);
    top_quant_ = std::make_shared<Quantizer<float, Dtype> >(quant_param);
    weight_quant_ = std::make_shared<Quantizer<float, Dtype> >(quant_param);
    bias_quant_ = std::make_shared<Quantizer<float, Dtype> >(quant_param);
  }

  virtual ~QuantComparativeConvTest() {
    delete blob_bottom_;
    delete blob_bottom_quant_;
    delete blob_top_;
    delete blob_top_quant_;
  }

  bool TestForward(int_tp testIdx) {
    std::cout << "==== Test Case " << testIdx << " ====" << std::endl;

    LayerParameter layer_param;
    ConvolutionParameter* convolution_param =
        layer_param.mutable_convolution_param();

    std::uniform_int_distribution<int_tp> dimsRand(1, 3);
    std::uniform_int_distribution<int_tp> dilationRand(1, 8);
    std::uniform_int_distribution<int_tp> kernelRand(1, 7);
    std::uniform_int_distribution<int_tp> padRand(0, 5);
    std::uniform_int_distribution<int_tp> strideRand(1, 6);
    std::uniform_int_distribution<int_tp> biasRand(0, 1);
    std::uniform_int_distribution<int_tp> groupRand(1, 4);

    std::uniform_int_distribution<int_tp> batchRand(1, 10);
    std::uniform_int_distribution<int_tp> fmapRand(1, 64);

    int_tp batchsize = batchRand(this->rng_);
    int_tp groups = groupRand(this->rng_);
    int_tp fmaps_in = fmapRand(this->rng_) * groups;
    int_tp fmaps_out = fmapRand(this->rng_) * groups;

    int dims = dimsRand(this->rng_);

    std::uniform_int_distribution<int_tp> sizeRand(1,
         std::max(2, static_cast<int_tp>(pow(ELEMENT_LIMIT /
                     (fmaps_in * fmaps_out * batchsize),
                     1.0 / (static_cast<double>(dims))))));

    BlobShape shape;
    shape.add_dim(batchsize);  // Batch
    shape.add_dim(fmaps_in);   // Channels

    convolution_param->set_group(groups);

    for (int_tp i = 0; i < dims; ++i) {
      convolution_param->add_kernel_size(kernelRand(this->rng_));
      convolution_param->add_dilation(dilationRand(this->rng_));
      convolution_param->add_pad(padRand(this->rng_));
      convolution_param->add_stride(strideRand(this->rng_));

      int_tp size = sizeRand(this->rng_);
      int_tp kernel_extent = convolution_param->dilation(i)
          * (convolution_param->kernel_size(i) - 1) + 1;
      size = std::max((int_tp)size,
                      (int_tp)(kernel_extent - 2 * convolution_param->pad(i)));
      shape.add_dim(size);
    }

    std::cout << "Shape in: [";
    for (int i = 0; i < dims + 2; ++i) {
      std::cout << shape.dim(i);
      if (i < dims + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Kernel: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->kernel_size(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Dilation: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->dilation(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Stride: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->stride(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Pad: [";
    for (int i = 0; i < dims; ++i) {
      std::cout << convolution_param->pad(i);
      if (i < dims - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    std::cout << "Group: " << groups << std::endl;

    blob_bottom_->Reshape(shape);
    blob_bottom_quant_->Reshape(shape);

    convolution_param->set_num_output(fmaps_out);
    convolution_param->set_axis(1);
    convolution_param->mutable_weight_filler()->set_type("gaussian");
    convolution_param->mutable_weight_filler()->set_value(1);
    int_tp grand = biasRand(this->rng_);
    if (grand == 0) {
      convolution_param->mutable_bias_filler()->set_type("constant");
      convolution_param->mutable_bias_filler()->set_value(0);
      convolution_param->set_bias_term(false);
    } else {
      convolution_param->mutable_bias_filler()->set_type("gaussian");
      convolution_param->mutable_bias_filler()->set_value(1);
      convolution_param->set_bias_term(true);
    }

    ConvolutionLayer<float, float, float> ref_layer(layer_param);
    ref_layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    shared_ptr<BaseConvolutionLayer<Dtype, Dtype, Dtype> > layer =
        layer_creator_func_(layer_param);
    layer->SetUp(this->blob_bottom_vec_quant_, this->blob_top_vec_quant_);

    vector<shared_ptr<QuantizerBase> > blobs_quants =
        layer->get_blobs_quantizers();

    vector<shared_ptr<QuantizerBase> > bottom_quants =
        layer->get_bottom_quantizers();

    vector<shared_ptr<QuantizerBase> > top_quants =
        layer->get_top_quantizers();

    caffe_rng_uniform(blob_bottom_->count(), float(-5.0), float(5.0),
                      blob_bottom_->mutable_cpu_data());

    ref_layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    bottom_quant_->ObserveIn_cpu(this->blob_bottom_->count(),
                                  this->blob_bottom_->cpu_data());
    bottom_quant_->update();
    bottom_quants[0]->update_param(bottom_quant_->quant_param());
    bottom_quant_->Forward_cpu(blob_bottom_->count(),
                               blob_bottom_->cpu_data(),
                               blob_bottom_quant_->mutable_cpu_data());

    top_quant_->ObserveIn_cpu(this->blob_top_->count(),
                               this->blob_top_->cpu_data());
    top_quant_->update();
    top_quants[0]->update_param(top_quant_->quant_param());

    for (int_tp i = 0; i < ref_layer.blobs().size(); ++i) {
      switch (i) {
        case 0:
          weight_quant_->ObserveIn_cpu(ref_layer.blobs()[i]->count(),
                                        ref_layer.blobs()[i]->cpu_data());
          weight_quant_->update();
          weight_quant_->Forward_cpu(ref_layer.blobs()[i]->count(),
                                     ref_layer.blobs()[i]->cpu_data(),
                                     layer->blobs()[i]->mutable_cpu_data());
          blobs_quants[i]->update_param(weight_quant_->quant_param());
          break;
        case 1:
          bias_quant_->ObserveIn_cpu(ref_layer.blobs()[i]->count(),
                                      ref_layer.blobs()[i]->cpu_data());
          bias_quant_->update();
          bias_quant_->Forward_cpu(ref_layer.blobs()[i]->count(),
                                   ref_layer.blobs()[i]->cpu_data(),
                                   layer->blobs()[i]->mutable_cpu_data());
          blobs_quants[i]->update_param(bias_quant_->quant_param());
          break;
        default:
          break;
      }
    }

    layer->Forward(this->blob_bottom_vec_quant_, this->blob_top_vec_quant_);

    blob_top_unquant_->ReshapeLike(blob_top_quant_);

    top_quant_->Backward_cpu(this->blob_top_->count(),
                             this->blob_top_quant_->cpu_data(),
                             this->blob_top_unquant_->mutable_cpu_data());

    EXPECT_EQ(blob_top_->count(), blob_top_quant_->count());
    EXPECT_EQ(blob_top_->count(), blob_top_unquant_->count());


    const float *top_data = blob_top_->cpu_data();
    const float *top_data_unquant = blob_top_unquant_->cpu_data();

    std::cout << "Shape out: [";
    for (int i = 0; i < dims + 2; ++i) {
      std::cout << blob_top_->shape()[i];
      if (i < dims + 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]"<< std::endl;

    bool failure = false;
    double tot_error = 0;
    double tot_value = 0;
    double tot_value_ref = 0;
    int_tp failure_count = 0;

    const QuantizerValues top_qv = top_quant_->in_quantizer_values();

    float eps = std::max(std::abs(top_qv.get_max<float>()),
                         std::abs(top_qv.get_min<float>())) * percentile_eps_;

    for (int_tp i = 0; i < blob_top_->count(); ++i) {
      bool fail = (fabs(top_data_unquant[i] - top_data[i]) >= eps);
      if (fail) {
        std::cout << "Value: " << top_data_unquant[i]
                  << ", expected: " << top_data[i] << " (at " << i << ")"
                  << std::endl;
        tot_error += fabs(top_data_unquant[i] - top_data[i]);
        tot_value += fabs(top_data_unquant[i]);
        tot_value_ref += fabs(top_data[i]);
        ++failure_count;
      }
      failure |= fail;
    }
    std::cout << "Error count: " << failure_count
              << "/" << blob_top_->count() << std::endl;
    std::cout << "Difference: " << tot_error
              << " (value: " << tot_value << " vs " << tot_value_ref << ")"
              << std::endl;

    EXPECT_EQ(failure, false);
    return failure;
  }

  shared_ptr<BaseConvolutionLayer<Dtype, Dtype, Dtype> >(*layer_creator_func_)
      (LayerParameter);

  shared_ptr<Quantizer<float, Dtype> > bottom_quant_;
  shared_ptr<Quantizer<float, Dtype> > top_quant_;
  shared_ptr<Quantizer<float, Dtype> > weight_quant_;
  shared_ptr<Quantizer<float, Dtype> > bias_quant_;

  Blob<float>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_quant_;
  Blob<float>* const blob_top_;
  Blob<Dtype>* const blob_top_quant_;
  Blob<float>* const blob_top_unquant_;

  vector<Blob<float>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_bottom_vec_quant_;
  vector<Blob<float>*> blob_top_vec_;
  vector<Blob<Dtype>*> blob_top_vec_quant_;

  float percentile_eps_;

  std::random_device rd_;
  std::mt19937 rng_;
};

TYPED_TEST_CASE(QuantComparativeConvTest, TestDtypesIntegerAndDevices);

TYPED_TEST(QuantComparativeConvTest, TestCaffeConv) {
  typedef typename TypeParam::Dtype Dtype;
  auto lmbd = [](LayerParameter layer_param) {
    shared_ptr<BaseConvolutionLayer<Dtype, Dtype, Dtype> > layer(
    new ConvolutionLayer<Dtype, Dtype, Dtype>(layer_param));
    return layer;
  };
  this->layer_creator_func_  = lmbd;
  for (int i = 0; i < 100; ++i) {
    if (this->TestForward(i)) {
      break;
    }
  }
}

#ifdef USE_LIBDNN
TYPED_TEST(QuantComparativeConvTest, TestLibDNNConv) {
  typedef typename TypeParam::Dtype Dtype;

  if (Caffe::mode() == Caffe::CPU) {
    // LibDNN not supported on CPU
    return;
  }

  auto lmbd = [](LayerParameter layer_param) {
    shared_ptr<BaseConvolutionLayer<Dtype, Dtype, Dtype> > layer(
    new LibDNNConvolutionLayer<Dtype, Dtype, Dtype>(layer_param));
    return layer;
  };
  this->layer_creator_func_  = lmbd;
  for (int i = 0; i < 100; ++i) {
    if (this->TestForward(i)) {
      break;
    }
  }
}

#endif  // USE_LIBDNN

}  // namespace caffe
