#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename TypeParam>
class InnerProductLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  InnerProductLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_bottom_nobatch_(new Blob<Dtype>(1, 2, 3, 4)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~InnerProductLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_nobatch_;
    delete blob_top_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_nobatch_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
};

TYPED_TEST_CASE(InnerProductLayerTest, TestDtypesAndDevices);

TYPED_TEST(InnerProductLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 10);
}

/** @brief TestSetUp while toggling transpose flag
 */
TYPED_TEST(InnerProductLayerTest, TestSetUpTransposeFalse) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->set_transpose(false);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(2, this->blob_top_->num());
  EXPECT_EQ(1, this->blob_top_->height());
  EXPECT_EQ(1, this->blob_top_->width());
  EXPECT_EQ(10, this->blob_top_->channels());
  EXPECT_EQ(2, layer->blobs()[0]->num_axes());
  EXPECT_EQ(10, layer->blobs()[0]->shape(0));
  EXPECT_EQ(60, layer->blobs()[0]->shape(1));
}

/** @brief TestSetUp while toggling transpose flag
 */
TYPED_TEST(InnerProductLayerTest, TestSetUpTransposeTrue) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->set_transpose(true);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(2, this->blob_top_->num());
  EXPECT_EQ(1, this->blob_top_->height());
  EXPECT_EQ(1, this->blob_top_->width());
  EXPECT_EQ(10, this->blob_top_->channels());
  EXPECT_EQ(2, layer->blobs()[0]->num_axes());
  EXPECT_EQ(60, layer->blobs()[0]->shape(0));
  EXPECT_EQ(10, layer->blobs()[0]->shape(1));
}

TYPED_TEST(InnerProductLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int_tp count = this->blob_top_->count();
  for (int_tp i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 1.);
  }
}

#if 0
TYPED_TEST(InnerProductLayerTest, TestForwardVGGFC6) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  UniformFiller<Dtype> filler(filler_param);
  caffe::Caffe::SetDevice(0);
  for (auto i = 1; i <= 64; i*=2) {
    Blob<Dtype>* const blob_bottom = new Blob<Dtype>(i, 392, 8, 8);
    Blob<Dtype>* const blob_top = new Blob<Dtype>();
    filler.Fill(blob_bottom);

    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(blob_bottom);
    this->blob_top_vec_.clear();
    this->blob_top_vec_.push_back(blob_top);
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(4096);
    inner_product_param->set_bias_term(false);
    inner_product_param->set_transpose(false);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    shared_ptr<InnerProductLayer<Dtype> > layer(
        new InnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    this->MakeReferenceTop(blob_top);
    const Dtype* A = blob_bottom->cpu_data();
    const Dtype* B = layer->blobs()[0]->cpu_data();
    Dtype* C = this->ref_blob_top_->mutable_cpu_data();
    int_tp M = blob_bottom->shape()[0];
    int_tp N = layer->blobs()[0]->shape(0);
    int_tp K = layer->blobs()[0]->shape(1);

    if (!std::is_same<Dtype, half_float::half>::value || i <= 2) {
      caffe_cpu_gemm(CblasNoTrans, CblasTrans, M, N, K,
                     (Dtype)1., A, B, (Dtype)0., C);

      const Dtype* data = blob_top->cpu_data();
      const int_tp count = blob_top->count();
      for (int_tp i = 0; i < count; ++i) {
        EXPECT_NEAR(data[i], C[i], 1e-1);
      }
    }
    if (Caffe::mode() == Caffe::GPU) {
      Timer timer;
      timer.initted();
      timer.Start();
      auto times = 10;
      for (auto i = 0; i < times; ++i) {
         layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      }
      timer.Stop();
      float elapsedTime = timer.MilliSeconds();
      elapsedTime /= times;
      std::cout << "MNK(" << M << "," << N << "," << K << ") Time is: "
                << elapsedTime
                << " ms" << std::endl;
      std::cout << "I/O: "
                << static_cast<double>(M*K + K*N + M*N) * sizeof(Dtype)
                   /elapsedTime / 1e6
                << "GB/s" << std::endl;
      std::cout << "FLOPS: "
                << static_cast<double>(M*N*(2*K-1)/elapsedTime/1e6) <<"GFLOPS"
                << std::endl;
    }
    delete blob_bottom;
    delete blob_top;
  }
}

TYPED_TEST(InnerProductLayerTest, TestForwardVGGFC6_AddEdge) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  UniformFiller<Dtype> filler(filler_param);
  caffe::Caffe::SetDevice(0);
  for (auto i = 1; i <= 64; i*=2) {
    Blob<Dtype>* const blob_bottom = new Blob<Dtype>(i, 25088+1, 1, 1);
    Blob<Dtype>* const blob_top = new Blob<Dtype>();
    filler.Fill(blob_bottom);

    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(blob_bottom);
    this->blob_top_vec_.clear();
    this->blob_top_vec_.push_back(blob_top);
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(4096+1);
    inner_product_param->set_bias_term(false);
    inner_product_param->set_transpose(false);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    shared_ptr<InnerProductLayer<Dtype> > layer(
        new InnerProductLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    this->MakeReferenceTop(blob_top);
    const Dtype* A = blob_bottom->cpu_data();
    const Dtype* B = layer->blobs()[0]->cpu_data();
    Dtype* C = this->ref_blob_top_->mutable_cpu_data();
    int_tp M = blob_bottom->shape()[0];
    int_tp N = layer->blobs()[0]->shape(0);
    int_tp K = layer->blobs()[0]->shape(1);

    if (!std::is_same<Dtype, half_float::half>::value || i <= 2) {
      caffe_cpu_gemm(CblasNoTrans, CblasTrans, M, N, K, (Dtype)1.,
                     A, B, (Dtype)0., C);

      const Dtype* data = blob_top->cpu_data();
      const int_tp count = blob_top->count();
      std::cout << blob_top->count() << std::endl;
      for (int_tp i = 0; i < count; ++i) {
        EXPECT_NEAR(data[i], C[i], 1e-1);
      }
    }
    if (Caffe::mode() == Caffe::GPU) {
      Timer timer;
      timer.initted();
      timer.Start();
      auto times = 10;
      for (auto i = 0; i < times; ++i) {
         layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
      }
      timer.Stop();
      float elapsedTime = timer.MilliSeconds();
      elapsedTime /= times;
      std::cout << "MNK(" << M << "," << N << "," << K << ") Time is: "
                << elapsedTime
                << " ms" << std::endl;
      std::cout << "I/O: "
                << static_cast<double>(M*K + K*N + M*N) * sizeof(Dtype)
                   /elapsedTime / 1e6
                << "GB/s" << std::endl;
      std::cout << "FLOPS: "
                << static_cast<double>(M*N*(2*K-1)/elapsedTime/1e6) <<"GFLOPS"
                << std::endl;
    }
    delete blob_bottom;
    delete blob_top;
  }
}
#endif

template <typename Dtype>
void gemv(const vector<shared_ptr<Blob<Dtype> > >& A,
          const int offA,
          const int M,
          const int N,
          const Blob<Dtype>* x,
          const int offx,
          Blob<Dtype>* y,
          const int offy,
          const float alpha,
          const float beta) {
  const unsigned rows = M;
  const unsigned cols = N;
  const Dtype* mat = A[0]->cpu_data() + offA;
  const Dtype* vec = x->cpu_data() + offx;
  Dtype* out_data = y->mutable_cpu_data() + offy;

  for (unsigned int r = 0; r < rows; r++) {
    out_data[r] = beta * out_data[r];
    for (unsigned int c = 0; c < cols; c++) {
      out_data[r] += alpha * mat[r * cols + c] * vec[c];
    }
  }
}

template void gemv(const vector<shared_ptr<Blob<float> > >& A,
          const int offA,
          const int M,
          const int N,
          const Blob<float>* x,
          const int offx,
          Blob<float>* y,
          const int offy,
          const float alpha,
          const float beta);

template void gemv(const vector<shared_ptr<Blob<double> > >& A,
          const int offA,
          const int M,
          const int N,
          const Blob<double>* x,
          const int offx,
          Blob<double>* y,
          const int offy,
          const float alpha,
          const float beta);

#if 0
TYPED_TEST(InnerProductLayerTest, TestForwardGemvFC6) {
  typedef typename TypeParam::Dtype Dtype;

  Blob<Dtype>* const blob_bottom = new Blob<Dtype>(1, 256, 6, 6);
  Blob<Dtype>* const blob_top = new Blob<Dtype>();
  FillerParameter filler_param;
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(blob_bottom);
  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(blob_bottom);
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(blob_top);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(4096);
  inner_product_param->set_bias_term(false);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  gemv(layer->blobs(), 0, layer->blobs()[0]->shape(0),
      layer->blobs()[0]->shape(1),
      blob_bottom, 0,
      this->MakeReferenceTop(blob_top), 0, 1., 0.);

  const Dtype* data = blob_top->cpu_data();
  const Dtype* ref_data = this->ref_blob_top_->cpu_data();
  const int_tp count = blob_top->count();
  for (int_tp i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], ref_data[i], 1e-1);
  }

  Timer timer;
  timer.initted();
  timer.Start();
  for (uint i = 0; i < 100; ++i) {
     layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  }
  timer.Stop();
  float elapsedTime = timer.MilliSeconds();
  std::cout << "GEMV(4096x9216) Time is: " << elapsedTime / 100.f
            <<" ms" << std::endl;

  delete blob_bottom;
  delete blob_top;
}

TYPED_TEST(InnerProductLayerTest, TestForwardGemvFC7) {
  typedef typename TypeParam::Dtype Dtype;

  Blob<Dtype>* const blob_bottom = new Blob<Dtype>(1, 4096, 1, 1);
  Blob<Dtype>* const blob_top = new Blob<Dtype>();
  FillerParameter filler_param;
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(blob_bottom);

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(blob_bottom);
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(blob_top);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(4096);
  inner_product_param->set_bias_term(false);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  gemv(layer->blobs(), 0, layer->blobs()[0]->shape(0),
      layer->blobs()[0]->shape(1),
      blob_bottom, 0,
      this->MakeReferenceTop(blob_top), 0, 1., 0.);

  const Dtype* data = blob_top->cpu_data();
  const Dtype* ref_data = this->ref_blob_top_->cpu_data();
  const int_tp count = blob_top->count();
  for (int_tp i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], ref_data[i], 1e-1);
  }

  Timer timer;
  timer.initted();
  timer.Start();
  for (uint i = 0; i < 100; ++i) {
     layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  }
  timer.Stop();
  float elapsedTime = timer.MilliSeconds();
  std::cout << "GEMV(4096x4096) Time is: " << elapsedTime / 100.f
            <<" ms" << std::endl;
  delete blob_bottom;
  delete blob_top;
}

TYPED_TEST(InnerProductLayerTest, TestForwardGemvFC8) {
  typedef typename TypeParam::Dtype Dtype;

  Blob<Dtype>* const blob_bottom = new Blob<Dtype>(1, 4096, 1, 1);
  Blob<Dtype>* const blob_top = new Blob<Dtype>();
  FillerParameter filler_param;
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(blob_bottom);

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(blob_bottom);
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(blob_top);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(1000);
  inner_product_param->set_bias_term(false);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  gemv(layer->blobs(), 0, layer->blobs()[0]->shape(0),
      layer->blobs()[0]->shape(1),
      blob_bottom, 0,
      this->MakeReferenceTop(blob_top), 0, 1., 0.);

  const Dtype* data = blob_top->cpu_data();
  const Dtype* ref_data = this->ref_blob_top_->cpu_data();
  const int_tp count = blob_top->count();
  for (int_tp i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], ref_data[i], 1e-1);
  }
  Timer timer;
  timer.initted();
  timer.Start();
  for (uint i = 0; i < 100; ++i) {
     layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  }
  timer.Stop();
  float elapsedTime = timer.MilliSeconds();
  std::cout << "GEMV(1000x4096) Time is: " << elapsedTime / 100.f
            <<" ms" << std::endl;

  delete blob_bottom;
  delete blob_top;
}
#endif

TYPED_TEST(InnerProductLayerTest, TestForwardGemvFC_dev1) {
  typedef typename TypeParam::Dtype Dtype;

  Blob<Dtype>* const blob_bottom = new Blob<Dtype>(1, 4099, 1, 1);
  Blob<Dtype>* const blob_top = new Blob<Dtype>();
  FillerParameter filler_param;
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(blob_bottom);

  this->blob_bottom_vec_.clear();
  this->blob_bottom_vec_.push_back(blob_bottom);
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(blob_top);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(1003);
  inner_product_param->set_bias_term(false);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  gemv(layer->blobs(), 0, layer->blobs()[0]->shape(0),
      layer->blobs()[0]->shape(1),
      blob_bottom, 0,
      this->MakeReferenceTop(blob_top), 0, 1., 0.);

  const Dtype* data = blob_top->cpu_data();
  const Dtype* ref_data = this->ref_blob_top_->cpu_data();
  const int_tp count = blob_top->count();
  for (int_tp i = 0; i < count; ++i) {
    EXPECT_NEAR(data[i], ref_data[i], 1e-1);
  }
  Timer timer;
  timer.initted();
  timer.Start();
  for (uint i = 0; i < 100; ++i) {
     layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  }
  timer.Stop();
  float elapsedTime = timer.MilliSeconds();
  std::cout << "GEMV(1003x4099) Time is: " << elapsedTime / 100.f
            <<" ms" << std::endl;
  delete blob_bottom;
  delete blob_top;
}

TYPED_TEST(InnerProductLayerTest, TestGEMV) {
  typedef typename TypeParam::Dtype Dtype;
  if (Caffe::mode() == Caffe::GPU) {
    Blob<Dtype>* const blob_bottom = new Blob<Dtype>(1, 4099, 1, 1);
    Blob<Dtype>* const blob_top = new Blob<Dtype>();
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom);

    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(blob_bottom);
    this->blob_top_vec_.clear();
    this->blob_top_vec_.push_back(blob_top);
    LayerParameter layer_param;
    InnerProductParameter* inner_product_param =
        layer_param.mutable_inner_product_param();
    inner_product_param->set_num_output(1003);
    inner_product_param->set_bias_term(false);
    inner_product_param->mutable_weight_filler()->set_type("uniform");
    shared_ptr<InnerProductLayer<Dtype> > layer(
        new InnerProductLayer<Dtype>(layer_param));

    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    device *dc = Caffe::GetDefaultDevice();
    const Dtype* x = this->blob_bottom_vec_[0]->gpu_data();
    Dtype* y = this->blob_top_vec_[0]->mutable_gpu_data();
    Dtype alpha = 2.5;
    Dtype beta = 2;
    unsigned int M = layer->blobs()[0]->shape(0);
    unsigned int N = layer->blobs()[0]->shape(1);
    //  add offset
    unsigned int offA = M * N / 2;
    unsigned int offx = 0;
    unsigned int offy = M / 2;
    M /= 2;
    greentea_gpu_gemv<Dtype>(dc->id(), CblasNoTrans,
                           M, N,
                           alpha,
                           (cl_mem)layer->blobs()[0]->gpu_data(),
                           offA, (cl_mem)x,
                           offx, beta, (cl_mem)y,
                           offy);
    gemv(layer->blobs(), offA, M, N,
        blob_bottom, offx, this->MakeReferenceTop(blob_top), offy, alpha, beta);

    const Dtype* data = blob_top->cpu_data();
    const Dtype* ref_data = this->ref_blob_top_->cpu_data();
    const int_tp count = blob_top->count();
    for (int_tp i = offy; i < count; ++i) {
      EXPECT_NEAR(data[i], ref_data[i], 1e-1);
    }

    delete blob_bottom;
    delete blob_top;
  }
}
/**
 * @brief Init. an IP layer without transpose + random weights,
 * run Forward, save the result.
 * Init. another IP layer with transpose.
 * manually copy and transpose the weights from the first IP layer,
 * then run Forward on the same input and check that the result is the same
 */
TYPED_TEST(InnerProductLayerTest, TestForwardTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  inner_product_param->set_transpose(false);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const int_tp count = this->blob_top_->count();
  Blob<Dtype>* const top = new Blob<Dtype>();
  top->ReshapeLike(*this->blob_top_);
  caffe_cpu_copy(count, this->blob_top_->cpu_data(), top->mutable_cpu_data());
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(new Blob<Dtype>());
  inner_product_param->set_transpose(true);
  shared_ptr<InnerProductLayer<Dtype> > ip_t(
      new InnerProductLayer<Dtype>(layer_param));
  ip_t->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  const int_tp count_w = layer->blobs()[0]->count();
  EXPECT_EQ(count_w, ip_t->blobs()[0]->count());
  // manually copy and transpose the weights from 1st IP layer into 2nd
  const Dtype* w = layer->blobs()[0]->cpu_data();
  Dtype* w_t = ip_t->blobs()[0]->mutable_cpu_data();
  const int_tp width = layer->blobs()[0]->shape(1);
  const int_tp width_t = ip_t->blobs()[0]->shape(1);
  for (int_tp i = 0; i < count_w; ++i) {
    int_tp r = i / width;
    int_tp c = i % width;
    w_t[c*width_t+r] = w[r*width+c];  // copy while transposing
  }
  // copy bias from 1st IP layer to 2nd IP layer
  ASSERT_EQ(layer->blobs()[1]->count(), ip_t->blobs()[1]->count());
  caffe_cpu_copy(layer->blobs()[1]->count(), layer->blobs()[1]->cpu_data(),
      ip_t->blobs()[1]->mutable_cpu_data());
  ip_t->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(count, this->blob_top_->count())
      << "Invalid count for top blob for IP with transpose.";
  Blob<Dtype>* const top_t = new Blob<Dtype>();\
  top_t->ReshapeLike(*this->blob_top_vec_[0]);
  caffe_cpu_copy(count,
    this->blob_top_vec_[0]->cpu_data(),
    top_t->mutable_cpu_data());
  const Dtype* data = top->cpu_data();
  const Dtype* data_t = top_t->cpu_data();
  for (int_tp i = 0; i < count; ++i) {
    EXPECT_FLOAT_EQ(data[i], data_t[i]);
  }
}

TYPED_TEST(InnerProductLayerTest, TestForwardNoBatch) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_nobatch_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();
  const int_tp count = this->blob_top_->count();
  for (int_tp i = 0; i < count; ++i) {
    EXPECT_GE(data[i], 1.);
  }
}

TYPED_TEST(InnerProductLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("gaussian");
  inner_product_param->mutable_bias_filler()->set_type("gaussian");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  InnerProductLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(InnerProductLayerTest, TestGradientTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(11);
  inner_product_param->mutable_weight_filler()->set_type("gaussian");
  inner_product_param->mutable_bias_filler()->set_type("gaussian");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  inner_product_param->set_transpose(true);
  InnerProductLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(InnerProductLayerTest, TestBackwardTranspose) {
  typedef typename TypeParam::Dtype Dtype;
  this->blob_bottom_vec_.push_back(this->blob_bottom_);
  LayerParameter layer_param;
  InnerProductParameter* inner_product_param =
      layer_param.mutable_inner_product_param();
  inner_product_param->set_num_output(10);
  inner_product_param->mutable_weight_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_type("uniform");
  inner_product_param->mutable_bias_filler()->set_min(1);
  inner_product_param->mutable_bias_filler()->set_max(2);
  inner_product_param->set_transpose(false);
  shared_ptr<InnerProductLayer<Dtype> > layer(
      new InnerProductLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // copy top blob
  Blob<Dtype>* const top = new Blob<Dtype>();
  top->CopyFrom(*this->blob_top_, false, true);
  // fake top diff
  Blob<Dtype>* const diff = new Blob<Dtype>();
  diff->ReshapeLike(*this->blob_top_);
  {
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(diff);
  }
  caffe_cpu_copy(this->blob_top_vec_[0]->count(),
    diff->cpu_data(),
    this->blob_top_vec_[0]->mutable_cpu_diff());
  vector<bool> propagate_down(1, true);
  layer->Backward(this->blob_top_vec_,
      propagate_down,
      this->blob_bottom_vec_);
  // copy first ip's weights and their diffs
  Blob<Dtype>* const w = new Blob<Dtype>();
  w->CopyFrom(*layer->blobs()[0], false, true);
  w->CopyFrom(*layer->blobs()[0], true, true);
  // copy bottom diffs
  Blob<Dtype>* const bottom_diff = new Blob<Dtype>();
  bottom_diff->CopyFrom(*this->blob_bottom_vec_[0], true, true);
  // repeat original top with transposed ip
  this->blob_top_vec_.clear();
  this->blob_top_vec_.push_back(new Blob<Dtype>());
  inner_product_param->set_transpose(true);
  shared_ptr<InnerProductLayer<Dtype> > ip_t(
      new InnerProductLayer<Dtype>(layer_param));
  ip_t->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // manually copy and transpose the weights from 1st IP layer into 2nd
  {
    const Dtype* w_src = w->cpu_data();
    Dtype* w_t = ip_t->blobs()[0]->mutable_cpu_data();
    const int_tp width = layer->blobs()[0]->shape(1);
    const int_tp width_t = ip_t->blobs()[0]->shape(1);
    for (int_tp i = 0; i < layer->blobs()[0]->count(); ++i) {
      int_tp r = i / width;
      int_tp c = i % width;
      w_t[c*width_t+r] = w_src[r*width+c];  // copy while transposing
    }
    // copy bias from 1st IP layer to 2nd IP layer
    ASSERT_EQ(layer->blobs()[1]->count(), ip_t->blobs()[1]->count());
    caffe_cpu_copy(layer->blobs()[1]->count(), layer->blobs()[1]->cpu_data(),
        ip_t->blobs()[1]->mutable_cpu_data());
  }
  ip_t->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_cpu_copy(this->blob_top_vec_[0]->count(),
    diff->cpu_data(),
    this->blob_top_vec_[0]->mutable_cpu_diff());
  ip_t->Backward(this->blob_top_vec_, propagate_down, this->blob_bottom_vec_);
  const Dtype* data = w->cpu_diff();
  const Dtype* data_t = ip_t->blobs()[0]->cpu_diff();
  const int_tp WIDTH = layer->blobs()[0]->shape(1);
  const int_tp WIDTH_T = ip_t->blobs()[0]->shape(1);
  for (int_tp i = 0; i < layer->blobs()[0]->count(); ++i) {
    int_tp r = i / WIDTH;
    int_tp c = i % WIDTH;
    EXPECT_NE(Dtype(0.), data[r*WIDTH+c]);
    EXPECT_FLOAT_EQ(data[r*WIDTH+c], data_t[c*WIDTH_T+r]);
  }
  data = bottom_diff->cpu_diff();
  data_t = this->blob_bottom_vec_[0]->cpu_diff();
  for (int_tp i = 0; i < this->blob_bottom_vec_[0]->count(); ++i) {
    EXPECT_NE(Dtype(0.), data[i]);
    EXPECT_FLOAT_EQ(data[i], data_t[i]);
  }
}

}  // namespace caffe
