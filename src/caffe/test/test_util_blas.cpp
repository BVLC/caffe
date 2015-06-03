#include <cstring>
#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/test/test_caffe_main.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "stdlib.h"

namespace caffe {

#ifdef USE_CUDA
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class GemmTest : public ::testing::Test {};

template<typename TypeParam>
class BLASTest: public MultiDeviceTest<TypeParam> {
	typedef typename TypeParam::Dtype Dtype;

protected:
	BLASTest() : A(new Blob<Dtype>()), B(new Blob<Dtype>()), C(new Blob<Dtype>()), C_(new Blob<Dtype>()) {
	}

	virtual void SetUp(int num_images, int num_channels, int im_width, int im_height) {
		A->Reshape(num_images, num_channels, im_width, im_height);
		B->Reshape(num_images, num_channels, im_width, im_height);
		C->Reshape(num_images, num_channels, im_width, im_height);

		// fill the values
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);

		filler.Fill(this->A);
		filler.Fill(this->B);
		filler.Fill(this->C);
    filler.Fill(this->C_);
	}

	virtual ~BLASTest() {
		delete(A);
		delete(B);
		delete(C);
    delete(C_);
	}

	void BLASTestPerformance(const int m, const int n, const int k) {

#ifdef USE_CUDA
  if ( ! ( sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2 ) ) {
    LOG(ERROR) << "Skipping test due to old architecture.";
  	return;
  }
#endif

		A->Reshape(1, 1, m, k);
		B->Reshape(1, 1, k, n);
		C->Reshape(1, 1, m, n);

		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);

		filler.Fill(this->A);
		filler.Fill(this->B);
		filler.Fill(this->C);

		record r;
		r.type 			= std::string(typeid(Dtype).name());
		r.num_images 		= 1;
		r.num_channels 	= 1;
		r.img_width			= m;
		r.img_height		= n;

#if defined(USE_CUDA) || defined(USE_OPENCL)

		if ( TypeParam::device == Caffe::GPU ) {
      A->gpu_data();
      B->gpu_data();
      C->mutable_gpu_data();
      BENCH(r, {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A->gpu_data(), B->gpu_data(), 0.0, C->mutable_gpu_data());
      });
		}

		if ( TypeParam::device == Caffe::CPU ) {
      BENCH(r, {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A->cpu_data(), B->cpu_data(), 0.0, C->mutable_cpu_data());
      });
    }

#else
		BENCH(r, {
				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A->cpu_data(), B->cpu_data(), 0.0, C->mutable_cpu_data());
		});
#endif

	}

  void BLASTestValidation(const int m, const int n, const int k) {

#ifdef USE_CUDA
  if ( ! ( sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2 ) ) {
    LOG(ERROR) << "Skipping test due to old architecture.";
    return;
  }
#endif

    A->Reshape(1, 1, m, k);
    B->Reshape(1, 1, k, n);
    C->Reshape(1, 1, m, n);
    C_->Reshape(1, 1, m, n);

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->A);
    filler.Fill(this->B);

    for ( int i=0; i<m; i++ ) {
      for ( int j=0; j<n; j++ ) {
        C->mutable_cpu_data()[i*n+j] = 0.0;
        C_->mutable_cpu_data()[i*n+j] = 0.0;
      }
    }

    record r;
    r.type      = std::string(typeid(Dtype).name());
    r.num_images    = 1;
    r.num_channels  = 1;
    r.img_width     = m;
    r.img_height    = n;

#if defined(USE_CUDA) || defined(USE_OPENCL)

    //snap("A", A->cpu_data(), k*m);
    //snap2D("A", A->cpu_data(), k, m);
    //snap("B", B->cpu_data(), n*k);
    //snap2D("B", B->cpu_data(), n, k);

    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A->gpu_data(), B->gpu_data(), 0.0, C->mutable_gpu_data());
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A->cpu_data(), B->cpu_data(), 0.0, C_->mutable_cpu_data());

    //snap("C(NEW)", C->cpu_data(), n*m);
    //snap2D("C(NEW)", C->cpu_data(), n, m);
    //snap("C_(OLD)", C_->cpu_data(), n*m);
    //snap2D("C_(OLD)", C_->cpu_data(), n, m);

    for (int i = 0; i < m*n; ++i) {
      EXPECT_NEAR(C->cpu_data()[i], C_->cpu_data()[i], 1e-4);
    }
    //diff2D("dC", C->cpu_data(), C_->cpu_data(), n, m);

#endif

  }

  Blob<Dtype>* A;
  Blob<Dtype>* B;
  Blob<Dtype>* C;
  Blob<Dtype>* C_;
};

TYPED_TEST_CASE(BLASTest, TestDtypesAndDevices);

TYPED_TEST(BLASTest, BLASTestPerformance) {

	for(int i=TEST_IMAGE_WIDTH_MIN; i<=2048; i*=2 ) {
		this->BLASTestPerformance(i,i,i);
	}
}

#if defined(USE_CUDA) || defined(USE_OPENCL)

TYPED_TEST(BLASTest, BLASTestValidation) {

  srand (time(NULL));
  int min = 1;
  int max = 128;

  int m;
  int n;
  int k;

  for ( int i = 0; i < 100; i++ ) {
    m = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    n = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    k = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    LOG(ERROR)<<m<<" x "<<n<<" x "<<k;
    this->BLASTestValidation(m,n,k);
  }

  /*
  this->BLASTestValidation(1,1,1);
  this->BLASTestValidation(8,8,8);
  this->BLASTestValidation(16,16,16);

  this->BLASTestValidation(32,32,32);
  this->BLASTestValidation(64,64,64);
  this->BLASTestValidation(16,16,17);
  this->BLASTestValidation(17,16,16);
  this->BLASTestValidation(16,17,16);
  */
}

#endif

#if defined(USE_CUDA) || defined(USE_OPENCL)

TYPED_TEST_CASE(GemmTest, TestDtypes);

TYPED_TEST(GemmTest, TestGemmCPUGPU) {

	Blob<TypeParam> A(1, 1, 2, 3);
  Blob<TypeParam> B(1, 1, 3, 4);
  Blob<TypeParam> C(1, 1, 2, 4);

  TypeParam data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  TypeParam A_reshape_data[6] = {1, 4, 2, 5, 3, 6};
  TypeParam B_reshape_data[12] = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
  TypeParam result[8] = {38, 44, 50, 56, 83, 98, 113, 128};
  caffe_copy(6, data, A.mutable_cpu_data());
  caffe_copy(12, data, B.mutable_cpu_data());

#ifdef USE_CUDA
  if ( ! ( sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2 ) ) {
    LOG(ERROR) << "Skipping test due to old architecture.";
  	return;
  }
#endif
    // [1, 2, 3; 4 5 6] * [1, 2, 3, 4; 5, 6, 7, 8; 9, 10, 11, 12];

    caffe_cpu_gemm<TypeParam>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());

    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    caffe_gpu_gemm<TypeParam>(CblasNoTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());

    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }

    // Test when we have a transposed A
    A.Reshape(1, 1, 3, 2);
    caffe_copy(6, A_reshape_data, A.mutable_cpu_data());
    caffe_cpu_gemm<TypeParam>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    caffe_gpu_gemm<TypeParam>(CblasTrans, CblasNoTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }

    // Test when we have a transposed A and a transposed B too
    B.Reshape(1, 1, 4, 3);
    caffe_copy(12, B_reshape_data, B.mutable_cpu_data());
    caffe_cpu_gemm<TypeParam>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    caffe_gpu_gemm<TypeParam>(CblasTrans, CblasTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }

    // Test when we have a transposed B
    A.Reshape(1, 1, 2, 3);
    caffe_copy(6, data, A.mutable_cpu_data());
    caffe_cpu_gemm<TypeParam>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
        A.cpu_data(), B.cpu_data(), 0., C.mutable_cpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
    caffe_gpu_gemm<TypeParam>(CblasNoTrans, CblasTrans, 2, 4, 3, 1.,
        A.gpu_data(), B.gpu_data(), 0., C.mutable_gpu_data());
    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(C.cpu_data()[i], result[i]);
    }
}

TYPED_TEST(GemmTest, TestGemvCPUGPU) {
  Blob<TypeParam> A(1, 1, 2, 3);
  Blob<TypeParam> x(1, 1, 1, 3);
  Blob<TypeParam> y(1, 1, 1, 2);
  TypeParam data[6] = {1, 2, 3, 4, 5, 6};
  TypeParam result_2[2] = {14, 32};
  TypeParam result_3[3] = {9, 12, 15};
  caffe_copy(6, data, A.mutable_cpu_data());
  caffe_copy(3, data, x.mutable_cpu_data());

#ifdef USE_CUDA
  if ( ! ( sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2 ) ) {
    LOG(ERROR) << "Skipping test due to old architecture.";
  	return;
  }
#endif
    caffe_cpu_gemv<TypeParam>(CblasNoTrans, 2, 3, 1., A.cpu_data(),
        x.cpu_data(), 0., y.mutable_cpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(y.cpu_data()[i], result_2[i]);
    }
    caffe_gpu_gemv<TypeParam>(CblasNoTrans, 2, 3, 1., A.gpu_data(),
        x.gpu_data(), 0., y.mutable_gpu_data());
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(y.cpu_data()[i], result_2[i]);
    }

    // Test transpose case
    caffe_copy(2, data, y.mutable_cpu_data());
    caffe_cpu_gemv<TypeParam>(CblasTrans, 2, 3, 1., A.cpu_data(),
        y.cpu_data(), 0., x.mutable_cpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(x.cpu_data()[i], result_3[i]);
    }
    caffe_gpu_gemv<TypeParam>(CblasTrans, 2, 3, 1., A.gpu_data(),
        y.gpu_data(), 0., x.mutable_gpu_data());
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(x.cpu_data()[i], result_3[i]);
    }
}
#endif

}  // namespace caffe

