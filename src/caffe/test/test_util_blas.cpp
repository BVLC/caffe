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
#include "omp.h"

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

	void BLASTestPerformanceGEMM(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int m, const int n, const int k) {

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

      caffe::Caffe::DeviceSync();

      BENCH(r, {
          caffe_gpu_gemm<Dtype>(TransA, TransB, m, n, k, 1.0, A->gpu_data(), B->gpu_data(), 0.0, C->mutable_gpu_data());
      });
		}

		if ( TypeParam::device == Caffe::CPU ) {
      BENCH(r, {
          caffe_cpu_gemm<Dtype>(TransA, TransB, m, n, k, 1.0, A->cpu_data(), B->cpu_data(), 0.0, C->mutable_cpu_data());
      });
    }

#else
		BENCH(r, {
				caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A->cpu_data(), B->cpu_data(), 0.0, C->mutable_cpu_data());
		});
#endif

	}

  void BLASTestValidationGEMM(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int m, const int n, const int k, size_t idx_offset_A, size_t idx_offset_B, size_t idx_offset_C, Dtype alpha, Dtype beta) {

#ifdef USE_CUDA
  if ( ! ( sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2 ) ) {
    LOG(ERROR) << "Skipping test due to old architecture.";
    return;
  }
#endif

    int scale      = 4;
    size_t sizeOfA = m*k;
    size_t sizeOfB = k*n;
    size_t sizeOfC = m*n;
    size_t sizeOfBufferA = scale*scale*sizeOfA;
    size_t sizeOfBufferB = scale*scale*sizeOfB;
    size_t sizeOfBufferC = scale*scale*sizeOfC;

    if ( sizeOfA + idx_offset_A > sizeOfBufferA ) {
      LOG(ERROR)<<"index offset for A = "<<idx_offset_A<<" out of range";
      EXPECT_TRUE(sizeOfA + idx_offset_A < sizeOfBufferA);
      return;
    }

    if ( sizeOfB + idx_offset_B > sizeOfBufferB ) {
      LOG(ERROR)<<"index offset for B = "<<idx_offset_B<<" out of range";
      EXPECT_TRUE(sizeOfB + idx_offset_B < sizeOfBufferB);
      return;
    }

    if ( sizeOfC + idx_offset_C > sizeOfBufferC ) {
      LOG(ERROR)<<"index offset for C = "<<idx_offset_C<<" out of range";
      EXPECT_TRUE(sizeOfC + idx_offset_C < sizeOfBufferC);
      return;
    }

    // initialize matrices
    A->Reshape(1, 1, scale*m, scale*k);
    B->Reshape(1, 1, scale*k, scale*n);
    C->Reshape(1, 1, scale*m, scale*n);
    C_->Reshape(1, 1, scale*m, scale*n);

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->A);
    filler.Fill(this->B);
    filler.Fill(this->C);

    for ( int i=0; i<scale*m*scale*k; i++ ) {
      if ( i < idx_offset_A ) {
        A->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_A + m*k ) {
        A->mutable_cpu_data()[i] = 0.0;
        continue;
      }
    }

    for ( int i=0; i<scale*k*scale*n; i++ ) {
      if ( i < idx_offset_B ) {
        B->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_B + n*k ) {
        B->mutable_cpu_data()[i] = 0.0;
        continue;
      }
    }

    for ( int i=0; i<scale*m*scale*n; i++ ) {
      if ( i < idx_offset_C ) {
        C->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_C + m*n ) {
        C->mutable_cpu_data()[i] = 0.0;
        continue;
      }
    }

    for ( int i=0; i<scale*m; i++ ) {
      for ( int j=0; j<scale*n; j++ ) {
        C_->mutable_cpu_data()[i*scale*n+j] = C->mutable_cpu_data()[i*scale*n+j];
      }
    }

#if defined(USE_CUDA) || defined(USE_OPENCL)

#if defined(USE_OPENCL)

    caffe_gpu_gemm<Dtype>(TransA, TransB, m, n, k, alpha, \
        A->gpu_data(), idx_offset_A, \
        B->gpu_data(), idx_offset_B, beta, \
        C->mutable_gpu_data(), idx_offset_C, NULL);
#endif

#if defined(USE_CUDA)
    caffe_gpu_gemm<Dtype>(TransA, TransB, m, n, k, alpha, \
        A->gpu_data() + idx_offset_A, \
        B->gpu_data() + idx_offset_B, beta, \
        C->mutable_gpu_data() + idx_offset_C);
#endif

    caffe_cpu_gemm<Dtype>(TransA, TransB, m, n, k, alpha, \
        A->cpu_data() + idx_offset_A, \
        B->cpu_data() + idx_offset_B, beta, \
        C_->mutable_cpu_data() + idx_offset_C);

    //SNAPSHOT2D("A", A->cpu_data(), scale*k, scale*m);
    //SNAPSHOT2D("B", B->cpu_data(), scale*n, scale*k);
    //SNAPSHOT2D("C(CPU)", C_->cpu_data() + idx_offset_C , n, m);
    //SNAPSHOT2D("C(GPU)", C->cpu_data() + idx_offset_C , n, m);

    for (int i = 0; i < scale*m*scale*n; ++i) {
      EXPECT_NEAR(C->cpu_data()[i], C_->cpu_data()[i], 1e-4);
    }
    //DIFFSHOT2D("delta", C->cpu_data() + idx_offset_C, C_->cpu_data() + idx_offset_C, n, m);

#endif

    // re-initialize matrices
    filler.Fill(this->A);
    filler.Fill(this->B);

    for ( int i=0; i<scale*m*scale*k; i++ ) {
      if ( i < idx_offset_A ) {
        A->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_A + m*k ) {
        A->mutable_cpu_data()[i] = 0.0;
        continue;
      }
    }

    for ( int i=0; i<scale*k*scale*n; i++ ) {
      if ( i < idx_offset_B ) {
        B->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_B + n*k ) {
        B->mutable_cpu_data()[i] = 0.0;
        continue;
      }
    }

    for ( int i=0; i<scale*m*scale*n; i++ ) {
      if ( i < idx_offset_C ) {
        C->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_C + m*n ) {
        C->mutable_cpu_data()[i] = 0.0;
        continue;
      }
    }

    for ( int i=0; i<scale*m; i++ ) {
      for ( int j=0; j<scale*n; j++ ) {
        C_->mutable_cpu_data()[i*scale*n+j] = C->mutable_cpu_data()[i*scale*n+j];
      }
    }

    #if defined(USE_CUDA) || defined(USE_OPENCL)

#if defined(USE_OPENCL)

    caffe_gpu_gemm<Dtype>(TransA, TransB, m, n, k, alpha, \
        A->gpu_data() + idx_offset_A, \
        B->gpu_data() + idx_offset_B, beta, \
        C->mutable_gpu_data() + idx_offset_C, NULL);
#endif

#if defined(USE_CUDA)
    caffe_gpu_gemm<Dtype>(TransA, TransB, m, n, k, alpha, \
        A->gpu_data() + idx_offset_A, \
        B->gpu_data() + idx_offset_B, beta, \
        C->mutable_gpu_data() + idx_offset_C);
#endif

    caffe_cpu_gemm<Dtype>(TransA, TransB, m, n, k, alpha, \
        A->cpu_data() + idx_offset_A, \
        B->cpu_data() + idx_offset_B, beta, \
        C_->mutable_cpu_data() + idx_offset_C);

    //SNAPSHOT2D("A", A->cpu_data(), scale*k, scale*m);
    //SNAPSHOT2D("B", B->cpu_data(), scale*n, scale*k);
    //SNAPSHOT2D("C(CPU)", C_->cpu_data() + idx_offset_C , n, m);
    //SNAPSHOT2D("C(GPU)", C->cpu_data() + idx_offset_C , n, m);

    for (int i = 0; i < scale*m*scale*n; ++i) {
      EXPECT_NEAR(C->cpu_data()[i], C_->cpu_data()[i], 1e-4);
    }
    //DIFFSHOT2D("delta", C->cpu_data() + idx_offset_C, C_->cpu_data() + idx_offset_C, n, m);

#endif
  }

  void BLASTestValidationGroupGEMM(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int m, const int n, const int k, const int gm, const int gn, const int gk, size_t idx_offset_A, size_t idx_offset_B, size_t idx_offset_C, Dtype alpha, Dtype beta) {

#if defined(USE_OPENCL)

    int scale      = 4;
    size_t sizeOfA = m*k;
    size_t sizeOfB = k*n;
    size_t sizeOfC = m*n;
    size_t sizeOfBufferA = scale*scale*sizeOfA;
    size_t sizeOfBufferB = scale*scale*sizeOfB;
    size_t sizeOfBufferC = scale*scale*sizeOfC;

    if ( sizeOfA + idx_offset_A > sizeOfBufferA ) {
      LOG(ERROR)<<"index offset for A = "<<idx_offset_A<<" out of range";
      EXPECT_TRUE(sizeOfA + idx_offset_A < sizeOfBufferA);
      return;
    }

    if ( sizeOfB + idx_offset_B > sizeOfBufferB ) {
      LOG(ERROR)<<"index offset for B = "<<idx_offset_B<<" out of range";
      EXPECT_TRUE(sizeOfB + idx_offset_B < sizeOfBufferB);
      return;
    }

    if ( sizeOfC + idx_offset_C > sizeOfBufferC ) {
      LOG(ERROR)<<"index offset for C = "<<idx_offset_C<<" out of range";
      EXPECT_TRUE(sizeOfC + idx_offset_C < sizeOfBufferC);
      return;
    }

    // initialize matrices
    A->Reshape(1, 1, scale*m, scale*k);
    B->Reshape(1, 1, scale*k, scale*n);
    C->Reshape(1, 1, scale*m, scale*n);
    C_->Reshape(1, 1, scale*m, scale*n);

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->A);
    filler.Fill(this->B);
    filler.Fill(this->C);

    for ( int i=0; i<scale*m*scale*k; i++ ) {
      if ( i < idx_offset_A ) {
        A->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_A + m*k ) {
        A->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      //A->mutable_cpu_data()[i] = 1.0;
    }

    for ( int i=0; i<scale*k*scale*n; i++ ) {
      if ( i < idx_offset_B ) {
        B->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_B + n*k ) {
        B->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      //B->mutable_cpu_data()[i] = (i - idx_offset_B) / (n*k/gn);
    }

    for ( int i=0; i<scale*m*scale*n; i++ ) {
      if ( i < idx_offset_C ) {
        C->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_C + m*n ) {
        C->mutable_cpu_data()[i] = 0.0;
        continue;
      }
    }

    for ( int i=0; i<scale*m; i++ ) {
      for ( int j=0; j<scale*n; j++ ) {
        C_->mutable_cpu_data()[i*scale*n+j] = C->mutable_cpu_data()[i*scale*n+j];
      }
    }

    caffe_gpu_group_gemm<Dtype>(TransA, TransB,
        m, n, k,
        gm, gn, gk,
        alpha,
        A->gpu_data(),
        B->gpu_data(),
        beta,
        C->mutable_gpu_data());

    for(int i = 0; i < gn; i++ ) {
      caffe_cpu_gemm<Dtype>(TransA, TransB,
          m, n/gn, k,
          alpha,
          A->cpu_data(),
          B->cpu_data() + i*k*n/gn,
          beta,
          C_->mutable_cpu_data() + i*m*n/gn);
    }

    //SNAPSHOT2D("A", A->cpu_data() + idx_offset_A, k, m);
    //SNAPSHOT2D("B", B->cpu_data() + idx_offset_B, n, k);
    //SNAPSHOT2D("C(CPU)", C_->cpu_data() + idx_offset_C , n, m);
    //SNAPSHOT2D("C(GPU)", C->cpu_data() + idx_offset_C , n, m);

    for (int i = 0; i < scale*m*scale*n; ++i) {
      EXPECT_NEAR(C->cpu_data()[i], C_->cpu_data()[i], 1e-4);
    }
    //LOG(INFO)<<m<<" x "<<n<<" x "<<k<<" into "<<gm<<" x "<<gn<<" x "<<gk;
    //DIFFSHOT2D("delta", C->cpu_data() + idx_offset_C, C_->cpu_data() + idx_offset_C, n, m);

#endif

  }

  void BLASTestPerformanceGroupGEMM(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int m, const int n, const int k, const int gm, const int gn, const int gk) {


#if defined(USE_OPENCL)

    A->Reshape(1, 1, m, k);
    B->Reshape(1, 1, k, n);
    C->Reshape(1, 1, m, n);

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);

    filler.Fill(this->A);
    filler.Fill(this->B);
    filler.Fill(this->C);

    record r;

    if ( TypeParam::device == Caffe::GPU ) {
      A->gpu_data();
      B->gpu_data();
      C->mutable_gpu_data();

      caffe::Caffe::DeviceSync();

      r.type      = std::string(typeid(Dtype).name());
      r.num_images    = 1;
      r.num_channels  = 1;
      r.img_width     = m;
      r.img_height    = n;

      BENCH(r, {
          caffe_gpu_group_gemm<Dtype>(TransA, TransB, m, n, k, gm, gn, gk, 1.0, A->gpu_data(), B->gpu_data(), 0.0, C->mutable_gpu_data());
      });

      r.type      = std::string(typeid(Dtype).name());
      r.num_images    = gn;
      r.num_channels  = 1;
      r.img_width     = m;
      r.img_height    = n/gn;

      BENCH(r, {
      for( int g = 0; g < gn; g++ ) {
        caffe_gpu_gemm<Dtype>(TransA, TransB, m, n/gn, k, 1.0, A->gpu_data(), B->gpu_data(), 0.0, C->mutable_gpu_data());
      }
      });
    }

#endif

  }


  void BLASTestValidationGEMV(const CBLAS_TRANSPOSE TransA, const int m, const int n, size_t idx_offset_A, size_t idx_offset_B, size_t idx_offset_C) {

  #ifdef USE_CUDA
  if ( ! ( sizeof(TypeParam) == 4 || CAFFE_TEST_CUDA_PROP.major >= 2 ) ) {
    LOG(ERROR) << "Skipping test due to old architecture.";
    return;
  }
  #endif

    int scale      = 2;
    size_t sizeOfA = m*n;
    size_t sizeOfB = n;
    size_t sizeOfC = m;

    size_t sizeOfBufferA = scale*scale*sizeOfA;
    size_t sizeOfBufferB = scale*sizeOfB;
    size_t sizeOfBufferC = scale*sizeOfC;

    if ( sizeOfA + idx_offset_A > sizeOfBufferA ) {
      LOG(ERROR)<<"index offset for A = "<<idx_offset_A<<" out of range";
      EXPECT_TRUE(sizeOfA + idx_offset_A < sizeOfBufferA);
      return;
    }

    if ( sizeOfB + idx_offset_B > sizeOfBufferB ) {
      LOG(ERROR)<<"index offset for B = "<<idx_offset_B<<" out of range";
      EXPECT_TRUE(sizeOfB + idx_offset_B < sizeOfBufferB);
      return;
    }

    if ( sizeOfC + idx_offset_C > sizeOfBufferC ) {
      LOG(ERROR)<<"index offset for C = "<<idx_offset_C<<" out of range";
      EXPECT_TRUE(sizeOfC + idx_offset_C < sizeOfBufferC);
      return;
    }

    int BLength = 0;
    int CLength = 0;

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);

    A->Reshape(1, 1, scale*m, scale*n);

    if( TransA == CblasNoTrans ) {
      BLength = n;
      CLength = m;
    } else {
      BLength = m;
      CLength = n;
    }
    B->Reshape(1, 1, 1, scale*BLength);
    C->Reshape(1, 1, 1, scale*CLength);
    C_->Reshape(1, 1, 1, scale*CLength);

    filler.Fill(this->A);
    filler.Fill(this->B);

    for ( int i=0; i<scale*m*scale*n; i++ ) {
      if ( i < idx_offset_A ) {
        A->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_A + m*n ) {
        A->mutable_cpu_data()[i] = 0.0;
        continue;
      }
    }

    for ( int i=0; i<scale*BLength; i++ ) {
      if ( i < idx_offset_B ) {
        B->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_B + BLength ) {
        B->mutable_cpu_data()[i] = 0.0;
        continue;
      }
    }

    for ( int i=0; i<scale*CLength; i++ ) {
      C->mutable_cpu_data()[i]  = 0.0;
      C_->mutable_cpu_data()[i] = 0.0;
    }

  #if defined(USE_CUDA) || defined(USE_OPENCL)

#if defined(USE_OPENCL)
    caffe_gpu_gemv(TransA, m, n,
        (Dtype) 1.0, \
        A->gpu_data(), idx_offset_A, \
        B->gpu_data(), idx_offset_B, \
        (Dtype) 0.0,\
        C->mutable_gpu_data(), idx_offset_C);
#endif
#if defined(USE_CUDA)
    caffe_gpu_gemv(TransA, m, n,
        (Dtype) 1.0, \
        A->gpu_data() + idx_offset_A, \
        B->gpu_data() + idx_offset_B, \
        (Dtype) 0.0,\
        C->mutable_gpu_data() + idx_offset_C);
#endif

    caffe_cpu_gemv(TransA, m, n, \
        (Dtype) 1.0, \
        A->cpu_data() + idx_offset_A, \
        B->cpu_data() + idx_offset_B, \
        (Dtype) 0.0, \
        C_->mutable_cpu_data() + idx_offset_C);


    //SNAPSHOT2D("A", A->cpu_data() + idx_offset_A, m, n);
    //SNAPSHOT("B", B->cpu_data() + idx_offset_B, BLength);
    //SNAPSHOT("C(CPU)", C_->cpu_data() + idx_offset_C, CLength);
    //SNAPSHOT("C(GPU)", C->cpu_data() + idx_offset_C, CLength);

    for (int i = 0; i < scale*CLength; ++i) {
      EXPECT_NEAR(C->cpu_data()[i], C_->cpu_data()[i], 1e-4);
    }
    //DIFFSHOT("delta", C->cpu_data() + idx_offset_C, C_->cpu_data() + idx_offset_C, CLength);

    A->Reshape(1, 1, scale*m, scale*n);

    if( TransA == CblasNoTrans ) {
      BLength = n;
      CLength = m;
    } else {
      BLength = m;
      CLength = n;
    }
    B->Reshape(1, 1, 1, scale*BLength);
    C->Reshape(1, 1, 1, scale*CLength);
    C_->Reshape(1, 1, 1, scale*CLength);

    filler.Fill(this->A);
    filler.Fill(this->B);

    for ( int i=0; i<scale*m*scale*n; i++ ) {
      if ( i < idx_offset_A ) {
        A->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_A + m*n ) {
        A->mutable_cpu_data()[i] = 0.0;
        continue;
      }
    }

    for ( int i=0; i<scale*BLength; i++ ) {
      if ( i < idx_offset_B ) {
        B->mutable_cpu_data()[i] = 0.0;
        continue;
      }
      if ( i >= idx_offset_B + BLength ) {
        B->mutable_cpu_data()[i] = 0.0;
        continue;
      }
    }

    for ( int i=0; i<scale*CLength; i++ ) {
      C->mutable_cpu_data()[i]  = 0.0;
      C_->mutable_cpu_data()[i] = 0.0;
    }

    for(int i = 0; i < scale*BLength; i++) {
      if( i < idx_offset_B ) {
        B->mutable_cpu_data()[i]  = 0.0;
      }
      if( i > idx_offset_B + BLength ) {
        B->mutable_cpu_data()[i]  = 0.0;
      }

    }

#if defined(USE_OPENCL)
    caffe_gpu_gemv(TransA, m, n,
        (Dtype) 1.0, \
        A->gpu_data() + idx_offset_A, \
        B->gpu_data() + idx_offset_B, \
        (Dtype) 0.0,\
        C->mutable_gpu_data() + idx_offset_C);
#endif
#if defined(USE_CUDA)
    caffe_gpu_gemv(TransA, m, n,
        (Dtype) 1.0, \
        A->gpu_data() + idx_offset_A, \
        B->gpu_data() + idx_offset_B, \
        (Dtype) 0.0,\
        C->mutable_gpu_data() + idx_offset_C);
#endif

    caffe_cpu_gemv(TransA, m, n, \
        (Dtype) 1.0, \
        A->cpu_data() + idx_offset_A, \
        B->cpu_data() + idx_offset_B, \
        (Dtype) 0.0, \
        C_->mutable_cpu_data() + idx_offset_C);


    //SNAPSHOT2D("A", A->cpu_data() + idx_offset_A, m, n);
    //SNAPSHOT("B", B->cpu_data() + idx_offset_B, BLength);
    //SNAPSHOT("C(CPU)", C_->cpu_data(), CLength);
    //SNAPSHOT("C(GPU)", C->cpu_data(), CLength);

    for (int i = 0; i < scale*CLength; ++i) {
      EXPECT_NEAR(C->cpu_data()[i], C_->cpu_data()[i], 1e-4);
    }
    //DIFFSHOT("delta", (Dtype*) C->cpu_data() + idx_offset_C, (Dtype*) C_->cpu_data() + idx_offset_C, CLength);

  #endif

  }


  Blob<Dtype>* A;
  Blob<Dtype>* B;
  Blob<Dtype>* C;
  Blob<Dtype>* C_;
};


TYPED_TEST_CASE(BLASTest, TestDtypesAndDevices);

TYPED_TEST(BLASTest, BLASTestPerformanceGEMM) {

  for ( int i = 0; i < 10; i++ ) {
    for(int size=16; size<=4096; size*=2 ) {
      this->BLASTestPerformanceGEMM(CblasNoTrans, CblasNoTrans, size,size,size);
      //this->BLASTestPerformanceGEMM(CblasTrans, CblasNoTrans, size,size,size);
      //this->BLASTestPerformanceGEMM(CblasNoTrans, CblasTrans, size,size,size);
      //this->BLASTestPerformanceGEMM(CblasTrans, CblasTrans, size,size,size);
    }

    for(int size=4128; size<=4128; size*=2 ) {
      this->BLASTestPerformanceGEMM(CblasNoTrans, CblasNoTrans, size,size,size);
      //this->BLASTestPerformanceGEMM(CblasTrans, CblasNoTrans, size,size,size);
      //this->BLASTestPerformanceGEMM(CblasNoTrans, CblasTrans, size,size,size);
      //this->BLASTestPerformanceGEMM(CblasTrans, CblasTrans, size,size,size);
    }
  }
}

#if defined(USE_CUDA) || defined(USE_OPENCL)

TYPED_TEST(BLASTest, BLASTestValidationGEMM) {

  srand (time(NULL));
  int min = 1;
  int max = 128;

  int m;
  int n;
  int k;

  for ( int i = 0; i < 10; i++ ) {
    m = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    n = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    k = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasNoTrans, m,n,k,0,0,0, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasNoTrans, m,n,k,0,0,0, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasTrans,   m,n,k,0,0,0, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasTrans,   m,n,k,0,0,0, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasNoTrans, m,n,k,0,0,0, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasNoTrans, m,n,k,0,0,0, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasTrans,   m,n,k,0,0,0, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasTrans,   m,n,k,0,0,0, 1.0, 1.0);
  }

  for ( int i = 0; i < 10; i++ ) {
    m = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    n = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    k = 1;
    this->BLASTestValidationGEMM(CblasNoTrans, CblasNoTrans, m,n,k,0,0,0, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasNoTrans, m,n,k,0,0,0, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasTrans,   m,n,k,0,0,0, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasTrans,   m,n,k,0,0,0, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasNoTrans, m,n,k,0,0,0, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasNoTrans, m,n,k,0,0,0, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasTrans,   m,n,k,0,0,0, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasTrans,   m,n,k,0,0,0, 1.0, 1.0);
  }
}

TYPED_TEST(BLASTest, BLASTestValidationGroupGEMM) {

  srand (time(NULL));
  int min = 1;
  int max = 128;

  int m;
  int n;
  int k;

  for ( int i = 0; i < 10; i++ ) {
    m = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    n = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    k = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    this->BLASTestValidationGroupGEMM(CblasNoTrans, CblasNoTrans, m,n,k,1,1,1,0,0,0, 1.0, 0.0);
    this->BLASTestValidationGroupGEMM(CblasNoTrans, CblasNoTrans, m,n,k,1,1,1,0,0,0, 1.0, 1.0);
  }

  for ( int i = 0; i < 10; i++ ) {
    m = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    n = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    k = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    this->BLASTestValidationGroupGEMM(CblasNoTrans, CblasNoTrans, m,2*n,k,1,2,1,0,0,0, 1.0, 0.0);
    this->BLASTestValidationGroupGEMM(CblasNoTrans, CblasNoTrans, m,2*n,k,1,2,1,0,0,0, 1.0, 1.0);
  }

  for ( int i = 0; i < 10; i++ ) {
    m = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    n = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    k = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    this->BLASTestValidationGroupGEMM(CblasNoTrans, CblasNoTrans, m,3*n,k,1,3,1,0,0,0, 1.0, 0.0);
    this->BLASTestValidationGroupGEMM(CblasNoTrans, CblasNoTrans, m,3*n,k,1,3,1,0,0,0, 1.0, 1.0);
  }

  /*
  LOG(ERROR)<<"The BIG one.";
  this->BLASTestValidationGroupGEMM(CblasNoTrans, CblasNoTrans, 96,30250,363,1,10,1,0,0,0, 1.0, 0.0);
  */
}

TYPED_TEST(BLASTest, BLASTestPerformanceGroupGEMM) {

  int repeat    = 1;
  int numImages = 10000;
  int m = 32;
  int n;
  int k = 32;

  for ( int im = 1; im <= numImages; im += 10 ) {
    n = im*32;
    for( int r = 0; r < repeat; r++ ) {
      this->BLASTestPerformanceGroupGEMM(CblasNoTrans, CblasNoTrans, m,n,k,1,n/32,1);
    }
  }


  /*
  LOG(ERROR)<<"The BIG one.";
  this->BLASTestValidationGroupGEMM(CblasNoTrans, CblasNoTrans, 96,30250,363,1,10,1,0,0,0, 1.0, 0.0);
  */
}

TYPED_TEST(BLASTest, BLASTestValidationGEMMOffset) {

  srand (time(NULL));
  int min = 1;
  int max = 128;

  int m;
  int n;
  int k;

  for ( int i = 0; i < 10; i++ ) {
    m = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    n = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    k = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasNoTrans, m,n,k,1,1,1, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasNoTrans, m,n,k,1,1,1, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasTrans,   m,n,k,1,1,1, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasTrans,   m,n,k,1,1,1, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasNoTrans, m,n,k,1,1,1, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasNoTrans, m,n,k,1,1,1, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasTrans,   m,n,k,1,1,1, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasTrans,   m,n,k,1,1,1, 1.0, 1.0);
  }

  for ( int i = 0; i < 10; i++ ) {
    m = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    n = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    k = 1;
    this->BLASTestValidationGEMM(CblasNoTrans, CblasNoTrans, m,n,k,1,1,1, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasNoTrans, m,n,k,1,1,1, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasTrans,   m,n,k,1,1,1, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasTrans,   m,n,k,1,1,1, 1.0, 0.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasNoTrans, m,n,k,1,1,1, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasNoTrans, m,n,k,1,1,1, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasNoTrans, CblasTrans,   m,n,k,1,1,1, 1.0, 1.0);
    this->BLASTestValidationGEMM(CblasTrans,   CblasTrans,   m,n,k,1,1,1, 1.0, 1.0);
  }

}

TYPED_TEST(BLASTest, BLASTestValidationGEMV) {

  srand (time(NULL));
  int min = 1;
  int max = 128;

  int m;
  int n;

  for ( int i = 0; i < 10; i++ ) {
    m = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    n = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    this->BLASTestValidationGEMV(CblasNoTrans, m, n, 0, 0, 0);
    this->BLASTestValidationGEMV(CblasTrans, m, n, 0, 0, 0);
  }
}

TYPED_TEST(BLASTest, BLASTestValidationGEMVOffset) {

  srand (time(NULL));
  int min = 1;
  int max = 128;

  int m;
  int n;

  for ( int i = 0; i < 10; i++ ) {
    m = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    n = min + rand() / (RAND_MAX / (max - min + 1) + 1);
    this->BLASTestValidationGEMV(CblasNoTrans, m, n, m, n/2, m/2);
    this->BLASTestValidationGEMV(CblasTrans, m, n, m, n/2, m/2);
  }
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

