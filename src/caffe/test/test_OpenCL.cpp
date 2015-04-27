#ifdef USE_OPENCL

#include <cstring>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "time.h"
#include "stdlib.h"

#include "caffe/test/test_caffe_main.hpp"
#include <caffe/util/OpenCL/OpenCLSupport.hpp>

namespace caffe {

template <typename Dtype>
class OpenCLSimpleTest : public ::testing::Test {
protected:
		OpenCLSimpleTest() {
			srand(time(NULL));
		};

		virtual ~OpenCLSimpleTest() {
		};
};

TYPED_TEST_CASE(OpenCLSimpleTest, TestDtypes);

TYPED_TEST(OpenCLSimpleTest, TestMalloc) {
	size_t bytes 	= 256;
	void* vPtr 		= NULL;
	EXPECT_TRUE(caffe::OpenCL::clMalloc(&vPtr, bytes));
	EXPECT_TRUE(caffe::OpenCL::clFree(vPtr));
}

TYPED_TEST(OpenCLSimpleTest, TestMaxMemory) {
	size_t MB 			= 1024*1024;
	int max_cnt			= 100000;
	int suc_cnt			= 0;

	void** vPtrArr = (void**) malloc(max_cnt*sizeof(void *));
	TypeParam pattern = 0.0;

	int i = 0;
	for(i = 0; i < max_cnt; i++ ) {
		vPtrArr[i] = NULL;

		if ( ! caffe::OpenCL::clMalloc(&vPtrArr[i], MB) ) {
			break;
		}
		try {
			caffe::OpenCL::clMemset(vPtrArr[i], pattern, MB);
		} catch (std::exception &e) {
			break;
		}
		suc_cnt++;
    if ( suc_cnt % 1024 == 0 ) {
      printf("OpenCL succesfully allocated %d buffers of 1MB\n", suc_cnt);
    }

	}
	printf("OpenCL successfully allocated %d buffers of 1MB\n", suc_cnt);

	for(i=suc_cnt-1; i >= 0; i-- ) {
		EXPECT_TRUE(caffe::OpenCL::clFree(vPtrArr[i]));
	}
	printf("OpenCL successfully released %d buffers of 1MB\n", suc_cnt);
}

TYPED_TEST(OpenCLSimpleTest, TestMaxBuffer) {
  size_t MB 			= 1024*1024;
  long int alloc_increment
                  = 100 * MB;
  size_t alloc		= alloc_increment;
  int max_cnt			= 1000;
  int suc_cnt			= 0;

  float pattern = 0.0;
  void* vPtr 		= NULL;

  int i = 0;
  for(i = 0; i < max_cnt; i++ ) {
    if ( ! caffe::OpenCL::clMalloc(&vPtr, alloc) ) {
      break;
    }
    try {
      caffe::OpenCL::clMemset(vPtr, pattern, alloc);
    } catch (std::exception &e) {
      break;
    }
    suc_cnt++;

    if (suc_cnt % 10 == 0)
    {
      printf("OpenCL succesfully allocated 1 buffer of %ldMB\n",
             suc_cnt * alloc_increment / MB);
    }

    alloc += alloc_increment;
    EXPECT_TRUE(caffe::OpenCL::clFree(vPtr));
  }
  long int megabytes= suc_cnt * alloc_increment / MB;
  printf("OpenCL successfully allocated one buffer of %ld MB\n", megabytes);

}

TYPED_TEST(OpenCLSimpleTest, TestMemcpy) {

	int		count	= 10;
	size_t bytes 	= count*sizeof(TypeParam);

	TypeParam* cpuPtrA = (TypeParam*) malloc(bytes);
	EXPECT_TRUE(cpuPtrA != NULL);

	TypeParam* cpuPtrB = (TypeParam*) malloc(bytes);
	EXPECT_TRUE(cpuPtrB != NULL);

	for(int i = 0; i < count; i++) {
		cpuPtrA[i] = rand() % count;
		cpuPtrB[i] = 0;
	}

	void* gpuPtrA 		= NULL;
	EXPECT_TRUE(caffe::OpenCL::clMalloc(&gpuPtrA, bytes));

	void* gpuPtrB 		= NULL;
	EXPECT_TRUE(caffe::OpenCL::clMalloc(&gpuPtrB, bytes));

	EXPECT_TRUE(caffe::OpenCL::clMemcpy(gpuPtrA, cpuPtrA, bytes, caffe::OpenCL::COPY_CPU_TO_GPU));
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(gpuPtrB, gpuPtrA, bytes, caffe::OpenCL::COPY_GPU_TO_GPU));
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(cpuPtrB, gpuPtrB, bytes, caffe::OpenCL::COPY_GPU_TO_CPU));

	for(int i = 0; i < count; i++) {
		EXPECT_EQ(cpuPtrA[i], cpuPtrB[i]);
	}

	EXPECT_TRUE(caffe::OpenCL::clFree(gpuPtrA));
	EXPECT_TRUE(caffe::OpenCL::clFree(gpuPtrB));
	free(cpuPtrA);
	free(cpuPtrB);
}

TYPED_TEST(OpenCLSimpleTest, TestMemcpyOffset) {

	int		count			= 10;
	int		offset			= 5;
	size_t bytes 			= count*sizeof(TypeParam);
	size_t offsetInBytes 	= offset*sizeof(TypeParam);

	TypeParam* cpuPtrA = (TypeParam*) malloc(bytes);
	EXPECT_TRUE(cpuPtrA != NULL);

	TypeParam* cpuPtrB = (TypeParam*) malloc(bytes);
	EXPECT_TRUE(cpuPtrB != NULL);

	for(int i = 0; i < count; i++) {
		cpuPtrA[i] = rand() % count;
		cpuPtrB[i] = 0;
	}

	void* gpuPtr 		= NULL;
	EXPECT_TRUE(caffe::OpenCL::clMalloc(&gpuPtr, bytes));

	EXPECT_TRUE(caffe::OpenCL::clMemcpy(gpuPtr, cpuPtrB, bytes, caffe::OpenCL::COPY_CPU_TO_GPU));
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(static_cast<char*>(gpuPtr) + offsetInBytes, cpuPtrA+offset, bytes-offsetInBytes, caffe::OpenCL::COPY_CPU_TO_GPU));
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(cpuPtrB, gpuPtr, bytes, caffe::OpenCL::COPY_GPU_TO_CPU));

	for(int i = offset; i < count; i++) {
		EXPECT_EQ(cpuPtrA[i], cpuPtrB[i]);
	}

	EXPECT_TRUE(caffe::OpenCL::clFree(gpuPtr));
	free(cpuPtrA);
	free(cpuPtrB);
}

TYPED_TEST(OpenCLSimpleTest, TestMemcpyOverlay) {

	int		count			= 10;
	int		offset			= 6;
	size_t bytes 			= count*sizeof(TypeParam);
	size_t offsetInBytes 	= offset*sizeof(TypeParam);

	TypeParam* cpuPtrA = (TypeParam*) malloc(bytes);
	EXPECT_TRUE(cpuPtrA != NULL);

	TypeParam* cpuPtrB = (TypeParam*) malloc(bytes);
	EXPECT_TRUE(cpuPtrB != NULL);

	for(int i = 0; i < count; i++) {
		cpuPtrA[i] = rand() % count;
		cpuPtrB[i] = 0;
	}

	void* gpuPtr 		= NULL;
	EXPECT_TRUE(caffe::OpenCL::clMalloc(&gpuPtr, bytes));

	EXPECT_TRUE(caffe::OpenCL::clMemcpy(gpuPtr, cpuPtrA, bytes, caffe::OpenCL::COPY_CPU_TO_GPU));
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(gpuPtr, static_cast<char*>(gpuPtr) + offsetInBytes, bytes-offsetInBytes, caffe::OpenCL::COPY_GPU_TO_GPU));
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(cpuPtrB, gpuPtr, bytes, caffe::OpenCL::COPY_GPU_TO_CPU));

	for(int i = offset; i < count; i++) {
		EXPECT_EQ(cpuPtrA[i], cpuPtrB[i-offset]);
	}

	EXPECT_TRUE(caffe::OpenCL::clFree(gpuPtr));
	free(cpuPtrA);
	free(cpuPtrB);
}

TYPED_TEST(OpenCLSimpleTest, TestMemset) {

	int		count	= 10;
	size_t bytes 	= count*sizeof(TypeParam);

	TypeParam* cpuPtrA = (TypeParam*) malloc(bytes);
	EXPECT_TRUE(cpuPtrA != NULL);

	TypeParam* cpuPtrB = (TypeParam*) malloc(bytes);
	EXPECT_TRUE(cpuPtrB != NULL);

	for(int i = 0; i < count; i++) {
		cpuPtrA[i] = rand() % count;
		cpuPtrB[i] = 0;
	}

	void* gpuPtr 		= NULL;
	EXPECT_TRUE(caffe::OpenCL::clMalloc(&gpuPtr, bytes));

	const TypeParam alpha = 0;
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(gpuPtr, cpuPtrA, bytes, caffe::OpenCL::COPY_CPU_TO_GPU));
	EXPECT_TRUE(caffe::OpenCL::clMemset(gpuPtr, alpha, bytes));
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(cpuPtrB, gpuPtr, bytes, caffe::OpenCL::COPY_GPU_TO_CPU));
	for(int i = 0; i < count; i++) {
		EXPECT_EQ(cpuPtrB[i], alpha);
	}

	const TypeParam beta = 1;
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(gpuPtr, cpuPtrA, bytes, caffe::OpenCL::COPY_CPU_TO_GPU));
	EXPECT_TRUE(caffe::OpenCL::clMemset(gpuPtr, beta, bytes));
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(cpuPtrB, gpuPtr, bytes, caffe::OpenCL::COPY_GPU_TO_CPU));
	for(int i = 0; i < count; i++) {
		EXPECT_EQ(cpuPtrB[i], beta);
	}

	EXPECT_TRUE(caffe::OpenCL::clFree(gpuPtr));
	free(cpuPtrA);
	free(cpuPtrB);
}

TYPED_TEST(OpenCLSimpleTest, TestMemsetOffset) {

	int		count			= 10;
	int		offset			= 5;
	size_t bytes 			= count*sizeof(TypeParam);
	size_t offsetInBytes 	= offset*sizeof(TypeParam);

	TypeParam* cpuPtrA = (TypeParam*) malloc(bytes);
	EXPECT_TRUE(cpuPtrA != NULL);

	TypeParam* cpuPtrB = (TypeParam*) malloc(bytes);
	EXPECT_TRUE(cpuPtrB != NULL);

	for(int i = 0; i < count; i++) {
		cpuPtrA[i] = rand() % count;
		cpuPtrB[i] = 0;
	}

	void* gpuPtr 		= NULL;
	EXPECT_TRUE(caffe::OpenCL::clMalloc(&gpuPtr, bytes));

	const TypeParam alpha = 0;
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(gpuPtr, cpuPtrA, bytes, caffe::OpenCL::COPY_CPU_TO_GPU));
	EXPECT_TRUE(caffe::OpenCL::clMemset(static_cast<char*>(gpuPtr) + offsetInBytes, alpha, bytes-offsetInBytes));
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(cpuPtrB, gpuPtr, bytes, caffe::OpenCL::COPY_GPU_TO_CPU));
	for(int i = offset; i < count; i++) {
		EXPECT_EQ(cpuPtrB[i], alpha);
	}

	const TypeParam beta = 1;
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(gpuPtr, cpuPtrA, bytes, caffe::OpenCL::COPY_CPU_TO_GPU));
	EXPECT_TRUE(caffe::OpenCL::clMemset(static_cast<char*>(gpuPtr) + offsetInBytes, beta, bytes-offsetInBytes));
	EXPECT_TRUE(caffe::OpenCL::clMemcpy(cpuPtrB, gpuPtr, bytes, caffe::OpenCL::COPY_GPU_TO_CPU));
	for(int i = offset; i < count; i++) {
		EXPECT_EQ(cpuPtrB[i], beta);
	}

	EXPECT_TRUE(caffe::OpenCL::clFree(gpuPtr));
	free(cpuPtrA);
	free(cpuPtrB);
}

TYPED_TEST(OpenCLSimpleTest, Test_clBLASgemv) {

	//bool clBLASgemv(const clblasTranspose TransA, const int m, const int n, const T alpha, const T* A, const T* x, const T beta, T* y)
	int		m			= 5;
	int		n			= 3;

	// create [m x n] matrix with random values
	size_t	mxn_size	= m*n*sizeof(TypeParam);

	SyncedMemory mem_mxn(mxn_size);
	TypeParam* cpuPtr_mxn = (TypeParam*) mem_mxn.mutable_cpu_data();
	EXPECT_TRUE(cpuPtr_mxn != NULL);

	int idx = 0;
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			cpuPtr_mxn[idx] = rand() % (m*n);
			idx++;
		}
	}

	TypeParam* gpuPtr_mxn	= (TypeParam*) mem_mxn.mutable_gpu_data();
	EXPECT_TRUE(gpuPtr_mxn != NULL);

	// create [n] vector
	size_t	n_size	= n*sizeof(TypeParam);

	SyncedMemory mem_n(n_size);
	TypeParam* cpuPtr_n = (TypeParam*) mem_n.mutable_cpu_data();
	EXPECT_TRUE(cpuPtr_n != NULL);

	for(int i = 0; i < n; i++) {
		cpuPtr_n[i] = rand() % (m*n);
	}
	//snap(cpuPtr_n, n);

	TypeParam* gpuPtr_n	= (TypeParam*) mem_n.mutable_gpu_data();
	EXPECT_TRUE(gpuPtr_n != NULL);

	// create [m] vector
	size_t	m_size	= m*sizeof(TypeParam);

	SyncedMemory mem_m(m_size);
	TypeParam* cpuPtr_m = (TypeParam*) mem_m.mutable_cpu_data();
	EXPECT_TRUE(cpuPtr_m != NULL);

	for(int i = 0; i < m; i++) {
		cpuPtr_m[i] = rand() % (m*n);
	}
	//snap(cpuPtr_m, m);

	TypeParam* gpuPtr_m	= (TypeParam*) mem_m.mutable_gpu_data();
	EXPECT_TRUE(gpuPtr_m != NULL);

	TypeParam alpha = 0.25;
	TypeParam beta  = 0.0;

	EXPECT_TRUE(caffe::OpenCL::clBLASgemv(clblasNoTrans, m, n, alpha, gpuPtr_mxn, gpuPtr_n, beta, gpuPtr_m));
	mem_m.cpu_data();
	//snap(cpuPtr_m, m);

	TypeParam val = 0;
	for(int i = 0; i < m; i++) {
		val = 0;
		for(int j = 0; j < n; j++) {
			val += alpha*cpuPtr_mxn[i*n+j]*cpuPtr_n[j];
		}
		EXPECT_EQ(val, cpuPtr_m[i]);
	}

	EXPECT_TRUE(caffe::OpenCL::clBLASgemv(clblasTrans, m, n, alpha, gpuPtr_mxn, gpuPtr_m, beta, gpuPtr_n));
	mem_n.cpu_data();
	//snap(cpuPtr_n, n);

	for(int j = 0; j < n; j++) {
		val = 0;
		for(int i = 0; i < m; i++) {
			val += alpha*cpuPtr_mxn[i*n+j]*cpuPtr_m[i];
		}
		EXPECT_EQ(val, cpuPtr_n[j]);
	}

}

TYPED_TEST(OpenCLSimpleTest, Test_clBLASdot) {

	//bool clBLASdot(const int n, const T* x, const int incx, const T* y, const int incy, T* out)
	int		n		= 10;
	size_t	n_size	= n*sizeof(TypeParam);

	// create [x] vector
	SyncedMemory mem_x(n_size);
	TypeParam* cpuPtr_x = (TypeParam*) mem_x.mutable_cpu_data();
	EXPECT_TRUE(cpuPtr_x != NULL);

	for(int i = 0; i < n; i++) {
		cpuPtr_x[i] = rand() % (n);
	}
	//snap(cpuPtr_x, n);

	TypeParam* gpuPtr_x	= (TypeParam*) mem_x.mutable_gpu_data();
	EXPECT_TRUE(gpuPtr_x != NULL);

	// create [x] vector
	SyncedMemory mem_y(n_size);
	TypeParam* cpuPtr_y = (TypeParam*) mem_y.mutable_cpu_data();
	EXPECT_TRUE(cpuPtr_y != NULL);

	for(int i = 0; i < n; i++) {
		cpuPtr_y[i] = rand() % (n);
	}
	//snap(cpuPtr_y, n);

	TypeParam* gpuPtr_y	= (TypeParam*) mem_y.mutable_gpu_data();
	EXPECT_TRUE(gpuPtr_y != NULL);

	// result
	TypeParam dot = 0;
	EXPECT_TRUE(caffe::OpenCL::clBLASdot(n, gpuPtr_x, 1, gpuPtr_y, 1, &dot));

	TypeParam val = 0;
	for(int i = 0; i < n; i++) {
		val += cpuPtr_x[i]*cpuPtr_y[i];
	}
	//LOG(ERROR)<<"result = "<<dot;
	EXPECT_NEAR(val, dot, 1e-4);
}

TYPED_TEST(OpenCLSimpleTest, Test_clsub) {

	//bool clsub(const int n, const void* array_GPU_x, const void* array_GPU_y, void* array_GPU_z)
	int		n		= 256;
	size_t	n_size	= n*sizeof(TypeParam);

	// create [x] vector
	SyncedMemory mem_x(n_size);
	TypeParam* cpuPtr_x = (TypeParam*) mem_x.mutable_cpu_data();
	EXPECT_TRUE(cpuPtr_x != NULL);

	for(int i = 0; i < n; i++) {
		cpuPtr_x[i] = rand() % (n);
	}
	//snap(cpuPtr_x, n);

	const TypeParam* gpuPtr_x	= (const TypeParam*) mem_x.gpu_data();
	EXPECT_TRUE(gpuPtr_x != NULL);

	// create [y] vector
	SyncedMemory mem_y(n_size);
	TypeParam* cpuPtr_y = (TypeParam*) mem_y.mutable_cpu_data();
	EXPECT_TRUE(cpuPtr_y != NULL);

	for(int i = 0; i < n; i++) {
		cpuPtr_y[i] = rand() % (n);
	}
	//snap(cpuPtr_y, n);

	const TypeParam* gpuPtr_y	= (const TypeParam*) mem_y.gpu_data();
	EXPECT_TRUE(gpuPtr_y != NULL);

	// create [z] vector
	SyncedMemory mem_z(n_size);
	TypeParam* cpuPtr_z = (TypeParam*) mem_z.mutable_cpu_data();
	EXPECT_TRUE(cpuPtr_z != NULL);

	for(int i = 0; i < n; i++) {
		cpuPtr_z[i] = 0;
	}
	//snap(cpuPtr_y, n);

	TypeParam* gpuPtr_z	= (TypeParam*) mem_z.mutable_gpu_data();
	EXPECT_TRUE(gpuPtr_z != NULL);

	// result
	EXPECT_TRUE(caffe::OpenCL::clsub(n, gpuPtr_x, gpuPtr_y, gpuPtr_z));
	mem_z.cpu_data();

	for(int i = 0; i < n; i++) {
		//LOG(INFO)<<cpuPtr_z[i];
		EXPECT_EQ( cpuPtr_x[i]-cpuPtr_y[i], cpuPtr_z[i]);
	}
}

TYPED_TEST(OpenCLSimpleTest, Test_clpowx) {

	//template<typename T> bool clpowx(const int n, const T* array_GPU_x, const T alpha, T* array_GPU_z);
	int		n		= 10;
	size_t	n_size	= n*sizeof(TypeParam);

	// create [x] vector
	SyncedMemory mem_x(n_size);
	TypeParam* cpuPtr_x = (TypeParam*) mem_x.mutable_cpu_data();
	EXPECT_TRUE(cpuPtr_x != NULL);

	for(int i = 0; i < n; i++) {
		cpuPtr_x[i] = rand() % (n);
	}
	//snap(cpuPtr_x, n);

	const TypeParam* gpuPtr_x	= (const TypeParam*) mem_x.gpu_data();
	EXPECT_TRUE(gpuPtr_x != NULL);

	// create [y] vector
	SyncedMemory mem_y(n_size);
	TypeParam* cpuPtr_y = (TypeParam*) mem_y.mutable_cpu_data();
	EXPECT_TRUE(cpuPtr_y != NULL);

	for(int i = 0; i < n; i++) {
		cpuPtr_y[i] = 0;
	}
	//snap(cpuPtr_y, n);

	TypeParam* gpuPtr_y	= (TypeParam*) mem_y.mutable_gpu_data();
	EXPECT_TRUE(gpuPtr_y != NULL);

	// result
	EXPECT_TRUE(caffe::OpenCL::clpowx(n, gpuPtr_x, (TypeParam) 2, gpuPtr_y));
	mem_y.mutable_cpu_data();

	for(int i = 0; i < n; i++) {
		//LOG(INFO)<<cpuPtr_y[i];
		EXPECT_NEAR( cpuPtr_x[i]*cpuPtr_x[i], cpuPtr_y[i], 1e-4);
	}
}


}  // namespace caffe

#endif // USE_OPENCL
