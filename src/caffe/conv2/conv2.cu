#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#define THREAD_INDEX threadIdx.x+blockIdx.x*blockDim.x
#if __CUDA_ARCH__ >= 200
const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
const int CAFFE_CUDA_NUM_THREADS = 512;
#endif
#define CUDA_CHECK(condition) \
	/* Code block avoids redefinition of cudaError_t error */ \
	cudaError_t error = (condition); \
	if(error!=cudaSuccess) \
{ printf(cudaGetErrorString(error)); \
	assert(0);\
}
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())
inline int CAFFE_GET_BLOCKS(const int N) {
	return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}
struct DeviceTempMemory
{
	void *mptr;
	int size;
	//cublasHandle_t _handle;
};
DeviceTempMemory pool={0,0};
__global__ void conv2_level1_kernel_(const int n,const float *M,int height,int width,
									 const float *kernel,int kernel_h,int kernel_w,
									 int pad_h,int pad_w,int stride_h,int stride_w,
									 int patch_h,int patch_w,float *r)
{
	__shared__ float img[512];
	__shared__ float k[100];
	int index=THREAD_INDEX;
	int image_size=width*height;
	int kernel_size=kernel_w*kernel_h;
	if (threadIdx.x<image_size)
	{
		img[threadIdx.x]=M[threadIdx.x];
	}

	if(threadIdx.x<kernel_size)
	{
		k[threadIdx.x]=kernel[threadIdx.x];
	}
	__syncthreads();
	if (index<n)
	{
		int w_index=index%patch_w;
		int h_index=index/patch_w;
		int w_start=0-pad_w+w_index*stride_w;
		int h_start=0-pad_h+h_index*stride_h;
		float sum=0;
		float *p_kernel=k;
		for (int y=0;y<kernel_h;y++)
		{
			int index_y=h_start+y;
			int offset_y=index_y*width;
			for (int x=0;x<kernel_w;x++)
			{
				int offset_x=w_start+x;
				float data=(index_y<0 || index_y>=height || offset_x<0 || offset_x>=width)?0:img[offset_y+offset_x];
				sum+=data*(*p_kernel);
				p_kernel++;
			}
		}
		r[index]=sum;
	}
}
__global__ void conv2_float_level2_kernel_(const int n,const float *M,int height,int width,
										   const float *kernel,int kernel_h,int kernel_w,
										   int pad_h,int pad_w,int stride_h,int stride_w,
										   int patch_h,int patch_w,float *r)
{
	int index=THREAD_INDEX;
	if(index<n)
	{
		int kernel_size=kernel_w*kernel_h;
		const float *p_kernel=kernel;
		__shared__ float k[1024];
		if(kernel_size<512)
		{
			if(threadIdx.x<kernel_size)
			{
				k[threadIdx.x]=kernel[threadIdx.x];
			}
			__syncthreads();
			p_kernel=k;
		}
		int w_index=index%patch_w;
		int h_index=index/patch_w;
		int w_start=0-pad_w+w_index*stride_w;
		int h_start=0-pad_h+h_index*stride_h;
		float sum=0;
		p_kernel=k;
		for (int y=0;y<kernel_h;y++)
		{
			int index_y=h_start+y;
			int offset_y=index_y*width;
			for (int x=0;x<kernel_w;x++)
			{
				int offset_x=w_start+x;
				float data=(index_y<0 || index_y>=height || index_y<0 || index_y>=width)?0:M[offset_y+offset_x];
				sum+=data*(*p_kernel);
				p_kernel++;
			}
		}
		r[index]=sum;

	}
}
__global__ void conv2_float_level3_kernel_(const int n,const float *M,int height,int width,
										   const float *kernel,int kernel_h,int kernel_w,
										   int pad_h,int pad_w,int stride_h,int stride_w,
										   int patch_h,int patch_w,float *r)
{
	int index=THREAD_INDEX;
	int patch_rows=kernel_h;
	if(index<n)
	{
		//计算patch的索引
		int patch_index=index/patch_rows;
		//计算当前标号所对应的patch的row的序号
		int patch_row_index=index%patch_rows;
		//计算patch的起始位置
		int w_index=patch_index%patch_w;
		int h_index=patch_index/patch_w;
		int w_start=0-pad_w+w_index*stride_w;
		int h_start=0-pad_h+h_index*stride_h;
		int index_y=h_start+patch_row_index;
		int offset_y=index_y*width;
		const float *p_kernel=kernel+patch_row_index*kernel_w;
		float sum=0;
		for (int x=0;x<kernel_w;x++)
		{
			int offset_x=w_start+x;
			float data=(index_y<0 || index_y>=height || offset_x<0 || offset_x>=width)?0:M[offset_y+offset_x];
			sum+=data*(*p_kernel);
			p_kernel++;
		}
		r[index]=sum;
	}
}
__global__ void conv2_float_sum_kernel_(const int n,const float *M,int height,int width,float *r)
{
	int index=THREAD_INDEX;
	if(index<n)
	{
		float sum=0;
		int offset_y=index*width;
		for (int x=0;x<width;x++)
		{
			sum+=M[offset_y+x];
		}
		r[index]=sum;
	}
}
void conv2_gpu(const float *data_image,int height,int width,
					 const float *kernel,int kernel_h,int kernel_w,
					 int pad_h,int pad_w,int stride_h,int stride_w,float *r)
{
	int patch_w=(width+2*pad_w-kernel_w)/stride_w+1;
	int patch_h=(height+2*pad_h-kernel_h)/stride_h+1;
	int num_kernels=patch_h*patch_w;
	int kernel_size=kernel_w*kernel_h;
	int image_size=height*width;
	if(image_size<512 && kernel_size<256)
	{
		conv2_level1_kernel_<<<CAFFE_GET_BLOCKS(num_kernels),CAFFE_CUDA_NUM_THREADS>>>(num_kernels,data_image,height,width,
			kernel,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,patch_h,patch_w,r);
	}
	else if(kernel_size<512)
	{
		conv2_float_level2_kernel_<<<CAFFE_GET_BLOCKS(num_kernels),CAFFE_CUDA_NUM_THREADS>>>(num_kernels,data_image,height,width,
			kernel,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,patch_h,patch_w,r);
	}
	else
	{
		int total_size=num_kernels*kernel_h*sizeof(float);
		if(pool.mptr)
		{
			if(pool.size<total_size)
			{
				cudaFree(pool.mptr);
				CUDA_CHECK(cudaMalloc(&pool.mptr,total_size));
				pool.size=total_size;
			}
		}
		else
		{
			CUDA_CHECK(cudaMalloc(&pool.mptr,total_size));
			pool.size=total_size;
		}
		int num_kernels2=num_kernels*kernel_h;
		conv2_float_level3_kernel_<<<CAFFE_GET_BLOCKS(num_kernels2),CAFFE_CUDA_NUM_THREADS>>>(num_kernels2,data_image,height,width,
			kernel,kernel_h,kernel_w,pad_h,pad_w,stride_h,stride_w,patch_h,patch_w,(float*)pool.mptr);
		//CUDA_POST_KERNEL_CHECK;
		conv2_float_sum_kernel_<<<CAFFE_GET_BLOCKS(num_kernels),CAFFE_CUDA_NUM_THREADS>>>(num_kernels,(float*)pool.mptr,num_kernels,kernel_h,r);
		//CUDA_CHECK(cudaThreadSynchronize());
	}
	//CUDA_POST_KERNEL_CHECK;
}