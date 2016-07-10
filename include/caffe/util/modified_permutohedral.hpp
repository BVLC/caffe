#ifndef CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
#define CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_

#include <cstdlib>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/hash_table.hpp"

/************************************************/
/***          ModifiedPermutohedral Lattice   ***/
/************************************************/

namespace caffe {

typedef struct MatrixEntry {
  int index;
  float weight;
} MatrixEntry;

class ModifiedPermutohedral
{
protected:
	struct Neighbors{
		int n1, n2;
		Neighbors( int n1=0, int n2=0 ):n1(n1),n2(n2){
		}
	};

  	// Check if GPU hash table if initialize
  	bool is_init;

	std::vector<int> offset_, rank_;
	std::vector<float> barycentric_;
	std::vector<Neighbors> blur_neighbors_;

	// GPU specific
	MatrixEntry *matrix;
	HashTable table;


	// Number of elements, size of sparse discretized space, dimension of features width and height
	int N_, M_, d_, w_, h_;

	void sseCompute(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
	void sseCompute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

	void seqCompute(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
	void seqCompute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

public:
	ModifiedPermutohedral();
	~ModifiedPermutohedral(){
	#ifndef CPU_ONLY
	if(is_init)
	  CUDA_CHECK(cudaFree(matrix));
	#endif
  }

  void init_cpu(const float* features, int num_dimensions, int num_points);
  void init_gpu(const float* features, int num_dimensions, int w, int h);

  void compute_cpu(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
  void compute_cpu(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

  void compute_gpu(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
  void compute_gpu(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

  void init (const float* features, int num_dimensions, int w, int h){
    switch (Caffe::mode()) {
      case Caffe::CPU:
        init_cpu(features, num_dimensions, w*h);
        break;
      #ifndef CPU_ONLY
      case Caffe::GPU:
        init_gpu(features, num_dimensions, w, h);
        is_init = true;
        break;
      #endif
      default:
        LOG(FATAL) << "Unknown caffe mode.";
    }
  }

  void init (const float* features, int num_dimensions, int num_pixels){
    switch (Caffe::mode()) {
      case Caffe::CPU:
        init_cpu(features, num_dimensions, num_pixels);
        break;
      case Caffe::GPU:
        LOG(FATAL) << "Cannot be in GPU mode for this function";
        break;
      default:
        LOG(FATAL) << "This function is only valid in CPU mode";
    }
  }  

  void compute(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const{
    switch (Caffe::mode()) {
      case Caffe::CPU:
        compute_cpu(out, in, value_size, reverse, add);
        break;
      #ifndef CPU_ONLY
      case Caffe::GPU:
        compute_gpu(out, in, value_size, reverse, add);
        break;
      #endif
      default:
        LOG(FATAL) << "Unknown caffe mode.";
    }
  }
  void compute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const{
    switch (Caffe::mode()) {
      case Caffe::CPU:
        compute_cpu(out, in, value_size, reverse, add);
        break;
      #ifndef CPU_ONLY
      case Caffe::GPU:
        compute_gpu(out, in, value_size, reverse, add);
        break;
      #endif
      default:
        LOG(FATAL) << "Unknown caffe mode.";
    }
  }

}; // class ModifiedPermutohedral
}//namespace caffe
#endif //CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_

