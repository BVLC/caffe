#ifndef CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
#define CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_

#include <cstdlib>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>

/************************************************/
/***          ModifiedPermutohedral Lattice   ***/
/************************************************/
namespace caffe {

class ModifiedPermutohedral
{
protected:
	struct Neighbors{
		int n1, n2;
		Neighbors( int n1=0, int n2=0 ):n1(n1),n2(n2){
		}
	};
	std::vector<int> offset_, rank_;
	std::vector<float> barycentric_;
	std::vector<Neighbors> blur_neighbors_;
	// Number of elements, size of sparse discretized space, dimension of features
	int N_, M_, d_;

	void sseCompute(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
  void sseCompute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

	void seqCompute(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
	void seqCompute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;

public:
	ModifiedPermutohedral();
	void init (const float* features, int num_dimensions, int num_points);
	void compute(float* out, const float* in, int value_size, bool reverse = false, bool add = false) const;
	void compute(double* out, const double* in, int value_size, bool reverse = false, bool add = false) const;
};
}
#endif //CAFFE_MODIFIED_PERMUTOHEDRAL_HPP_
