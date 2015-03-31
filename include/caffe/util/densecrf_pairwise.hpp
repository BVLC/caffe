#ifndef _DENSECRF_PAIRWISE_H
#define _DENSECRF_PAIRWISE_H

#include <vector>
#include <cstdlib>

#include "caffe/util/permutohedral.hpp"

class PairwisePotential {
 public:
  virtual ~PairwisePotential();
  virtual void apply(float * out_values, const float * in_values, float * tmp, int value_size) const = 0;
};

class SemiMetricFunction {
 public:
  virtual ~SemiMetricFunction();
  // For two probabilities apply
  // the semi metric transform: v_i = sum_j mu_ij u_j
  virtual void apply(float * out_values, const float * in_values, int value_size) const = 0;
};

class PottsPotential: public PairwisePotential{
protected:
  Permutohedral lattice_;
  PottsPotential( const PottsPotential& ){}
  int N_;
  float w_;
  float *norm_;
public:
  virtual ~PottsPotential();
  PottsPotential(const float* features, int D, int N, float w, bool per_pixel_normalization=true);

  virtual void apply(float* out_values, const float* in_values, float* tmp, int value_size) const;
};

class SemiMetricPotential: public PottsPotential{
protected:
  const SemiMetricFunction * function_;
public:
  virtual ~SemiMetricPotential();
  virtual void apply(float* out_values, const float* in_values, float* tmp, int value_size) const;
  SemiMetricPotential(const float* features, int D, int N, float w, const SemiMetricFunction* function, bool per_pixel_normalization=true);
};



#endif
