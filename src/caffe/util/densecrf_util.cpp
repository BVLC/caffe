#include <cstring>

#include "caffe/util/densecrf_util.hpp"

float* allocate(size_t N) {
  float * r = NULL;
  if (N>0) {
#ifdef SSE_DENSE_CRF
    r = (float*)_mm_malloc( N*sizeof(float)+16, 16 );
#else
    r = new float[N];
#endif
  }

  memset( r, 0, sizeof(float)*N);
  return r;
}

void deallocate(float*& ptr) {
  if (ptr)
#ifdef SSE_DENSE_CRF
    _mm_free( ptr );
#else
  delete[] ptr;
#endif
  ptr = NULL;
}

